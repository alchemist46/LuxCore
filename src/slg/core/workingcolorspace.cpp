/***************************************************************************
 * Copyright 1998-2026 by authors (see AUTHORS.txt)                        *
 *                                                                         *
 *   This file is part of LuxCoreRender.                                   *
 *                                                                         *
 * Licensed under the Apache License, Version 2.0 (the "License");         *
 * you may not use this file except in compliance with the License.        *
 * You may obtain a copy of the License at                                 *
 *                                                                         *
 *     http://www.apache.org/licenses/LICENSE-2.0                          *
 *                                                                         *
 * Unless required by applicable law or agreed to in writing, software     *
 * distributed under the License is distributed on an "AS IS" BASIS,       *
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.*
 * See the License for the specific language governing permissions and     *
 * limitations under the License.                                          *
 ***************************************************************************/

#include <cmath>
#include <cstdio>

#include <OpenColorIO/OpenColorIO.h>
namespace OCIO = OCIO_NAMESPACE;

#include "luxrays/core/color/spds/blackbodyspd.h"
#include "luxrays/core/geometry/matrix3x3.h"
#include "luxrays/utils/strutils.h"
#include "slg/core/workingcolorspace.h"
#include "slg/core/workingcolorspace_matrices.h"
#include "slg/utils/filenameresolver.h"

using namespace std;
using namespace luxrays;
using namespace slg;

//------------------------------------------------------------------------------
// WorkingColorSpace
//------------------------------------------------------------------------------

const float (&WorkingColorSpace::GetFromRec709Matrix() const)[3][3] {
	return workingcolorspace::FromRec709[type];
}

const float (&WorkingColorSpace::GetToRec709Matrix() const)[3][3] {
	return workingcolorspace::ToRec709[type];
}

const float (&WorkingColorSpace::GetToXYZMatrix() const)[3][3] {
	return workingcolorspace::ToXYZ[type];
}

// The Rec.709 luminance weights of RGBColor::Y() and Spectrum_Y()
static const double rec709LuminanceWeights[3] = { 0.212671, 0.715160, 0.072169 };

static Spectrum ConvertWeights(const float (&toRec709)[3][3], const double (&w)[3]) {
	// (ToRec709^T * w)[x] = sum over y of ToRec709[y][x] * w[y]
	double result[3];
	for (u_int x = 0; x < 3; ++x)
		result[x] = toRec709[0][x] * w[0] + toRec709[1][x] * w[1] + toRec709[2][x] * w[2];

	return Spectrum((float)result[0], (float)result[1], (float)result[2]);
}

const Spectrum &WorkingColorSpace::GetLuminanceWeights() const {
	static const Spectrum weights[3] = {
		Spectrum(0.212671f, 0.715160f, 0.072169f),
		ConvertWeights(workingcolorspace::ToRec709[REC2020], rec709LuminanceWeights),
		ConvertWeights(workingcolorspace::ToRec709[ACESCG], rec709LuminanceWeights)
	};

	return weights[type];
}

Spectrum WorkingColorSpace::ConvertRec709Weights(const Spectrum &w) const {
	if (IsRec709())
		return w;

	const double wd[3] = { w.c[0], w.c[1], w.c[2] };
	return ConvertWeights(GetToRec709Matrix(), wd);
}

const float *WorkingColorSpace::GetChromaticities() const {
	static const float chromaticities[3][8] = {
		// Rec.709 / sRGB, D65
		{ .64f, .33f, .30f, .60f, .15f, .06f, .3127f, .3290f },
		// Rec.2020, D65
		{ .708f, .292f, .170f, .797f, .131f, .046f, .3127f, .3290f },
		// ACES AP1, ACES white
		{ .713f, .293f, .165f, .830f, .128f, .044f, .32168f, .33767f }
	};

	return chromaticities[type];
}

static void AddMatrixKernelParameters(vector<string> &params, const string &name, const float (&m)[3][3]) {
	for (u_int y = 0; y < 3; ++y) {
		for (u_int x = 0; x < 3; ++x) {
			char buf[64];
			snprintf(buf, sizeof(buf), "-D %s_%u%u=%.9gf", name.c_str(), y, x, m[y][x]);
			params.push_back(buf);
		}
	}
}

void WorkingColorSpace::AddKernelParameters(vector<string> &params) const {
	if (IsRec709())
		return;

	// PARAM_WCS_FROM_REC709_<row><column>: Rec.709 to the working space, for
	// the colours the kernels compute in Rec.709 (dispersion, thin film, sky2)
	AddMatrixKernelParameters(params, "PARAM_WCS_FROM_REC709", GetFromRec709Matrix());
	// PARAM_WCS_TO_REC709_<row><column>: the working space to Rec.709, for the
	// formulas written for Rec.709 colours (the Disney tint)
	AddMatrixKernelParameters(params, "PARAM_WCS_TO_REC709", GetToRec709Matrix());

	// PARAM_WCS_Y_R/G/B: the luminance weights of Spectrum_Y()
	const Spectrum &w = GetLuminanceWeights();
	const char *names[3] = { "R", "G", "B" };
	for (u_int i = 0; i < 3; ++i) {
		char buf[64];
		snprintf(buf, sizeof(buf), "-D PARAM_WCS_Y_%s=%.9gf", names[i], w.c[i]);
		params.push_back(buf);
	}
}

void WorkingColorSpace::SetKernelConstants(luxrays::ocl::KernelConstants &kc) const {
	kc.wcsIsRec709 = IsRec709() ? 1u : 0u;

	const float (&from)[3][3] = GetFromRec709Matrix();
	const float (&to)[3][3] = GetToRec709Matrix();
	for (u_int y = 0; y < 3; ++y) {
		for (u_int x = 0; x < 3; ++x) {
			kc.wcsFromRec709[y * 3 + x] = from[y][x];
			kc.wcsToRec709[y * 3 + x] = to[y][x];
		}
	}

	const Spectrum &w = GetLuminanceWeights();
	for (u_int i = 0; i < 3; ++i)
		kc.wcsY[i] = w.c[i];
}

string WorkingColorSpace::FindOpenColorIOColorSpace(const string &configFileName) const {
	static const char *interopIDs[] = {
		"lin_rec709_scene", "lin_rec2020_scene", "lin_ap1_scene"
	};
	// Blender's names first, then the ones of the ACES and older configs
	static const vector<string> names[] = {
		{ "Linear Rec.709", "Linear Rec.709 (sRGB)", "lin_rec709", "Utility - Linear - sRGB" },
		{ "Linear Rec.2020", "lin_rec2020", "Utility - Linear - Rec.2020" },
		{ "ACEScg", "lin_ap1", "ACES - ACEScg" }
	};

	try {
		OCIO::ConstConfigRcPtr config = (configFileName == "") ?
			OCIO::GetCurrentConfig() :
			OCIO::Config::CreateFromFile(SLG_FileNameResolver.ResolveFile(configFileName).c_str());

		const int count = config->getNumColorSpaces(OCIO::SEARCH_REFERENCE_SPACE_SCENE, OCIO::COLORSPACE_ALL);
		for (int i = 0; i < count; ++i) {
			const char *name = config->getColorSpaceNameByIndex(OCIO::SEARCH_REFERENCE_SPACE_SCENE, OCIO::COLORSPACE_ALL, i);
			OCIO::ConstColorSpaceRcPtr cs = config->getColorSpace(name);
			if (cs && (string(cs->getInteropID()) == interopIDs[type]))
				return name;
		}

		// getColorSpace() resolves the aliases too
		for (auto const &name : names[type]) {
			OCIO::ConstColorSpaceRcPtr cs = config->getColorSpace(name.c_str());
			if (cs)
				return cs->getName();
		}
	} catch (OCIO::Exception &exception) {
		throw runtime_error("OpenColorIO Error in WorkingColorSpace::FindOpenColorIOColorSpace(): " + string(exception.what()));
	}

	return "";
}

WorkingColorSpace::WorkingColorSpaceType WorkingColorSpace::String2WorkingColorSpaceType(const string &type) {
	if (type == "rec709")
		return REC709;
	else if (type == "rec2020")
		return REC2020;
	else if (type == "acescg")
		return ACESCG;
	else
		throw runtime_error("Unknown working color space type: " + type);
}

string WorkingColorSpace::WorkingColorSpaceType2String(const WorkingColorSpaceType type) {
	switch (type) {
		case REC709:
			return "rec709";
		case REC2020:
			return "rec2020";
		case ACESCG:
			return "acescg";
		default:
			throw runtime_error("Unknown working color space type in WorkingColorSpace::WorkingColorSpaceType2String(): " + ToString(type));
	}
}

//------------------------------------------------------------------------------
// Blackbody temperatures
//------------------------------------------------------------------------------

// CIE XYZ to Linear Rec.709: the inverse of the ToXYZ matrix of REC709
namespace {
struct XYZToRec709Matrix {
	XYZToRec709Matrix() {
		double toXYZ[3][3];
		for (u_int y = 0; y < 3; ++y)
			for (u_int x = 0; x < 3; ++x)
				toXYZ[y][x] = workingcolorspace::ToXYZ[WorkingColorSpace::REC709][y][x];

		if (!Invert3x3(toXYZ, m))
			throw runtime_error("Internal error in XYZToRec709Matrix(): singular matrix");
	}

	double m[3][3];
};
}

Spectrum WorkingColorSpace::FromTemperature(const float temperature,
		const TemperatureNormalization normalization) const {
	if (normalization != TEMPERATURE_NORMALIZE_LUMINANCE)
		return FromRec709(TemperatureToWhitePoint(temperature, normalization == TEMPERATURE_NORMALIZE_MAX));

	// As Cycles' blackbody: the Rec.709 colour of the spectrum (not
	// desaturated into the gamut) scaled to luminance 1, converted to the
	// working space, its negative components (outside the gamut of the
	// working space) set to 0
	BlackbodySPD spd(temperature);
	const XYZColor xyz = spd.ToXYZ();

	static const XYZToRec709Matrix xyzToRec709;
	const double (&m)[3][3] = xyzToRec709.m;
	double rgb[3];
	for (u_int y = 0; y < 3; ++y)
		rgb[y] = m[y][0] * xyz.c[0] + m[y][1] * xyz.c[1] + m[y][2] * xyz.c[2];

	// The luminance of RGBColor::Y(), so the working space luminance is 1
	const double y = rec709LuminanceWeights[0] * rgb[0] + rec709LuminanceWeights[1] * rgb[1] +
			rec709LuminanceWeights[2] * rgb[2];
	// No light (a temperature of 0 K or close to it)
	if (!(y > 0.0) || !std::isfinite(y))
		return Spectrum(0.f);

	const Spectrum c = FromRec709(Spectrum((float)(rgb[0] / y), (float)(rgb[1] / y), (float)(rgb[2] / y)));
	return c.Clamp(0.f, INFINITY);
}

TemperatureNormalization WorkingColorSpace::GetTemperatureNormalization(
		const Properties &props, const string &prefix) {
	if (props.IsDefined(prefix + ".normalization"))
		return String2TemperatureNormalization(props.Get(prefix + ".normalization").Get<string>());

	return props.Get(Property(prefix + ".normalize")(false)).Get<bool>() ?
		TEMPERATURE_NORMALIZE_MAX : TEMPERATURE_NORMALIZE_NONE;
}

void WorkingColorSpace::AddTemperatureNormalization(Properties &props,
		const string &prefix, const TemperatureNormalization normalization) {
	props.Set(Property(prefix + ".normalize")(normalization != TEMPERATURE_NORMALIZE_NONE));
	if (normalization == TEMPERATURE_NORMALIZE_LUMINANCE)
		props.Set(Property(prefix + ".normalization")(TemperatureNormalization2String(normalization)));
}

TemperatureNormalization WorkingColorSpace::String2TemperatureNormalization(const string &type) {
	if (type == "none")
		return TEMPERATURE_NORMALIZE_NONE;
	else if (type == "max")
		return TEMPERATURE_NORMALIZE_MAX;
	else if (type == "luminance")
		return TEMPERATURE_NORMALIZE_LUMINANCE;
	else
		throw runtime_error("Unknown temperature normalization: " + type);
}

string WorkingColorSpace::TemperatureNormalization2String(const TemperatureNormalization type) {
	switch (type) {
		case TEMPERATURE_NORMALIZE_NONE:
			return "none";
		case TEMPERATURE_NORMALIZE_MAX:
			return "max";
		case TEMPERATURE_NORMALIZE_LUMINANCE:
			return "luminance";
		default:
			throw runtime_error("Unknown temperature normalization in WorkingColorSpace::TemperatureNormalization2String(): " + ToString(type));
	}
}

// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
