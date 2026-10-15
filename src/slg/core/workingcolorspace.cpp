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

#include <cstdio>

#include <OpenColorIO/OpenColorIO.h>
namespace OCIO = OCIO_NAMESPACE;

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

void WorkingColorSpace::AddKernelParameters(vector<string> &params) const {
	if (IsRec709())
		return;

	// PARAM_WCS_FROM_REC709_<row><column>: Rec.709 to the working space, for
	// the colours the kernels compute in Rec.709 (dispersion, thin film, sky2)
	const float (&m)[3][3] = GetFromRec709Matrix();
	for (u_int y = 0; y < 3; ++y) {
		for (u_int x = 0; x < 3; ++x) {
			char buf[64];
			snprintf(buf, sizeof(buf), "-D PARAM_WCS_FROM_REC709_%u%u=%.9gf", y, x, m[y][x]);
			params.push_back(buf);
		}
	}
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

// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
