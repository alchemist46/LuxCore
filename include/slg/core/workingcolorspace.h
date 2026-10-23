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

#ifndef _SLG_WORKINGCOLORSPACE_H
#define	_SLG_WORKINGCOLORSPACE_H

#include <string>
#include <vector>

#include "luxrays/core/color/color.h"
#include "luxrays/core/kernelconstants.h"
#include "luxrays/utils/properties.h"

namespace slg {

//------------------------------------------------------------------------------
// TemperatureNormalization
//
// How the colour of a blackbody temperature (lights, emission, the blackbody
// texture) is scaled. The property "<name>.normalize" (a bool) selects NONE or
// MAX, as it always did; "<name>.normalization" ("none", "max" or
// "luminance"), if defined, takes precedence.
//------------------------------------------------------------------------------

typedef enum {
	// The RGB of the blackbody spectrum as is (very large values)
	TEMPERATURE_NORMALIZE_NONE = 0,
	// Divided by the largest RGB value of any blackbody up to 13000 K: the
	// brightness follows the temperature (luminance 0.73 at 6500 K)
	TEMPERATURE_NORMALIZE_MAX = 1,
	// Luminance 1 in the working space, as Cycles' light temperature and
	// Blackbody node: the temperature changes only the colour
	TEMPERATURE_NORMALIZE_LUMINANCE = 2
} TemperatureNormalization;

//------------------------------------------------------------------------------
// WorkingColorSpace
//
// The RGB space the light transport runs in (scene.colorspace.working), the
// same choice as Blender's per file working space. Constant RGB values in a
// scene are taken as already in the working space. The inputs with a known
// space (image maps, gamma encoded constants, OpenColorIO colour spaces) and
// the values LuxCore computes in Rec.709 (the spectral to RGB conversions:
// temperature, sun, sky2, fresnel data, dispersion, thin film) are converted
// to it.
//
// REC709 is the identity: every conversion is skipped, so the render is
// bit-identical to the builds without a working space.
//------------------------------------------------------------------------------

class WorkingColorSpace {
public:
	typedef enum {
		REC709 = 0,
		REC2020 = 1,
		ACESCG = 2
	} WorkingColorSpaceType;

	WorkingColorSpace(const WorkingColorSpaceType t = REC709) : type(t) { }

	WorkingColorSpaceType GetType() const { return type; }
	bool IsRec709() const { return type == REC709; }

	bool operator==(const WorkingColorSpace &other) const { return type == other.type; }
	bool operator!=(const WorkingColorSpace &other) const { return type != other.type; }

	// Linear Rec.709 to the working space. It returns c unchanged in REC709.
	luxrays::Spectrum FromRec709(const luxrays::Spectrum &c) const {
		return IsRec709() ? c : Mul(GetFromRec709Matrix(), c);
	}
	// The working space to Linear Rec.709. It returns c unchanged in REC709.
	luxrays::Spectrum ToRec709(const luxrays::Spectrum &c) const {
		return IsRec709() ? c : Mul(GetToRec709Matrix(), c);
	}

	// The luminance of a working space colour: RGBColor::Y() in REC709 (the
	// Rec.709 weights LuxCore has always used), the same weights applied to
	// the colour converted to Rec.709 otherwise.
	float Y(const luxrays::Spectrum &c) const {
		if (IsRec709())
			return c.Y();
		const luxrays::Spectrum &w = GetLuminanceWeights();
		return w.c[0] * c.c[0] + w.c[1] * c.c[1] + w.c[2] * c.c[2];
	}
	const luxrays::Spectrum &GetLuminanceWeights() const;

	// Weights written for Rec.709 colours (a dot product with them), as
	// weights of working space colours: ToRec709^T * w. It returns w
	// unchanged in REC709.
	luxrays::Spectrum ConvertRec709Weights(const luxrays::Spectrum &w) const;

	// Row-major 3x3 matrices (see workingcolorspace_matrices.h)
	const float (&GetFromRec709Matrix() const)[3][3];
	const float (&GetToRec709Matrix() const)[3][3];
	const float (&GetToXYZMatrix() const)[3][3];

	// The primaries and white point (CIE xy: red, green, blue, white), as the
	// OpenEXR chromaticities attribute: Rec.709 and Rec.2020 with D65, ACES
	// AP1 with the ACES white point
	const float *GetChromaticities() const;

	// The kernel parameters of the working space (none in REC709): the
	// Rec.709 to working and working to Rec.709 matrices and the luminance
	// weights
	void AddKernelParameters(std::vector<std::string> &params) const;
	// The same values in the kernel constants, for the devices that use them
	// instead of the kernel parameters
	void SetKernelConstants(luxrays::ocl::KernelConstants &kc) const;

	// The name of the working space in an OpenColorIO config, found by its
	// interop id (OCIO 2.5) or by one of its usual names. It returns an empty
	// string if the config has no such colour space.
	std::string FindOpenColorIOColorSpace(const std::string &configFileName) const;

	static WorkingColorSpaceType String2WorkingColorSpaceType(const std::string &type);
	static std::string WorkingColorSpaceType2String(const WorkingColorSpaceType type);

	// The colour of a blackbody at the temperature (Kelvin), in the working
	// space
	luxrays::Spectrum FromTemperature(const float temperature,
			const TemperatureNormalization normalization) const;

	// The normalization of a temperature from the properties
	// <prefix>.normalize and <prefix>.normalization
	static TemperatureNormalization GetTemperatureNormalization(
			const luxrays::Properties &props, const std::string &prefix);
	// The same two properties: .normalize always, .normalization only for
	// LUMINANCE (so the scenes of the other modes stay readable by the older
	// builds)
	static void AddTemperatureNormalization(luxrays::Properties &props,
			const std::string &prefix, const TemperatureNormalization normalization);
	static TemperatureNormalization String2TemperatureNormalization(const std::string &type);
	static std::string TemperatureNormalization2String(const TemperatureNormalization type);

private:
	static luxrays::Spectrum Mul(const float (&m)[3][3], const luxrays::Spectrum &c) {
		return luxrays::Spectrum(
				m[0][0] * c.c[0] + m[0][1] * c.c[1] + m[0][2] * c.c[2],
				m[1][0] * c.c[0] + m[1][1] * c.c[1] + m[1][2] * c.c[2],
				m[2][0] * c.c[0] + m[2][1] * c.c[1] + m[2][2] * c.c[2]);
	}

	WorkingColorSpaceType type;
};

}

#endif	/* _SLG_WORKINGCOLORSPACE_H */
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
