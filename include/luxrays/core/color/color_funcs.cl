#line 2 "color_funcs.cl"

/***************************************************************************
 * Copyright 1998-2020 by authors (see AUTHORS.txt)                        *
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

OPENCL_FORCE_INLINE bool Spectrum_IsEqual(const float3 a, const float3 b) {
	return all(isequal(a, b));
}

OPENCL_FORCE_INLINE bool Spectrum_IsBlack(const float3 a) {
	return Spectrum_IsEqual(a, BLACK);
}

OPENCL_FORCE_INLINE bool Spectrum_IsNan(const float3 a) {
	return any(isnan(a));
}

OPENCL_FORCE_INLINE bool Spectrum_IsInf(const float3 a) {
	return any(isinf(a));
}

OPENCL_FORCE_INLINE bool Spectrum_IsNanOrInf(const float3 a) {
	return Spectrum_IsNan(a) || Spectrum_IsInf(a);
}

OPENCL_FORCE_INLINE float Spectrum_Filter(const float3 s)  {
	return (s.x + s.y + s.z) * 0.33333333f;
}

// The luminance of a colour in the working colour space (see
// WorkingColorSpace::Y())
OPENCL_FORCE_INLINE float Spectrum_Y(const float3 s) {
#if defined(PARAM_WCS_Y_R)
	return PARAM_WCS_Y_R * s.x + PARAM_WCS_Y_G * s.y + PARAM_WCS_Y_B * s.z;
#else
	return 0.212671f * s.x + 0.715160f * s.y + 0.072169f * s.z;
#endif
}

OPENCL_FORCE_INLINE float3 Spectrum_Clamp(const float3 s) {
	return clamp(s, BLACK, WHITE);
}

OPENCL_FORCE_INLINE float3 Spectrum_Exp(const float3 s) {
	return MAKE_FLOAT3(exp(s.x), exp(s.y), exp(s.z));
}

OPENCL_FORCE_INLINE float3 Spectrum_Pow(const float3 s, const float e) {
	return MAKE_FLOAT3(pow(s.x, e), pow(s.y, e), pow(s.z, e));
}

// Linear Rec.709 to the working colour space, for the colours the kernels
// compute in Rec.709 (dispersion, thin film, sky2). The matrix is set by the
// host (WorkingColorSpace::AddKernelParameters()), only when the working space
// is not Rec.709.
#if defined(PARAM_WCS_FROM_REC709_00)
OPENCL_FORCE_INLINE float3 WorkingColorSpace_FromRec709(const float3 c) {
	return MAKE_FLOAT3(
		PARAM_WCS_FROM_REC709_00 * c.x + PARAM_WCS_FROM_REC709_01 * c.y + PARAM_WCS_FROM_REC709_02 * c.z,
		PARAM_WCS_FROM_REC709_10 * c.x + PARAM_WCS_FROM_REC709_11 * c.y + PARAM_WCS_FROM_REC709_12 * c.z,
		PARAM_WCS_FROM_REC709_20 * c.x + PARAM_WCS_FROM_REC709_21 * c.y + PARAM_WCS_FROM_REC709_22 * c.z);
}
#else
#define WorkingColorSpace_FromRec709(c) (c)
#endif

// The working colour space to Linear Rec.709, for the formulas written for
// Rec.709 colours
#if defined(PARAM_WCS_TO_REC709_00)
OPENCL_FORCE_INLINE float3 WorkingColorSpace_ToRec709(const float3 c) {
	return MAKE_FLOAT3(
		PARAM_WCS_TO_REC709_00 * c.x + PARAM_WCS_TO_REC709_01 * c.y + PARAM_WCS_TO_REC709_02 * c.z,
		PARAM_WCS_TO_REC709_10 * c.x + PARAM_WCS_TO_REC709_11 * c.y + PARAM_WCS_TO_REC709_12 * c.z,
		PARAM_WCS_TO_REC709_20 * c.x + PARAM_WCS_TO_REC709_21 * c.y + PARAM_WCS_TO_REC709_22 * c.z);
}
#else
#define WorkingColorSpace_ToRec709(c) (c)
#endif

OPENCL_FORCE_INLINE float3 Spectrum_Sqrt(const float3 s) {
	return MAKE_FLOAT3(sqrt(s.x), sqrt(s.y), sqrt(s.z));
}

OPENCL_FORCE_INLINE float3 Spectrum_ScaledClamp(const float3 c, const float low, const float high) {
	float3 ret = c;

	const float maxValue = fmax(c.x, fmax(c.y, c.z));
	if (maxValue > 0.f) {
		if (maxValue > high) {
			const float scale = high / maxValue;
			ret *= scale;
		}

		if (maxValue < low) {
			const float scale = low / maxValue;
			ret *= scale;
		}
	}

	return ret;
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
