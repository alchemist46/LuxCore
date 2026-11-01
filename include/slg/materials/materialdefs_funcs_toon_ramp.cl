#line 2 "materialdefs_funcs_toon_ramp.cl"

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

//------------------------------------------------------------------------------
// The toon material, the parts shared by the CPU (toon.cpp) and the OpenCL
// kernels (see luxrays/devices/cpudevice_oclemul.h)
//
// For one light direction, the material reflects
//
//   (kd ramp + shadowColor max(1 - ramp, 0) + ks highlight + rim rimStep) / pi
//
// in place of the kd cos / pi of a matte material, the toon part: ramp is the
// colour of the ramp at the cosine of the light direction (the direction of
// the centre of the light with ToonParam::lightCenter), the shadow colour
// fills what the factor ramps leave, highlight the step of the toon specular
// on the angle between the normal and the half vector, rimStep the step of the
// rim on the angle between the eye direction and the surface. The steps are
// the ones of Cycles' Toon BSDF: 1 within an angle, falling linearly to 0 over
// smooth * pi/2 after it. The angle of the highlight is the one of the GGX
// highlight of the same roughness (ToonMaterial_HighlightAngle()), the one of
// the rim rimSize * pi/2.
//------------------------------------------------------------------------------

// From 0 below the threshold to 1 at threshold + smooth, linear in between
OPENCL_FORCE_INLINE float ToonMaterial_Step(const float x, const float threshold, const float smooth) {
	if (smooth > 0.f)
		return clamp((x - threshold) / smooth, 0.f, 1.f);
	else
		return (x >= threshold) ? 1.f : 0.f;
}

OPENCL_FORCE_INLINE ToonRampInterpolation ToonMaterial_RampInterpolation(__global const ToonParam *toon,
		const uint key) {
	return (ToonRampInterpolation)((toon->rampInterpolations >> (2 * key)) & 3);
}

// The colour of the ramp at x, the cosine of the light direction: the base
// colour below the first key, then the colour of each key, blended into the
// next one by its interpolation (a constant key steps to the next one at its
// position, over rampSmooth)
OPENCL_FORCE_INLINE float3 ToonMaterial_Ramp(__global const ToonParam *toon, const float x) {
	float3 prev = MAKE_FLOAT3(toon->rampBase[0], toon->rampBase[1], toon->rampBase[2]);
	float3 result = prev;
	for (uint i = 0; i < toon->rampCount; ++i) {
		const float3 color = MAKE_FLOAT3(toon->rampColors[3 * i], toon->rampColors[3 * i + 1],
				toon->rampColors[3 * i + 2]);
		const float position = toon->rampPositions[i];

		float w;
		const ToonRampInterpolation mode = (i == 0) ? TOON_RAMP_CONSTANT :
				ToonMaterial_RampInterpolation(toon, i - 1);
		if (mode == TOON_RAMP_CONSTANT)
			w = ToonMaterial_Step(x, position, toon->rampSmooth);
		else {
			const float start = toon->rampPositions[i - 1];
			const float span = position - start;
			w = (span > 0.f) ? clamp((x - start) / span, 0.f, 1.f) : ((x >= position) ? 1.f : 0.f);
			if (mode == TOON_RAMP_SMOOTH)
				w = w * w * (3.f - 2.f * w);
		}

		result += (color - prev) * w;
		prev = color;
	}

	return result;
}

// 1 within maxAngle, falling linearly to 0 over smooth * pi/2 after it
// (Cycles' bsdf_toon_get_intensity())
OPENCL_FORCE_INLINE float ToonMaterial_Intensity(const float angle, const float maxAngle, const float smooth) {
	const float smoothAngle = smooth * (M_PI_F * .5f);

	if (angle < maxAngle)
		return 1.f;
	else if ((smoothAngle > 0.f) && (angle < maxAngle + smoothAngle))
		return 1.f - (angle - maxAngle) / smoothAngle;
	else
		return 0.f;
}

// The angle between the normal and the half vector within which the toon
// highlight of a roughness is full: where the GGX distribution of alpha =
// roughness^2 falls to half its peak, tan^2 = alpha^2 (sqrt(2) - 1) /
// (1 - sqrt(2) alpha^2) (roughness 0.2: 1.5 degrees, 0.5: 9.6, 0.7: 21, the
// whole hemisphere from 0.917)
OPENCL_FORCE_INLINE float ToonMaterial_HighlightAngle(const float roughness) {
	const float r = clamp(roughness, 0.f, 1.f);
	const float a2 = r * r * r * r;
	const float d = 1.f - 1.41421356f * a2;

	return (d > 0.f) ? atan(sqrt(a2 * .41421356f / d)) : (M_PI_F * .5f);
}

// The step of the toon specular of a light direction and an eye direction (in
// the shading frame, eyeDir.z > 0): on the angle between the normal and the
// half vector, stretched by 1 + stretch along the U and V directions
OPENCL_FORCE_INLINE float ToonMaterial_Highlight(__global const ToonParam *toon, const float roughness,
		const float3 lightDir, const float3 eyeDir) {
	const float3 h = normalize(lightDir + eyeDir);
	const float hx = h.x / fmax(1.f + toon->specularStretchU, .05f);
	const float hy = h.y / fmax(1.f + toon->specularStretchV, .05f);

	return ToonMaterial_Intensity(atan2(sqrt(hx * hx + hy * hy), fabs(h.z)),
			ToonMaterial_HighlightAngle(roughness), toon->specularSmooth);
}

// The step of the rim on the angle between the eye direction and the surface
OPENCL_FORCE_INLINE float ToonMaterial_RimStep(__global const ToonParam *toon, const float cosEye) {
	return ToonMaterial_Intensity(asin(clamp(cosEye, 0.f, 1.f)), toon->rimSize * (M_PI_F * .5f),
			toon->rimSmooth);
}

// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
