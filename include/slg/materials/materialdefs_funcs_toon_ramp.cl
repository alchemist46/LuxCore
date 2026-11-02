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
//   (kd baseWeight ramp + shadowColor max(1 - ramp, 0)
//     + ks specularWeight specularRamp + rim rimWeight rimStep rimRamp) / pi
//
// in place of the kd cos / pi of a matte material, the toon part (the
// parameters of Redshift's Toon material): ramp is the colour of the base
// ramp at the cosine of the light direction (the direction of the centre of
// the light with ToonParam::lightCenter), the shadow colour fills what the
// factor ramps leave, specularRamp the colour of the specular ramp at the
// intensity of the glossy highlight of the roughness
// (ToonMaterial_GlossyIntensity(), Redshift's reflection tonemap), rimStep the
// step of the rim on the angle between the eye direction and the surface and
// rimRamp the colour of the rim ramp across it (1 without one). The steps are
// the ones of Cycles' Toon BSDF: 1 within an angle, falling linearly to 0 over
// smooth * pi/2 after it; the one of the rim is within rimSize * pi/2. Without
// a specular ramp, the highlight is the step on the angle between the normal
// and the half vector, within the angle where the GGX highlight of the
// roughness falls to half its peak (ToonMaterial_HighlightAngle()).
//------------------------------------------------------------------------------

// From 0 below the threshold to 1 at threshold + smooth, linear in between
OPENCL_FORCE_INLINE float ToonMaterial_Step(const float x, const float threshold, const float smooth) {
	if (smooth > 0.f)
		return clamp((x - threshold) / smooth, 0.f, 1.f);
	else
		return (x >= threshold) ? 1.f : 0.f;
}

OPENCL_FORCE_INLINE ToonRampInterpolation ToonMaterial_RampInterpolation(__global const ToonRampParam *ramp,
		const uint key) {
	return (ToonRampInterpolation)((ramp->interpolations >> (2 * key)) & 3);
}

// The colour of the ramp at x: the base colour below the first key, then the
// colour of each key, blended into the next one by its interpolation (a
// constant key steps to the next one at its position, over smooth)
OPENCL_FORCE_INLINE float3 ToonMaterial_Ramp(__global const ToonRampParam *ramp, const float x) {
	float3 prev = MAKE_FLOAT3(ramp->base[0], ramp->base[1], ramp->base[2]);
	float3 result = prev;
	for (uint i = 0; i < ramp->count; ++i) {
		const float3 color = MAKE_FLOAT3(ramp->colors[3 * i], ramp->colors[3 * i + 1],
				ramp->colors[3 * i + 2]);
		const float position = ramp->positions[i];

		float w;
		const ToonRampInterpolation mode = (i == 0) ? TOON_RAMP_CONSTANT :
				ToonMaterial_RampInterpolation(ramp, i - 1);
		if (mode == TOON_RAMP_CONSTANT)
			w = ToonMaterial_Step(x, position, ramp->smooth);
		else {
			const float start = ramp->positions[i - 1];
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

// The intensity of the glossy highlight of a light direction and an eye
// direction (in the shading frame, z > 0), the input of the specular ramp
// (Redshift's "specular intensity of the lighting"): the light the GGX
// microfacet reflection of the roughness reflects, without the Fresnel, times
// pi, so that 1 is the light of a white matte surface facing the light:
//
//   pi D(h) G2 / (4 cos(eye)) = G2 / (4 ax ay (hx^2/ax^2 + hy^2/ay^2 + hz^2)^2 cos(eye))
//
// with alpha = roughness^2 stretched by 1 + stretch along the U and V
// directions and Smith's height-correlated G2. Its peak, facing the light and
// the eye, is 1 / (4 alpha^2): a small roughness has a small bright highlight,
// a large one a wide dim one (roughness 0.5: 4 at the peak, 0.84: 0.5, 1: 0.25).
//
// lightDir is the direction of the centre of a light of angular radius
// lightRadius: the highlight is the reflection of the whole light, the lobe
// widened by the light as for the sphere lights of Karis, "Real Shading in
// Unreal Engine 4" (2013). The half vector is turned towards the normal by
// spread, the radius of the half vectors of the light, lightRadius / (2
// sqrt(h.eye)) (dw_h = dw_light / (4 h.eye)): the direction of the light
// nearest the mirror one. The intensity is times ax ay / sqrt((ax^2 +
// spread^2) (ay^2 + spread^2)), so that a mirror reflects a disc of the
// light's size and of the light's intensity, about 1 / lightRadius^2 = pi /
// (its solid angle). Against the average over the light disc (numerically):
// within 10-20% at its centre, down to roughness 0 and at grazing eye
// directions.
OPENCL_FORCE_INLINE float ToonMaterial_GlossyIntensity(__global const ToonParam *toon, const float roughness,
		const float lightRadius, const float3 lightDir, const float3 eyeDir) {
	const float r = clamp(roughness, 0.f, 1.f);
	const float alpha = fmax(r * r, 1e-3f);
	const float ax = alpha * fmax(1.f + toon->specularStretchU, .05f);
	const float ay = alpha * fmax(1.f + toon->specularStretchV, .05f);

	float3 h = normalize(lightDir + eyeDir);
	float spread = 0.f;
	if (lightRadius > 0.f) {
		spread = .5f * lightRadius / sqrt(fmax(dot(h, eyeDir), 1e-4f));
		const float sinH = sqrt(h.x * h.x + h.y * h.y);
		const float theta = fmax(atan2(sinH, h.z) - spread, 0.f);
		const float scale = (sinH > 0.f) ? (sin(theta) / sinH) : 0.f;
		h = MAKE_FLOAT3(h.x * scale, h.y * scale, cos(theta));
	}
	const float hx = h.x / ax;
	const float hy = h.y / ay;
	const float e = hx * hx + hy * hy + h.z * h.z;

	// Smith's lambda of the two directions
	const float lx = ax * lightDir.x, ly = ay * lightDir.y;
	const float ex = ax * eyeDir.x, ey = ay * eyeDir.y;
	const float lambdaLight = .5f * (sqrt(1.f + (lx * lx + ly * ly) / (lightDir.z * lightDir.z)) - 1.f);
	const float lambdaEye = .5f * (sqrt(1.f + (ex * ex + ey * ey) / (eyeDir.z * eyeDir.z)) - 1.f);

	const float spread2 = spread * spread;
	return 1.f / (4.f * sqrt((ax * ax + spread2) * (ay * ay + spread2)) * e * e *
			(1.f + lambdaLight + lambdaEye) * eyeDir.z);
}

// The toon specular of a light direction and an eye direction (in the shading
// frame, eyeDir.z > 0), times its weight: the colour of the specular ramp at
// the intensity of the glossy highlight of the light (of angular radius
// lightRadius), or without a ramp the step on the angle between the normal
// and the half vector, stretched by 1 + stretch along the U and V directions
OPENCL_FORCE_INLINE float3 ToonMaterial_Highlight(__global const ToonParam *toon, const float roughness,
		const float lightRadius, const float3 lightDir, const float3 eyeDir) {
	if (toon->specularRamp.count > 0) {
		if (lightDir.z <= 0.f)
			return MAKE_FLOAT3(0.f, 0.f, 0.f);

		const float t = ToonMaterial_GlossyIntensity(toon, roughness, lightRadius, lightDir, eyeDir);
		return ToonMaterial_Ramp(&toon->specularRamp, fmin(t, 1.f)) * toon->specularWeight;
	}

	const float3 h = normalize(lightDir + eyeDir);
	const float hx = h.x / fmax(1.f + toon->specularStretchU, .05f);
	const float hy = h.y / fmax(1.f + toon->specularStretchV, .05f);
	const float step = ToonMaterial_Intensity(atan2(sqrt(hx * hx + hy * hy), fabs(h.z)),
			ToonMaterial_HighlightAngle(roughness), toon->specularSmooth) * toon->specularWeight;

	return MAKE_FLOAT3(step, step, step);
}

// The rim: its step on the angle between the eye direction and the surface,
// times the colour of the rim ramp across it (0 at its inner edge, 1 at the
// silhouette)
OPENCL_FORCE_INLINE float3 ToonMaterial_Rim(__global const ToonParam *toon, const float cosEye) {
	const float angle = asin(clamp(cosEye, 0.f, 1.f));
	const float width = toon->rimSize * (M_PI_F * .5f);
	const float step = ToonMaterial_Intensity(angle, width, toon->rimSmooth) * toon->rimWeight;
	if (step <= 0.f)
		return MAKE_FLOAT3(0.f, 0.f, 0.f);

	if (toon->rimRamp.count > 0) {
		const float t = (width > 0.f) ? clamp(1.f - angle / width, 0.f, 1.f) : 0.f;
		return ToonMaterial_Ramp(&toon->rimRamp, t) * step;
	} else
		return MAKE_FLOAT3(step, step, step);
}

// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
