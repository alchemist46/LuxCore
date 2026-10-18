#line 2 "sheen_funcs.cl"

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
// The microfiber sheen lobe (Zeltner, Burley & Chiang 2022), as Blender Cycles'
// Sheen BSDF (bsdf_sheen.h): f(eye, light) |cos light| = albedo(mu_eye, r) *
// LTC(light), the LTC oriented by the eye direction. The LTC is a normalized
// distribution, so it is also the pdf of Sheen_SampleLTC(). Pure functions on
// local directions (the shading normal is z, both directions above the
// surface), shared by the CPU (see sheen.cpp) and the GPU; the parameters come
// from SheenLTC_Lookup() (sheen_ltc_tables.cl).
//------------------------------------------------------------------------------

// The frame of the LTC (Cycles' make_orthonormals_safe_tangent(N, wi)): T is
// the tangential part of the eye direction, B = N x T
OPENCL_FORCE_INLINE void Sheen_Frame(const float3 eyeDir, float3 *T, float3 *B) {
	const float len = sqrt(eyeDir.x * eyeDir.x + eyeDir.y * eyeDir.y);
	if (len > 1e-6f) {
		*T = MAKE_FLOAT3(eyeDir.x / len, eyeDir.y / len, 0.f);
		*B = MAKE_FLOAT3(-eyeDir.y / len, eyeDir.x / len, 0.f);
	} else {
		*T = MAKE_FLOAT3(1.f, 0.f, 0.f);
		*B = MAKE_FLOAT3(0.f, 1.f, 0.f);
	}
}

// The LTC density of lightDir, f |cos light| / albedo
OPENCL_FORCE_INLINE float Sheen_LTC(const float a, const float b,
		const float3 eyeDir, const float3 lightDir) {
	if (lightDir.z <= 0.f)
		return 0.f;

	float3 T, B;
	Sheen_Frame(eyeDir, &T, &B);
	const float ox = lightDir.x * T.x + lightDir.y * T.y;
	const float oy = lightDir.x * B.x + lightDir.y * B.y;
	const float oz = lightDir.z;

	const float u = a * ox + b * oz;
	const float v = a * oy;
	const float lenSqr = u * u + v * v + oz * oz;
	const float ratio = a / lenSqr;

	return M_1_PI_F * oz * ratio * ratio;
}

// A light direction distributed as the LTC (it can be below the surface,
// then the caller rejects it)
OPENCL_FORCE_INLINE float3 Sheen_SampleLTC(const float a, const float b,
		const float3 eyeDir, const float u0, const float u1) {
	// A uniform point of the unit disk (Shirley-Chiu concentric mapping,
	// which keeps the stratification of the samples)
	const float sx = 2.f * u0 - 1.f;
	const float sy = 2.f * u1 - 1.f;
	float dx = 0.f, dy = 0.f;
	if ((sx != 0.f) || (sy != 0.f)) {
		float r, theta;
		if (fabs(sx) > fabs(sy)) {
			r = sx;
			theta = (M_PI_F / 4.f) * (sy / sx);
		} else {
			r = sy;
			theta = (M_PI_F / 2.f) - (M_PI_F / 4.f) * (sx / sy);
		}
		dx = r * cos(theta);
		dy = r * sin(theta);
	}
	const float diskZ = sqrt(fmax(0.f, 1.f - dx * dx - dy * dy));

	// The cosine-distributed direction mapped by the LTC
	float ox = dx - diskZ * b;
	float oy = dy;
	float oz = diskZ * a;
	const float len = sqrt(ox * ox + oy * oy + oz * oz);
	ox /= len;
	oy /= len;
	oz /= len;

	float3 T, B;
	Sheen_Frame(eyeDir, &T, &B);
	return MAKE_FLOAT3(ox * T.x + oy * B.x, ox * T.y + oy * B.y, oz);
}
