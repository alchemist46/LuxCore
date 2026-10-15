#line 2 "sphericaltriangle_funcs.cl"

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
// Solid angle sampling of a triangle seen from a point: the spherical triangle
// sampling of J. Arvo ("Stratified Sampling of Spherical Triangles", SIGGRAPH
// 1995), in the form of pbrt-v4's SampleSphericalTriangle(). The triangle
// lights use it for the direct light of the path engines: uniform area
// sampling has 6-16x its variance on the walls of a Cornell box.
//
// It is the same code on the CPU (compiled through the OpenCL emulation, see
// src/slg/lights/sphericaltriangle.cpp) and on the GPU.
//------------------------------------------------------------------------------

// Below this solid angle the float precision of the sampling breaks down (the
// triangle is small or far: area sampling is as good there), above it the
// triangle covers most of the sphere (the pbrt-v4 bounds)
#define SPHERICALTRIANGLE_MIN_SOLIDANGLE 3e-4f
#define SPHERICALTRIANGLE_MAX_SOLIDANGLE 6.22f

// The angle between two unit vectors, accurate for small and large angles
OPENCL_FORCE_INLINE float SphericalTriangle_AngleBetween(const float3 a, const float3 b) {
	return atan2(length(cross(a, b)), dot(a, b));
}

// The solid angle of the triangle (v0, v1, v2) seen from p (A. van Oosterom
// and J. Strackee, 1983)
OPENCL_FORCE_INLINE float SphericalTriangle_SolidAngle(const float3 v0, const float3 v1,
		const float3 v2, const float3 p) {
	const float3 a = normalize(v0 - p);
	const float3 b = normalize(v1 - p);
	const float3 c = normalize(v2 - p);

	return fabs(2.f * atan2(dot(a, cross(b, c)), 1.f + dot(a, b) + dot(a, c) + dot(b, c)));
}

// True if a triangle of this solid angle is sampled by solid angle, false if
// by area. The sampling and the pdf of a direct hit must take the same branch.
OPENCL_FORCE_INLINE bool SphericalTriangle_UseSolidAngle(const float solidAngle) {
	return (solidAngle >= SPHERICALTRIANGLE_MIN_SOLIDANGLE) && (solidAngle <= SPHERICALTRIANGLE_MAX_SOLIDANGLE);
}

// The two coordinates of a point of the unit square from one sample: the even
// and the odd bits of its 24 bit fraction. The triangle lights sample their
// point from one dimension of the sampler (as Triangle_LowDiscrepancySample()
// does): a stratified 1D sample gives a stratified 2D point.
OPENCL_FORCE_INLINE void SphericalTriangle_SplitSample(const float u, float *s, float *t) {
	const uint m = min((uint)(u * 16777216.f), 16777215u);

	uint x = m & 0x555555u;
	x = (x | (x >> 1)) & 0x333333u;
	x = (x | (x >> 2)) & 0x0f0f0fu;
	x = (x | (x >> 4)) & 0xff00ffu;
	x = (x | (x >> 8)) & 0x000fffu;

	uint y = (m >> 1) & 0x555555u;
	y = (y | (y >> 1)) & 0x333333u;
	y = (y | (y >> 2)) & 0x0f0f0fu;
	y = (y | (y >> 4)) & 0xff00ffu;
	y = (y | (y >> 8)) & 0x000fffu;

	// The odd bits are the higher ones of each pair: y is the coarser coordinate
	*s = (y + .5f) * (1.f / 4096.f);
	*t = (x + .5f) * (1.f / 4096.f);
}

// Samples a direction uniformly in the solid angle of the triangle (v0, v1, v2)
// seen from p and returns the barycentric coordinates of the point it hits
// (point = b0 * v0 + b1 * v1 + b2 * v2) and the solid angle pdf. Returns false
// for a degenerate spherical triangle.
OPENCL_FORCE_INLINE bool SphericalTriangle_Sample(const float3 v0, const float3 v1, const float3 v2,
		const float3 p, const float u0, const float u1,
		float *b0, float *b1, float *b2, float *pdfW) {
	const float3 a = normalize(v0 - p);
	const float3 b = normalize(v1 - p);
	const float3 c = normalize(v2 - p);

	// The normals of the planes of the great circles through the vertices
	float3 nab = cross(a, b);
	float3 nbc = cross(b, c);
	float3 nca = cross(c, a);
	if ((dot(nab, nab) == 0.f) || (dot(nbc, nbc) == 0.f) || (dot(nca, nca) == 0.f))
		return false;
	nab = normalize(nab);
	nbc = normalize(nbc);
	nca = normalize(nca);

	// The angles at the vertices of the spherical triangle
	const float alpha = SphericalTriangle_AngleBetween(nab, -nca);
	const float beta = SphericalTriangle_AngleBetween(nbc, -nab);
	const float gamma = SphericalTriangle_AngleBetween(nca, -nbc);

	// Sample the sub-triangle area A' uniformly in [0, A]
	const float areaPlusPi = alpha + beta + gamma;
	const float area = areaPlusPi - M_PI_F;
	if (area <= 0.f)
		return false;
	*pdfW = 1.f / area;
	const float subAreaPlusPi = M_PI_F + u0 * area;

	// The cosine of the arc from a to the third vertex c' of the sub-triangle
	const float cosAlpha = cos(alpha);
	const float sinAlpha = sin(alpha);
	const float sinSub = sin(subAreaPlusPi);
	const float cosSub = cos(subAreaPlusPi);
	const float sinPhi = sinSub * cosAlpha - cosSub * sinAlpha;
	const float cosPhi = cosSub * cosAlpha + sinSub * sinAlpha;
	const float k1 = cosPhi + cosAlpha;
	const float k2 = sinPhi - sinAlpha * dot(a, b);
	const float cosBp = clamp((k2 + (k2 * cosPhi - k1 * sinPhi) * cosAlpha) /
			((k2 * sinPhi + k1 * cosPhi) * sinAlpha), -1.f, 1.f);
	const float sinBp = sqrt(fmax(0.f, 1.f - cosBp * cosBp));
	const float3 cp = cosBp * a + sinBp * normalize(c - dot(c, a) * a);

	// Sample the direction along the arc from b to c'
	const float cosTheta = 1.f - u1 * (1.f - dot(cp, b));
	const float sinTheta = sqrt(fmax(0.f, 1.f - cosTheta * cosTheta));
	const float3 w = cosTheta * b + sinTheta * normalize(cp - dot(cp, b) * b);

	// The barycentric coordinates of the hit of the ray (p, w) on the triangle
	const float3 e1 = v1 - v0;
	const float3 e2 = v2 - v0;
	const float3 s1 = cross(w, e2);
	const float divisor = dot(s1, e1);
	if (divisor == 0.f)
		return false;
	const float invDivisor = 1.f / divisor;
	const float3 s = p - v0;
	float bb1 = clamp(dot(s, s1) * invDivisor, 0.f, 1.f);
	float bb2 = clamp(dot(w, cross(s, e1)) * invDivisor, 0.f, 1.f);
	const float sum = bb1 + bb2;
	if (sum > 1.f) {
		bb1 /= sum;
		bb2 /= sum;
	}

	*b0 = 1.f - bb1 - bb2;
	*b1 = bb1;
	*b2 = bb2;

	return true;
}
