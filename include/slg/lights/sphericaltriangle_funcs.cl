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

//------------------------------------------------------------------------------
// The quad lights: the two triangles of a planar parallelogram of one light
// mesh (every Blender area light) sampled as one light (see
// LightSourceDefinitions::Preprocess()). A point of the quad is
// c + u * ex + v * ey with (u, v) in the unit square. A rectangle is sampled
// uniformly in the solid angle it covers, as a spherical rectangle (C. Urena,
// M. Fajardo and A. King, "An Area-Preserving Parametrization for Spherical
// Rectangles", EGSR 2013, in the form of Cycles' area_light_rect_sample()),
// from a 2D sample; any other parallelogram, a rectangle far away or seen
// edge-on, and a point in the plane of the quad, by area.
//------------------------------------------------------------------------------

// The corners of a rectangle are right angles within this cosine
#define QUADLIGHT_MAX_CORNER_COS 1e-4f
// Below this solid angle the float precision of the spherical rectangle
// breaks down (its solid angle is the sum of the 4 angles minus 2 pi: 1e-4
// relative error at 0.01 sr, 5e-4 at 1e-3 sr, the 99th percentile next to the
// plane of the quad), and area sampling is as good.
// There is no upper bound (unlike the spherical triangles): close to the
// plane of the quad, in front of it, the solid angle tends to 2 pi and stays
// accurate (1e-4 at 1e-4 of the size of the quad from it), where area
// sampling has fireflies.
#define QUADLIGHT_MIN_SOLIDANGLE 1e-2f

typedef struct {
	// The frame of the rectangle: x and y along its edges, z its normal
	// pointing away from the shading point
	float3 x, y, z;
	// The rectangle in that frame, the shading point at the origin
	float x0, x1, y0, y1, z0;
	float lenU, lenV;
	// The constants of the sampling
	float b0, b1, k;
	float solidAngle;
} SphericalRectangle;

// Sets up the solid angle sampling of the quad (c, ex, ey) seen from p.
// Returns false if the quad is sampled by area: not a rectangle, p in its
// plane or a solid angle below the bound. The sampling and the pdf of a direct hit must take
// the same branch.
OPENCL_FORCE_INLINE bool SphericalRectangle_Init(SphericalRectangle *r,
		const float3 c, const float3 ex, const float3 ey, const float3 p) {
	const float lenU = length(ex);
	const float lenEy = length(ey);
	if (!(lenU > 0.f) || !(lenEy > 0.f))
		return false;
	const float3 x = ex / lenU;
	const float cosCorner = dot(x, ey) / lenEy;
	if (fabs(cosCorner) > QUADLIGHT_MAX_CORNER_COS)
		return false;
	// Exactly orthogonal to x
	const float3 eyOrtho = ey - (lenEy * cosCorner) * x;
	const float lenV = length(eyOrtho);
	const float3 y = eyOrtho / lenV;
	float3 z = cross(x, y);

	const float3 d = c - p;
	float z0 = dot(d, z);
	if (z0 > 0.f) {
		z = -z;
		z0 = -z0;
	}
	// p in the plane of the quad, within the float precision of the points
	// (a point of the light itself: the spherical rectangle of a point a
	// float error away from it covers the whole hemisphere, and the point
	// would light itself)
	const float planeEpsilon = 1e-4f * (lenU + lenEy) + 1e-6f * (fabs(p.x) + fabs(p.y) + fabs(p.z));
	if (!(-z0 > planeEpsilon))
		return false;

	const float x0 = dot(d, x);
	const float y0 = dot(d, y);
	const float x1 = x0 + lenU;
	const float y1 = y0 + lenV;

	// The normals of the planes through p and the edges (in the frame of the
	// rectangle), and the internal angles of the spherical rectangle. The
	// angles come from atan2() of the whole normals, not from acos() of their
	// dot products (Cycles), which loses 3e-4 radians next to 0 and pi: close
	// to the plane of a quad, outside of it, the solid angle was off by 1%.
	const float dx = x0 - x1;
	const float dy = y1 - y0;
	const float3 n0 = normalize(MAKE_FLOAT3(0.f, -z0 * dx, y0 * dx));
	const float3 n1 = normalize(MAKE_FLOAT3(-z0 * dy, 0.f, x1 * dy));
	const float3 n2 = normalize(MAKE_FLOAT3(0.f, z0 * dx, -y1 * dx));
	const float3 n3 = normalize(MAKE_FLOAT3(z0 * dy, 0.f, -x0 * dy));
	const float g0 = SphericalTriangle_AngleBetween(n0, -n1);
	const float g1 = SphericalTriangle_AngleBetween(n1, -n2);
	const float g2 = SphericalTriangle_AngleBetween(n2, -n3);
	const float g3 = SphericalTriangle_AngleBetween(n3, -n0);

	const float k = 2.f * M_PI_F - g2 - g3;
	const float solidAngle = g0 + g1 - k;
	// Also false for a NaN
	if (!(solidAngle >= QUADLIGHT_MIN_SOLIDANGLE))
		return false;

	r->x = x;
	r->y = y;
	r->z = z;
	r->x0 = x0;
	r->x1 = x1;
	r->y0 = y0;
	r->y1 = y1;
	r->z0 = z0;
	r->lenU = lenU;
	r->lenV = lenV;
	r->b0 = n0.z;
	r->b1 = n2.z;
	r->k = k;
	r->solidAngle = solidAngle;

	return true;
}

// Samples a direction uniformly in the spherical rectangle: returns the point
// it hits and its (u, v)
OPENCL_FORCE_INLINE float3 SphericalRectangle_Sample(const SphericalRectangle *r,
		const float3 p, const float u0, const float u1, float *u, float *v) {
	// The x of the point
	const float au = u0 * r->solidAngle + r->k;
	const float fu = (cos(au) * r->b0 - r->b1) / sin(au);
	const float cu = clamp(((fu > 0.f) ? 1.f : -1.f) / sqrt(fu * fu + r->b0 * r->b0), -1.f, 1.f);
	const float xu = clamp(-(cu * r->z0) / fmax(sqrt(1.f - cu * cu), 1e-7f), r->x0, r->x1);

	// The y of the point
	const float dd = sqrt(xu * xu + r->z0 * r->z0);
	const float h0 = r->y0 / sqrt(dd * dd + r->y0 * r->y0);
	const float h1 = r->y1 / sqrt(dd * dd + r->y1 * r->y1);
	const float hv = h0 + u1 * (h1 - h0);
	const float hv2 = hv * hv;
	const float yv = (hv2 < 1.f - 1e-6f) ? ((hv * dd) / sqrt(1.f - hv2)) : r->y1;

	*u = clamp((xu - r->x0) / r->lenU, 0.f, 1.f);
	*v = clamp((yv - r->y0) / r->lenV, 0.f, 1.f);

	return p + xu * r->x + yv * r->y + r->z0 * r->z;
}

// Samples a point of the quad (c, ex, ey) seen from p from the 2D sample
// (u0, u1): returns the point, its (u, v) and the solid angle pdf, or 0 if the
// point was sampled by area (the area pdf is then the inverse of the area of
// the quad)
OPENCL_FORCE_INLINE float QuadLight_Sample(const float3 c, const float3 ex, const float3 ey,
		const float3 p, const float u0, const float u1,
		float3 *point, float *u, float *v) {
	SphericalRectangle r;
	if (SphericalRectangle_Init(&r, c, ex, ey, p)) {
		*point = SphericalRectangle_Sample(&r, p, u0, u1, u, v);
		return 1.f / r.solidAngle;
	}

	*u = u0;
	*v = u1;
	*point = c + u0 * ex + u1 * ey;
	return 0.f;
}

// The solid angle pdf of QuadLight_Sample() from p, or 0 if it samples by area
OPENCL_FORCE_INLINE float QuadLight_SolidAnglePdf(const float3 c, const float3 ex, const float3 ey,
		const float3 p) {
	SphericalRectangle r;
	return SphericalRectangle_Init(&r, c, ex, ey, p) ? (1.f / r.solidAngle) : 0.f;
}

// The corner c and the edges ex, ey of the quad of the triangle (p0, p1, p2)
// whose vertex number corner is c (the other two are c + ex and c + ey)
OPENCL_FORCE_INLINE void QuadLight_GetEdges(const float3 p0, const float3 p1, const float3 p2,
		const uint corner, float3 *c, float3 *ex, float3 *ey) {
	if (corner == 0) {
		*c = p0;
		*ex = p1 - p0;
		*ey = p2 - p0;
	} else if (corner == 1) {
		*c = p1;
		*ex = p2 - p1;
		*ey = p0 - p1;
	} else {
		*c = p2;
		*ex = p0 - p2;
		*ey = p1 - p2;
	}
}

// The barycentric coordinates b1, b2 of a triangle whose vertex number corner
// has the weight w0 and the next two (in their cyclic order) w1 and w2
OPENCL_FORCE_INLINE void QuadLight_Barycentrics(const uint corner,
		const float w0, const float w1, const float w2, float *b1, float *b2) {
	if (corner == 0) {
		*b1 = w1;
		*b2 = w2;
	} else if (corner == 1) {
		*b1 = w0;
		*b2 = w1;
	} else {
		*b1 = w2;
		*b2 = w0;
	}
}

// The triangle of the point (u, v) of a quad and its barycentric coordinates
// b1, b2: the first triangle (c, c + ex, c + ey) has the corner c at its
// vertex number corner, the second one (c + ex + ey, c + ey, c + ex) has
// c + ex + ey at its vertex number partnerCorner. Returns true for the second.
OPENCL_FORCE_INLINE bool QuadLight_GetTriangle(const float u, const float v,
		const uint corner, const uint partnerCorner, float *b1, float *b2) {
	if (u + v <= 1.f) {
		QuadLight_Barycentrics(corner, 1.f - u - v, u, v, b1, b2);
		return false;
	} else {
		QuadLight_Barycentrics(partnerCorner, u + v - 1.f, 1.f - u, 1.f - v, b1, b2);
		return true;
	}
}
