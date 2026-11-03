#line 2 "lens_funcs.cl"

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
// The lens models of the cameras, shared by the CPU (compiled through the
// OpenCL emulation, see luxrays/devices/cpudevice_oclemul.h) and the GPU
// (camera_funcs.cl):
//  - the thin lens model of PerspectiveCamera (ROADMAP F11): the optical
//    vignetting, the Seidel aberrations, the tilt of the plane of focus and
//    the dispersion;
//  - the sequential trace of RealisticCamera (ROADMAP F12) through the
//    spherical surfaces of a lens prescription.
//------------------------------------------------------------------------------

// The lens samples the optical vignetting tries before it gives the ray the
// weight 0
#define LENS_VIGNETTING_TRIES 16
// The table of the share of the aperture inside the barrel: the offset of the
// barrel from 0 to the radius of the aperture + the one of the barrel (in lens
// radii) in LENS_VIGNETTING_RADII steps, by LENS_VIGNETTING_ANGLES directions
#define LENS_VIGNETTING_RADII 64
#define LENS_VIGNETTING_ANGLES 32

// The floats of a surface of RealisticCamera, from the film to the front:
// the radius of curvature (0 for the aperture stop, a plane), the z of its
// vertex (camera space), its semi-aperture, then the medium on its object
// side: its IOR at the d line and its Cauchy A and B (B in um^2, 0 for no
// dispersion)
#define LENS_SURFACE_SIZE 6
// The exit pupil bounds of RealisticCamera: LENS_PUPIL_BUCKETS rings of the
// film, 4 floats each (x min, x max, y min, y max on the plane of the rear
// vertex, for the film points on the +x axis)
#define LENS_PUPIL_BUCKETS 64

// The Fraunhofer d, F and C lines (nm)
#define LENS_WAVELENGTH_D 587.56f
#define LENS_WAVELENGTH_F 486.13f
#define LENS_WAVELENGTH_C 656.27f

//------------------------------------------------------------------------------
// Common
//------------------------------------------------------------------------------

// The dispersion of a two term Cauchy glass at the wavelength (nm): (n - n_d) /
// (n_F - n_C), the same for any IOR and Abbe number: 0 at the d line, 0.70 at
// the F line (486 nm), -0.30 at the C line (656 nm), 2.1 at 380 nm, -0.66 at
// 780 nm
OPENCL_FORCE_INLINE float Lens_NormalizedDispersion(const float wavelength) {
	const float invD2 = 1.f / (LENS_WAVELENGTH_D * LENS_WAVELENGTH_D);
	const float invF2 = 1.f / (LENS_WAVELENGTH_F * LENS_WAVELENGTH_F);
	const float invC2 = 1.f / (LENS_WAVELENGTH_C * LENS_WAVELENGTH_C);

	return (1.f / (wavelength * wavelength) - invD2) / (invF2 - invC2);
}

// The radius of the aperture in the direction theta (radians), in lens radii:
// a regular polygon of blades sides (a circle below 3 blades), its vertices on
// the unit circle, turned by rotation (radians) and blended toward the
// circle by roundness
OPENCL_FORCE_INLINE float Lens_ApertureRadius(const uint blades, const float rotation,
		const float roundness, const float theta) {
	if (blades < 3)
		return 1.f;

	const float halfAngle = M_PI_F / blades;
	float a = theta - rotation;
	a -= (2.f * M_PI_F) * floor(a * (1.f / (2.f * M_PI_F)));

	const uint sector = (uint)(a / halfAngle);
	const float rho = (sector % 2 == 0) ? (a - sector * halfAngle) :
			((sector + 1) * halfAngle - a);
	const float r = cos(halfAngle) / cos(rho);

	return r + roundness * (1.f - r);
}

// A uniform float of the 32 bits x (the lowbias32 hash of Chris Wellons)
OPENCL_FORCE_INLINE float Lens_HashFloat(uint x) {
	x ^= x >> 16;
	x *= 0x7feb352du;
	x ^= x >> 15;
	x *= 0x846ca68bu;
	x ^= x >> 16;

	return (x >> 8) * (1.f / 16777216.f);
}

//------------------------------------------------------------------------------
// The thin lens model of PerspectiveCamera (ROADMAP F11)
//------------------------------------------------------------------------------

// The lens sample of the try (> 0) of the optical vignetting, from the first
// one: independent of it and of the other tries
OPENCL_FORCE_INLINE void Lens_VignettingRetrySample(const float u0, const float u1,
		const uint retry, float *v0, float *v1) {
	*v0 = Lens_HashFloat(as_uint(u0) ^ (retry * 0x9e3779b9u));
	*v1 = Lens_HashFloat(as_uint(u1) + retry * 0x85ebca6bu + 0x632be5abu);
}

// True if the lens point (lensX, lensY) (lens radii) passes the barrel of
// radius scale centered at (barrelX, barrelY) (lens radii)
OPENCL_FORCE_INLINE bool Lens_InsideBarrel(const float lensX, const float lensY,
		const float barrelX, const float barrelY, const float scale) {
	const float dx = lensX - barrelX;
	const float dy = lensY - barrelY;

	return dx * dx + dy * dy <= scale * scale;
}

// The share of the aperture inside the barrel centered at (barrelX, barrelY)
// (lens radii), bilinear in the table (see
// PerspectiveCamera::InitVignettingTable()) of the offsets up to maxOffset
// (the radius of the aperture + the one of the barrel)
OPENCL_FORCE_INLINE float Lens_VignettingShare(__global const float *table,
		const float barrelX, const float barrelY, const float maxOffset) {
	const float d = sqrt(barrelX * barrelX + barrelY * barrelY);
	if (d >= maxOffset)
		return 0.f;

	const float fr = d / maxOffset * (LENS_VIGNETTING_RADII - 1);
	const uint r0 = min((uint)fr, (uint)(LENS_VIGNETTING_RADII - 2));
	const float tr = fr - r0;

	float phi = atan2(barrelY, barrelX);
	if (phi < 0.f)
		phi += 2.f * M_PI_F;
	const float fa = phi * (LENS_VIGNETTING_ANGLES / (2.f * M_PI_F));
	const uint a0 = min((uint)fa, (uint)(LENS_VIGNETTING_ANGLES - 1));
	const uint a1 = (a0 + 1) % LENS_VIGNETTING_ANGLES;
	const float ta = fa - a0;

	const float v00 = table[r0 * LENS_VIGNETTING_ANGLES + a0];
	const float v01 = table[r0 * LENS_VIGNETTING_ANGLES + a1];
	const float v10 = table[(r0 + 1) * LENS_VIGNETTING_ANGLES + a0];
	const float v11 = table[(r0 + 1) * LENS_VIGNETTING_ANGLES + a1];

	return (1.f - tr) * ((1.f - ta) * v00 + ta * v01) + tr * ((1.f - ta) * v10 + ta * v11);
}

// The weight of a lens sample that passed the barrel at one of
// LENS_VIGNETTING_TRIES independent tries: share / P(one try passes)
OPENCL_FORCE_INLINE float Lens_VignettingWeight(const float share) {
	if (share <= 0.f)
		return 0.f;

	const float q = 1.f - share;
	const float q2 = q * q;
	const float q4 = q2 * q2;
	const float q8 = q4 * q4;
	const float q16 = q8 * q8;

	return share / (1.f - q16);
}

// The point the ray of the lens point lensPoint (camera space, on the plane z
// = 0) aims at, for the pinhole ray dir of the film point (camera space): on
// the plane of focus, through (0, 0, focalDistance) with the normal
// focusNormal, moved along dir by the spherical aberration, the field
// curvature, the astigmatism and focusShift (fractions of the distance), and
// across by the coma (lens radii, toward the axis: the image of a point
// moves the other way, away from the axis for a positive coma). h is the
// image height (1 at the corner of the frame) in the direction (mx, my)
OPENCL_FORCE_INLINE float3 Lens_FocusPoint(const float3 dir,
		const float lensX, const float lensY, const float lensRadius,
		const float focalDistance, const float3 focusNormal,
		const float h, const float mx, const float my,
		const float spherical, const float coma, const float astigmatism,
		const float fieldCurvature, const float focusShift) {
	// The plane of focus along the pinhole ray (at infinity if the ray does
	// not reach it)
	const float cosN = dot(focusNormal, dir);
	const float t = (cosN > 0.f) ? (focalDistance * focusNormal.z / cosN) : 1e20f;

	// The lens point in lens radii, along the meridional direction (m) and
	// across it (s)
	const float invR = (lensRadius > 0.f) ? (1.f / lensRadius) : 0.f;
	const float lm = (lensX * mx + lensY * my) * invR;
	const float ls = (lensY * mx - lensX * my) * invR;
	const float rho2 = lm * lm + ls * ls;
	const float cos2 = (rho2 > 0.f) ? (lm * lm / rho2) : 0.f;

	const float h2 = h * h;
	const float shift = spherical * rho2 + fieldCurvature * h2 + astigmatism * h2 * cos2 + focusShift;
	float3 p = dir * (t * fmax(1.f + shift, .01f));

	if (coma != 0.f) {
		// The Seidel coma: rho^2 (2 + cos(2 theta)) along m, rho^2 sin(2
		// theta) across, the focus point toward the axis for the image away
		// from it
		const float k = -coma * h * lensRadius;
		const float alongM = k * (2.f * rho2 + lm * lm - ls * ls);
		const float acrossM = k * (2.f * lm * ls);
		p.x += alongM * mx - acrossM * my;
		p.y += alongM * my + acrossM * mx;
	}

	return p;
}

//------------------------------------------------------------------------------
// RealisticCamera (ROADMAP F12)
//------------------------------------------------------------------------------

// The IOR of the medium of a surface (its object side) at the wavelength
OPENCL_FORCE_INLINE float Lens_SurfaceIOR(__global const float *surface,
		const float wavelength, const int dispersion) {
	const float cauchyB = surface[5];
	if (dispersion && (cauchyB > 0.f)) {
		const float l = wavelength * 1e-3f;
		return surface[4] + cauchyB / (l * l);
	} else
		return surface[3];
}

// Traces the ray (o, d) (camera space, d normalized) through the surfaces of
// the lens, from the film to the front (toFront) or the other way: false if a
// semi-aperture, the aperture stop or a total internal reflection stops it.
// The stop is a regular polygon (Lens_ApertureRadius()) of radius stopRadius,
// with a central obstruction of the fraction obstruction of its radius
OPENCL_FORCE_INLINE bool Lens_Trace(__global const float *surfaces, const uint surfaceCount,
		const bool toFront, const float stopRadius, const uint blades,
		const float rotation, const float roundness, const float obstruction,
		const float wavelength, const int dispersion,
		float3 *o, float3 *d) {
	float3 rayOrig = *o;
	float3 rayDir = *d;

	for (uint k = 0; k < surfaceCount; ++k) {
		const uint i = toFront ? k : (surfaceCount - 1 - k);
		__global const float *surface = &surfaces[i * LENS_SURFACE_SIZE];
		const float radius = surface[0];
		const float z = surface[1];

		float t;
		float3 center = MAKE_FLOAT3(0.f, 0.f, 0.f);
		if (radius == 0.f) {
			// The aperture stop
			if ((toFront && (rayDir.z <= 0.f)) || (!toFront && (rayDir.z >= 0.f)))
				return false;
			t = (z - rayOrig.z) / rayDir.z;
		} else {
			// A spherical surface: its center of curvature is radius away
			// from its vertex toward the film (the lens design convention,
			// with the light from the object)
			center.z = z - radius;
			const float3 oc = rayOrig - center;
			const float b = dot(oc, rayDir);
			const float c = dot(oc, oc) - radius * radius;
			const float discriminant = b * b - c;
			if (discriminant < 0.f)
				return false;

			// The numerically stable roots
			const float q = -b - ((b >= 0.f) ? 1.f : -1.f) * sqrt(discriminant);
			float t0 = q;
			float t1 = (q != 0.f) ? (c / q) : 0.f;
			if (t0 > t1) {
				const float tmp = t0;
				t0 = t1;
				t1 = tmp;
			}

			// The vertex side of the sphere: the near root for a ray toward
			// +z on a surface curved toward the object (radius < 0), and so on
			const bool nearRoot = (rayDir.z < 0.f) != (radius < 0.f);
			t = nearRoot ? t0 : t1;
		}
		if (t < 0.f)
			return false;

		const float3 p = rayOrig + rayDir * t;
		const float r2 = p.x * p.x + p.y * p.y;
		if (radius == 0.f) {
			const float edge = stopRadius * Lens_ApertureRadius(blades, rotation, roundness, atan2(p.y, p.x));
			const float inner = obstruction * edge;
			if ((r2 > edge * edge) || (r2 < inner * inner))
				return false;
		} else {
			const float semiAperture = surface[2];
			if (r2 > semiAperture * semiAperture)
				return false;
		}
		rayOrig = p;

		if (radius != 0.f) {
			float3 n = normalize(p - center);
			if (dot(n, rayDir) > 0.f)
				n = -n;

			// The media on the two sides of the surface
			const float iorObject = Lens_SurfaceIOR(surface, wavelength, dispersion);
			const float iorFilm = (i > 0) ? Lens_SurfaceIOR(&surfaces[(i - 1) * LENS_SURFACE_SIZE], wavelength, dispersion) : 1.f;
			const float eta = toFront ? (iorFilm / iorObject) : (iorObject / iorFilm);

			const float cosi = -dot(n, rayDir);
			const float k2 = 1.f - eta * eta * (1.f - cosi * cosi);
			if (k2 < 0.f)
				return false;
			rayDir = normalize(rayDir * eta + n * (eta * cosi - sqrt(k2)));
		}
	}

	*o = rayOrig;
	*d = rayDir;

	return true;
}

// The ray of RealisticCamera from the point filmPoint of the sensor (camera
// space) through the lens (camera space), its point on the exit pupil sampled
// in the bounds of the ring of the film point: the weight of the ray (the
// irradiance of the film, cos^4(theta) A / d^2, normalized to 1 at the center
// of the image), 0 if the lens stops it
OPENCL_FORCE_INLINE float RealisticCamera_LensRay(__global const RealisticCamera *camera,
		__global const float *cameraData, const float3 filmPoint,
		const float u0, const float u1, const float wavelength,
		float3 *rayOrig, float3 *rayDir) {
	const float r = sqrt(filmPoint.x * filmPoint.x + filmPoint.y * filmPoint.y);
	const uint bucket = min((uint)(r / camera->maxFilmRadius * LENS_PUPIL_BUCKETS), (uint)(LENS_PUPIL_BUCKETS - 1));
	__global const float *bounds = &cameraData[camera->pupilBoundsOffset + bucket * 4];
	const float area = (bounds[1] - bounds[0]) * (bounds[3] - bounds[2]);

	if (area > 0.f) {
		// The bounds are the ones of the film points on the +x axis
		const float px = bounds[0] + u0 * (bounds[1] - bounds[0]);
		const float py = bounds[2] + u1 * (bounds[3] - bounds[2]);
		const float cosPhi = (r > 0.f) ? (filmPoint.x / r) : 1.f;
		const float sinPhi = (r > 0.f) ? (filmPoint.y / r) : 0.f;
		const float3 pupilPoint = MAKE_FLOAT3(px * cosPhi - py * sinPhi, px * sinPhi + py * cosPhi, camera->rearZ);

		float3 o = filmPoint;
		float3 d = normalize(pupilPoint - filmPoint);
		const float cosTheta = d.z;
		if (Lens_Trace(&cameraData[camera->surfacesOffset], camera->surfaceCount, true,
				camera->stopRadius, camera->stopBlades, camera->stopRotation,
				camera->stopRoundness, camera->stopObstruction,
				wavelength, camera->dispersion, &o, &d)) {
			*rayOrig = o;
			*rayDir = d;

			const float cos2 = cosTheta * cosTheta;
			return cos2 * cos2 * area * camera->weightScale;
		}
	}

	// The ray the pixel would have through a pinhole at the entrance pupil,
	// with the weight 0
	*rayOrig = MAKE_FLOAT3(0.f, 0.f, 0.f);
	*rayDir = normalize(MAKE_FLOAT3(-filmPoint.x, -filmPoint.y, camera->focalLength));

	return 0.f;
}

// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
