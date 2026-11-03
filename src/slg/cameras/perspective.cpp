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

#include <cstddef>

#include "slg/cameras/perspective.h"
#include "slg/film/film.h"
#include "slg/core/sdl.h"
#include "slg/scene/scene.h"

// The lens model shared with the GPU (camera_funcs.cl), compiled through the
// OpenCL emulation (see luxrays/devices/cpudevice_oclemul.h)
namespace slg { namespace ocl { namespace {
#include "luxrays/devices/cpudevice_oclemul.h"
#include "slg/cameras/lens_funcs.cl"
} } }

using namespace std;
using namespace luxrays;
using namespace slg;

//------------------------------------------------------------------------------
// PerspectiveCamera
//------------------------------------------------------------------------------

PerspectiveCamera::PerspectiveCamera(const Point &o, const Point &t,
		const Vector &u, const float *region) :
		ProjectiveCamera(PERSPECTIVE, region, o, t, u),
		screenOffsetX(0.f), screenOffsetY(0.f), fieldOfView(45.f),
		bokehBlades(0), bokehPower(0), bokehDistribution(DIST_EXPONENTIAL),
		bokehDistributionImageMap(nullptr), bokehDistributionMap(nullptr),
		bokehScaleX(1.f), bokehScaleY(1.f),
		bokehRotation(0.f), bokehRoundness(0.f), bokehObstruction(0.f),
		enableOculusRiftBarrel(false),
		vignettingDistance(0.f), vignettingScale(1.f),
		distortionK1(0.f), distortionK2(0.f),
		aberrationSpherical(0.f), aberrationComa(0.f), aberrationAstigmatism(0.f), aberrationFieldCurvature(0.f),
		tiltX(0.f), tiltY(0.f),
		dispersionLongitudinal(0.f), dispersionLateral(0.f),
		enableLensModel(false), enableVignetting(false),
		vignettingMaxOffset(0.f), invHalfDiagonal(1.f), focusNormal(0.f, 0.f, 1.f) {
}

PerspectiveCamera::PerspectiveCamera(const CameraType camType,
		const Point &o, const Point &t,
		const Vector &u, const float *region) :
		ProjectiveCamera(camType, region, o, t, u),
		screenOffsetX(0.f), screenOffsetY(0.f), fieldOfView(45.f),
		bokehBlades(0), bokehPower(0), bokehDistribution(DIST_EXPONENTIAL),
		bokehDistributionImageMap(nullptr), bokehDistributionMap(nullptr),
		bokehScaleX(1.f), bokehScaleY(1.f),
		bokehRotation(0.f), bokehRoundness(0.f), bokehObstruction(0.f),
		enableOculusRiftBarrel(false),
		vignettingDistance(0.f), vignettingScale(1.f),
		distortionK1(0.f), distortionK2(0.f),
		aberrationSpherical(0.f), aberrationComa(0.f), aberrationAstigmatism(0.f), aberrationFieldCurvature(0.f),
		tiltX(0.f), tiltY(0.f),
		dispersionLongitudinal(0.f), dispersionLateral(0.f),
		enableLensModel(false), enableVignetting(false),
		vignettingMaxOffset(0.f), invHalfDiagonal(1.f), focusNormal(0.f, 0.f, 1.f) {
}

PerspectiveCamera::~PerspectiveCamera() {
}

void PerspectiveCamera::InitCameraTransforms(CameraTransforms *trans) {
	// This is a trick I use from LuxCoreRenderer to set cameraToWorld to
	// identity matrix.
	if (orig == target)
		trans->cameraToWorld = Transform();
	else {
		const Transform worldToCamera = LookAt(orig, target, up);
		trans->cameraToWorld = Inverse(worldToCamera);
	}

	// Compute projective camera transformations
	trans->screenToCamera = Inverse(Perspective(fieldOfView, clipHither, clipYon));
	trans->screenToWorld = trans->cameraToWorld * trans->screenToCamera;
	// Compute projective camera screen transformations
	trans->rasterToScreen = luxrays::Translate(Vector(screenWindow[0] + screenOffsetX, screenWindow[3] + screenOffsetY, 0.f)) *
		Scale(screenWindow[1] - screenWindow[0], screenWindow[2] - screenWindow[3], 1.f) *
		Scale(1.f / filmWidth, 1.f / filmHeight, 1.f);
	trans->rasterToCamera = trans->screenToCamera * trans->rasterToScreen;
	trans->rasterToWorld = trans->screenToWorld * trans->rasterToScreen;
}

void PerspectiveCamera::InitCameraData() {
	const float tanAngle = tanf(Radians(fieldOfView) / 2.f) * 2.f;
	const float xPixelWidth = tanAngle * ((screenWindow[1] - screenWindow[0]) / 2.f);
	const float yPixelHeight = tanAngle * ((screenWindow[3] - screenWindow[2]) / 2.f);
	pixelArea = xPixelWidth * yPixelHeight;

	if (bokehDistributionImageMap) {
		// Initialize bokeh Distribution2D

		bokehDistributionMap = nullptr;

		const u_int distributionWidth = bokehDistributionImageMap->GetWidth();
		const u_int distributionHeight = bokehDistributionImageMap->GetHeight();
		vector<float> data(distributionWidth * distributionHeight);
		for (u_int y = 0; y < distributionHeight; ++y) {
			for (u_int x = 0; x < distributionWidth; ++x) {
				const u_int index = x + y * distributionWidth;

				data[index] = bokehDistributionImageMap->GetStorage().GetFloat(index);
			}
		}

		bokehDistributionMap = std::make_unique<Distribution2D>(data, distributionWidth, distributionHeight);
	}
	
	// Normalize bokeh scale vector
	const float s = 1.f / sqrtf(Sqr(bokehScaleX) + Sqr(bokehScaleY));
	bokehScaleX *= s;
	bokehScaleY *= s;

	//--------------------------------------------------------------------------
	// The thin lens model (ROADMAP F11)
	//--------------------------------------------------------------------------

	const bool hasLens = (lensRadius > 0.f) && (focalDistance > 0.f);
	enableVignetting = hasLens && (vignettingDistance != 0.f);
	enableLensModel = enableVignetting ||
			(distortionK1 != 0.f) || (distortionK2 != 0.f) ||
			(aberrationSpherical != 0.f) || (aberrationComa != 0.f) ||
			(aberrationAstigmatism != 0.f) || (aberrationFieldCurvature != 0.f) ||
			(tiltX != 0.f) || (tiltY != 0.f) ||
			(dispersionLongitudinal != 0.f) || (dispersionLateral != 0.f);

	// The image height 1 at the farthest corner of the frame, in tangent
	// units (the camera point at z = 1)
	const float tanHalfAngle = tanf(Radians(fieldOfView) / 2.f);
	const float halfWidth = Max(fabsf(screenWindow[0] + screenOffsetX), fabsf(screenWindow[1] + screenOffsetX));
	const float halfHeight = Max(fabsf(screenWindow[2] + screenOffsetY), fabsf(screenWindow[3] + screenOffsetY));
	invHalfDiagonal = 1.f / (tanHalfAngle * sqrtf(Sqr(halfWidth) + Sqr(halfHeight)));

	// The normal of the plane of focus: the z axis turned by tiltX about the
	// x axis, then by tiltY about the y axis
	const float tx = Radians(tiltX);
	const float ty = Radians(tiltY);
	focusNormal = Normalize(Vector(cosf(tx) * sinf(ty), -sinf(tx), cosf(tx) * cosf(ty)));

	worldToCamera = Inverse(camTrans.cameraToWorld);
	cameraToRaster = Inverse(camTrans.rasterToCamera);

	InitVignettingTable();
}

void PerspectiveCamera::InitVignettingTable() {
	vignettingTable.clear();
	vignettingMaxOffset = 0.f;
	if (!enableVignetting)
		return;

	// A stratified set of points of the aperture, with its distribution
	const u_int n = 64;
	vector<float> pointsX(n * n), pointsY(n * n);
	float maxRadius = 0.f;
	for (u_int j = 0; j < n; ++j) {
		for (u_int i = 0; i < n; ++i) {
			const u_int index = j * n + i;
			SampleAperture((i + .5f) / n, (j + .5f) / n, &pointsX[index], &pointsY[index]);
			maxRadius = Max(maxRadius, sqrtf(Sqr(pointsX[index]) + Sqr(pointsY[index])));
		}
	}

	// The share of them inside the barrel at each offset of the table
	vignettingMaxOffset = maxRadius + vignettingScale;
	vignettingTable.resize(LENS_VIGNETTING_RADII * LENS_VIGNETTING_ANGLES);
	const float scale2 = Sqr(vignettingScale);
	for (u_int r = 0; r < LENS_VIGNETTING_RADII; ++r) {
		const float offset = vignettingMaxOffset * r / (LENS_VIGNETTING_RADII - 1);
		for (u_int a = 0; a < LENS_VIGNETTING_ANGLES; ++a) {
			const float phi = 2.f * M_PI * a / LENS_VIGNETTING_ANGLES;
			const float barrelX = offset * cosf(phi);
			const float barrelY = offset * sinf(phi);

			u_int inside = 0;
			for (u_int i = 0; i < n * n; ++i) {
				if (Sqr(pointsX[i] - barrelX) + Sqr(pointsY[i] - barrelY) <= scale2)
					++inside;
			}
			vignettingTable[r * LENS_VIGNETTING_ANGLES + a] = inside / (float)(n * n);
		}
	}
}

void PerspectiveCamera::InitRay(Ray *ray, const float filmX, const float filmY) const {
	Point Pras;
	if (enableOculusRiftBarrel) {
		OculusRiftBarrelPostprocess(filmX / filmWidth, (filmHeight - filmY) / filmHeight,
				&Pras.x, &Pras.y);

		Pras.x = Min(Pras.x * filmWidth, (float)(filmWidth - 1));
		Pras.y = Min(Pras.y * filmHeight, (float)(filmHeight - 1));
	} else
		Pras = Point(filmX, filmHeight - filmY, 0.f);

	const Point Pcamera = Point(camTrans.rasterToCamera * Pras);

	ray->Update(Pcamera, Vector(Pcamera.x, Pcamera.y, Pcamera.z));
}

bool PerspectiveCamera::TintsPath() const {
	return (dispersionLongitudinal != 0.f) || (dispersionLateral != 0.f);
}

bool PerspectiveCamera::SupportsLightTracing() const {
	// The aberrations and the dispersion move the focus point of each lens
	// sample: the lens mapping has no exact inverse
	return (aberrationSpherical == 0.f) && (aberrationComa == 0.f) &&
			(aberrationAstigmatism == 0.f) && (aberrationFieldCurvature == 0.f) &&
			!TintsPath();
}

Spectrum PerspectiveCamera::GenerateRay(const float time,
		const float filmX, const float filmY,
		Ray *ray, PathVolumeInfo *volInfo,
		const float u0, const float u1, const float wavelength) const {
	if (!enableLensModel)
		return ProjectiveCamera::GenerateRay(time, filmX, filmY, ray, volInfo, u0, u1, wavelength);

	InitRay(ray, filmX, filmY);
	if (HasVolume())
		volInfo->AddVolume(GetVolume());

	// The pinhole ray of the film point (z = 1), moved by the distortion and
	// the lateral dispersion
	const float tanX = ray->o.x / ray->o.z;
	const float tanY = ray->o.y / ray->o.z;
	const float tanLength = sqrtf(Sqr(tanX) + Sqr(tanY));
	const float h = tanLength * invHalfDiagonal;
	const float h2 = Sqr(h);
	const float dispersion = TintsPath() ? ocl::Lens_NormalizedDispersion(wavelength) : 0.f;
	// The image of the wavelength is larger by 1 + dispersionLateral D: its
	// film point sees a smaller angle
	const float scale = (1.f + distortionK1 * h2 + distortionK2 * h2 * h2) /
			(1.f + dispersionLateral * dispersion);
	Vector dir(tanX * scale, tanY * scale, 1.f);
	Point orig;

	float weight = 1.f;
	if ((lensRadius > 0.f) && (focalDistance > 0.f)) {
		// A point of the aperture (lens radii)
		float lensX, lensY;
		SampleAperture(u0, u1, &lensX, &lensY);

		if (enableVignetting) {
			// The barrel opening seen from the lens along the pinhole ray
			const float barrelX = -vignettingDistance * dir.x / lensRadius;
			const float barrelY = -vignettingDistance * dir.y / lensRadius;

			bool inside = ocl::Lens_InsideBarrel(lensX, lensY, barrelX, barrelY, vignettingScale);
			for (u_int i = 1; (i < LENS_VIGNETTING_TRIES) && !inside; ++i) {
				float v0, v1;
				ocl::Lens_VignettingRetrySample(u0, u1, i, &v0, &v1);
				SampleAperture(v0, v1, &lensX, &lensY);
				inside = ocl::Lens_InsideBarrel(lensX, lensY, barrelX, barrelY, vignettingScale);
			}

			weight = inside ? ocl::Lens_VignettingWeight(ocl::Lens_VignettingShare(
					&vignettingTable[0], barrelX, barrelY, vignettingMaxOffset)) : 0.f;
		}

		lensX *= lensRadius;
		lensY *= lensRadius;

		// The meridional direction of the image point
		const float mx = (tanLength > 0.f) ? (tanX / tanLength) : 1.f;
		const float my = (tanLength > 0.f) ? (tanY / tanLength) : 0.f;

		const ocl::float3 focusPoint = ocl::Lens_FocusPoint(
				ocl::make_float3(dir.x, dir.y, dir.z), lensX, lensY, lensRadius,
				focalDistance, ocl::make_float3(focusNormal.x, focusNormal.y, focusNormal.z),
				h, mx, my,
				aberrationSpherical, aberrationComa, aberrationAstigmatism,
				aberrationFieldCurvature, dispersionLongitudinal * dispersion);

		orig = Point(lensX, lensY, 0.f);
		dir = Vector(focusPoint.x, focusPoint.y, focusPoint.z) - Vector(orig);
	}

	// The ray starts on the plane of the hither clip
	ray->o = orig + dir * (clipHither / dir.z);
	ray->d = Normalize(dir);
	ray->mint = MachineEpsilon::E(ray->o);
	ray->maxt = (clipYon - clipHither) / ray->d.z;
	ray->time = time;

	if (motionSystem) {
		*ray = motionSystem->Sample(ray->time) * (camTrans.cameraToWorld * (*ray));
		// I need to normalize the direction vector again because the motion
		// system could include some kind of scale
		ray->d = Normalize(ray->d);
	} else
		*ray = camTrans.cameraToWorld * (*ray);

	// World arbitrary clipping plane support
	if (enableClippingPlane)
		ApplyArbitraryClippingPlane(ray);

	return Spectrum(weight);
}

void PerspectiveCamera::ClampRay(Ray *ray) const {
	Vector globalDir = dir;
	if (motionSystem)
		globalDir *= motionSystem->Sample(ray->time);
	const float cosi = Dot(ray->d, globalDir);

	ray->mint = Max(ray->mint, clipHither / cosi);
	ray->maxt = Min(ray->maxt, clipYon / cosi);
}

bool PerspectiveCamera::GetSamplePosition(Ray *ray, float *x, float *y) const {
	if (enableLensModel)
		return GetSamplePositionLensModel(ray, x, y);

	Vector globalDir = dir;
	if (motionSystem)
		globalDir *= motionSystem->Sample(ray->time);
	const float cosi = Dot(ray->d, globalDir);

	if ((cosi <= 0.f) || (!isinf(ray->maxt) && (ray->maxt * cosi < clipHither ||
		ray->maxt * cosi > clipYon)))
		return false;

	Point pO = (ray->o + ((lensRadius > 0.f) ? (ray->d * (focalDistance / cosi)) : ray->d));
	if (motionSystem)
		pO *= motionSystem->SampleInverse(ray->time);
	pO *= Inverse(camTrans.rasterToWorld);

	*x = pO.x;
	*y = filmHeight - pO.y;

	// Check if we are inside the image plane
	if ((*x < filmSubRegion[0]) || (*x >= filmSubRegion[1] + 1) ||
			(*y < filmSubRegion[2]) || (*y >= filmSubRegion[3] + 1))
		return false;
	else {
		// World arbitrary clipping plane support
		if (enableClippingPlane) {
			// Check if the ray end point is on the not visible side of the plane
			const Point endPoint = (*ray)(ray->maxt);
			if (Dot(clippingPlaneNormal, endPoint - clippingPlaneCenter) <= 0.f)
				return false;
			// Update ray mint/maxt
			ApplyArbitraryClippingPlane(ray);
		}

		return true;
	}
}

bool PerspectiveCamera::GetSamplePositionLensModel(Ray *ray, float *x, float *y) const {
	if (!SupportsLightTracing())
		return false;

	Vector globalDir = dir;
	if (motionSystem)
		globalDir *= motionSystem->Sample(ray->time);
	const float cosi = Dot(ray->d, globalDir);

	if ((cosi <= 0.f) || (!isinf(ray->maxt) && (ray->maxt * cosi < clipHither ||
		ray->maxt * cosi > clipYon)))
		return false;

	// The ray in camera space
	Point o = ray->o;
	Vector d = ray->d;
	if (motionSystem) {
		const Matrix4x4 inv = motionSystem->SampleInverse(ray->time);
		o *= inv;
		d *= inv;
	}
	o = worldToCamera * o;
	d = worldToCamera * d;
	if (d.z <= 0.f)
		return false;

	// The point of the plane of focus the ray comes from, or a point of the
	// pinhole ray
	Point focusPoint;
	if ((lensRadius > 0.f) && (focalDistance > 0.f)) {
		const float cosN = Dot(focusNormal, d);
		if (cosN <= 0.f)
			return false;
		const float t = Dot(focusNormal, Point(0.f, 0.f, focalDistance) - o) / cosN;
		focusPoint = o + d * t;
	} else
		focusPoint = o + d;
	if (focusPoint.z <= 0.f)
		return false;

	// The pinhole direction of the film point (z = 1), back from the
	// distortion by Newton on rho + k1 rho^3 + k2 rho^5 = the image height
	const float tanX = focusPoint.x / focusPoint.z;
	const float tanY = focusPoint.y / focusPoint.z;
	const float hUndistorted = sqrtf(Sqr(tanX) + Sqr(tanY)) * invHalfDiagonal;
	float filmScale = 1.f;
	if ((hUndistorted > 0.f) && ((distortionK1 != 0.f) || (distortionK2 != 0.f))) {
		float h = hUndistorted;
		bool converged = false;
		for (u_int i = 0; i < 16; ++i) {
			const float h2 = Sqr(h);
			const float f = h * (1.f + distortionK1 * h2 + distortionK2 * h2 * h2) - hUndistorted;
			const float df = 1.f + 3.f * distortionK1 * h2 + 5.f * distortionK2 * h2 * h2;
			if (df <= 0.f)
				return false;
			h -= f / df;
			if (fabsf(f) < 1e-6f * hUndistorted) {
				converged = true;
				break;
			}
		}
		if (!converged || (h <= 0.f))
			return false;
		filmScale = h / hUndistorted;
	}

	if (enableVignetting) {
		// The lens point must pass the barrel (see GenerateRay())
		if (!ocl::Lens_InsideBarrel(o.x / lensRadius, o.y / lensRadius,
				-vignettingDistance * tanX / lensRadius, -vignettingDistance * tanY / lensRadius,
				vignettingScale))
			return false;
	}

	const Point pRaster = cameraToRaster * Point(tanX * filmScale * clipHither,
			tanY * filmScale * clipHither, clipHither);
	*x = pRaster.x;
	*y = filmHeight - pRaster.y;

	// Check if we are inside the image plane
	if ((*x < filmSubRegion[0]) || (*x >= filmSubRegion[1] + 1) ||
			(*y < filmSubRegion[2]) || (*y >= filmSubRegion[3] + 1))
		return false;
	else {
		// World arbitrary clipping plane support
		if (enableClippingPlane) {
			// Check if the ray end point is on the not visible side of the plane
			const Point endPoint = (*ray)(ray->maxt);
			if (Dot(clippingPlaneNormal, endPoint - clippingPlaneCenter) <= 0.f)
				return false;
			// Update ray mint/maxt
			ApplyArbitraryClippingPlane(ray);
		}

		return true;
	}
}

float PerspectiveCamera::DistortionJacobian(const float filmX, const float filmY) const {
	if ((distortionK1 == 0.f) && (distortionK2 == 0.f))
		return 1.f;

	const Point pCamera = camTrans.rasterToCamera * Point(filmX, filmHeight - filmY, 0.f);
	const float h2 = (Sqr(pCamera.x) + Sqr(pCamera.y)) / Sqr(pCamera.z) * Sqr(invHalfDiagonal);
	// The radial map h (1 + k1 h^2 + k2 h^4): its radial derivative times
	// its ratio
	const float g = 1.f + distortionK1 * h2 + distortionK2 * h2 * h2;

	return fabsf(g * (g + 2.f * distortionK1 * h2 + 4.f * distortionK2 * h2 * h2));
}

void PerspectiveCamera::SampleAperture(const float u1, const float u2,
		float *x, float *y) const {
	Point lensPoint(0.f, 0.f, 0.f);
	{
		if ((bokehDistribution == DIST_NONE) || ((bokehDistribution != DIST_CUSTOM) && (bokehBlades < 3))) {
			ConcentricSampleDisk(u1, u2, &lensPoint.x, &lensPoint.y);

			if (bokehObstruction > 0.f) {
				// The disk mapped to the annulus, keeping the uniform density
				const float r2 = Sqr(lensPoint.x) + Sqr(lensPoint.y);
				if (r2 > 0.f) {
					const float k = sqrtf((Sqr(bokehObstruction) + (1.f - Sqr(bokehObstruction)) * r2) / r2);
					lensPoint.x *= k;
					lensPoint.y *= k;
				}
			}
		} else {
			// Bokeh support

			if (bokehDistribution == DIST_CUSTOM) {
				assert (bokehDistributionMap);

				float uv[2];
				float distPdf;
				bokehDistributionMap->SampleContinuous(u1, u2, uv, &distPdf);
				if (distPdf > 0.f) {
					lensPoint.x = uv[0] * bokehScaleX;
					lensPoint.y = uv[1] * bokehScaleY;

					if (bokehRotation != 0.f) {
						const float a = Radians(bokehRotation);
						const float px = lensPoint.x;
						lensPoint.x = px * cosf(a) - lensPoint.y * sinf(a);
						lensPoint.y = px * sinf(a) + lensPoint.y * cosf(a);
					}
				}
			} else {
				const float halfAngle = M_PI / bokehBlades;
				const float honeyRadius = cosf(halfAngle);

				const float theta = 2.f * M_PI * u2;

				// The angle in the polygon turned by the rotation
				float a = theta - Radians(bokehRotation);
				if (a < 0.f)
					a += 2.f * M_PI;
				const u_int sector = Floor2UInt(a / halfAngle);
				const float rho = (sector % 2 == 0) ? (a - sector * halfAngle) :
						((sector + 1) * halfAngle - a);

				float r = honeyRadius / cosf(rho);
				// Blended toward the circumscribed circle
				r += bokehRoundness * (1.f - r);

				float r2;
				switch (bokehDistribution) {
					case DIST_UNIFORM:
						r2 = u1;
						break;
					case DIST_EXPONENTIAL:
						r2 = ExponentialSampleDisk(u1, bokehPower);
						break;
					case DIST_INVERSEEXPONENTIAL:
						r2 = InverseExponentialSampleDisk(u1, bokehPower);
						break;
					case DIST_GAUSSIAN:
						r2 = GaussianSampleDisk(u1);
						break;
					case DIST_INVERSEGAUSSIAN:
						r2 = InverseGaussianSampleDisk(u1);
						break;
					case DIST_TRIANGULAR:
						r2 = TriangularSampleDisk(u1);
						break;
					default:
						throw runtime_error("Unknown bokeh distribution in PerspectiveCamera::SampleAperture(): " + ToString(bokehDistribution));
				}
				// The disk mapped to the annulus of the obstruction
				if (bokehObstruction > 0.f)
					r2 = Sqr(bokehObstruction) + (1.f - Sqr(bokehObstruction)) * r2;
				r *= sqrtf(r2);

				lensPoint.x = r * cosf(theta) * bokehScaleX;
				lensPoint.y = r * sinf(theta) * bokehScaleY;
			}
		}
	}

	*x = lensPoint.x;
	*y = lensPoint.y;
}

bool PerspectiveCamera::LocalSampleLens(const float time,
		const float u1, const float u2,
		Point *lensp) const {
	Point lensPoint(0.f, 0.f, 0.f);
	if (lensRadius > 0.f) {
		SampleAperture(u1, u2, &lensPoint.x, &lensPoint.y);

		lensPoint.x *= lensRadius;
		lensPoint.y *= lensRadius;
	}

	*lensp = lensPoint;

	return true;
}

bool PerspectiveCamera::SampleLens(const float time,
		const float u1, const float u2,
		Point *lensp) const {
	Point lensPoint(0.f, 0.f, 0.f);
	LocalSampleLens(time, u1, u2, &lensPoint);

	if (motionSystem)
		*lensp = motionSystem->Sample(time) * (camTrans.cameraToWorld * lensPoint);
	else
		*lensp = camTrans.cameraToWorld * lensPoint;

	return true;
}

void PerspectiveCamera::GetPDF(const Ray &eyeRay, const float eyeDistance,
		const float filmX, const float filmY,
		float *pdfW, float *fluxToRadianceFactor) const {
	Vector globalDir = dir;
	if (motionSystem)
		globalDir *= motionSystem->Sample(eyeRay.time);

	const float cosAtCamera = Dot(eyeRay.d, globalDir);
	if (cosAtCamera <= 0.f) {
		if (pdfW)
			*pdfW = 0.f;
		if (fluxToRadianceFactor)
			*fluxToRadianceFactor = 0.f;
	} else {
		// The distortion changes the solid angle of the pixels
		const float area = enableLensModel ? (pixelArea * DistortionJacobian(filmX, filmY)) : pixelArea;
		const float cameraPdfW = 1.f / (cosAtCamera * cosAtCamera * cosAtCamera * area);

		if (pdfW)
			*pdfW = cameraPdfW;
		if (fluxToRadianceFactor)
			*fluxToRadianceFactor = cameraPdfW / (eyeDistance * eyeDistance);
	}
}

PropertiesUPtr PerspectiveCamera::ToProperties(const ImageMapCache &imgMapCache, const bool useRealFileName) const {
	PropertiesUPtr props = ProjectiveCamera::ToProperties(imgMapCache, useRealFileName);

	props->Set(Property("scene.camera.type")("perspective"));
	props->Set(Property("scene.camera.oculusrift.barrelpostpro.enable")(enableOculusRiftBarrel));
	props->Set(Property("scene.camera.fieldofview")(fieldOfView));
	props->Set(Property("scene.camera.bokeh.blades")(bokehBlades));
	props->Set(Property("scene.camera.bokeh.power")(bokehPower));
	props->Set(Property("scene.camera.bokeh.distribution.type")(BokehDistributionType2String(bokehDistribution)));

	if (bokehDistributionImageMap) {
		const string fileName = useRealFileName ?
			bokehDistributionImageMap->GetName() :
			imgMapCache.GetSequenceFileName(*bokehDistributionImageMap);
		props->Set(Property("scene.camera.bokeh.distribution.image")(fileName));
	}

	props->Set(Property("scene.camera.bokeh.scale.x")(bokehScaleX));
	props->Set(Property("scene.camera.bokeh.scale.y")(bokehScaleY));

	// The aperture shape and the thin lens model (ROADMAP F11), when not at
	// their defaults
	const auto setIf = [&props](const string &name, const float value, const float defaultValue) {
		if (value != defaultValue)
			props->Set(Property(name)(value));
	};
	setIf("scene.camera.bokeh.rotation", bokehRotation, 0.f);
	setIf("scene.camera.bokeh.roundness", bokehRoundness, 0.f);
	setIf("scene.camera.bokeh.obstruction", bokehObstruction, 0.f);
	setIf("scene.camera.opticalvignetting.distance", vignettingDistance, 0.f);
	setIf("scene.camera.opticalvignetting.scale", vignettingScale, 1.f);
	setIf("scene.camera.distortion.k1", distortionK1, 0.f);
	setIf("scene.camera.distortion.k2", distortionK2, 0.f);
	setIf("scene.camera.aberration.spherical", aberrationSpherical, 0.f);
	setIf("scene.camera.aberration.coma", aberrationComa, 0.f);
	setIf("scene.camera.aberration.astigmatism", aberrationAstigmatism, 0.f);
	setIf("scene.camera.aberration.fieldcurvature", aberrationFieldCurvature, 0.f);
	setIf("scene.camera.tilt.x", tiltX, 0.f);
	setIf("scene.camera.tilt.y", tiltY, 0.f);
	setIf("scene.camera.dispersion.longitudinal", dispersionLongitudinal, 0.f);
	setIf("scene.camera.dispersion.lateral", dispersionLateral, 0.f);

	return props;
}

PerspectiveCamera::BokehDistributionType PerspectiveCamera::String2BokehDistributionType(string type) {
	if (type == "NONE")
		return DIST_NONE;
	else if (type == "UNIFORM")
		return DIST_UNIFORM;
	else if (type == "EXPONENTIAL")
		return DIST_EXPONENTIAL;
	else if (type == "INVERSEEXPONENTIAL")
		return DIST_INVERSEEXPONENTIAL;
	else if (type == "GAUSSIAN")
		return DIST_GAUSSIAN;
	else if (type == "INVERSEGAUSSIAN")
		return DIST_INVERSEGAUSSIAN;
	else if (type == "TRIANGULAR")
		return DIST_TRIANGULAR;
	else if (type == "CUSTOM")
		return DIST_CUSTOM;
	else
		throw runtime_error("Unknown bokeh distribution type: " + type);
}

string PerspectiveCamera::BokehDistributionType2String(const BokehDistributionType type) {
	switch (type) {
		case DIST_NONE:
			return "NONE";
		case DIST_UNIFORM:
			return "UNIFORM";
		case DIST_EXPONENTIAL:
			return "EXPONENTIAL";
		case DIST_INVERSEEXPONENTIAL:
			return "INVERSEEXPONENTIAL";
		case DIST_GAUSSIAN:
			return "GAUSSIAN";
		case DIST_INVERSEGAUSSIAN:
			return "INVERSEGAUSSIAN";
		case DIST_TRIANGULAR:
			return "TRIANGULAR";
		case DIST_CUSTOM:
			return "CUSTOM";
		default:
			throw runtime_error("Unsupported bokeh distribution type in PerspectiveCamera::String2BokehDistributionType(): " + ToString(type));
	}
}
	
//------------------------------------------------------------------------------
// Oculus Rift post-processing pixel shader
//------------------------------------------------------------------------------

void PerspectiveCamera::OculusRiftBarrelPostprocess(const float x, const float y, float *barrelX, float *barrelY) {
	// Express the sample in coordinates relative to the eye center
	float ex = x * 2.f - 1.f;
	float ey = y * 2.f - 1.f;

	if ((ex == 0.f) && (ey == 0.f)) {
		*barrelX = 0.f;
		*barrelY = 0.f;
		return;
	}

	// Distance from the eye center
	const float distance = sqrtf(ex * ex + ey * ey);

	// "Push" the sample away based on the distance from the center
	const float scale = 1.f / 1.4f;
	const float k0 = 1.f;
	const float k1 = .22f;
	const float k2 = .23f;
	const float k3 = 0.f;
	const float distance2 = distance * distance;
	const float distance4 = distance2 * distance2;
	const float distance6 = distance2 * distance4;
	const float fr = scale * (k0 + k1 * distance2 + k2 * distance4 + k3 * distance6);

	ex *= fr;
	ey *= fr;

	// Clamp the coordinates
	ex = Clamp(ex, -1.f, 1.f);
	ey = Clamp(ey, -1.f, 1.f);

	*barrelX = (ex + 1.f) * .5f;
	*barrelY = (ey + 1.f) * .5f;
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
