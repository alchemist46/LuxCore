#line 2 "camera_funcs.cl"

/***************************************************************************
 * Copyright 1998-2016 by authors (see AUTHORS.txt)                        *
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

OPENCL_FORCE_INLINE void Camera_OculusRiftBarrelPostprocess(const float x, const float y, float *barrelX, float *barrelY) {
	// Express the sample in coordinates relative to the eye center
	float ex = x * 2.f - 1.f;
	float ey = y * 2.f - 1.f;

	if ((ex == 0.f) && (ey == 0.f)) {
		*barrelX = 0.f;
		*barrelY = 0.f;
		return;
	}

	// Distance from the eye center
	const float distance = sqrt(ex * ex + ey * ey);

	// "Push" the sample away base on the distance from the center
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
	ex = clamp(ex, -1.f, 1.f);
	ey = clamp(ey, -1.f, 1.f);

	*barrelX = (ex + 1.f) * .5f;
	*barrelY = (ey + 1.f) * .5f;
}

OPENCL_FORCE_INLINE void Camera_ApplyArbitraryClippingPlane(
		__global Ray *ray,
		const float3 clippingPlaneCenter, const float3 clippingPlaneNormal) {
	const float3 rayOrig = MAKE_FLOAT3(ray->o.x, ray->o.y, ray->o.z);
	const float3 rayDir = MAKE_FLOAT3(ray->d.x, ray->d.y, ray->d.z);

	// Intersect the ray with clipping plane
	const float denom = dot(clippingPlaneNormal, rayDir);
	const float3 pr = clippingPlaneCenter - rayOrig;
	float d = dot(pr, clippingPlaneNormal);

	if (fabs(denom) > DEFAULT_COS_EPSILON_STATIC) {
		// There is a valid intersection
		d /= denom; 

		if (d > 0.f) {
			// The plane is in front of the camera
			if (denom < 0.f) {
				// The plane points toward the camera
				ray->maxt = clamp(d, ray->mint, ray->maxt);
			} else {
				// The plane points away from the camera
				ray->mint = clamp(d, ray->mint, ray->maxt);
			}
		} else {
			if ((denom < 0.f) && (d < 0.f)) {
				// No intersection possible, I use a trick here to avoid any
				// intersection by setting mint=maxt
				ray->mint = ray->maxt;
			} else {
				// Nothing to do
			}
		}
	} else {
		// The plane is parallel to the view directions. Check if I'm on the
		// visible side of the plane or not
		if (d >= 0.f) {
			// No intersection possible, I use a trick here to avoid any
			// intersection by setting mint=maxt
			ray->mint = ray->maxt;
		} else {
			// Nothing to do
		}
	}
}

//------------------------------------------------------------------------------
// Perspective camera
//------------------------------------------------------------------------------

// A point of the aperture in lens radii (see PerspectiveCamera::SampleAperture())
OPENCL_FORCE_INLINE void PerspectiveCamera_SampleAperture(
		__global const PerspectiveCamera* restrict cameraPersp,
		__global const float* restrict cameraBokehDistribution,
		const float u1, const float u2,
		float *lensU, float *lensV) {
	*lensU = 0.f;
	*lensV = 0.f;

	{
		const float bokehBlades = cameraPersp->bokehBlades;
		const BokehDistributionType bokehDistribution = cameraPersp->bokehDistribution;
		const float bokehObstruction = cameraPersp->bokehObstruction;
		if ((bokehDistribution == DIST_NONE) || ((bokehDistribution != DIST_CUSTOM) && (bokehBlades < 3))) {
			ConcentricSampleDisk(u1, u2, lensU, lensV);

			if (bokehObstruction > 0.f) {
				// The disk mapped to the annulus, keeping the uniform density
				const float r2 = (*lensU) * (*lensU) + (*lensV) * (*lensV);
				if (r2 > 0.f) {
					const float k = sqrt((bokehObstruction * bokehObstruction + (1.f - bokehObstruction * bokehObstruction) * r2) / r2);
					*lensU *= k;
					*lensV *= k;
				}
			}
		} else {
			// Bokeh support
			
			if (bokehDistribution == DIST_CUSTOM) {
				float2 sampleUV;
				float distPdf;
				Distribution2D_SampleContinuous(cameraBokehDistribution, u1, u2, &sampleUV, &distPdf);
				if (distPdf > 0.f) {
					// The image on [-1, 1]^2, centred on the lens axis
					*lensU = (2.f * sampleUV.x - 1.f) * cameraPersp->bokehScaleX;
					*lensV = (2.f * sampleUV.y - 1.f) * cameraPersp->bokehScaleY;

					const float bokehRotation = cameraPersp->bokehRotation;
					if (bokehRotation != 0.f) {
						const float pu = *lensU;
						*lensU = pu * cos(bokehRotation) - (*lensV) * sin(bokehRotation);
						*lensV = pu * sin(bokehRotation) + (*lensV) * cos(bokehRotation);
					}
				}
			} else {
				const float halfAngle = M_PI_F / bokehBlades;
				const float honeyRadius = cos(halfAngle);

				const float theta = 2.f * M_PI_F * u2;

				// The angle in the polygon turned by the rotation
				float a = theta - cameraPersp->bokehRotation;
				if (a < 0.f)
					a += 2.f * M_PI_F;
				const uint sector = Floor2UInt(a / halfAngle);
				const float rho = (sector % 2 == 0) ? (a - sector * halfAngle) :
						((sector + 1) * halfAngle - a);

				float r = honeyRadius / cos(rho);
				// Blended toward the circumscribed circle
				r += cameraPersp->bokehRoundness * (1.f - r);

				float r2;
				switch (cameraPersp->bokehDistribution) {
					case DIST_UNIFORM:
						r2 = u1;
						break;
					case DIST_EXPONENTIAL:
						r2 = ExponentialSampleDisk(u1, cameraPersp->bokehPower);
						break;
					case DIST_INVERSEEXPONENTIAL:
						r2 = InverseExponentialSampleDisk(u1, cameraPersp->bokehPower);
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
						// Something wrong here
						r2 = 0.f;
						break;
				}
				// The disk mapped to the annulus of the obstruction
				if (bokehObstruction > 0.f)
					r2 = bokehObstruction * bokehObstruction + (1.f - bokehObstruction * bokehObstruction) * r2;
				r *= sqrt(r2);

				*lensU = r * cos(theta) * cameraPersp->bokehScaleX;
				*lensV = r * sin(theta) * cameraPersp->bokehScaleY;
			}
		}
	}
}

OPENCL_FORCE_INLINE void PerspectiveCamera_LocalSampleLens(
		__global const PerspectiveCamera* restrict cameraPersp,
		__global const float* restrict cameraBokehDistribution,
		const float u1, const float u2,
		float *lensU, float *lensV) {
	*lensU = 0.f;
	*lensV = 0.f;

	if (cameraPersp->projCamera.lensRadius > 0.f) {
		PerspectiveCamera_SampleAperture(cameraPersp, cameraBokehDistribution, u1, u2, lensU, lensV);

		const float lensRadius = cameraPersp->projCamera.lensRadius;
		*lensU *= lensRadius;
		*lensV *= lensRadius;
	}
}

// The ray of the thin lens model (ROADMAP F11, see PerspectiveCamera::GenerateRay())
// in camera space from the pinhole point pCamera of the film point (on the
// hither plane): returns the weight of the optical vignetting
OPENCL_FORCE_INLINE float PerspectiveCamera_LensModelRay(
		__global const CameraBase* restrict cameraBase,
		__global const PerspectiveCamera* restrict cameraPersp,
		__global const float* restrict cameraBokehDistribution,
		const float3 pCamera,
		const float dofSampleX, const float dofSampleY, const float wavelength,
		float3 *rayOrig, float3 *rayDir) {
	// The pinhole ray of the film point (z = 1), moved by the distortion and
	// the lateral dispersion
	const float tanX = pCamera.x / pCamera.z;
	const float tanY = pCamera.y / pCamera.z;
	const float tanLength = sqrt(tanX * tanX + tanY * tanY);
	const float h = tanLength * cameraPersp->invHalfDiagonal;
	const float h2 = h * h;
	const bool dispersive = (cameraPersp->dispersionLongitudinal != 0.f) || (cameraPersp->dispersionLateral != 0.f);
	const float dispersion = dispersive ? Lens_NormalizedDispersion(wavelength) : 0.f;
	// The image of the wavelength is larger by 1 + dispersionLateral D: its
	// film point sees a smaller angle
	const float scale = (1.f + cameraPersp->distortionK1 * h2 + cameraPersp->distortionK2 * h2 * h2) /
			(1.f + cameraPersp->dispersionLateral * dispersion);
	float3 dir = MAKE_FLOAT3(tanX * scale, tanY * scale, 1.f);
	float3 orig = MAKE_FLOAT3(0.f, 0.f, 0.f);

	float weight = 1.f;
	const float lensRadius = cameraPersp->projCamera.lensRadius;
	const float focalDistance = cameraPersp->projCamera.focalDistance;
	if ((lensRadius > 0.f) && (focalDistance > 0.f)) {
		// A point of the aperture (lens radii)
		float lensX, lensY;
		PerspectiveCamera_SampleAperture(cameraPersp, cameraBokehDistribution,
				dofSampleX, dofSampleY, &lensX, &lensY);

		const uint vignettingTableOffset = cameraPersp->vignettingTableOffset;
		if (vignettingTableOffset != NULL_INDEX) {
			// The barrel opening seen from the lens along the pinhole ray
			const float vignettingScale = cameraPersp->vignettingScale;
			const float barrelX = -cameraPersp->vignettingDistance * dir.x / lensRadius;
			const float barrelY = -cameraPersp->vignettingDistance * dir.y / lensRadius;

			bool inside = Lens_InsideBarrel(lensX, lensY, barrelX, barrelY, vignettingScale);
			for (uint i = 1; (i < LENS_VIGNETTING_TRIES) && !inside; ++i) {
				float v0, v1;
				Lens_VignettingRetrySample(dofSampleX, dofSampleY, i, &v0, &v1);
				PerspectiveCamera_SampleAperture(cameraPersp, cameraBokehDistribution,
						v0, v1, &lensX, &lensY);
				inside = Lens_InsideBarrel(lensX, lensY, barrelX, barrelY, vignettingScale);
			}

			weight = inside ? Lens_VignettingWeight(Lens_VignettingShare(
					&cameraBokehDistribution[vignettingTableOffset], barrelX, barrelY,
					cameraPersp->vignettingMaxOffset)) : 0.f;
		}

		lensX *= lensRadius;
		lensY *= lensRadius;

		// The meridional direction of the image point
		const float mx = (tanLength > 0.f) ? (tanX / tanLength) : 1.f;
		const float my = (tanLength > 0.f) ? (tanY / tanLength) : 0.f;

		const float3 focusPoint = Lens_FocusPoint(dir, lensX, lensY, lensRadius,
				focalDistance, VLOAD3F(&cameraPersp->focusNormal.x),
				h, mx, my,
				cameraPersp->aberrationSpherical, cameraPersp->aberrationComa,
				cameraPersp->aberrationAstigmatism, cameraPersp->aberrationFieldCurvature,
				cameraPersp->dispersionLongitudinal * dispersion);

		orig = MAKE_FLOAT3(lensX, lensY, 0.f);
		dir = focusPoint - orig;
	}

	// The ray starts on the plane of the hither clip
	*rayOrig = orig + dir * (cameraBase->hither / dir.z);
	*rayDir = dir;

	return weight;
}

OPENCL_FORCE_INLINE float PerspectiveCamera_GenerateRayImpl(
		__global const CameraBase* restrict cameraBase,
		__global const PerspectiveCamera* restrict cameraPersp,
		__global const Transform* restrict rasterToCamera,
		__global const Transform* restrict cameraToWorld,
		__global const float* restrict cameraBokehDistribution,
		const uint filmWidth, const uint filmHeight,
		__global Ray *ray,
		__global PathVolumeInfo *volInfo,
		const float filmX, const float filmY, const float timeSample,
		const float dofSampleX, const float dofSampleY, const float wavelength) {
	PathVolumeInfo_StartVolume(volInfo, cameraBase->volumeIndex);

	float3 Pras;
	if (cameraPersp->enableOculusRiftBarrel) {
		float ssx, ssy;
		Camera_OculusRiftBarrelPostprocess(filmX / filmWidth, (filmHeight - filmY) / filmHeight, &ssx, &ssy);
		Pras = MAKE_FLOAT3(min(ssx * filmWidth, (float) (filmWidth - 1)), min(ssy * filmHeight, (float) (filmHeight - 1)), 0.f);
	} else
		Pras = MAKE_FLOAT3(filmX, filmHeight - filmY, 0.f);

	float3 rayOrig = Transform_ApplyPoint(rasterToCamera, Pras);
	float3 rayDir = rayOrig;

	const float hither = cameraBase->hither;

	float weight = 1.f;
	const float lensRadius = cameraPersp->projCamera.lensRadius;
	const float focalDistance = cameraPersp->projCamera.focalDistance;
	if (cameraPersp->enableLensModel) {
		weight = PerspectiveCamera_LensModelRay(cameraBase, cameraPersp, cameraBokehDistribution,
				rayOrig, dofSampleX, dofSampleY, wavelength, &rayOrig, &rayDir);
	} else if ((lensRadius > 0.f) && (focalDistance > 0.f)) {
		// Sample point on lens
		float lensU, lensV;
		PerspectiveCamera_LocalSampleLens(cameraPersp, cameraBokehDistribution, dofSampleX, dofSampleY, &lensU, &lensV);

		// Compute point on plane of focus
		const float dist = focalDistance - hither;

		const float ft = dist / rayDir.z;
		const float3 Pfocus = rayOrig + rayDir * ft;

		// Update ray for effect of lens
		const float k = dist / focalDistance;
		rayOrig.x += lensU * k;
		rayOrig.y += lensV * k;

		rayDir = Pfocus - rayOrig;
	}

	rayDir = normalize(rayDir);

	const float maxt = (cameraBase->yon - hither) / rayDir.z;
	const float time = mix(cameraBase->shutterOpen, cameraBase->shutterClose, timeSample);

	// Transform ray in world coordinates
	rayOrig = Transform_ApplyPoint(cameraToWorld, rayOrig);
	rayDir = Transform_ApplyVector(cameraToWorld, rayDir);

	const uint interpolatedTransformFirstIndex = cameraBase->motionSystem.interpolatedTransformFirstIndex;
	if (interpolatedTransformFirstIndex != NULL_INDEX) {
		Matrix4x4 m;
		MotionSystem_Sample(&cameraBase->motionSystem, time, cameraBase->interpolatedTransforms, &m);

		rayOrig = Matrix4x4_ApplyPoint_Private(&m, rayOrig);
		rayDir = Matrix4x4_ApplyVector_Private(&m, rayDir);
	}

	Ray_Init3(ray, rayOrig, rayDir, maxt, time);
	
	if (cameraPersp->projCamera.enableClippingPlane) {
		Camera_ApplyArbitraryClippingPlane(ray,
				VLOAD3F(&cameraPersp->projCamera.clippingPlaneCenter.x),
				VLOAD3F(&cameraPersp->projCamera.clippingPlaneNormal.x));
	}

	/*printf("(%f, %f, %f) (%f, %f, %f) [%f, %f]\n",
		ray->o.x, ray->o.y, ray->o.z, ray->d.x, ray->d.y, ray->d.z,
		ray->mint, ray->maxt);*/

	return weight;
}

OPENCL_FORCE_INLINE float PerspectiveCamera_GenerateRay(
		__global const Camera* restrict camera,
		__global const float* restrict cameraBokehDistribution,
		const uint filmWidth, const uint filmHeight,
		__global Ray *ray,
		__global PathVolumeInfo *volInfo,
		const float filmX, const float filmY, const float timeSample,
		const float dofSampleX, const float dofSampleY, const float wavelength) {
	return PerspectiveCamera_GenerateRayImpl(&camera->base, &camera->persp,
			&camera->base.rasterToCamera, &camera->base.cameraToWorld,
			cameraBokehDistribution,
			filmWidth, filmHeight, ray, volInfo, filmX, filmY,
			timeSample, dofSampleX, dofSampleY, wavelength);
}

//------------------------------------------------------------------------------
// Realistic camera (ROADMAP F12, see RealisticCamera)
//------------------------------------------------------------------------------

OPENCL_FORCE_INLINE float RealisticCamera_GenerateRay(
		__global const Camera* restrict camera,
		__global const float* restrict cameraData,
		const uint filmWidth, const uint filmHeight,
		__global Ray *ray,
		__global PathVolumeInfo *volInfo,
		const float filmX, const float filmY, const float timeSample,
		const float dofSampleX, const float dofSampleY, const float wavelength) {
	PathVolumeInfo_StartVolume(volInfo, camera->base.volumeIndex);

	// The point of the sensor
	const float3 Pras = MAKE_FLOAT3(filmX, filmHeight - filmY, 0.f);
	const float3 filmPoint = Transform_ApplyPoint(&camera->base.rasterToCamera, Pras);

	float3 rayOrig, rayDir;
	const float weight = RealisticCamera_LensRay(&camera->realistic, cameraData, filmPoint,
			dofSampleX, dofSampleY, wavelength, &rayOrig, &rayDir);

	// The clipping planes across the axis at their distances from the
	// entrance pupil (see RealisticCamera::GenerateRay())
	float tHither = 0.f;
	float maxt = camera->base.yon;
	if (rayDir.z > 0.f) {
		tHither = fmax((camera->base.hither - rayOrig.z) / rayDir.z, 0.f);
		maxt = fmax((camera->base.yon - rayOrig.z) / rayDir.z, tHither);
	}
	const float time = mix(camera->base.shutterOpen, camera->base.shutterClose, timeSample);

	// Transform ray in world coordinates
	rayOrig = Transform_ApplyPoint(&camera->base.cameraToWorld, rayOrig);
	rayDir = Transform_ApplyVector(&camera->base.cameraToWorld, rayDir);

	const uint interpolatedTransformFirstIndex = camera->base.motionSystem.interpolatedTransformFirstIndex;
	if (interpolatedTransformFirstIndex != NULL_INDEX) {
		Matrix4x4 m;
		MotionSystem_Sample(&camera->base.motionSystem, time, camera->base.interpolatedTransforms, &m);

		rayOrig = Matrix4x4_ApplyPoint_Private(&m, rayOrig);
		rayDir = normalize(Matrix4x4_ApplyVector_Private(&m, rayDir));
	}

	Ray_Init3(ray, rayOrig, rayDir, maxt, time);
	ray->mint = fmax(ray->mint, tHither);

	if (camera->realistic.projCamera.enableClippingPlane) {
		Camera_ApplyArbitraryClippingPlane(ray,
				VLOAD3F(&camera->realistic.projCamera.clippingPlaneCenter.x),
				VLOAD3F(&camera->realistic.projCamera.clippingPlaneNormal.x));
	}

	return weight;
}

//------------------------------------------------------------------------------
// Orthographic camera
//------------------------------------------------------------------------------

OPENCL_FORCE_INLINE float OrthographicCamera_GenerateRay(
		__global const Camera* restrict camera,
		const uint filmWidth, const uint filmHeight,
		__global Ray *ray,
		__global PathVolumeInfo *volInfo,
		const float filmX, const float filmY, const float timeSample,
		const float dofSampleX, const float dofSampleY) {
	PathVolumeInfo_StartVolume(volInfo, camera->base.volumeIndex);

	const float3 Pras = MAKE_FLOAT3(filmX, filmHeight - filmY, 0.f);
	float3 rayOrig = Transform_ApplyPoint(&camera->base.rasterToCamera, Pras);
	float3 rayDir = MAKE_FLOAT3(0.f, 0.f, 1.f);

	const float hither = camera->base.hither;

	const float lensRadius = camera->ortho.projCamera.lensRadius;
	const float focalDistance = camera->ortho.projCamera.focalDistance;
	if ((lensRadius > 0.f) && (focalDistance > 0.f)) {
		// Sample point on lens
		float lensU, lensV;
		ConcentricSampleDisk(dofSampleX, dofSampleY, &lensU, &lensV);
		lensU *= lensRadius;
		lensV *= lensRadius;

		// Compute point on plane of focus
		const float dist = focalDistance - hither;

		const float ft = dist;
		const float3 Pfocus = rayOrig + rayDir * ft;

		// Update ray for effect of lens
		const float k = dist / focalDistance;
		rayOrig.x += lensU * k;
		rayOrig.y += lensV * k;

		rayDir = Pfocus - rayOrig;
	}

	rayDir = normalize(rayDir);

	const float maxt = (camera->base.yon - hither);
	const float time = mix(camera->base.shutterOpen, camera->base.shutterClose, timeSample);

	// Transform ray in world coordinates
	rayOrig = Transform_ApplyPoint(&camera->base.cameraToWorld, rayOrig);
	rayDir = Transform_ApplyVector(&camera->base.cameraToWorld, rayDir);

	const uint interpolatedTransformFirstIndex = camera->base.motionSystem.interpolatedTransformFirstIndex;
	if (interpolatedTransformFirstIndex != NULL_INDEX) {
		Matrix4x4 m;
		MotionSystem_Sample(&camera->base.motionSystem, time, camera->base.interpolatedTransforms, &m);

		rayOrig = Matrix4x4_ApplyPoint_Private(&m, rayOrig);
		rayDir = Matrix4x4_ApplyVector_Private(&m, rayDir);
	}

	Ray_Init3(ray, rayOrig, rayDir, maxt, time);

	if (camera->ortho.projCamera.enableClippingPlane) {
		Camera_ApplyArbitraryClippingPlane(ray,
				VLOAD3F(&camera->ortho.projCamera.clippingPlaneCenter.x),
				VLOAD3F(&camera->ortho.projCamera.clippingPlaneNormal.x));
	}

	return 1.f;


	/*printf("(%f, %f, %f) (%f, %f, %f) [%f, %f]\n",
		ray->o.x, ray->o.y, ray->o.z, ray->d.x, ray->d.y, ray->d.z,
		ray->mint, ray->maxt);*/
}

//------------------------------------------------------------------------------
// Environment camera
//------------------------------------------------------------------------------

OPENCL_FORCE_INLINE void EnvironmentCamera_GenerateRayImpl(
		__global const CameraBase* restrict cameraBase,
		const float degrees,
		__global const Transform* restrict cameraToWorld,
		const uint filmWidth, const uint filmHeight,
		__global Ray *ray,
		__global PathVolumeInfo *volInfo,
		const float filmX, const float filmY, const float timeSample,
		const float dofSampleX, const float dofSampleY) {
	PathVolumeInfo_StartVolume(volInfo, cameraBase->volumeIndex);

	const float theta = M_PI_F * (filmHeight - filmY) / filmHeight;
	const float phi = Radians(360.f - ((360.f - degrees) * .5f + degrees * filmX / filmWidth)) - .5 * M_PI_F;

	float3 rayOrig = MAKE_FLOAT3(0.f, 0.f, 0.f);
	float3 rayDir = MAKE_FLOAT3(sin(theta)*cos(phi), cos(theta), sin(theta)*sin(phi));
	
	const float maxt = (cameraBase->yon - cameraBase->hither);
	const float time = mix(cameraBase->shutterOpen, cameraBase->shutterClose, timeSample);

	// Transform ray in world coordinates
	rayOrig = Transform_ApplyPoint(cameraToWorld, rayOrig);
	rayDir = Transform_ApplyVector(cameraToWorld, rayDir);

	const uint interpolatedTransformFirstIndex = cameraBase->motionSystem.interpolatedTransformFirstIndex;
	if (interpolatedTransformFirstIndex != NULL_INDEX) {
		Matrix4x4 m;
		MotionSystem_Sample(&cameraBase->motionSystem, time, cameraBase->interpolatedTransforms, &m);

		rayOrig = Matrix4x4_ApplyPoint_Private(&m, rayOrig);
		rayDir = Matrix4x4_ApplyVector_Private(&m, rayDir);
	}

	Ray_Init3(ray, rayOrig, rayDir, maxt, time);

	/*printf("(%f, %f, %f) (%f, %f, %f) [%f, %f]\n",
		ray->o.x, ray->o.y, ray->o.z, ray->d.x, ray->d.y, ray->d.z,
		ray->mint, ray->maxt);*/
}

OPENCL_FORCE_INLINE void EnvironmentCamera_GenerateRay(
		__global const Camera* restrict camera,
		const uint filmWidth, const uint filmHeight,
		__global Ray *ray,
		__global PathVolumeInfo *volInfo,
		const float filmX, const float filmY, const float timeSample,
		const float dofSampleX, const float dofSampleY) {
	EnvironmentCamera_GenerateRayImpl(&camera->base, camera->env.degrees, &camera->base.cameraToWorld,
			filmWidth, filmHeight, ray, volInfo,
			filmX, filmY, timeSample, dofSampleX, dofSampleY);
}

//------------------------------------------------------------------------------
// Stereo camera
//------------------------------------------------------------------------------

OPENCL_FORCE_INLINE float StereoCamera_GenerateRay(
		__global const Camera* restrict camera,
		__global const float* restrict cameraBokehDistribution,
		const uint origFilmWidth, const uint origFilmHeight,
		__global Ray *ray,
		__global PathVolumeInfo *volInfo,
		const float origFilmX, const float origFilmY, const float timeSample,
		const float dofSampleX, const float dofSampleY, const float wavelength) {
	float weight = 1.f;

	PathVolumeInfo_StartVolume(volInfo, camera->base.volumeIndex);

	switch(camera->stereo.stereoCameraType) {
		case STEREO_PERSPECTIVE: {
			__global const Transform* restrict rasterToCamera;
			__global const Transform* restrict cameraToWorld;
			float filmX, filmY;
			const float filmWidth = origFilmWidth / 2;
			const float filmHeight = origFilmHeight;
			if (origFilmX < filmWidth) {
				rasterToCamera = &camera->stereo.leftEyeRasterToCamera;
				cameraToWorld = &camera->stereo.leftEyeCameraToWorld;
				filmX = origFilmX;
			} else {
				rasterToCamera = &camera->stereo.rightEyeRasterToCamera;
				cameraToWorld = &camera->stereo.rightEyeCameraToWorld;
				filmX = origFilmX - filmWidth;
			}
			filmY = origFilmY;

			weight = PerspectiveCamera_GenerateRayImpl(&camera->base, &camera->stereo.perspCamera,
					rasterToCamera, cameraToWorld, cameraBokehDistribution,
					filmWidth, filmHeight, ray, volInfo, filmX, filmY, timeSample, dofSampleX, dofSampleY,
					wavelength);
			break;
		}
		case STEREO_ENVIRONMENT_180: {
			__global const Transform* restrict rasterToCamera;
			__global const Transform* restrict cameraToWorld;
			float filmX, filmY;
			const float filmWidth = origFilmWidth / 2;
			const float filmHeight = origFilmHeight;
			if (origFilmX < filmWidth) {
				rasterToCamera = &camera->stereo.leftEyeRasterToCamera;
				cameraToWorld = &camera->stereo.leftEyeCameraToWorld;
				filmX = origFilmX;
			} else {
				rasterToCamera = &camera->stereo.rightEyeRasterToCamera;
				cameraToWorld = &camera->stereo.rightEyeCameraToWorld;
				filmX = origFilmX - filmWidth;
			}
			filmY = origFilmY;

			EnvironmentCamera_GenerateRayImpl(&camera->base, 180.f, cameraToWorld,
					filmWidth, filmHeight, ray, volInfo,
					filmX, filmY, timeSample, dofSampleX, dofSampleY);
			break;
		}
		case STEREO_ENVIRONMENT_360: {
			__global const Transform* restrict rasterToCamera;
			__global const Transform* restrict cameraToWorld;
			float filmX, filmY;
			const float filmWidth = origFilmWidth;
			const float filmHeight = origFilmHeight / 2;
			filmX = origFilmX;
			if (origFilmY < filmHeight) {
				rasterToCamera = &camera->stereo.leftEyeRasterToCamera;
				cameraToWorld = &camera->stereo.leftEyeCameraToWorld;
				filmY = origFilmY;
			} else {
				rasterToCamera = &camera->stereo.rightEyeRasterToCamera;
				cameraToWorld = &camera->stereo.rightEyeCameraToWorld;
				filmY = origFilmY - filmHeight;
			}

			EnvironmentCamera_GenerateRayImpl(&camera->base, 360.f, cameraToWorld,
					filmWidth, filmHeight, ray, volInfo,
					filmX, filmY, timeSample, dofSampleX, dofSampleY);
			break;
		}
		default:
			// Something very wrong here...
			break;
	}

	/*printf("(%f, %f, %f) (%f, %f, %f) [%f, %f]\n",
		ray->o.x, ray->o.y, ray->o.z, ray->d.x, ray->d.y, ray->d.z,
		ray->mint, ray->maxt);*/

	return weight;
}

//------------------------------------------------------------------------------
// Generic function
//------------------------------------------------------------------------------

// Returns the weight of the ray (see Camera::GenerateRay()). The camera data
// buffer has the custom bokeh distribution at its start, the table of the
// optical vignetting and the lens of the realistic camera at their offsets
OPENCL_HOT_FUNCTION float Camera_GenerateRay(
		__global const Camera* restrict camera,
		__global const float* restrict cameraBokehDistribution,
		const uint filmWidth, const uint filmHeight,
		__global Ray *ray,
		__global PathVolumeInfo *volInfo,
		const float filmX, const float filmY, const float timeSample,
		const float dofSampleX, const float dofSampleY, const float wavelength) {
	switch (camera->type) {
		case PERSPECTIVE:
			return PerspectiveCamera_GenerateRay(camera, cameraBokehDistribution, filmWidth, filmHeight, ray, volInfo, filmX, filmY, timeSample, dofSampleX, dofSampleY, wavelength);
		case ORTHOGRAPHIC:
			return OrthographicCamera_GenerateRay(camera, filmWidth, filmHeight, ray, volInfo, filmX, filmY, timeSample, dofSampleX, dofSampleY);
		case STEREO:
			return StereoCamera_GenerateRay(camera, cameraBokehDistribution, filmWidth, filmHeight, ray, volInfo, filmX, filmY, timeSample, dofSampleX, dofSampleY, wavelength);
		case ENVIRONMENT:
			EnvironmentCamera_GenerateRay(camera, filmWidth, filmHeight, ray, volInfo, filmX, filmY, timeSample, dofSampleX, dofSampleY);
			return 1.f;
		case REALISTIC:
			return RealisticCamera_GenerateRay(camera, cameraBokehDistribution, filmWidth, filmHeight, ray, volInfo, filmX, filmY, timeSample, dofSampleX, dofSampleY, wavelength);
		default:
			// Something really wrong here
			return 0.f;
	}
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
