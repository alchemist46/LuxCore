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

#if !defined(LUXRAYS_DISABLE_OPENCL)

#include <iosfwd>
#include <limits>

#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>

#include "slg/engines/pathoclbase/compiledscene.h"
#include "slg/kernels/kernels.h"

#include "slg/cameras/orthographic.h"
#include "slg/cameras/perspective.h"
#include "slg/cameras/realistic.h"
#include "slg/cameras/stereo.h"
#include "slg/cameras/environment.h"

using namespace std;
using namespace luxrays;
using namespace slg;

// The aperture shape and the thin lens model (ROADMAP F11) of a perspective
// camera, the table of its optical vignetting appended to the camera data
static void CompilePerspectiveLensModel(const PerspectiveCamera &perspCamera,
		slg::ocl::PerspectiveCamera &camera, vector<float> &cameraData) {
	camera.bokehRotation = Radians(perspCamera.bokehRotation);
	camera.bokehRoundness = perspCamera.bokehRoundness;
	camera.bokehObstruction = perspCamera.bokehObstruction;

	camera.enableLensModel = perspCamera.HasLensModel();
	camera.invHalfDiagonal = perspCamera.GetInvHalfDiagonal();
	camera.distortionK1 = perspCamera.distortionK1;
	camera.distortionK2 = perspCamera.distortionK2;
	camera.vignettingDistance = perspCamera.vignettingDistance;
	camera.vignettingScale = perspCamera.vignettingScale;
	camera.vignettingMaxOffset = perspCamera.GetVignettingMaxOffset();
	if (perspCamera.HasVignetting()) {
		camera.vignettingTableOffset = cameraData.size();
		const vector<float> &table = perspCamera.GetVignettingTable();
		cameraData.insert(cameraData.end(), table.begin(), table.end());
	} else
		camera.vignettingTableOffset = NULL_INDEX;
	ASSIGN_VECTOR(camera.focusNormal, perspCamera.GetFocusNormal());
	camera.aberrationSpherical = perspCamera.aberrationSpherical;
	camera.aberrationComa = perspCamera.aberrationComa;
	camera.aberrationAstigmatism = perspCamera.aberrationAstigmatism;
	camera.aberrationFieldCurvature = perspCamera.aberrationFieldCurvature;
	camera.dispersionLongitudinal = perspCamera.dispersionLongitudinal;
	camera.dispersionLateral = perspCamera.dispersionLateral;
}

void CompiledScene::CompileCamera() {
	wasCameraCompiled = true;

	// The camera data buffer: the custom bokeh distribution at its start,
	// then the data of the lens
	cameraBokehDistribution.clear();
	cameraBokehDistributionSize = 0;

	//SLG_LOG("Compile Camera");

	//--------------------------------------------------------------------------
	// Camera definition
	//--------------------------------------------------------------------------

	auto& sceneCamera = scene.GetCamera();


	// Initialize CameraBase

	camera.base.yon = sceneCamera.clipYon;
	camera.base.hither = sceneCamera.clipHither;
	camera.base.shutterOpen = sceneCamera.shutterOpen;
	camera.base.shutterClose = sceneCamera.shutterClose;
	camera.base.volumeIndex = sceneCamera.HasVolume() ?
		scene.GetMaterials().GetMaterialIndex(sceneCamera.GetVolume()) : NULL_INDEX;
	camera.base.tintPath = sceneCamera.TintsPath();

	if (sceneCamera.motionSystem) {
		if (sceneCamera.motionSystem->interpolatedTransforms.size() > CAMERA_MAX_INTERPOLATED_TRANSFORM)
			throw runtime_error("Too many interpolated transformations in camera motion system: " +
					ToString(sceneCamera.motionSystem->interpolatedTransforms.size()));

		for (u_int i = 0; i < sceneCamera.motionSystem->interpolatedTransforms.size(); ++i) {
			const InterpolatedTransform &it = sceneCamera.motionSystem->interpolatedTransforms[i];

			// Here, I assume that luxrays::ocl::InterpolatedTransform and
			// luxrays::InterpolatedTransform are the same
			camera.base.interpolatedTransforms[i] = *((const luxrays::ocl::InterpolatedTransform *)&it);
		}

		camera.base.motionSystem.interpolatedTransformFirstIndex = 0;
		camera.base.motionSystem.interpolatedTransformLastIndex = sceneCamera.motionSystem->interpolatedTransforms.size() - 1;
	} else {
		camera.base.motionSystem.interpolatedTransformFirstIndex = NULL_INDEX;
		camera.base.motionSystem.interpolatedTransformLastIndex = NULL_INDEX;
	}

	// Initialize Camera specific data

	switch (sceneCamera.GetType()) {
		case Camera::ORTHOGRAPHIC: {
			auto& orthoCamera = dynamic_cast<const OrthographicCamera&>(sceneCamera);
			camera.type = slg::ocl::ORTHOGRAPHIC;

			memcpy(camera.base.rasterToCamera.m.m, orthoCamera.GetRasterToCamera().m.m, 4 * 4 * sizeof(float));
			memcpy(camera.base.cameraToWorld.m.m, orthoCamera.GetCameraToWorld().m.m, 4 * 4 * sizeof(float));

			camera.ortho.projCamera.lensRadius = orthoCamera.lensRadius;
			camera.ortho.projCamera.focalDistance = orthoCamera.focalDistance;

			if (orthoCamera.enableClippingPlane) {
				camera.ortho.projCamera.enableClippingPlane = true;
				ASSIGN_VECTOR(camera.ortho.projCamera.clippingPlaneCenter, orthoCamera.clippingPlaneCenter);
				ASSIGN_VECTOR(camera.ortho.projCamera.clippingPlaneNormal, orthoCamera.clippingPlaneNormal);
			} else
				camera.ortho.projCamera.enableClippingPlane = false;
			break;
		}
		case Camera::PERSPECTIVE: {
			auto& perspCamera = dynamic_cast<const PerspectiveCamera&>(sceneCamera);
			camera.type = slg::ocl::PERSPECTIVE;

			memcpy(camera.base.rasterToCamera.m.m, perspCamera.GetRasterToCamera().m.m, 4 * 4 * sizeof(float));
			memcpy(camera.base.cameraToWorld.m.m, perspCamera.GetCameraToWorld().m.m, 4 * 4 * sizeof(float));

			camera.persp.projCamera.lensRadius = perspCamera.lensRadius;
			camera.persp.projCamera.focalDistance = perspCamera.focalDistance;

			camera.persp.screenOffsetX = perspCamera.screenOffsetX;
			camera.persp.screenOffsetY = perspCamera.screenOffsetY;

			camera.persp.enableOculusRiftBarrel = perspCamera.enableOculusRiftBarrel;
			if (perspCamera.enableClippingPlane) {
				camera.persp.projCamera.enableClippingPlane = true;
				ASSIGN_VECTOR(camera.persp.projCamera.clippingPlaneCenter, perspCamera.clippingPlaneCenter);
				ASSIGN_VECTOR(camera.persp.projCamera.clippingPlaneNormal, perspCamera.clippingPlaneNormal);
			} else
				camera.persp.projCamera.enableClippingPlane = false;

			camera.persp.bokehBlades = perspCamera.bokehBlades;
			camera.persp.bokehPower = perspCamera.bokehPower;
			switch (perspCamera.bokehDistribution) {
				case PerspectiveCamera::DIST_NONE:
					camera.persp.bokehDistribution = slg::ocl::DIST_NONE;
					break;
				case PerspectiveCamera::DIST_UNIFORM:
					camera.persp.bokehDistribution = slg::ocl::DIST_UNIFORM;
					break;
				case PerspectiveCamera::DIST_EXPONENTIAL:
					camera.persp.bokehDistribution = slg::ocl::DIST_EXPONENTIAL;
					break;
				case PerspectiveCamera::DIST_INVERSEEXPONENTIAL:
					camera.persp.bokehDistribution = slg::ocl::DIST_INVERSEEXPONENTIAL;
					break;
				case PerspectiveCamera::DIST_GAUSSIAN:
					camera.persp.bokehDistribution = slg::ocl::DIST_GAUSSIAN;
					break;
				case PerspectiveCamera::DIST_INVERSEGAUSSIAN:
					camera.persp.bokehDistribution = slg::ocl::DIST_INVERSEGAUSSIAN;
					break;
				case PerspectiveCamera::DIST_TRIANGULAR:
					camera.persp.bokehDistribution = slg::ocl::DIST_TRIANGULAR;
					break;
				case PerspectiveCamera::DIST_CUSTOM: {
					camera.persp.bokehDistribution = slg::ocl::DIST_CUSTOM;

					std::tie(cameraBokehDistribution, cameraBokehDistributionSize) =
						CompileDistribution2D(*perspCamera.bokehDistributionMap);
					break;
				}
				default:
					throw runtime_error("Unknown bokeh distribution type in CompiledScene::CompileCamera(): " + ToString(perspCamera.bokehDistribution));
			}
			camera.persp.bokehScaleX = perspCamera.bokehScaleX;
			camera.persp.bokehScaleY = perspCamera.bokehScaleY;

			CompilePerspectiveLensModel(perspCamera, camera.persp, cameraBokehDistribution);
			break;
		}
		case Camera::REALISTIC: {
			auto& realisticCamera = dynamic_cast<const RealisticCamera&>(sceneCamera);
			camera.type = slg::ocl::REALISTIC;

			memcpy(camera.base.rasterToCamera.m.m, realisticCamera.GetRasterToCamera().m.m, 4 * 4 * sizeof(float));
			memcpy(camera.base.cameraToWorld.m.m, realisticCamera.GetCameraToWorld().m.m, 4 * 4 * sizeof(float));

			camera.realistic = realisticCamera.GetLensCamera();
			// After the custom bokeh distribution, if any
			const u_int offset = cameraBokehDistribution.size();
			camera.realistic.surfacesOffset += offset;
			camera.realistic.pupilBoundsOffset += offset;
			const vector<float> &lensData = realisticCamera.GetLensData();
			cameraBokehDistribution.insert(cameraBokehDistribution.end(), lensData.begin(), lensData.end());

			camera.realistic.projCamera.lensRadius = realisticCamera.lensRadius;
			camera.realistic.projCamera.focalDistance = realisticCamera.focalDistance;
			if (realisticCamera.enableClippingPlane) {
				camera.realistic.projCamera.enableClippingPlane = true;
				ASSIGN_VECTOR(camera.realistic.projCamera.clippingPlaneCenter, realisticCamera.clippingPlaneCenter);
				ASSIGN_VECTOR(camera.realistic.projCamera.clippingPlaneNormal, realisticCamera.clippingPlaneNormal);
			} else
				camera.realistic.projCamera.enableClippingPlane = false;
			break;
		}
		case Camera::ENVIRONMENT: {
			auto& envCamera = dynamic_cast<const EnvironmentCamera&>(sceneCamera);
			camera.type = slg::ocl::ENVIRONMENT;

			memcpy(camera.base.rasterToCamera.m.m, envCamera.GetRasterToCamera().m.m, 4 * 4 * sizeof(float));
			memcpy(camera.base.cameraToWorld.m.m, envCamera.GetCameraToWorld().m.m, 4 * 4 * sizeof(float));

			camera.env.degrees = envCamera.degrees;
			break;
		}
		case Camera::STEREO: {
			auto& stereoCamera = dynamic_cast<const StereoCamera&>(sceneCamera);

			camera.type = slg::ocl::STEREO;

			switch(stereoCamera.GetStereoType()) {
				case StereoCamera::STEREO_PERSPECTIVE:
					camera.stereo.stereoCameraType = slg::ocl::STEREO_PERSPECTIVE;
					break;
				case StereoCamera::STEREO_ENVIRONMENT_180:
					camera.stereo.stereoCameraType = slg::ocl::STEREO_ENVIRONMENT_180;
					break;
				case StereoCamera::STEREO_ENVIRONMENT_360:
					camera.stereo.stereoCameraType = slg::ocl::STEREO_ENVIRONMENT_360;
					break;
				default:
					throw runtime_error("Unknown StereoCamera type in CompiledScene::CompileCamera(): " + ToString(stereoCamera.GetStereoType()));
			}

			camera.stereo.perspCamera.projCamera.lensRadius = stereoCamera.lensRadius;
			camera.stereo.perspCamera.projCamera.focalDistance = stereoCamera.focalDistance;

			memcpy(camera.stereo.leftEyeRasterToCamera.m.m, stereoCamera.GetRasterToCamera(0).m.m, 4 * 4 * sizeof(float));
			memcpy(camera.stereo.leftEyeCameraToWorld.m.m, stereoCamera.GetCameraToWorld(0).m.m, 4 * 4 * sizeof(float));
			memcpy(camera.stereo.rightEyeRasterToCamera.m.m, stereoCamera.GetRasterToCamera(1).m.m, 4 * 4 * sizeof(float));
			memcpy(camera.stereo.rightEyeCameraToWorld.m.m, stereoCamera.GetCameraToWorld(1).m.m, 4 * 4 * sizeof(float));

			camera.stereo.perspCamera.enableOculusRiftBarrel = stereoCamera.enableOculusRiftBarrel;
			// No aperture shape nor thin lens model
			camera.stereo.perspCamera.bokehRotation = 0.f;
			camera.stereo.perspCamera.bokehRoundness = 0.f;
			camera.stereo.perspCamera.bokehObstruction = 0.f;
			camera.stereo.perspCamera.enableLensModel = false;
			camera.stereo.perspCamera.vignettingTableOffset = NULL_INDEX;
			if (stereoCamera.enableClippingPlane) {
				camera.stereo.perspCamera.projCamera.enableClippingPlane = true;
				ASSIGN_VECTOR(camera.stereo.perspCamera.projCamera.clippingPlaneCenter, stereoCamera.clippingPlaneCenter);
				ASSIGN_VECTOR(camera.stereo.perspCamera.projCamera.clippingPlaneNormal, stereoCamera.clippingPlaneNormal);
			} else
				camera.stereo.perspCamera.projCamera.enableClippingPlane = false;
			break;
		}
		default:
			throw runtime_error("Unknown camera type in CompiledScene::CompileCamera(): " + ToString(sceneCamera.GetType()));
	}

	cameraBokehDistributionSize = cameraBokehDistribution.size() * sizeof(float);
}

#endif
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
