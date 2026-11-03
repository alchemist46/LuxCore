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

#include <memory>

#include "slg/cameras/environment.h"
#include "slg/cameras/perspective.h"
#include "slg/cameras/orthographic.h"
#include "slg/cameras/realistic.h"
#include "slg/cameras/stereo.h"
#include "slg/scene/scene.h"
#include "slg/utils/filenameresolver.h"

using namespace std;
using namespace luxrays;
using namespace slg;

void Scene::ParseCamera(const Properties &props) {
	if (!props.HaveNames("scene.camera")) {
		// There is no camera definition
		return;
	}

	auto newCamera = CreateCamera(props);

	// Use the new camera
	camera = std::move(newCamera);

	editActions.AddAction(CAMERA_EDIT);
}

CameraUPtr Scene::CreateCamera(const Properties &props) {
	Point orig, target;
	Vector up;
	if (props.IsDefined("scene.camera.lookat")) {
		SDL_LOG("WARNING: deprecated property scene.camera.lookat");

		const Property &prop = props.Get("scene.camera.lookat");
		orig.x = prop.Get<double>(0);
		orig.y = prop.Get<double>(1);
		orig.z = prop.Get<double>(2);
		target.x = prop.Get<double>(3);
		target.y = prop.Get<double>(4);
		target.z = prop.Get<double>(5);
		up = Vector(0.f, 0.f, 1.f);
	} else {
		if (!props.IsDefined("scene.camera.lookat.orig")) {
			// This is a trick I use from LuxCoreRenderer to set cameraToWorld to
			// identity matrix.
			//
			// I leave orig, target and up all set to (0.f, 0.f, 0.f)
		} else {
			orig = props.Get(Property("scene.camera.lookat.orig")(0.f, 10.f, 0.f)).Get<Point>();
			target = props.Get(Property("scene.camera.lookat.target")(0.f, 0.f, 0.f)).Get<Point>();
			up = props.Get(Property("scene.camera.up")(0.f, 0.f, 1.f)).Get<Vector>();
		}
	}

	const string type = props.Get(Property("scene.camera.type")("perspective")).Get<string>();

	SDL_LOG("Camera type: " << type);
	SDL_LOG("Camera position: " << orig);
	SDL_LOG("Camera target: " << target);

	CameraUPtr camera;
	if ((type == "environment") || (type == "orthographic") || (type == "perspective") || (type == "stereo") ||
			(type == "realistic")) {
		if (type == "orthographic") {
			OrthographicCamera *orthoCamera;

			if (props.IsDefined("scene.camera.screenwindow")) {
				float screenWindow[4];

				const Property defaultProp = Property("scene.camera.screenwindow")(0.f, 1.f, 0.f, 1.f);
				const Property &prop = props.Get(defaultProp);
				screenWindow[0] = prop.Get<double>(0);
				screenWindow[1] = prop.Get<double>(1);
				screenWindow[2] = prop.Get<double>(2);
				screenWindow[3] = prop.Get<double>(3);

				orthoCamera = new OrthographicCamera(orig, target, up, &screenWindow[0]);
			} else
				orthoCamera = new OrthographicCamera(orig, target, up);

			camera.reset(orthoCamera);
		} else if (type == "perspective") {
			PerspectiveCamera *perspCamera;

			if (props.IsDefined("scene.camera.screenwindow")) {
				float screenWindow[4];

				const Property &prop = props.Get(Property("scene.camera.screenwindow")(0.f, 1.f, 0.f, 1.f));
				screenWindow[0] = prop.Get<double>(0);
				screenWindow[1] = prop.Get<double>(1);
				screenWindow[2] = prop.Get<double>(2);
				screenWindow[3] = prop.Get<double>(3);

				perspCamera = new PerspectiveCamera(orig, target, up, &screenWindow[0]);
			} else 
				perspCamera = new PerspectiveCamera(orig, target, up);
			camera.reset(perspCamera);

			perspCamera->fieldOfView = Clamp(props.Get(Property("scene.camera.fieldofview")(45.0)).Get<double>(),
					DEFAULT_EPSILON_STATIC_DBL, 180.0 - DEFAULT_EPSILON_STATIC_DBL);

			perspCamera->bokehBlades = props.Get(Property("scene.camera.bokeh.blades")(0u)).Get<u_int>();
			perspCamera->bokehPower = props.Get(Property("scene.camera.bokeh.power")(3u)).Get<u_int>();

			perspCamera->bokehDistribution = PerspectiveCamera::String2BokehDistributionType(
					props.Get(Property("scene.camera.bokeh.distribution.type")("NONE")).Get<string>());
			if (perspCamera->bokehDistribution == PerspectiveCamera::DIST_CUSTOM) {
				const string imgMapName = props.Get(Property("scene.camera.bokeh.distribution.image")("image.png")).Get<string>();

				ImageMapConfig imgCfg(props, "scene.camera.bokeh.distribution.image");
				// Force float storage
				imgCfg.SetStorageType(ImageMapStorage::FLOAT);

				perspCamera->bokehDistributionImageMap = &imgMapCache.GetImageMap(imgMapName, imgCfg, false);

				if (perspCamera->bokehDistributionImageMap->GetSpectrumMean() == 0.f)
					throw runtime_error("Used a black image in camera bokeh distribution: " + imgMapName);
			}

			perspCamera->bokehScaleX = props.Get(Property("scene.camera.bokeh.scale.x")(1.0)).Get<double>();
			perspCamera->bokehScaleY = props.Get(Property("scene.camera.bokeh.scale.y")(1.0)).Get<double>();

			perspCamera->enableOculusRiftBarrel = props.Get(Property("scene.camera.oculusrift.barrelpostpro.enable")(false)).Get<bool>();

			// The aperture shape and the thin lens model (ROADMAP F11)
			perspCamera->bokehRotation = props.Get(Property("scene.camera.bokeh.rotation")(0.0)).Get<double>();
			perspCamera->bokehRoundness = Clamp(props.Get(Property("scene.camera.bokeh.roundness")(0.0)).Get<double>(), 0.0, 1.0);
			perspCamera->bokehObstruction = Clamp(props.Get(Property("scene.camera.bokeh.obstruction")(0.0)).Get<double>(), 0.0, .95);
			perspCamera->vignettingDistance = props.Get(Property("scene.camera.opticalvignetting.distance")(0.0)).Get<double>();
			perspCamera->vignettingScale = Max(props.Get(Property("scene.camera.opticalvignetting.scale")(1.0)).Get<double>(), 1e-3);
			perspCamera->distortionK1 = props.Get(Property("scene.camera.distortion.k1")(0.0)).Get<double>();
			perspCamera->distortionK2 = props.Get(Property("scene.camera.distortion.k2")(0.0)).Get<double>();
			perspCamera->aberrationSpherical = props.Get(Property("scene.camera.aberration.spherical")(0.0)).Get<double>();
			perspCamera->aberrationComa = props.Get(Property("scene.camera.aberration.coma")(0.0)).Get<double>();
			perspCamera->aberrationAstigmatism = props.Get(Property("scene.camera.aberration.astigmatism")(0.0)).Get<double>();
			perspCamera->aberrationFieldCurvature = props.Get(Property("scene.camera.aberration.fieldcurvature")(0.0)).Get<double>();
			perspCamera->tiltX = Clamp(props.Get(Property("scene.camera.tilt.x")(0.0)).Get<double>(), -80.0, 80.0);
			perspCamera->tiltY = Clamp(props.Get(Property("scene.camera.tilt.y")(0.0)).Get<double>(), -80.0, 80.0);
			perspCamera->dispersionLongitudinal = props.Get(Property("scene.camera.dispersion.longitudinal")(0.0)).Get<double>();
			perspCamera->dispersionLateral = props.Get(Property("scene.camera.dispersion.lateral")(0.0)).Get<double>();
		} else if (type == "realistic") {
			// ROADMAP F12
			RealisticCamera *realisticCamera;
			if (props.IsDefined("scene.camera.screenwindow")) {
				float screenWindow[4];

				const Property &prop = props.Get(Property("scene.camera.screenwindow")(0.f, 1.f, 0.f, 1.f));
				screenWindow[0] = prop.Get<double>(0);
				screenWindow[1] = prop.Get<double>(1);
				screenWindow[2] = prop.Get<double>(2);
				screenWindow[3] = prop.Get<double>(3);

				realisticCamera = new RealisticCamera(orig, target, up, &screenWindow[0]);
			} else
				realisticCamera = new RealisticCamera(orig, target, up);
			camera.reset(realisticCamera);

			realisticCamera->lensFileName = props.Get(Property("scene.camera.realistic.lens.file")("")).Get<string>();
			if (!realisticCamera->lensFileName.empty())
				realisticCamera->lens = RealisticCamera::ReadLensFile(SLG_FileNameResolver.ResolveFile(realisticCamera->lensFileName));
			else {
				realisticCamera->lensPreset = props.Get(Property("scene.camera.realistic.lens.preset")("dgauss")).Get<string>();
				realisticCamera->lens = RealisticCamera::GetLensPreset(realisticCamera->lensPreset);
			}
			realisticCamera->focalLength = Max(props.Get(Property("scene.camera.realistic.focallength")(0.0)).Get<double>(), 0.0);
			realisticCamera->sensorWidth = Max(props.Get(Property("scene.camera.realistic.sensorwidth")(36.0)).Get<double>(), .1);
			realisticCamera->fStop = Max(props.Get(Property("scene.camera.realistic.fstop")(0.0)).Get<double>(), 0.0);
			realisticCamera->dispersion = props.Get(Property("scene.camera.realistic.dispersion")(false)).Get<bool>();

			// The shape of the aperture stop
			realisticCamera->stopBlades = Min(props.Get(Property("scene.camera.bokeh.blades")(0u)).Get<u_int>(), 64u);
			realisticCamera->stopRotation = props.Get(Property("scene.camera.bokeh.rotation")(0.0)).Get<double>();
			realisticCamera->stopRoundness = Clamp(props.Get(Property("scene.camera.bokeh.roundness")(0.0)).Get<double>(), 0.0, 1.0);
			realisticCamera->stopObstruction = Clamp(props.Get(Property("scene.camera.bokeh.obstruction")(0.0)).Get<double>(), 0.0, .95);
		} else if (type == "stereo")  {
			const string stereoTypeStr = props.Get(Property("scene.camera.stereo.type")("perspective")).Get<string>();	

			StereoCamera *stereoCamera;
			if (stereoTypeStr == "perspective")
				stereoCamera = new StereoCamera(StereoCamera::STEREO_PERSPECTIVE, orig, target, up);
			else if (stereoTypeStr == "environment_180")
				stereoCamera = new StereoCamera(StereoCamera::STEREO_ENVIRONMENT_180, orig, target, up);
			else if (stereoTypeStr == "environment_360")
				stereoCamera = new StereoCamera(StereoCamera::STEREO_ENVIRONMENT_360, orig, target, up);
			else
				throw runtime_error("Unknown StereoCamera type: " + stereoTypeStr);
			camera.reset(stereoCamera);
			stereoCamera->enableOculusRiftBarrel = props.Get(Property("scene.camera.oculusrift.barrelpostpro.enable")(false)).Get<bool>();
			stereoCamera->horizStereoEyesDistance = props.Get(Property("scene.camera.eyesdistance")(.0626f)).Get<double>();
			stereoCamera->horizStereoLensDistance = props.Get(Property("scene.camera.lensdistance")(.2779f)).Get<double>();
		} else {
			EnvironmentCamera *environmentCamera;

			if (props.IsDefined("scene.camera.screenwindow")) {
				float screenWindow[4];

				const Property &prop = props.Get(Property("scene.camera.screenwindow")(-1.f, 1.f, -1.f, 1.f));
				screenWindow[0] = prop.Get<double>(0);
				screenWindow[1] = prop.Get<double>(1);
				screenWindow[2] = prop.Get<double>(2);
				screenWindow[3] = prop.Get<double>(3);

				environmentCamera = new EnvironmentCamera(orig, target, up, &screenWindow[0]);
			} else
				environmentCamera = new EnvironmentCamera(orig, target, up);

			camera.reset(environmentCamera);
			
			environmentCamera->degrees = Clamp(props.Get(Property("scene.camera.environment.degrees")(360.0)).Get<double>(),
					0.0, 360.0);
		};

		if (type != "environment") {
			ProjectiveCamera *projCamera = (ProjectiveCamera *)camera.get();

			projCamera->lensRadius = props.Get(Property("scene.camera.lensradius")(0.0)).Get<double>();
			projCamera->focalDistance = props.Get(Property("scene.camera.focaldistance")(10.0)).Get<double>();
			projCamera->autoFocus = props.Get(Property("scene.camera.autofocus.enable")(false)).Get<bool>();

			// Check if I have to arbitrary clipping plane
			if (props.Get(Property("scene.camera.clippingplane.enable")(false)).Get<bool>()) {
				projCamera->clippingPlaneCenter = props.Get(Property("scene.camera.clippingplane.center")(Point())).Get<Point>();
				projCamera->clippingPlaneNormal = props.Get(Property("scene.camera.clippingplane.normal")(Normal())).Get<Normal>();
				projCamera->enableClippingPlane = true;
				SDL_LOG("Camera clipping plane enabled");
			}
			else {
				projCamera->enableClippingPlane = false;
				SDL_LOG("Camera clipping plane disabled");
			}
		}
	} else
		throw runtime_error("Unknown camera type: " + type);
	
	camera->clipHither = props.Get(Property("scene.camera.cliphither")(1e-3)).Get<double>();
	camera->clipYon = props.Get(Property("scene.camera.clipyon")(1e30)).Get<double>();
	camera->shutterOpen = props.Get(Property("scene.camera.shutteropen")(0.0)).Get<double>();
	camera->shutterClose = props.Get(Property("scene.camera.shutterclose")(1.0)).Get<double>();

	camera->autoVolume = props.Get(Property("scene.camera.autovolume.enable")(true)).Get<bool>();
	if (!camera->autoVolume && props.IsDefined("scene.camera.volume")) {
		MaterialConstRef vol = matDefs.GetMaterial(props.Get("scene.camera.volume").Get<string>());
		try {
			camera->SetVolume(dynamic_cast<const Volume&>(vol));
		} catch (std::bad_cast&) {
			throw runtime_error("Camera volume is a material: " + vol.GetName());
		}
	}

	// Check if I have to use a motion system
	if (props.IsDefined("scene.camera.motion.0.time")) {
		// Build the motion system
		vector<float> times;
		vector<Transform> transforms;
		for (u_int i = 0;; ++i) {
			const string prefix = "scene.camera.motion." + ToString(i);
			if (!props.IsDefined(prefix +".time"))
				break;

			const float t = props.Get(prefix +".time").Get<double>();
			if (i > 0 && t <= times.back())
				throw runtime_error("Motion camera time must be monotonic");
			times.push_back(t);

			const Matrix4x4 mat = props.Get(Property(prefix +
				".transformation")(Matrix4x4::MAT_IDENTITY)).Get<Matrix4x4>();
			// NOTE: Transform for MotionSystem are global2local for scene objects
			// and not local2global for camera
			transforms.push_back(Transform(mat));
		}

		camera->motionSystem = new MotionSystem(times, transforms);
	}
	
	// I update the camera with dummy film width/height to initialize at least
	// all no raster related transformations
	camera->Update(100u, 100u, nullptr);

	return camera;
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
