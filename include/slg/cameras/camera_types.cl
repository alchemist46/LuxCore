#line 2 "camera_types.cl"

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

#define CAMERA_MAX_INTERPOLATED_TRANSFORM 8

typedef enum {
	PERSPECTIVE, ORTHOGRAPHIC, STEREO, ENVIRONMENT, REALISTIC
} CameraType;

typedef struct {
	// Placed here for Transform memory alignment
	Transform rasterToCamera;
	Transform cameraToWorld;

	float yon, hither;
	float shutterOpen, shutterClose;

	unsigned int volumeIndex;

	// The camera tints the path with the colour of its wavelength (a
	// dispersive lens, see GlassMaterial::WaveLengthColor())
	int tintPath;

	// Used for camera motion blur
	MotionSystem motionSystem;
	InterpolatedTransform interpolatedTransforms[CAMERA_MAX_INTERPOLATED_TRANSFORM];
} CameraBase;

typedef struct {
	// Used for camera clipping plane
	Point clippingPlaneCenter;
	Normal clippingPlaneNormal;
	int enableClippingPlane;

	float lensRadius;
	float focalDistance;
} ProjectiveCamera;

typedef enum {
	DIST_NONE,
	DIST_UNIFORM,
	DIST_EXPONENTIAL,
	DIST_INVERSEEXPONENTIAL,
	DIST_GAUSSIAN,
	DIST_INVERSEGAUSSIAN,
	DIST_TRIANGULAR,
	DIST_CUSTOM
} BokehDistributionType;

typedef struct {
	ProjectiveCamera projCamera;
	
	float screenOffsetX, screenOffsetY;
	float fieldOfView;

	unsigned int bokehBlades, bokehPower;
	BokehDistributionType bokehDistribution;
	float bokehScaleX, bokehScaleY;
	// The rotation in radians
	float bokehRotation, bokehRoundness, bokehObstruction;

	int enableOculusRiftBarrel;

	// The thin lens model (ROADMAP F11, see PerspectiveCamera), used if
	// enableLensModel is true
	int enableLensModel;
	// 1 / the half diagonal of the frame in tangent units (the image height
	// 1 at the corner)
	float invHalfDiagonal;
	float distortionK1, distortionK2;
	// The barrel distance (scene units) and radius (lens radii), the offset
	// of the table in the camera data buffer (NULL_INDEX without vignetting)
	float vignettingDistance, vignettingScale, vignettingMaxOffset;
	unsigned int vignettingTableOffset;
	Vector focusNormal;
	float aberrationSpherical, aberrationComa, aberrationAstigmatism, aberrationFieldCurvature;
	float dispersionLongitudinal, dispersionLateral;
} PerspectiveCamera;

// ROADMAP F12, see RealisticCamera: the sensor point of a pixel is
// rasterToCamera * the raster point
typedef struct {
	ProjectiveCamera projCamera;

	// The plane of the exit pupil bounds (the rear vertex), the film radius of
	// the last ring of the bounds, the effective focal length
	float rearZ, maxFilmRadius, focalLength;
	// The normalization of the ray weight: (center exposure)^-1
	float weightScale;
	float stopRadius, stopRotation, stopRoundness, stopObstruction;
	unsigned int stopBlades;
	// In the camera data buffer
	unsigned int surfacesOffset, surfaceCount, pupilBoundsOffset;
	int dispersion;
} RealisticCamera;

typedef struct {
	ProjectiveCamera projCamera;
} OrthographicCamera;

typedef struct {
	float degrees;
} EnvironmentCamera;

typedef enum {
	STEREO_PERSPECTIVE,
	STEREO_ENVIRONMENT_180,
	STEREO_ENVIRONMENT_360
} StereoCameraType;

typedef struct {
	StereoCameraType stereoCameraType;

	PerspectiveCamera perspCamera;

	Transform leftEyeRasterToCamera;
	Transform leftEyeCameraToWorld;

	Transform rightEyeRasterToCamera;
	Transform rightEyeCameraToWorld;
} StereoCamera;

typedef struct {
	CameraType type;

	CameraBase base;

	union {
		PerspectiveCamera persp;
		OrthographicCamera ortho;
		StereoCamera stereo;
		EnvironmentCamera env;
		RealisticCamera realistic;
	};
} Camera;
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
