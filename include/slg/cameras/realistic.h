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

#ifndef _SLG_REALISTIC_CAMERA_H
#define	_SLG_REALISTIC_CAMERA_H

#include <string>
#include <vector>

#include "slg/cameras/projective.h"

namespace slg {

//------------------------------------------------------------------------------
// RealisticCamera (ROADMAP F12): the rays traced from the sensor through the
// spherical surfaces of a lens prescription (Kolb et al. 1995, pbrt-v4's
// RealisticCamera). The vignetting, the aberrations, the distortion and the
// field curvature come from the lens, with no other control.
//
// The camera origin is the entrance pupil of the lens: the framing matches
// the perspective camera of the same focal length and sensor, the focus
// distance is measured from it. The film moves to focus. The scene units are
// meters. The exposure is normalized to the one of the thin lens at the
// center of the image.
//
// The lens mapping has no inverse: no light tracing (LIGHTCPU, BIDIRCPU, the
// hybrid back/forward of the path tracers).
//------------------------------------------------------------------------------

class RealisticCamera : public ProjectiveCamera {
public:
	// A surface of a lens prescription, from the front (mm): the radius of
	// curvature (0 for the aperture stop), the distance to the next surface
	// (to the film for the last one, which the focus overrides), the IOR at
	// the d line and the Abbe number of the medium after it (1 and 0 for
	// air), the semi-aperture
	typedef struct {
		float radius, thickness, ior, abbe, semiAperture;
	} LensSurface;

	RealisticCamera(const luxrays::Point &o, const luxrays::Point &t,
			const luxrays::Vector &u, const float *screenWindow = nullptr);
	virtual ~RealisticCamera();

	virtual void UpdateAuto(SceneConstRef scene, luxrays::IntersectionDevice *device = nullptr);

	virtual luxrays::Spectrum GenerateRay(const float time,
		const float filmX, const float filmY,
		luxrays::Ray *ray, PathVolumeInfo *volInfo,
		const float u0, const float u1, const float wavelength = 587.56f) const;
	virtual bool TintsPath() const { return dispersion; }
	virtual bool SupportsLightTracing() const { return false; }

	virtual bool GetSamplePosition(luxrays::Ray *eyeRay, float *filmX, float *filmY) const;
	virtual bool LocalSampleLens(const float time, const float u1, const float u2,
		luxrays::Point *lensPoint) const;
	virtual bool SampleLens(const float time, const float u1, const float u2,
		luxrays::Point *lensPoint) const;
	virtual void GetPDF(const luxrays::Ray &eyeRay, const float eyeDistance,
		const float filmX, const float filmY,
		float *pdfW, float *fluxToRadianceFactor) const;

	virtual luxrays::PropertiesUPtr ToProperties(const ImageMapCache &imgMapCache, const bool useRealFileName) const;

	// The lens file: a surface per line, "radius thickness nd vd
	// semi-aperture" (mm), or pbrt's "radius thickness eta
	// aperture-diameter"; # starts a comment
	static std::vector<LensSurface> ReadLensFile(const std::string &fileName);
	// The built-in lens designs (see realistic.cpp): "dgauss", "triplet",
	// "tessar", "heliar", "petzval", "telephoto", "retrofocus", "fisheye"
	static std::vector<LensSurface> GetLensPreset(const std::string &name);

	// User defined values
	std::string lensFileName, lensPreset;
	std::vector<LensSurface> lens;
	// The focal length the lens is scaled to (mm, 0 for its own), the sensor
	// width (mm), the f-number (0 for the widest)
	float focalLength, sensorWidth, fStop;
	// The shape of the aperture stop (see Lens_ApertureRadius()), the
	// rotation in degrees
	u_int stopBlades;
	float stopRotation, stopRoundness, stopObstruction;
	// The IOR of the glasses at the wavelength of the path
	bool dispersion;

	// Set by Update(): the data of the GPU camera (camera_types.cl) and of the
	// camera data buffer (the surfaces, the exit pupil bounds), which the CPU
	// uses too
	const ocl::RealisticCamera &GetLensCamera() const { return lensCamera; }
	const std::vector<float> &GetLensData() const { return lensData; }
	// The effective focal length (mm) and f-number of the lens
	float GetEffectiveFocalLength() const { return effectiveFocalLength; }
	float GetEffectiveFStop() const { return effectiveFStop; }

private:
	virtual void InitCameraTransforms(CameraTransforms *trans);
	virtual void InitCameraData();
	virtual void InitRay(luxrays::Ray *ray, const float filmX, const float filmY) const;

	// Sets up the lens for the focus, the f-number and the film, unless they
	// did not change since the last call
	void UpdateLens();

	ocl::RealisticCamera lensCamera;
	std::vector<float> lensData;
	float filmZ, effectiveFocalLength, effectiveFStop;
	// The values UpdateLens() set up the lens for
	std::string lensKey;
};

}

#endif	/* _SLG_REALISTIC_CAMERA_H */

// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
