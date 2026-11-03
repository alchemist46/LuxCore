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

#ifndef _SLG_PERSPECTIVE_CAMERA_H
#define	_SLG_PERSPECTIVE_CAMERA_H

#include <string>

#include "luxrays/utils/mcdistribution.h"

#include "slg/imagemap/imagemap.h"
#include "slg/cameras/projective.h"

namespace slg {

//------------------------------------------------------------------------------
// PerspectiveCamera
//------------------------------------------------------------------------------

class PerspectiveCamera : public ProjectiveCamera {
public:
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

	PerspectiveCamera(const luxrays::Point &o, const luxrays::Point &t,
			const luxrays::Vector &u, const float *screenWindow = NULL);
	virtual ~PerspectiveCamera();

	virtual luxrays::Spectrum GenerateRay(const float time,
		const float filmX, const float filmY,
		luxrays::Ray *ray, PathVolumeInfo *volInfo,
		const float u0, const float u1, const float wavelength = 587.56f) const;
	virtual bool TintsPath() const;
	virtual bool SupportsLightTracing() const;

	virtual void ClampRay(luxrays::Ray *ray) const;
	virtual bool GetSamplePosition(luxrays::Ray *eyeRay, float *filmX, float *filmY) const;
	virtual bool LocalSampleLens(const float time, const float u1, const float u2,
		luxrays::Point *lensPoint) const;
	virtual bool SampleLens(const float time, const float u1, const float u2,
		luxrays::Point *lensPoint) const;
	virtual void GetPDF(const luxrays::Ray &eyeRay, const float eyeDistance,
		const float filmX, const float filmY,
		float *pdfW, float *fluxToRadianceFactor) const;

	virtual luxrays::PropertiesUPtr ToProperties(const ImageMapCache &imgMapCache, const bool useRealFileName) const;

	static BokehDistributionType String2BokehDistributionType(std::string type);
	static std::string BokehDistributionType2String(const BokehDistributionType type);
	
	float screenOffsetX, screenOffsetY;
	float fieldOfView;

	u_int bokehBlades, bokehPower;
	BokehDistributionType bokehDistribution;
	ImageMapConstPtr bokehDistributionImageMap;
	luxrays::Distribution2DUPtr bokehDistributionMap;
	float bokehScaleX, bokehScaleY;
	// The aperture shape: the rotation of the blades (degrees), the blend of
	// the polygon toward the circle, the radius of a central obstruction (a
	// catadioptric lens) in lens radii
	float bokehRotation, bokehRoundness, bokehObstruction;

	bool enableOculusRiftBarrel;

	// The thin lens model (ROADMAP F11), all of it off at 0:
	//  - the optical vignetting (cat's eye bokeh): a barrel opening of radius
	//    vignettingScale lens radii, vignettingDistance (scene units) in front
	//    of the lens;
	//  - the radial distortion of the film point (Brown-Conrady, the image
	//    height 1 at the corner of the frame), positive for barrel;
	//  - the Seidel aberrations: the focus point of a lens sample moves along
	//    the ray by spherical rho^2 + fieldCurvature h^2 + astigmatism h^2
	//    cos^2(theta) (fractions of the distance), and the image of a point
	//    across by the Seidel coma h rho^2 (2 + cos(2 theta), sin(2 theta))
	//    (lens radii, away from the axis for a positive coma), rho the lens
	//    radius fraction of the sample, theta its angle to the direction of
	//    the image point, h the image height;
	//  - the tilt (degrees) of the plane of focus about the x and y axes of
	//    the image (Scheimpflug);
	//  - the dispersion at the wavelength of the path, its normalized
	//    dispersion D (Lens_NormalizedDispersion(), 0.70 at 486 nm and -0.30
	//    at 656 nm): the focus moves by dispersionLongitudinal D (fractions of
	//    the distance), the image scales by 1 + dispersionLateral D (a
	//    positive value makes the blue image larger)
	float vignettingDistance, vignettingScale;
	float distortionK1, distortionK2;
	float aberrationSpherical, aberrationComa, aberrationAstigmatism, aberrationFieldCurvature;
	float tiltX, tiltY;
	float dispersionLongitudinal, dispersionLateral;

	// Set by Update()
	bool HasLensModel() const { return enableLensModel; }
	bool HasVignetting() const { return enableVignetting; }
	const std::vector<float> &GetVignettingTable() const { return vignettingTable; }
	float GetVignettingMaxOffset() const { return vignettingMaxOffset; }
	float GetInvHalfDiagonal() const { return invHalfDiagonal; }
	const luxrays::Vector &GetFocusNormal() const { return focusNormal; }

protected:
	// Used by sub-classes
	PerspectiveCamera(const CameraType type, const luxrays::Point &o, const luxrays::Point &t,
			const luxrays::Vector &u, const float *region = NULL);

private:
	static void OculusRiftBarrelPostprocess(const float x, const float y, float *barrelX, float *barrelY);

	virtual void InitCameraTransforms(CameraTransforms *trans);
	virtual void InitCameraData();
	virtual void InitRay(luxrays::Ray *ray, const float filmX, const float filmY) const;

	// A point of the aperture in lens radii
	void SampleAperture(const float u1, const float u2, float *x, float *y) const;
	void InitVignettingTable();
	bool GetSamplePositionLensModel(luxrays::Ray *eyeRay, float *filmX, float *filmY) const;
	// The determinant of the Jacobian of the distortion at the film point
	float DistortionJacobian(const float filmX, const float filmY) const;

	float pixelArea;

	bool enableLensModel, enableVignetting;
	std::vector<float> vignettingTable;
	float vignettingMaxOffset;
	float invHalfDiagonal;
	luxrays::Vector focusNormal;
	luxrays::Transform worldToCamera, cameraToRaster;
};

}

#endif	/* _SLG_PERSPECTIVE_CAMERA_H */
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
