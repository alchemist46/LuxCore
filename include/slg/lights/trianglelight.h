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

#ifndef _SLG_TRIANGLELIGHT_H
#define	_SLG_TRIANGLELIGHT_H

#include "slg/usings.h"
#include "slg/lights/light.h"
#include "slg/scene/sceneobject.h"

namespace slg {


//------------------------------------------------------------------------------
// TriangleLight implementation
//------------------------------------------------------------------------------

// The solid angle sampling of a triangle seen from a point (the code shared
// with the GPU, sphericaltriangle_funcs.cl). SphericalTriangle_Sample() takes
// one sample, split in two coordinates.
float SphericalTriangle_SolidAngle(const luxrays::Point &v0, const luxrays::Point &v1,
		const luxrays::Point &v2, const luxrays::Point &p);
bool SphericalTriangle_UseSolidAngle(const float solidAngle);
bool SphericalTriangle_Sample(const luxrays::Point &v0, const luxrays::Point &v1,
		const luxrays::Point &v2, const luxrays::Point &p, const float u,
		float *b0, float *b1, float *b2, float *pdfW);

class TriangleLight : public IntersectableLightSource {
public:
	TriangleLight();
	virtual ~TriangleLight();

	virtual void Preprocess();

	virtual LightSourceType GetType() const { return TYPE_TRIANGLE; }

	virtual bool IsDirectLightSamplingEnabled() const;

	float GetTriangleArea() const { return triangleArea; }
	float GetMeshArea() const { return meshArea; }

	virtual float GetArea() const { return triangleArea; }
	virtual float GetPower(SceneConstRef scene) const;

	virtual luxrays::Spectrum Emit(SceneConstRef scene,
		const float time, const float u0, const float u1,
		const float u2, const float u3, const float passThroughEvent,
		luxrays::Ray &ray, float &emissionPdfW,
		float *directPdfA = NULL, float *cosThetaAtLight = NULL) const;

    virtual luxrays::Spectrum Illuminate(SceneConstRef scene, const BSDF &bsdf,
		const float time, const float u0, const float u1, const float passThroughEvent,
        luxrays::Ray &shadowRay, float &directPdfW,
		float *emissionPdfW = NULL, float *cosThetaAtLight = NULL) const;

	virtual bool IsAlwaysInShadow(SceneConstRef scene,
			const luxrays::Point &p, const luxrays::Normal &n) const;

	virtual luxrays::Spectrum GetRadiance(const HitPoint &hitPoint,
			float *directPdfA = NULL,
			float *emissionPdfW = NULL) const;

	// The solid angle pdf of Illuminate() for a direct hit on the light from
	// refPoint (the MIS weight of the BSDF sampling)
	float GetDirectPdfW(const HitPoint &hitPoint, const luxrays::Point &refPoint,
			const float distance, const float cosAtLight) const;

	bool UsesSolidAngleSampling() const;

	SceneObjectConstPtr sceneObject;
	// Note: meshIndex is initialized in LightSourceDefinitions::Preprocess()
	u_int meshIndex, triangleIndex;
	// Solid angle sampling of the direct light, set by
	// LightSourceDefinitions::Preprocess() (lightstrategy.trianglelight.solidangle)
	bool solidAngleSampling;

private:
	float triangleArea, invTriangleArea;
	float meshArea, invMeshArea;
};

}

#endif	/* _SLG_TRIANGLELIGHT_H */
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
