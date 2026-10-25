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

// The sampling of the quad lights (sphericaltriangle_funcs.cl): the quad is
// c + u * ex + v * ey. QuadLight_Sample() returns the solid angle pdf, or 0
// for a point sampled by area.
float QuadLight_Sample(const luxrays::Point &c, const luxrays::Vector &ex,
		const luxrays::Vector &ey, const luxrays::Point &p, const float u0, const float u1,
		luxrays::Point *point, float *u, float *v);
float QuadLight_SolidAnglePdf(const luxrays::Point &c, const luxrays::Vector &ex,
		const luxrays::Vector &ey, const luxrays::Point &p);
bool QuadLight_GetTriangle(const float u, const float v,
		const u_int corner, const u_int partnerCorner, float *b1, float *b2);

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

	// The quads (see quadLight below)
	bool IsQuadPrimary() const { return quadLight == this; }
	bool IsQuadSecondary() const { return quadLight && (quadLight != this); }
	// The corner and the edges of the quad of the first triangle
	void GetQuadEdges(const luxrays::Transform &localToWorld, luxrays::Point *c,
			luxrays::Vector *ex, luxrays::Vector *ey) const;
	// The pdf of Illuminate() of the first triangle for a direct hit on the
	// quad from refPoint
	float GetQuadDirectPdfW(const luxrays::Transform &localToWorld, const luxrays::Point &refPoint,
			const float distance, const float cosAtLight) const;

	SceneObjectConstPtr sceneObject;
	// Note: meshIndex is initialized in LightSourceDefinitions::Preprocess()
	u_int meshIndex, triangleIndex;
	// Solid angle sampling of the direct light, set by
	// LightSourceDefinitions::Preprocess() (lightstrategy.trianglelight.solidangle)
	bool solidAngleSampling;

	// Two triangles of the mesh that make a planar parallelogram are one light
	// for the direct light sampling of the path engines, set by
	// LightSourceDefinitions::Preprocess() (lightstrategy.trianglelight.quad):
	// the strategies pick the quad as its first triangle (quadLight, nullptr
	// for a triangle out of any quad) with the importance of the quad and
	// never pick the second one, Illuminate() of the first triangle samples
	// the whole quad and a BSDF hit on either triangle takes its pdf. The
	// first triangle also holds quadPartner, the second triangle, the vertex
	// number of the corner c of the quad in its triangle (the other two are
	// c + ex and c + ey, in their cyclic order) and the vertex number of the
	// opposite corner c + ex + ey in the second triangle.
	const TriangleLight *quadLight;
	const TriangleLight *quadPartner;
	u_int quadCorner, quadPartnerCorner;

private:
	// The opacity of the light point seen from hitPoint.fixedDir (an estimate
	// of 0 or 1 for a transparency texture, see Material::GetPassThroughTransparency())
	luxrays::Spectrum GetOpacity(const HitPoint &hitPoint, const float passThroughEvent) const;

	float triangleArea, invTriangleArea;
	float meshArea, invMeshArea;
};

// For the strategies of the direct light sampling, which pick the quads of
// two triangle lights as one light (see TriangleLight::quadLight): the light
// picked for l (the first triangle of its quad), true for the second triangle
// of a quad (never picked) and the power of l (the one of the quad for its
// first triangle)
LightSourceConstRef GetIlluminatePickLight(LightSourceConstRef l);
bool IsIlluminateQuadSecondary(LightSourceConstRef l);
float GetIlluminatePower(SceneConstRef scene, LightSourceConstRef l);

}

#endif	/* _SLG_TRIANGLELIGHT_H */
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
