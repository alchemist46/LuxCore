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

#ifndef _SLG_BSDF_H
#define	_SLG_BSDF_H

#include "luxrays/luxrays.h"
#include "luxrays/core/geometry/frame.h"
#include "luxrays/core/exttrianglemesh.h"
#include "luxrays/core/color/color.h"
#include "luxrays/usings.h"
#include "slg/lights/light.h"
#include "slg/lights/trianglelight.h"
#include "slg/materials/material.h"
#include "slg/volumes/volume.h"
#include "slg/bsdf/bsdfevents.h"
#include "slg/bsdf/hitpoint.h"
#include "slg/scene/sceneobject.h"
#include "slg/utils/pathvolumeinfo.h"

namespace slg {

// OpenCL data types
namespace ocl {
#include "slg/bsdf/bsdf_types.cl"
}

typedef enum {
	NO_REFLECT_TRANSMIT, ONLY_REFLECT, ONLY_TRANSMIT, REFLECT_TRANSMIT
} AlbedoSpecularSetting;

extern AlbedoSpecularSetting String2AlbedoSpecularSetting(const std::string &type);
extern const std::string AlbedoSpecularSetting2String(const AlbedoSpecularSetting type);

class Scene;

class BSDF {
public:
	// An empty BSDF
	BSDF() { };

	// A BSDF initialized from a ray hit
	BSDF(const bool fixedFromLight, const bool throughShadowTransparency,
		SceneConstRef scene, const luxrays::Ray &ray,
		const luxrays::RayHit &rayHit, const float passThroughEvent,
		const PathVolumeInfo *volInfo)
	{
		assert (!rayHit.Miss());
		Init(fixedFromLight, throughShadowTransparency, scene,
				ray, rayHit, passThroughEvent, volInfo);
	}
	// A BSDF initialized with a point on a surface
	BSDF(SceneConstRef scene,
		const u_int meshIndex, const u_int triangleIndex,
		const luxrays::Point &surfacePoint,
		const float surfacePointBary1, const float surfacePointBary2,
		const float time,
		const float passThroughEvent, const PathVolumeInfo *volInfo) {
		Init(scene, meshIndex, triangleIndex,
				surfacePoint, surfacePointBary1, surfacePointBary2,
				time, passThroughEvent, volInfo);
	}
	// A BSDF initialized with a volume scattering point
	BSDF(const bool fixedFromLight, const bool throughShadowTransparency,
		SceneConstRef scene, const luxrays::Ray &ray,
		VolumeConstRef volume, const float t, const float passThroughEvent) {
		Init(fixedFromLight, throughShadowTransparency,
				scene, ray, volume, t, passThroughEvent);
	}

	// Used when hitting a surface
	void Init(
		const bool fixedFromLight,
		const bool throughShadowTransparency,
		SceneConstRef scene,
		const luxrays::Ray &ray,
		const luxrays::RayHit &rayHit,
		const float passThroughEvent,
		const PathVolumeInfo *volInfo
	);
	// Used when have a point of a surface
	void Init(SceneConstRef scene,
		const u_int meshIndex, const u_int triangleIndex,
		const luxrays::Point &surfacePoint,
		const float surfacePointBary1, const float surfacePointBary2,
		const float time,
		const float passThroughEvent, const PathVolumeInfo *volInfo);
	// Used when hitting a volume scatter point
	void Init(const bool fixedFromLight, const bool throughShadowTransparency,
		SceneConstRef scene, const luxrays::Ray &ray,
		VolumeConstRef volume, const float t, const float passThroughEvent);

	void MoveHitPoint(const luxrays::Point &p, const luxrays::Normal &n);
	// A new pass-through event for a surface hit point: the volumes of a
	// material may depend on it (Mix), so they are chosen again
	void SetPassThroughEvent(SceneConstRef scene, const float passThroughEvent,
			const PathVolumeInfo *volInfo);
	
	bool IsEmpty() const { return (not material); }
	bool IsLightSource() const { return material->IsLightSource(); }
	// At a regularized vertex (see HitPoint::regularization) the delta lobes
	// of glass and mirror are GGX lobes
	bool IsDelta() const {
		return (hitPoint.regularization > 0.f) ? material->IsRegularizedDelta() : material->IsDelta();
	}
	bool IsVisibleIndirectDiffuse() const { return material->IsVisibleIndirectDiffuse(); }
	bool IsVisibleIndirectGlossy() const { return material->IsVisibleIndirectGlossy(); }
	bool IsVisibleIndirectSpecular() const { return material->IsVisibleIndirectSpecular(); }
	bool IsShadowCatcher() const { return material->IsShadowCatcher(); }
	bool IsShadowCatcherOnlyInfiniteLights() const { return material->IsShadowCatcherOnlyInfiniteLights(); }
	bool IsCameraInvisible() const;
	bool IsVolume() const {
		auto ptr = dynamic_observer_cast<const Volume>(material);
		return bool(ptr);
	}
	bool IsPhotonGIEnabled() const { return material->IsPhotonGIEnabled(); }
	bool IsHoldout() const { return material->IsHoldout(); }
	bool IsAlbedoEndPoint(const AlbedoSpecularSetting albedoSpecularSetting,
		const float albedoSpecularGlossinessThreshold) const;
	u_int GetObjectID() const;
	const std::string &GetMaterialName() const;
	u_int GetMaterialID() const { return material->GetID(); }
	u_int GetLightID() const { return material->GetLightID(); }

	VolumeConstPtr GetMaterialInteriorVolume() const {
		return material->GetInteriorVolume(hitPoint, hitPoint.passThroughEvent);
	}
	VolumeConstPtr GetMaterialExteriorVolume() const {
		return material->GetExteriorVolume(hitPoint, hitPoint.passThroughEvent);
	}

	float GetGlossiness() const {
		return (hitPoint.regularization > 0.f) ?
			luxrays::Max(material->GetGlossiness(), hitPoint.regularization) : material->GetGlossiness();
	}
	// The glossiness without the path regularization. The eye path classifies
	// its caustic paths with it (the light tracing of the hybrid back/forward
	// rendering and the PhotonGI caustic cache render them, on paths that
	// are never regularized): a regularized delta lobe is a GLOSSY event of
	// glossiness 0, so it stays nearly specular.
	float GetUnregularizedGlossiness() const { return material->GetGlossiness(); }
	// The factor undoing the eta^2 radiance scaling of a transmission through
	// this surface (1 for the materials without refraction), see
	// EyePathInfo::etaScale
	float GetRefractionEtaScale() const;
	SceneObjectConstRef GetSceneObject() const { return *sceneObject; }

	BSDFEvent GetEventTypes() const {
		return (hitPoint.regularization > 0.f) ? material->GetRegularizedEventTypes() : material->GetEventTypes();
	}
	MaterialType GetMaterialType() const { return material->GetType(); }
	// An eye path entering the material starts the subsurface random walk
	// (Material::GetSubsurfaceWalkMaterial()), in the medium of
	// GetSubsurfaceWalkParams() at this vertex
	bool HasSubsurfaceWalk() const;
	// Material::HasDiffuseBesideSubsurfaceWalk()
	bool HasDiffuseBesideSubsurfaceWalk() const;
	void GetSubsurfaceWalkParams(luxrays::Spectrum *color, luxrays::Spectrum *radius,
		float *anisotropy) const;

	luxrays::Spectrum GetPassThroughTransparency(const bool backTracing) const;
	const luxrays::Spectrum &GetPassThroughShadowTransparency() const { return material->GetPassThroughShadowTransparency(); }
	const bool &GetPassThroughShadowTransparencyOverride() const { return material->GetPassThroughShadowTransparencyOverride(); }
	const luxrays::Frame &GetFrame() const { return frame; }

	luxrays::Spectrum Albedo() const;
	luxrays::Spectrum EvaluateTotal() const;
	luxrays::Spectrum Evaluate(const luxrays::Vector &generatedDir,
		BSDFEvent *event, float *directPdfW = NULL, float *reversePdfW = NULL) const;
	luxrays::Spectrum Sample(luxrays::Vector *sampledDir,
		const float u0, const float u1,
		float *pdfW, float *absCosSampledDir,
		BSDFEvent *event) const;
	luxrays::Spectrum ShadowCatcherSample(Vector *sampledDir,
		float *pdfW, float *absCosSampledDir, BSDFEvent *event) const;
	void Pdf(const luxrays::Vector &sampledDir, float *directPdfW, float *reversePdfW) const;

	luxrays::Spectrum GetEmittedRadiance(float *directPdfA = NULL, float *emissionPdfW = NULL) const ;

	LightSourceConstPtr GetLightSource() const { return triangleLightSource; }
	TriangleLightConstPtr GetTriangleLightSource() const { return triangleLightSource; }

	luxrays::Point GetRayOrigin(const luxrays::Vector &sampleDir) const {
		if (IsVolume())
			return hitPoint.p;
		else {
			// Rise the ray origin along the geometry normal to avoid self intersection
			const float riseDirection = (Dot(sampleDir, hitPoint.geometryN) > 0.f) ? 1.f : -1.f;

			return hitPoint.p + riseDirection * luxrays::Vector(hitPoint.geometryN * luxrays::MachineEpsilon::E(hitPoint.p));
		}
	}

	bool HasBakeMap(const BakeMapType type) const;
	luxrays::Spectrum GetBakeMapValue() const;

	HitPoint hitPoint;

private:
	// Nota: following properties are intrisecally const. However, due to
	// design particulars, they are not initialized in constructor, but in
	// Init(). Therefore they have to be modifiable somehow, thus the mutable
	// attribute (not fully satisfying, however).
	mutable SceneObjectConstPtr sceneObject;  // Optional reference, owned by scene
	mutable MaterialConstPtr material;  // Optional reference, owned by scene
	mutable TriangleLightConstPtr triangleLightSource; // != NULL only if it is an area light, optional, owned by scen
	luxrays::Frame frame;
};

}

#endif	/* _SLG_BSDF_H */
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
