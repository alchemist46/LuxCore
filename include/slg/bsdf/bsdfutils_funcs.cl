#line 2 "bsdfutils_funcs.cl"

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

// At a regularized vertex (see HitPoint::regularization) the delta lobes of
// glass and mirror are GGX lobes
OPENCL_FORCE_INLINE BSDFEvent BSDF_GetEventTypes(__global const BSDF *bsdf
		MATERIALS_PARAM_DECL) {
	__global const Material *mat = &mats[bsdf->materialIndex];

	return (bsdf->hitPoint.regularization > 0.f) ? mat->regularizedEventTypes : mat->eventTypes;
}

OPENCL_FORCE_INLINE bool BSDF_IsDelta(__global const BSDF *bsdf
		MATERIALS_PARAM_DECL) {
	__global const Material *mat = &mats[bsdf->materialIndex];

	return (bsdf->hitPoint.regularization > 0.f) ? mat->isRegularizedDelta : mat->isDelta;
}

// The factor undoing the eta^2 radiance scaling of a transmission through
// this surface (1 for the materials without refraction), see
// EyePathInfo::etaScale and BSDF::GetRefractionEtaScale()
OPENCL_FORCE_INLINE float BSDF_GetRefractionEtaScale(__global const BSDF *bsdf
		MATERIALS_PARAM_DECL) {
	__global const Material *mat = &mats[bsdf->materialIndex];
	if (!mat->isRefractive)
		return 1.f;

	const float nc = ExtractExteriorIors(&bsdf->hitPoint, mat->refractionExteriorIorTexIndex
			TEXTURES_PARAM);
	const float nt = ExtractInteriorIors(&bsdf->hitPoint, mat->refractionInteriorIorTexIndex
			TEXTURES_PARAM);
	if (!(nc > 0.f) || !(nt > 0.f))
		return 1.f;

	// Entering the object, the glass scales the radiance by (nc / nt)^2
	const float eta = bsdf->hitPoint.intoObject ? (nt / nc) : (nc / nt);

	return eta * eta;
}

OPENCL_FORCE_INLINE uint BSDF_GetObjectID(__global const BSDF *bsdf, __global const SceneObject* restrict sceneObjs) {
	const uint sceneObjectIndex = bsdf->sceneObjectIndex;

	return (sceneObjectIndex != NULL_INDEX) ? sceneObjs[sceneObjectIndex].objectID : NULL_INDEX;
}

OPENCL_FORCE_INLINE uint BSDF_GetMaterialID(__global const BSDF *bsdf
		MATERIALS_PARAM_DECL) {
	return mats[bsdf->materialIndex].matID;
}

OPENCL_FORCE_INLINE uint BSDF_GetLightID(__global const BSDF *bsdf
		MATERIALS_PARAM_DECL) {
	return mats[bsdf->materialIndex].lightID;
}

OPENCL_FORCE_INLINE uint BSDF_GetMaterialInteriorVolume(__global const BSDF *bsdf
		MATERIALS_PARAM_DECL) {
	return Material_GetInteriorVolume(bsdf->materialIndex, &bsdf->hitPoint,
			bsdf->hitPoint.passThroughEvent
			MATERIALS_PARAM);
}

OPENCL_FORCE_INLINE uint BSDF_GetMaterialExteriorVolume(__global const BSDF *bsdf
		MATERIALS_PARAM_DECL) {
	return Material_GetExteriorVolume(bsdf->materialIndex, &bsdf->hitPoint,
			bsdf->hitPoint.passThroughEvent
			MATERIALS_PARAM);
}

OPENCL_FORCE_INLINE float BSDF_GetGlossiness(__global const BSDF *bsdf
		MATERIALS_PARAM_DECL) {
	const float glossiness = Material_GetGlossiness(bsdf->materialIndex
			MATERIALS_PARAM);

	return (bsdf->hitPoint.regularization > 0.f) ? fmax(glossiness, bsdf->hitPoint.regularization) : glossiness;
}

// The glossiness without the path regularization, for the caustic path
// classification of the eye path (see BSDF::GetUnregularizedGlossiness())
OPENCL_FORCE_INLINE float BSDF_GetUnregularizedGlossiness(__global const BSDF *bsdf
		MATERIALS_PARAM_DECL) {
	return Material_GetGlossiness(bsdf->materialIndex
			MATERIALS_PARAM);
}

OPENCL_FORCE_INLINE float3 BSDF_GetPassThroughShadowTransparency(__global const BSDF *bsdf
		MATERIALS_PARAM_DECL) {
	return VLOAD3F(&mats[bsdf->materialIndex].passThroughShadowTransparency.c[0]);
}

OPENCL_FORCE_INLINE float3 BSDF_GetLandingGeometryN(__global const BSDF *bsdf) {
	return HitPoint_GetGeometryN(&bsdf->hitPoint);
}

OPENCL_FORCE_INLINE float3 BSDF_GetLandingInterpolatedN(__global const BSDF *bsdf) {
	return HitPoint_GetInterpolatedN(&bsdf->hitPoint);
}

OPENCL_FORCE_INLINE float3 BSDF_GetLandingShadeN(__global const BSDF *bsdf) {
	return HitPoint_GetShadeN(&bsdf->hitPoint);
}

OPENCL_FORCE_INLINE float3 BSDF_GetRayOrigin(__global const BSDF *bsdf, const float3 sampleDir) {
	const float3 p = VLOAD3F(&bsdf->hitPoint.p.x);

	if (bsdf->isVolume)
		return p;
	else {
		// Rise the ray origin along the geometry normal to avoid self intersection
		const float3 geometryN = VLOAD3F(&bsdf->hitPoint.geometryN.x);
		const float riseDirection = (dot(sampleDir, geometryN) > 0.f) ? 1.f : -1.f;
		
		return p + riseDirection * (geometryN * MachineEpsilon_E_Float3(p));
	}
}

OPENCL_FORCE_INLINE bool BSDF_IsAlbedoEndPoint(__global const BSDF *bsdf,
		const AlbedoSpecularSetting albedoSpecularSetting,
		const float albedoSpecularGlossinessThreshold
		MATERIALS_PARAM_DECL) {
	const BSDFEvent eventTypes = BSDF_GetEventTypes(bsdf MATERIALS_PARAM);
	if (!BSDF_IsDelta(bsdf MATERIALS_PARAM) &&
			!((eventTypes & GLOSSY) && (BSDF_GetGlossiness(bsdf MATERIALS_PARAM) < albedoSpecularGlossinessThreshold)))
		return true;

	switch (albedoSpecularSetting) {
		case NO_REFLECT_TRANSMIT:
			return true;
		case ONLY_REFLECT:
			return !((eventTypes & REFLECT) && !(eventTypes & TRANSMIT));
		case ONLY_TRANSMIT:
			return !(!(eventTypes & REFLECT) && (eventTypes & TRANSMIT));
		case REFLECT_TRANSMIT:
			return !((eventTypes & REFLECT) || (eventTypes & TRANSMIT));
		default:
			return true;
	}
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
