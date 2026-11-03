#line 2 "bsdf_funcs.cl"

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

// The interpolated and the shading normals of a hit point made valid for its
// eye direction (see BSDF_EnsureValidReflection())
OPENCL_FORCE_INLINE void BSDF_EnsureValidNormals(__global HitPoint *hitPoint) {
	const float3 fixedDir = VLOAD3F(&hitPoint->fixedDir.x);
	// The geometric normal on the side of the eye
	const float side = hitPoint->intoObject ? 1.f : -1.f;
	const float3 ng = side * VLOAD3F(&hitPoint->geometryN.x);
	if (dot(fixedDir, ng) < DEFAULT_COS_EPSILON_STATIC)
		return;

	const float3 interpolatedN = VLOAD3F(&hitPoint->interpolatedN.x);
	const float3 shadeN = VLOAD3F(&hitPoint->shadeN.x);
	const bool sameN = (shadeN.x == interpolatedN.x) && (shadeN.y == interpolatedN.y) && (shadeN.z == interpolatedN.z);
	const float3 validInterpolatedN = side * BSDF_EnsureValidReflection(ng, fixedDir, side * interpolatedN);
	VSTORE3F(validInterpolatedN, &hitPoint->interpolatedN.x);
	VSTORE3F(sameN ? validInterpolatedN : (side * BSDF_EnsureValidReflection(ng, fixedDir, side * shadeN)),
			&hitPoint->shadeN.x);
}

// Used when hitting a surface
// The offset of a hit point p to the tangent plane of a vertex v of normal n,
// only outwards (on the side sideN of the eye)
OPENCL_FORCE_INLINE float3 BSDF_TangentPlaneOffset(const float3 p, const float3 v, const float3 vn,
		const float3 sideN) {
	const float3 n = (dot(vn, sideN) < 0.f) ? -normalize(vn) : normalize(vn);
	const float d = dot(p - v, n);

	return (d < 0.f) ? (-d * n) : MAKE_FLOAT3(0.f, 0.f, 0.f);
}

// "Hacking the Shadow Terminator" by Johannes Hanika (Ray Tracing Gems II,
// Cycles' shadow terminator geometry offset): the point of a toon vertex is
// moved to the smooth surface above its flat triangle, the barycentric blend
// of its projections on the tangent planes of the three vertices. The rays
// that leave it see the smooth surface, so the shadow terminator follows the
// shading normal instead of the edges of the triangles of a coarse mesh
// (the steps of the toon ramp end where the light is, and show the
// staircase of the flat triangles a matte material hides near its
// terminator). See SmoothSurfacePoint() in bsdf.cpp.
OPENCL_FORCE_NOT_INLINE void BSDF_SetSmoothSurfacePoint(__global HitPoint *hitPoint,
		const uint meshIndex, const uint triangleIndex
		EXTMESH_PARAM_DECL) {
	__global const ExtMesh* restrict meshDesc = &meshDescs[meshIndex];
	if (meshDesc->normalsOffset == NULL_INDEX)
		return;

	Transform motionLocalToWorld;
	ExtMesh_GetMotionLocal2World(hitPoint->time, meshIndex, &motionLocalToWorld EXTMESH_PARAM);
	float3 v0, v1, v2;
	ExtMesh_GetTriangleVertices(&motionLocalToWorld, meshIndex, triangleIndex, &v0, &v1, &v2 EXTMESH_PARAM);

	__global const Triangle* restrict tri = &triangles[meshDesc->trisOffset + triangleIndex];
	__global const Normal* restrict vns = &vertNormals[meshDesc->normalsOffset];
	const float3 n0 = ExtMesh_LocalToWorldNormal(meshDesc, &motionLocalToWorld, VLOAD3F(&vns[tri->v[0]].x));
	const float3 n1 = ExtMesh_LocalToWorldNormal(meshDesc, &motionLocalToWorld, VLOAD3F(&vns[tri->v[1]].x));
	const float3 n2 = ExtMesh_LocalToWorldNormal(meshDesc, &motionLocalToWorld, VLOAD3F(&vns[tri->v[2]].x));

	const float3 geometryN = VLOAD3F(&hitPoint->geometryN.x);
	const float3 sideN = hitPoint->intoObject ? geometryN : -geometryN;
	const float3 p = VLOAD3F(&hitPoint->p.x);
	const float b1 = hitPoint->triangleBariCoord1;
	const float b2 = hitPoint->triangleBariCoord2;
	const float3 offset = (1.f - b1 - b2) * BSDF_TangentPlaneOffset(p, v0, n0, sideN) +
			b1 * BSDF_TangentPlaneOffset(p, v1, n1, sideN) +
			b2 * BSDF_TangentPlaneOffset(p, v2, n2, sideN);

	VSTORE3F(p + offset, &hitPoint->p.x);
}

OPENCL_FORCE_INLINE void BSDF_Init(
		__global BSDF *bsdf,
		const bool throughShadowTransparency,
		__global Ray *ray,
		__global const RayHit *rayHit,
		const float passThroughEvent,
		__global PathVolumeInfo *volInfo
		MATERIALS_PARAM_DECL
		) {
	const uint meshIndex = rayHit->meshIndex;
	const uint triangleIndex = rayHit->triangleIndex;
	const float time = ray->time;

	const float3 rayOrig = VLOAD3F(&ray->o.x);
	const float3 rayDir = VLOAD3F(&ray->d.x);
	const float3 hitPointP = rayOrig + rayHit->t * rayDir;

	const uint matIndex = sceneObjs[meshIndex].materialIndex;
	if (mats[matIndex].nullPassThrough) {
		// A null material that lets every ray through (a volume's shell):
		// Scene_Intersect() continues the ray and reads only the side, the
		// volumes and the material of the hit point, so the normals, UV,
		// differentials, bump and frame are not built
		__global HitPoint *hitPoint = &bsdf->hitPoint;
		hitPoint->time = time;
		hitPoint->throughShadowTransparency = throughShadowTransparency;
		hitPoint->passThroughEvent = passThroughEvent;
		hitPoint->regularization = 0.f;
		hitPoint->subsurface = HITPOINT_SSS_SURFACE;
		hitPoint->wavelength = 0.f;
		hitPoint->dispersionTinted = 0;
		VSTORE3F(hitPointP, &hitPoint->p.x);
		VSTORE3F(-rayDir, &hitPoint->fixedDir.x);
		hitPoint->objectID = sceneObjs[meshIndex].objectID;
		Transform motionLocalToWorld;
		ExtMesh_GetMotionLocal2World(time, meshIndex, &motionLocalToWorld EXTMESH_PARAM);
		const float3 geometryN = ExtMesh_GetGeometryNormal(&motionLocalToWorld, meshIndex, triangleIndex EXTMESH_PARAM);
		VSTORE3F(geometryN, &hitPoint->geometryN.x);
		// As HitPoint_Init()
		hitPoint->intoObject = (dot(rayDir, geometryN) < 0.f);
		hitPoint->meshIndex = meshIndex;
		hitPoint->triangleIndex = triangleIndex;
		hitPoint->triangleBariCoord1 = rayHit->b1;
		hitPoint->triangleBariCoord2 = rayHit->b2;

		bsdf->sceneObjectIndex = meshIndex;
		bsdf->materialIndex = matIndex;
		PathVolumeInfo_SetHitPointVolumes(
				volInfo,
				hitPoint,
				Material_GetInteriorVolume(matIndex, hitPoint,
					passThroughEvent
					MATERIALS_PARAM),
				Material_GetExteriorVolume(matIndex, hitPoint,
					passThroughEvent
					MATERIALS_PARAM)
				MATERIALS_PARAM);
		const uint offset = lightIndexOffsetByMeshIndex[meshIndex];
		bsdf->triangleLightSourceIndex = (offset == NULL_INDEX) ? NULL_INDEX : lightIndexByTriIndex[offset + triangleIndex];
		bsdf->isVolume = false;

		return;
	}

	HitPoint_Init(&bsdf->hitPoint, throughShadowTransparency,
		meshIndex, triangleIndex, time,
		hitPointP, -rayDir,
		rayHit->b1, rayHit->b2,
		passThroughEvent
		MATERIALS_PARAM);

	// Save the scene object index
	bsdf->sceneObjectIndex = meshIndex;

	// Get the material
	bsdf->materialIndex = matIndex;

	// A toon vertex: its point on the smooth surface
	if (mats[matIndex].type == TOON)
		BSDF_SetSmoothSurfacePoint(&bsdf->hitPoint, meshIndex, triangleIndex EXTMESH_PARAM);

	//--------------------------------------------------------------------------
	// Set interior and exterior volumes
	//--------------------------------------------------------------------------

	PathVolumeInfo_SetHitPointVolumes(
			volInfo,
			&bsdf->hitPoint,
			Material_GetInteriorVolume(matIndex, &bsdf->hitPoint,
				passThroughEvent
				MATERIALS_PARAM),
			Material_GetExteriorVolume(matIndex, &bsdf->hitPoint,
				passThroughEvent
				MATERIALS_PARAM)
			MATERIALS_PARAM);

	//--------------------------------------------------------------------------

	// Check if it is a light source
	const uint offset = lightIndexOffsetByMeshIndex[meshIndex];
	bsdf->triangleLightSourceIndex = (offset == NULL_INDEX) ? NULL_INDEX : lightIndexByTriIndex[offset + triangleIndex];

	//--------------------------------------------------------------------------
	// Apply bump or normal mapping
	//--------------------------------------------------------------------------

	// The bump cache holds the shading normals of the sub-materials of the
	// previous vertex
	bsdf->hitPoint.bumpCacheValidSlots = 0;
	Material_Bump(matIndex, &bsdf->hitPoint
			MATERIALS_PARAM);
	BSDF_EnsureValidNormals(&bsdf->hitPoint);
	// Re-read the shadeN modified by Material_Bump() and BSDF_EnsureValidNormals()
	const float3 shadeN = VLOAD3F(&bsdf->hitPoint.shadeN.x);
	const float3 dpdu = VLOAD3F(&bsdf->hitPoint.dpdu.x);
	const float3 dpdv = VLOAD3F(&bsdf->hitPoint.dpdv.x);

	//--------------------------------------------------------------------------
	// Build the local reference system
	//--------------------------------------------------------------------------
	Frame_Set(&bsdf->frame, dpdu, dpdv, shadeN);

	bsdf->isVolume = false;
}

// Used when hitting a volume scatter point
OPENCL_FORCE_INLINE void BSDF_InitVolume(
		__global BSDF *bsdf,
		const bool throughShadowTransparency,
		__global const Material* restrict mats,
		__global Ray *ray,
		const uint volumeIndex, const float t, const float passThroughEvent) {
	bsdf->hitPoint.throughShadowTransparency = throughShadowTransparency;
	bsdf->hitPoint.passThroughEvent = passThroughEvent;
	bsdf->hitPoint.regularization = 0.f;
	bsdf->hitPoint.subsurface = HITPOINT_SSS_SURFACE;
	bsdf->hitPoint.wavelength = 0.f;
	bsdf->hitPoint.dispersionTinted = 0;

	const float3 rayOrig = VLOAD3F(&ray->o.x);
	const float3 rayDir = VLOAD3F(&ray->d.x);

	const float3 hitPointP = rayOrig + t * rayDir;
	VSTORE3F(hitPointP, &bsdf->hitPoint.p.x);
	// A unit vector, see BSDF::Init()
	const float3 geometryN = -normalize(rayDir);
	VSTORE3F(geometryN, &bsdf->hitPoint.fixedDir.x);

	bsdf->sceneObjectIndex = NULL_INDEX;
	// No mesh (meshIndex below): the local to world transformation is the
	// identity
	bsdf->hitPoint.time = ray->time;
	bsdf->materialIndex = volumeIndex;

	VSTORE3F(geometryN, &bsdf->hitPoint.geometryN.x);
	VSTORE3F(geometryN, &bsdf->hitPoint.interpolatedN.x);
	VSTORE3F(geometryN, &bsdf->hitPoint.shadeN.x);

	bsdf->hitPoint.intoObject = true;
	bsdf->hitPoint.interiorVolumeIndex = volumeIndex;
	bsdf->hitPoint.exteriorVolumeIndex = volumeIndex;
	const uint iorTexIndex = (volumeIndex != NULL_INDEX) ? mats[volumeIndex].volume.iorTexIndex : NULL_INDEX;
	bsdf->hitPoint.interiorIorTexIndex = iorTexIndex;
	bsdf->hitPoint.exteriorIorTexIndex = iorTexIndex;

	bsdf->triangleLightSourceIndex = NULL_INDEX;

	VSTORE2F(MAKE_FLOAT2(0.f, 0.f), &bsdf->hitPoint.defaultUV.u);

	float3 dpdu, dpdv;
	CoordinateSystem(geometryN, &dpdu, &dpdv);
	VSTORE3F(dpdu, &bsdf->hitPoint.dpdu.x);
	VSTORE3F(dpdv, &bsdf->hitPoint.dpdv.x);
	VSTORE3F(MAKE_FLOAT3(0.f, 0.f, 0.f), &bsdf->hitPoint.dndu.x);
	VSTORE3F(MAKE_FLOAT3(0.f, 0.f, 0.f), &bsdf->hitPoint.dndv.x);

	bsdf->hitPoint.meshIndex = NULL_INDEX;
	bsdf->hitPoint.triangleIndex = NULL_INDEX;
	bsdf->hitPoint.triangleBariCoord1 = 0.f;
	bsdf->hitPoint.triangleBariCoord2 = 0.f;

	bsdf->hitPoint.objectID = NULL_INDEX;

	// Build the local reference system
	Frame_SetFromZ(&bsdf->frame, geometryN);

	bsdf->isVolume = true;
}

OPENCL_FORCE_INLINE float3 BSDF_Albedo(__global const BSDF *bsdf
		MATERIALS_PARAM_DECL) {
	return Material_Albedo(bsdf->materialIndex, &bsdf->hitPoint
			MATERIALS_PARAM);
}

//------------------------------------------------------------------------------
// "Taming the Shadow Terminator"
// by Matt Jen-Yuan Chiang, Yining Karl Li and Brent Burley
// https://www.yiningkarlli.com/projects/shadowterminator.html
//------------------------------------------------------------------------------

OPENCL_FORCE_INLINE float BSDF_ShadowTerminatorAvoidanceFactor(const float3 Ni, const float3 Ns,
		const float3 lightDir) {
	const float dotNsLightDir = dot(Ns, lightDir);
	if (dotNsLightDir <= 0.f)
		return 0.f;

	const float dotNiNs = dot(Ni, Ns);
	if (dotNiNs <= 0.f)
		return 0.f;

	const float G = fmin(1.f, dot(Ni, lightDir) / (dotNsLightDir * dotNiNs));
	if (G <= 0.f)
		return 0.f;
	
	const float G2 = G * G;
	const float G3 = G2 * G;

	return -G3 + G2 + G;
}

OPENCL_FORCE_INLINE float3 BSDF_Evaluate(__global const BSDF *bsdf,
		const float3 generatedDir, BSDFEvent *event, float *directPdfW
		MATERIALS_PARAM_DECL) {
	//const Vector &eyeDir = fromLight ? generatedDir : hitPoint.fixedDir;
	//const Vector &lightDir = fromLight ? hitPoint.fixedDir : generatedDir;
	const float3 eyeDir = VLOAD3F(&bsdf->hitPoint.fixedDir.x);
	const float3 lightDir = generatedDir;
	const float3 geometryN = VLOAD3F(&bsdf->hitPoint.geometryN.x);
	const float3 interpolatedN =  VLOAD3F(&bsdf->hitPoint.interpolatedN.x);
	const float3 shadeN =  VLOAD3F(&bsdf->hitPoint.shadeN.x);

	const float dotLightDirNG = dot(lightDir, geometryN);
	const float absDotLightDirNG = fabs(dotLightDirNG);
	const float dotEyeDirNG = dot(eyeDir, geometryN);
	const float absDotEyeDirNG = fabs(dotEyeDirNG);

	if (!bsdf->isVolume) {
		// These kind of tests make sense only for materials

		// Avoid glancing angles
		if ((absDotLightDirNG < DEFAULT_COS_EPSILON_STATIC) ||
				(absDotEyeDirNG < DEFAULT_COS_EPSILON_STATIC))
			return BLACK;

		// Check geometry normal and light direction side
		const float sideTest = dotEyeDirNG * dotLightDirNG;
		const BSDFEvent matEvents = BSDF_GetEventTypes(bsdf
				MATERIALS_PARAM);
		if (((sideTest > 0.f) && !(matEvents & REFLECT)) ||
				((sideTest < 0.f) && !(matEvents & TRANSMIT)))
			return BLACK;

		// Check shading normal and light direction side
		const float sideTestIS = dot(eyeDir, interpolatedN) * dot(lightDir, interpolatedN);
		if (((sideTestIS > 0.f) && !(matEvents & REFLECT)) ||
				((sideTestIS < 0.f) && !(matEvents & TRANSMIT)))
			return BLACK;
	}

	__global const Frame *frame = &bsdf->frame;
	const float3 localLightDir = Frame_ToLocal(frame, lightDir);
	const float3 localEyeDir = Frame_ToLocal(frame, eyeDir);
	float3 result = Material_Evaluate(bsdf->materialIndex, &bsdf->hitPoint,
			localLightDir, localEyeDir,	event, directPdfW
			MATERIALS_PARAM);
	if (Spectrum_IsBlack(result))
		return BLACK;

	if (!bsdf->isVolume) {
		// Shadow terminator artefact avoidance
		if ((*event & REFLECT) &&
				(*event & (DIFFUSE | GLOSSY)) &&
				((shadeN.x != interpolatedN.x) || (shadeN.y != interpolatedN.y) || (shadeN.z != interpolatedN.z)))
			result *= BSDF_ShadowTerminatorAvoidanceFactor(BSDF_GetLandingInterpolatedN(bsdf),
					BSDF_GetLandingShadeN(bsdf), lightDir);
	}

	return result;
}

// The direct light of a toon vertex (see BSDF_IsDirectLightSamplingOnly()):
// the toon part of the ramp direction, lightDir or the direction of the
// centre of the light with ToonParam::lightCenter (see BSDF::EvaluateToon()).
// Inlined: a call costs the direct light of every material more scratch and
// time (2026-11-01: +32 B, 1-4% slower).
OPENCL_FORCE_INLINE float3 BSDF_EvaluateToon(__global const BSDF *bsdf,
		const float3 lightDir, __global const LightSource* restrict light, BSDFEvent *event
		MATERIALS_PARAM_DECL) {
	float3 rampDir = lightDir;
	float lightRadius = 0.f;
	if (mats[bsdf->materialIndex].toon.lightCenter) {
		float3 centerDir;
		float centerRadius;
		if (Light_GetCenterDir(light, VLOAD3F(&bsdf->hitPoint.p.x), &centerDir, &centerRadius)) {
			rampDir = centerDir;
			lightRadius = centerRadius;
		}
	}

	const float3 eyeDir = VLOAD3F(&bsdf->hitPoint.fixedDir.x);
	const float3 geometryN = VLOAD3F(&bsdf->hitPoint.geometryN.x);
	const float3 interpolatedN =  VLOAD3F(&bsdf->hitPoint.interpolatedN.x);
	const float3 shadeN =  VLOAD3F(&bsdf->hitPoint.shadeN.x);

	// The tests of BSDF_Evaluate() for a material that only reflects, the side
	// of the light on the interpolated normal: the light of a direction below
	// a flat triangle but above the smooth surface reaches its point on the
	// smooth surface (see BSDF_SetSmoothSurfacePoint())
	if ((fabs(dot(eyeDir, geometryN)) < DEFAULT_COS_EPSILON_STATIC) ||
			(fabs(dot(lightDir, interpolatedN)) < DEFAULT_COS_EPSILON_STATIC) ||
			(dot(eyeDir, interpolatedN) * dot(lightDir, interpolatedN) < 0.f))
		return BLACK;

	__global const Frame *frame = &bsdf->frame;
	float3 result = ToonMaterial_EvaluateToon(&mats[bsdf->materialIndex], &bsdf->hitPoint,
			Frame_ToLocal(frame, lightDir), Frame_ToLocal(frame, rampDir), Frame_ToLocal(frame, eyeDir),
			lightRadius
			MATERIALS_PARAM);
	*event = DIFFUSE | REFLECT;

	// Shadow terminator artefact avoidance
	if ((shadeN.x != interpolatedN.x) || (shadeN.y != interpolatedN.y) || (shadeN.z != interpolatedN.z))
		result *= BSDF_ShadowTerminatorAvoidanceFactor(BSDF_GetLandingInterpolatedN(bsdf),
				BSDF_GetLandingShadeN(bsdf), lightDir);

	return result;
}

OPENCL_FORCE_INLINE float3 BSDF_Sample(__global const BSDF *bsdf, const float u0, const float u1,
		float3 *sampledDir, float *pdfW, float *absCosSampledDir, BSDFEvent *event
		MATERIALS_PARAM_DECL) {
	const float3 fixedDir = VLOAD3F(&bsdf->hitPoint.fixedDir.x);
	const float3 localFixedDir = Frame_ToLocal(&bsdf->frame, fixedDir);
	float3 localSampledDir;

	float3 result = Material_Sample(bsdf->materialIndex, &bsdf->hitPoint,
			localFixedDir, &localSampledDir, u0, u1,
			bsdf->hitPoint.passThroughEvent,
			pdfW, event
			MATERIALS_PARAM);
	if (Spectrum_IsBlack(result))
		return BLACK;

	*absCosSampledDir = fabs(CosTheta(localSampledDir));
	*sampledDir = Frame_ToWorld(&bsdf->frame, localSampledDir);

	if (!bsdf->isVolume) {
		// Shadow terminator artefact avoidance
		const float3 shadeN = VLOAD3F(&bsdf->hitPoint.shadeN.x);
		const float3 interpolatedN = VLOAD3F(&bsdf->hitPoint.interpolatedN.x);
		if ((*event & REFLECT) &&
				(*event & (DIFFUSE | GLOSSY)) &&
				((shadeN.x != interpolatedN.x) || (shadeN.y != interpolatedN.y) || (shadeN.z != interpolatedN.z)))
			result *= BSDF_ShadowTerminatorAvoidanceFactor(BSDF_GetLandingInterpolatedN(bsdf),
					BSDF_GetLandingShadeN(bsdf), *sampledDir);
	}

	return result;
}

// The part of BSDF_Evaluate() of an eye path that comes from the diffuse
// reflection lobe of the material (Material_EvaluateDiffuse()), with the
// lobe's share of the pdf of BSDF_Sample() (see BSDF::EvaluateDiffuse())
OPENCL_FORCE_INLINE float3 BSDF_EvaluateDiffuse(__global const BSDF *bsdf,
		const float3 lightDir, float *pdfW
		MATERIALS_PARAM_DECL) {
	*pdfW = 0.f;
	if (bsdf->isVolume)
		return BLACK;

	const float3 localLightDir = Frame_ToLocal(&bsdf->frame, lightDir);
	const float3 localEyeDir = Frame_ToLocal(&bsdf->frame, VLOAD3F(&bsdf->hitPoint.fixedDir.x));
	float diffusePdfW;
	float3 result = Material_EvaluateDiffuse(bsdf->materialIndex, &bsdf->hitPoint,
			localLightDir, localEyeDir, &diffusePdfW
			MATERIALS_PARAM);
	if (Spectrum_IsBlack(result) || !(diffusePdfW > 0.f))
		return BLACK;

	// Shadow terminator artefact avoidance, as BSDF_Evaluate() and
	// BSDF_Sample() of a diffuse reflection
	const float3 shadeN = VLOAD3F(&bsdf->hitPoint.shadeN.x);
	const float3 interpolatedN = VLOAD3F(&bsdf->hitPoint.interpolatedN.x);
	if ((shadeN.x != interpolatedN.x) || (shadeN.y != interpolatedN.y) || (shadeN.z != interpolatedN.z))
		result *= BSDF_ShadowTerminatorAvoidanceFactor(BSDF_GetLandingInterpolatedN(bsdf),
				BSDF_GetLandingShadeN(bsdf), lightDir);

	*pdfW = diffusePdfW;

	return result;
}

// The part of a BSDF sample (BSDF_Sample() of sampledDir, event and pdfW) that
// comes from the diffuse lobe of the material, with the share of the pdf of
// the lobe (see BSDF::GetDiffuseSample())
OPENCL_FORCE_INLINE float3 BSDF_GetDiffuseSample(__global const BSDF *bsdf,
		const float3 sampledDir, const BSDFEvent event, const float pdfW,
		float *diffusePdfShare
		MATERIALS_PARAM_DECL) {
	*diffusePdfShare = 0.f;
	if ((event & SPECULAR) || !(pdfW > 0.f))
		return BLACK;

	float diffusePdfW;
	const float3 result = BSDF_EvaluateDiffuse(bsdf, sampledDir, &diffusePdfW
			MATERIALS_PARAM);
	if (Spectrum_IsBlack(result))
		return BLACK;

	*diffusePdfShare = fmin(diffusePdfW / pdfW, 1.f);

	return result / pdfW;
}

OPENCL_FORCE_INLINE bool BSDF_IsLightSource(__global const BSDF *bsdf) {
	return (bsdf->triangleLightSourceIndex != NULL_INDEX);
}

// True if the emission of the light hit by a BSDF ray depends on the opacity
// of the side it hits (as BSDF::HasEmissionOpacity()): a null material keeps
// its emission and lets every ray through
OPENCL_FORCE_INLINE bool BSDF_HasEmissionOpacity(__global const BSDF *bsdf
		MATERIALS_PARAM_DECL) {
	const uint matIndex = bsdf->materialIndex;

	return (bsdf->triangleLightSourceIndex != NULL_INDEX) &&
			!mats[matIndex].noPassThroughTransparency && (mats[matIndex].type != NULLMAT);
}

OPENCL_FORCE_INLINE float3 BSDF_GetEmittedRadiance(__global const BSDF *bsdf, float *directPdfA
		LIGHTS_PARAM_DECL) {
	const uint triangleLightSourceIndex = bsdf->triangleLightSourceIndex;
	if (triangleLightSourceIndex == NULL_INDEX)
		return BLACK;
	else
		return IntersectableLight_GetRadiance(&lights[triangleLightSourceIndex],
				&bsdf->hitPoint, directPdfA
				LIGHTS_PARAM);
}

// vertexPassThroughEvent: see Material_GetPassThroughTransparency(), left as
// it is when the material has no transparency to evaluate
OPENCL_FORCE_INLINE float3 BSDF_GetPassThroughTransparency(__global const BSDF *bsdf, const bool backTracing,
		float *vertexPassThroughEvent
		MATERIALS_PARAM_DECL) {
	if (mats[bsdf->materialIndex].noPassThroughTransparency)
		return BLACK;
	if (mats[bsdf->materialIndex].nullPassThrough)
		return WHITE;

	const float3 localFixedDir = Frame_ToLocal(&bsdf->frame, VLOAD3F(&bsdf->hitPoint.fixedDir.x));

	return Material_GetPassThroughTransparency(bsdf->materialIndex,
			&bsdf->hitPoint, localFixedDir, bsdf->hitPoint.passThroughEvent, backTracing,
			vertexPassThroughEvent
			MATERIALS_PARAM);
}

OPENCL_FORCE_INLINE float BSDF_IsPhotonGIEnabled(__global const BSDF *bsdf
		MATERIALS_PARAM_DECL) {
	return Material_IsPhotonGIEnabled(bsdf->materialIndex
			MATERIALS_PARAM);
}

OPENCL_FORCE_INLINE bool BSDF_IsHoldout(__global const BSDF *bsdf
		MATERIALS_PARAM_DECL) {
	return Material_IsHoldout(bsdf->materialIndex
			MATERIALS_PARAM);
}

//------------------------------------------------------------------------------
// Shadow catcher related functions
//------------------------------------------------------------------------------

OPENCL_FORCE_INLINE bool BSDF_IsShadowCatcher(__global const BSDF *bsdf
		MATERIALS_PARAM_DECL) {
	const uint matIndex = bsdf->materialIndex;

	return (matIndex == NULL_INDEX) ? false : mats[matIndex].isShadowCatcher;
}

OPENCL_FORCE_INLINE bool BSDF_IsShadowCatcherOnlyInfiniteLights(__global const BSDF *bsdf
		MATERIALS_PARAM_DECL) {
	const uint matIndex = bsdf->materialIndex;

	return (matIndex == NULL_INDEX) ? false : (mats[matIndex].isShadowCatcher && mats[matIndex].isShadowCatcherOnlyInfiniteLights);
}

OPENCL_FORCE_INLINE float3 BSDF_ShadowCatcherSample(__global const BSDF *bsdf,
		float3 *sampledDir, float *pdfW, float *absCosSampledDir, BSDFEvent *event
		MATERIALS_PARAM_DECL) {
	// Just continue to trace the ray
	*sampledDir = -VLOAD3F(&bsdf->hitPoint.fixedDir.x);
	*absCosSampledDir = fabs(dot(*sampledDir, VLOAD3F(&bsdf->hitPoint.geometryN.x)));

	*pdfW = 1.f;
	*event = SPECULAR | TRANSMIT;
	const float3 result = WHITE;

	// Adjoint BSDF
//	if (hitPoint.fromLight) {
//		const float absDotFixedDirNG = AbsDot(hitPoint.fixedDir, hitPoint.geometryN);
//		const float absDotSampledDirNG = AbsDot(*sampledDir, hitPoint.geometryN);
//		return result * (absDotSampledDirNG / absDotFixedDirNG);
//	} else
		return result;
}

//------------------------------------------------------------------------------
// Bake related functions
//------------------------------------------------------------------------------

OPENCL_FORCE_INLINE bool BSDF_HasBakeMap(__global const BSDF *bsdf, const BakeMapType type
		MATERIALS_PARAM_DECL) {
	return (bsdf->sceneObjectIndex != NULL_INDEX) &&
			(sceneObjs[bsdf->sceneObjectIndex].bakeMapIndex != NULL_INDEX) &&
			(sceneObjs[bsdf->sceneObjectIndex].bakeMapType == type);
}

OPENCL_FORCE_INLINE float3 BSDF_GetBakeMapValue(__global const BSDF *bsdf
		MATERIALS_PARAM_DECL) {
	const uint mapIndex = sceneObjs[bsdf->sceneObjectIndex].bakeMapIndex;
	__global const ImageMap *imageMap = &imageMapDescs[mapIndex];

	const uint uvIndex = sceneObjs[bsdf->sceneObjectIndex].bakeMapUVIndex;
	const float2 uv = HitPoint_GetUV(&bsdf->hitPoint, uvIndex EXTMESH_PARAM);

	return ImageMap_GetSpectrum(
			imageMap,
			uv.x, uv.y
			IMAGEMAPS_PARAM);
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
