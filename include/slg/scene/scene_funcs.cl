#line 2 "scene_funcs.cl"

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

OPENCL_HOT_FUNCTION bool Scene_Intersect(
		__constant const GPUTaskConfiguration* restrict taskConfig,
		const SceneRayType rayType,
		int *throughShadowTransparency,
		__global PathVolumeInfo *volInfo,
		__global HitPoint *tmpHitPoint,
		const float passThrough,
		__global Ray *ray,
		__global RayHit *rayHit,
		__global BSDF *bsdf,
		float3 *connectionThroughput,  const float3 pathThroughput,
		__global SampleResult *sampleResult,
		const bool backTracing,
		__global Seed *seedPassThroughEvent
		MATERIALS_PARAM_DECL
		) {
	*connectionThroughput = WHITE;
	
	const bool cameraRay = rayType & CAMERA_RAY;
	const bool shadowRay = rayType & SHADOW_RAY;

	const bool hit = (rayHit->meshIndex != NULL_INDEX);

	uint rayVolumeIndex = volInfo->currentVolumeIndex;

	if (hit) {
		// Initialize the BSDF of the hit point
		BSDF_Init(bsdf,
				*throughShadowTransparency,
				ray, rayHit,
				passThrough,
				volInfo
				MATERIALS_PARAM
				);

		rayVolumeIndex = bsdf->hitPoint.intoObject ? bsdf->hitPoint.exteriorVolumeIndex : bsdf->hitPoint.interiorVolumeIndex;
	} else if (rayVolumeIndex == NULL_INDEX) {
		// No volume information, I use the default volume
		rayVolumeIndex = scene->defaultVolumeIndex;
	}

	// Check if there is volume scatter event
	if (rayVolumeIndex != NULL_INDEX) {
		// This applies volume transmittance too
		// Note: by using passThrough here, I introduce subtle correlation
		// between scattering events and pass-through events
		float3 connectionEmission = BLACK;

		// A shadow ray takes the transmittance of the segment, with the
		// Russian roulette of Scene::Intersect() on a low one (the random
		// numbers come from the pass-through sequence of the shadow ray, as
		// on the CPU)
		float3 segmentTransmittance = WHITE;
		// The pass-through event for the surface at the end of the segment
		// (see Volume::Scatter())
		float surfacePassThrough = passThrough;
		const float t = Volume_Scatter(&mats[rayVolumeIndex], ray,
				hit ? rayHit->t : ray->maxt,
				passThrough, volInfo->scatteredStart, shadowRay,
				pathThroughput, shadowRay ? &segmentTransmittance : connectionThroughput, &connectionEmission,
				&surfacePassThrough, tmpHitPoint
				TEXTURES_PARAM);

		if (shadowRay) {
			const float rrProb = fmax(segmentTransmittance.x, fmax(segmentTransmittance.y, segmentTransmittance.z));
			if (rrProb < .1f) {
				bool blocked = (rrProb <= 0.f);
				if (!blocked) {
					Seed seed = *seedPassThroughEvent;
					blocked = (Rnd_FloatValue(&seed) >= rrProb);
					*seedPassThroughEvent = seed;
				}

				if (blocked) {
					// Not a miss (see the scatter event below)
					rayHit->meshIndex = 0xfffffffeu;
					return false;
				}

				segmentTransmittance /= rrProb;
			}
			*connectionThroughput *= segmentTransmittance;
		}

		// Add the volume emitted light to the appropriate light group
		if (!Spectrum_IsBlack(connectionEmission) && sampleResult)
			SampleResult_AddEmission(&taskConfig->film, sampleResult, BSDF_GetLightID(bsdf
				MATERIALS_PARAM),
					pathThroughput, connectionEmission);

		if (t > 0.f) {
			// There was a volume scatter event

			// I have to set RayHit fields even if there wasn't a real
			// ray hit
			rayHit->t = t;
			// This is a trick in order to have RayHit::Miss() return
			// false. I assume 0xfffffffeu will trigger a memory fault if
			// used (and the bug will be noticed)
			rayHit->meshIndex = 0xfffffffeu;

			BSDF_InitVolume(bsdf, *throughShadowTransparency, mats, ray, rayVolumeIndex, t, passThrough);
			volInfo->scatteredStart = true;

			return false;
		}

		// A new pass-through event for the surface: the volumes of a material
		// may depend on it (Mix), so they are chosen again (as
		// BSDF::SetPassThroughEvent())
		if (hit && (surfacePassThrough != passThrough)) {
			bsdf->hitPoint.passThroughEvent = surfacePassThrough;
			const uint matIndex = bsdf->materialIndex;
			if (!mats[matIndex].constVolumes)
				PathVolumeInfo_SetHitPointVolumes(
						volInfo,
						&bsdf->hitPoint,
						Material_GetInteriorVolume(matIndex, &bsdf->hitPoint,
							surfacePassThrough
							MATERIALS_PARAM),
						Material_GetExteriorVolume(matIndex, &bsdf->hitPoint,
							surfacePassThrough
							MATERIALS_PARAM)
						MATERIALS_PARAM);
		}
	}

	if (hit) {
		bool continueToTrace =
			// Check if the volume priority system tells me to continue to trace the ray
			PathVolumeInfo_ContinueToTrace(volInfo, bsdf
				MATERIALS_PARAM) ||
			// Check if it is a camera invisible object and we are a tracing a camera ray
			(cameraRay && sceneObjs[rayHit->meshIndex].cameraInvisible);

		// Check if it is a pass through point
		if (!continueToTrace) {
			float vertexPassThroughEvent = -1.f;
			const float3 passThroughTrans = BSDF_GetPassThroughTransparency(bsdf, backTracing,
				&vertexPassThroughEvent
				MATERIALS_PARAM);
			if (!Spectrum_IsBlack(passThroughTrans)) {
				*connectionThroughput *= passThroughTrans;

				// It is a pass through point, continue to trace the ray
				continueToTrace = true;
			} else if (vertexPassThroughEvent >= 0.f) {
				// A vertex: the material's own choices need the pass-through
				// event uniform again (as BSDF::SetVertexPassThroughEvent())
				bsdf->hitPoint.passThroughEvent = vertexPassThroughEvent;
				const uint matIndex = bsdf->materialIndex;
				if (!mats[matIndex].constVolumes)
					PathVolumeInfo_SetHitPointVolumes(
							volInfo,
							&bsdf->hitPoint,
							Material_GetInteriorVolume(matIndex, &bsdf->hitPoint,
								vertexPassThroughEvent
								MATERIALS_PARAM),
							Material_GetExteriorVolume(matIndex, &bsdf->hitPoint,
								vertexPassThroughEvent
								MATERIALS_PARAM)
							MATERIALS_PARAM);
			}
		}

		if (!continueToTrace && shadowRay) {
			const float3 shadowTransparency = BSDF_GetPassThroughShadowTransparency(bsdf
				MATERIALS_PARAM);

			if (!Spectrum_IsBlack(shadowTransparency)) {
				*connectionThroughput *= shadowTransparency;
				*throughShadowTransparency = true;
				continueToTrace = true;
			}
		}

		if (continueToTrace) {
			// Update volume information
			const BSDFEvent eventTypes = BSDF_GetEventTypes(bsdf
						MATERIALS_PARAM);
			PathVolumeInfo_Update(volInfo, eventTypes, bsdf
					MATERIALS_PARAM);

			// It is a transparent material, continue to trace the ray
			ray->mint = rayHit->t + MachineEpsilon_E(rayHit->t);

			// A safety check in case of not enough numerical precision
			if ((ray->mint == rayHit->t) || (ray->mint >= ray->maxt))
				return false;
			else
				return true;
		} else
			return false;
	} else {
		// Nothing was hit, stop tracing the ray
		return false;
	}
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
