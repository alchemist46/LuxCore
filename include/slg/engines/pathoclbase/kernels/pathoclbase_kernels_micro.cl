#line 2 "pathoclbase_kernels_micro.cl"

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

//------------------------------------------------------------------------------
// AdvancePaths (Micro-Kernels)
//------------------------------------------------------------------------------

//#define DEBUG_PRINTF_KERNEL_NAME 1

//------------------------------------------------------------------------------
// The steps of a path vertex, called by AdvancePaths_MK_RT
//------------------------------------------------------------------------------

#if defined(LUXRAYS_CUDA_DEVICE)
// The members of the vertex BSDF that the next iteration reads when it doesn't
// sample the BSDF: AdvancePaths_ShadowRayHit() (event types, shading normal),
// AdvancePaths_HitNothing() (the environment light visibility cache, the shadow
// transparency flag) and Scene_Intersect() on a miss (the light ID of a volume
// emission)
OPENCL_FORCE_INLINE void AdvancePaths_SaveVertexBSDF(__global BSDF *dst, const BSDF *src) {
	dst->materialIndex = src->materialIndex;
	dst->isVolume = src->isVolume;
	dst->hitPoint.p = src->hitPoint.p;
	dst->hitPoint.shadeN = src->hitPoint.shadeN;
	dst->hitPoint.intoObject = src->hitPoint.intoObject;
	dst->hitPoint.throughShadowTransparency = src->hitPoint.throughShadowTransparency;
}
#endif

// The next vertex ray: sample the BSDF, Russian roulette (the next state is
// MK_RT_NEXT_VERTEX or MK_SPLAT_SAMPLE)
OPENCL_FORCE_INLINE PathState AdvancePaths_GenerateNextVertexRay(
		__constant const GPUTaskConfiguration* restrict taskConfig,
		__global GPUTaskState *taskState,
		__global const BSDF *bsdf,
		__global EyePathInfo *pathInfo,
		__global Ray *ray,
		__global SampleResult *sampleResult,
		const DirectLightResult directLightResult
		SAMPLER_PARAM_DECL
		MATERIALS_PARAM_DECL) {
	const uint sampleOffset = taskConfig->pathTracer.eyeSampleBootSize + pathInfo->depth.depth * taskConfig->pathTracer.eyeSampleStepSize;

	// Sample the BSDF
	float3 sampledDir;
	float3 bsdfSample;
	float cosSampledDir;
	float bsdfPdfW;
	BSDFEvent bsdfEvent;

	if (BSDF_IsShadowCatcher(bsdf MATERIALS_PARAM) && (directLightResult != SHADOWED)) {
		bsdfSample = BSDF_ShadowCatcherSample(bsdf,
				&sampledDir, &bsdfPdfW, &cosSampledDir, &bsdfEvent
				MATERIALS_PARAM);

		if (sampleResult->firstPathVertex) {
			// In this case I have also to set the value of the alpha channel to 0.0
			sampleResult->alpha = 0.f;
		}
	} else {
		const float3 shadowTransparency = BSDF_GetPassThroughShadowTransparency(bsdf
				MATERIALS_PARAM);
		if (!sampleResult->firstPathVertex && !Spectrum_IsBlack(shadowTransparency) && !pathInfo->isNearlyS) {
			sampledDir = -VLOAD3F(&bsdf->hitPoint.fixedDir.x);
			bsdfSample = shadowTransparency;
			bsdfPdfW = pathInfo->lastBSDFPdfW;
			cosSampledDir = -1.f;
			bsdfEvent = pathInfo->lastBSDFEvent;
		} else {
			bsdfSample = BSDF_Sample(bsdf,
					Sampler_GetSample(taskConfig, sampleOffset + IDX_BSDF_X SAMPLER_PARAM),
					Sampler_GetSample(taskConfig, sampleOffset + IDX_BSDF_Y SAMPLER_PARAM),
					&sampledDir, &bsdfPdfW, &cosSampledDir, &bsdfEvent
					MATERIALS_PARAM);

			pathInfo->isPassThroughPath = false;
		}
	}

	if (sampleResult->firstPathVertex)
		sampleResult->firstPathVertexEvent = bsdfEvent;

	EyePathInfo_AddVertex(pathInfo, bsdf, bsdfEvent, bsdfPdfW,
			taskConfig->pathTracer.hybridBackForward.glossinessThreshold
			MATERIALS_PARAM);

	// Russian Roulette
	const bool rrEnabled = EyePathInfo_UseRR(pathInfo, taskConfig->pathTracer.rrDepth);
	const float rrProb = rrEnabled ?
		RussianRouletteProb(taskConfig->pathTracer.rrImportanceCap, bsdfSample) :
		1.f;
	const bool rrContinuePath = !rrEnabled ||
		!(rrProb < Sampler_GetSample(taskConfig, sampleOffset + IDX_RR SAMPLER_PARAM));

	// Max. path depth
	const bool maxPathDepth = (pathInfo->depth.depth >= taskConfig->pathTracer.maxPathDepth.depth);

	const bool continuePath = !Spectrum_IsBlack(bsdfSample) && rrContinuePath && !maxPathDepth;
	if (continuePath) {
		// The pdf of the BSDF technique, for the MIS weight of a hit on a
		// light, includes the Russian Roulette
		pathInfo->lastBSDFPdfW *= rrProb;

		float3 throughputFactor = WHITE;

		// RR increases path contribution
		throughputFactor /= rrProb;
		throughputFactor *= bsdfSample;

		VSTORE3F(throughputFactor * VLOAD3F(taskState->throughput.c), taskState->throughput.c);

		// This is valid for irradiance AOV only if it is not a SPECULAR material and
		// first path vertex. Set or update sampleResult.irradiancePathThroughput
		if (!taskConfig->film.hasChannelIrradiance) {
		} else if (sampleResult->firstPathVertex) {
			if (!(BSDF_GetEventTypes(bsdf
						MATERIALS_PARAM) & SPECULAR))
				VSTORE3F(TO_FLOAT3(M_1_PI_F * fabs(dot(
						VLOAD3F(&bsdf->hitPoint.shadeN.x),
						sampledDir)) / rrProb),
						sampleResult->irradiancePathThroughput.c);
			else
				VSTORE3F(BLACK, sampleResult->irradiancePathThroughput.c);
		} else
			VSTORE3F(throughputFactor * VLOAD3F(sampleResult->irradiancePathThroughput.c), sampleResult->irradiancePathThroughput.c);

		Ray_Init2(ray, BSDF_GetRayOrigin(bsdf, sampledDir), sampledDir, ray->time);

		sampleResult->firstPathVertex = false;

		// Initialize the pass-through event seed
		//
		// Note: I use the IDX_PASSTHROUGH of the next path depth
		const uint nextSampleOffset = taskConfig->pathTracer.eyeSampleBootSize + pathInfo->depth.depth * taskConfig->pathTracer.eyeSampleStepSize;
		const float passThroughEvent = Sampler_GetSample(taskConfig, nextSampleOffset + IDX_PASSTHROUGH SAMPLER_PARAM);
		Seed seedPassThroughEvent;
		Rnd_InitFloat(passThroughEvent, &seedPassThroughEvent);
		taskState->seedPassThroughEvent = seedPassThroughEvent;

		// Initialize the trough a shadow transparency flag used by Scene_Intersect()
		taskState->throughShadowTransparency = false;

		return MK_RT_NEXT_VERTEX;
	} else
		return MK_SPLAT_SAMPLE;
}

// The trace result of the shadow ray: add the direct light if the light is
// visible (the next state is MK_GENERATE_NEXT_VERTEX_RAY or MK_SPLAT_SAMPLE,
// or MK_RT_DL to continue the trace). The BSDF is the one of the vertex of the
// shadow ray (only the members of AdvancePaths_SaveVertexBSDF() are read), but
// with a taskDirectLightVertex the path has moved to the next vertex: the path
// values of the vertex come from it.
OPENCL_FORCE_INLINE PathState AdvancePaths_ShadowRayHit(
		__constant const GPUTaskConfiguration* restrict taskConfig,
		__global GPUTaskTmp *taskTmp,
		__global HitPoint *intersectHitPoint,
		__global GPUTaskDirectLight *taskDirectLight,
		__global GPUTaskState *taskState,
		__global const BSDF *bsdf,
		__global PathVolumeInfo *directLightVolInfo,
		__global Ray *ray,
		__global RayHit *rayHit,
		__global SampleResult *sampleResult,
		__global const GPUTaskDirectLightVertex *taskDirectLightVertex
		MATERIALS_PARAM_DECL) {
	// Count the shadow ray traced by the RT kernel
	if (taskConfig->film.hasChannelRayCount)
		sampleResult->rayCount += 1;

	float3 connectionThroughput = WHITE;

	Seed seedPassThroughEvent = taskDirectLight->seedPassThroughEvent;
	const float passThroughEvent = Rnd_FloatValue(&seedPassThroughEvent);
	taskDirectLight->seedPassThroughEvent = seedPassThroughEvent;

	int throughShadowTransparency = taskDirectLight->throughShadowTransparency;
	// With the any-hit shadow rays (nothing in the scene can let a shadow ray
	// through: no transparency, no volume) the trace result is final and the
	// BSDF of the blocker is not needed
	const bool continueToTrace = scene->shadowRayAnyHit ? false :
		Scene_Intersect(taskConfig,
			EYE_RAY | SHADOW_RAY,
			&throughShadowTransparency,
			directLightVolInfo,
			intersectHitPoint,
			passThroughEvent,
			ray, rayHit, &taskTmp->tmpBsdf,
			&connectionThroughput, WHITE,
			NULL,
			true
			MATERIALS_PARAM
			);
	taskDirectLight->throughShadowTransparency = throughShadowTransparency;
	VSTORE3F(connectionThroughput * VLOAD3F(taskDirectLight->illumInfo.lightRadiance.c), taskDirectLight->illumInfo.lightRadiance.c);
	VSTORE3F(connectionThroughput * VLOAD3F(taskDirectLight->illumInfo.lightIrradiance.c), taskDirectLight->illumInfo.lightIrradiance.c);

	const bool rayMiss = (rayHit->meshIndex == NULL_INDEX);

	// If continueToTrace, there is nothing to do, just keep the same state
	if (!continueToTrace) {
		if (rayMiss) {
			// Nothing was hit, the light source is visible
			if (!BSDF_IsShadowCatcher(bsdf MATERIALS_PARAM)) {
				const bool firstPathVertex = taskDirectLightVertex ?
					taskDirectLightVertex->firstPathVertex : sampleResult->firstPathVertex;

				const float3 lightRadiance = VLOAD3F(taskDirectLight->illumInfo.lightRadiance.c);
				SampleResult_AddDirectLight(&taskConfig->film,
						sampleResult, taskDirectLight->illumInfo.lightID,
						BSDF_GetEventTypes(bsdf
							MATERIALS_PARAM),
						VLOAD3F(taskDirectLightVertex ? taskDirectLightVertex->throughput.c : taskState->throughput.c),
						lightRadiance, 1.f, firstPathVertex,
						VLOAD3F(taskDirectLightVertex ? taskDirectLightVertex->irradiancePathThroughput.c : sampleResult->irradiancePathThroughput.c));

				// The first path vertex is not handled by AddDirectLight(). This is valid
				// for irradiance AOV only if it is not a SPECULAR material.
				//
				// Note: irradiance samples the light sources only here (i.e. no
				// direct hit, no MIS, it would be useless)
				if (taskConfig->film.hasChannelIrradiance && firstPathVertex && !(BSDF_GetEventTypes(bsdf
							MATERIALS_PARAM) & SPECULAR)) {
					const float3 irradiance = (M_1_PI_F * fabs(dot(
								VLOAD3F(&bsdf->hitPoint.shadeN.x),
								VLOAD3F(&ray->d.x)))) *
							VLOAD3F(taskDirectLight->illumInfo.lightIrradiance.c);
					VSTORE3F(irradiance, sampleResult->irradiance.c);
				}
			}

			taskDirectLight->directLightResult = ILLUMINATED;
		} else
			taskDirectLight->directLightResult = SHADOWED;

		// Check if this is the last path vertex
		return (sampleResult->lastPathVertex) ? MK_SPLAT_SAMPLE : MK_GENERATE_NEXT_VERTEX_RAY;
	}

	return MK_RT_DL;
}

// Nothing was hit (the next state is MK_SPLAT_SAMPLE). The BSDF is the one of
// the previous path vertex (only the members of AdvancePaths_SaveVertexBSDF()
// are read).
OPENCL_FORCE_INLINE void AdvancePaths_HitNothing(
		__constant const GPUTaskConfiguration* restrict taskConfig,
		__global GPUTaskState *taskState,
		__global const BSDF *bsdf,
		__global EyePathInfo *pathInfo,
		__global const Ray *ray,
		__global SampleResult *sampleResult
		LIGHTS_PARAM_DECL) {
	bool checkDirectLightHit = true;
	
	checkDirectLightHit = checkDirectLightHit &&
			(!(taskConfig->pathTracer.forceBlackBackground && pathInfo->isPassThroughPath) || !pathInfo->isPassThroughPath);

	checkDirectLightHit = checkDirectLightHit &&
			// Avoid to render caustic path if hybridBackForwardEnable
			(!taskConfig->pathTracer.hybridBackForward.enabled || !EyePathInfo_IsCausticPath(pathInfo));

	checkDirectLightHit = checkDirectLightHit &&
			((!taskConfig->pathTracer.pgic.indirectEnabled && !taskConfig->pathTracer.pgic.causticEnabled) ||
			PhotonGICache_IsDirectLightHitVisible(taskConfig, pathInfo, taskState->photonGICausticCacheUsed));

	if (checkDirectLightHit) {
		DirectHitInfiniteLight(
				&taskConfig->film,
				pathInfo,
				&taskState->throughput,
				ray,
				sampleResult->firstPathVertex ? NULL : bsdf,
				sampleResult
				LIGHTS_PARAM);
	}

	if (pathInfo->depth.depth == 0) {
		sampleResult->alpha = 0.f;
		sampleResult->depth = INFINITY;
		sampleResult->position.x = INFINITY;
		sampleResult->position.y = INFINITY;
		sampleResult->position.z = INFINITY;
		sampleResult->geometryNormal.x = 0.f;
		sampleResult->geometryNormal.y = 0.f;
		sampleResult->geometryNormal.z = 0.f;
		sampleResult->shadingNormal.x = 0.f;
		sampleResult->shadingNormal.y = 0.f;
		sampleResult->shadingNormal.z = 0.f;
		sampleResult->materialID = 0;
		sampleResult->objectID = 0;
		sampleResult->uv.u = INFINITY;
		sampleResult->uv.v = INFINITY;
		sampleResult->isHoldout = false;
	} else if (!sampleResult->isHoldout && pathInfo->isTransmittedPath) {
		// I set to 0.0 also the alpha all purely transmitted paths hitting nothing
		sampleResult->alpha = 0.f;
	}
}

// Something was hit: the shading of the hit and the direct light sampling of
// the vertex (the next state is MK_RT_DL, MK_GENERATE_NEXT_VERTEX_RAY or
// MK_SPLAT_SAMPLE)
OPENCL_FORCE_INLINE PathState AdvancePaths_HitObject(
		__constant const GPUTaskConfiguration* restrict taskConfig,
		__global GPUTask *task,
		__global GPUTaskTmp *taskTmp,
		__global GPUTaskDirectLight *taskDirectLight,
		__global GPUTaskState *taskState,
		__global const BSDF *bsdf,
		__global EyePathInfo *pathInfo,
		__global PathVolumeInfo *directLightVolInfo,
		__global Ray *ray,
		__global const RayHit *rayHit,
		__global Ray *shadowRay,
		__global SampleResult *sampleResult,
		__global void *samplerSharedDataBuff,
		__global void *samplesBuff,
		__global float *samplesDataBuff,
		__global SampleResult *sampleResultsBuff
		KERNEL_ARGS_INFINITELIGHTS
		KERNEL_ARGS_PHOTONGI
		LIGHTS_PARAM_DECL) {
	if (taskState->albedoToDo && BSDF_IsAlbedoEndPoint(bsdf, taskConfig->pathTracer.albedo.specularSetting,
			taskConfig->pathTracer.albedo.specularGlossinessThreshold MATERIALS_PARAM)) {
		const float3 albedo = VLOAD3F(taskState->throughput.c) * BSDF_Albedo(bsdf
				MATERIALS_PARAM);
		VSTORE3F(albedo, sampleResult->albedo.c);
		sampleResult->shadingNormal = bsdf->hitPoint.shadeN;

		taskState->albedoToDo = false;
	}

	if (pathInfo->depth.depth == 0) {
		const bool isHoldout = BSDF_IsHoldout(bsdf
				MATERIALS_PARAM);
		sampleResult->alpha = isHoldout ? 0.f : 1.f;
		sampleResult->depth = rayHit->t;
		sampleResult->position = bsdf->hitPoint.p;
		sampleResult->geometryNormal = bsdf->hitPoint.geometryN;
		sampleResult->materialID = BSDF_GetMaterialID(bsdf
				MATERIALS_PARAM);
		sampleResult->objectID = BSDF_GetObjectID(bsdf, sceneObjs);
		sampleResult->uv = bsdf->hitPoint.defaultUV;
		sampleResult->isHoldout = isHoldout;
	}

	//----------------------------------------------------------------------
	// Check if it is a baked material
	//----------------------------------------------------------------------

	if (BSDF_HasBakeMap(bsdf, COMBINED MATERIALS_PARAM)) {
		const float3 radiance = VLOAD3F(&taskState->throughput.c[0]) * BSDF_GetBakeMapValue(bsdf MATERIALS_PARAM);
		VADD3F(sampleResult->radiancePerPixelNormalized[0].c, radiance);

		return MK_SPLAT_SAMPLE;
	} else if (BSDF_HasBakeMap(bsdf, LIGHTMAP MATERIALS_PARAM)) {
		const float3 radiance = VLOAD3F(&taskState->throughput.c[0]) *
				BSDF_Albedo(bsdf MATERIALS_PARAM) *
				BSDF_GetBakeMapValue(bsdf MATERIALS_PARAM);
		VADD3F(sampleResult->radiancePerPixelNormalized[0].c, radiance);

		return MK_SPLAT_SAMPLE;
	}

	//--------------------------------------------------------------------------
	// Check if it is a light source and I have to add light emission
	//--------------------------------------------------------------------------

	bool checkDirectLightHit = true;

	checkDirectLightHit = checkDirectLightHit &&
			// Avoid to render caustic path if hybridBackForwardEnable
			(!taskConfig->pathTracer.hybridBackForward.enabled || !EyePathInfo_IsCausticPath(pathInfo));

	checkDirectLightHit = checkDirectLightHit &&
			((!taskConfig->pathTracer.pgic.indirectEnabled && !taskConfig->pathTracer.pgic.causticEnabled) ||
			PhotonGICache_IsDirectLightHitVisible(taskConfig, pathInfo, taskState->photonGICausticCacheUsed));

	// Check if it is a light source (note: I can hit only triangle area light sources)
	if (BSDF_IsLightSource(bsdf) && checkDirectLightHit) {
		DirectHitFiniteLight(
				&taskConfig->film,
				pathInfo,
				&taskState->throughput,
				ray,
				rayHit->t,
				bsdf,
				sampleResult
				LIGHTS_PARAM);
	}

	//----------------------------------------------------------------------
	// Check if I can use the photon cache
	//----------------------------------------------------------------------

	if (taskConfig->pathTracer.pgic.indirectEnabled || taskConfig->pathTracer.pgic.causticEnabled) {
		const bool isPhotonGIEnabled = PhotonGICache_IsPhotonGIEnabled(bsdf,
				taskConfig->pathTracer.pgic.glossinessUsageThreshold
				MATERIALS_PARAM);

		switch (taskConfig->pathTracer.pgic.debugType) {
			case PGIC_DEBUG_SHOWINDIRECT: {
				if (isPhotonGIEnabled) {
					__global const Spectrum* restrict radiance = PhotonGICache_GetIndirectRadiance(bsdf,
							pgicRadiancePhotons, pgicLightGroupCounts, pgicRadiancePhotonsValues, pgicRadiancePhotonsBVHNodes,
							taskConfig->pathTracer.pgic.indirectLookUpRadius * taskConfig->pathTracer.pgic.indirectLookUpRadius,
							taskConfig->pathTracer.pgic.indirectLookUpNormalCosAngle);
					if (radiance) {
						for (uint i = 0; i < pgicLightGroupCounts; ++i)
							VADD3F(sampleResult->radiancePerPixelNormalized[i].c, VLOAD3F(radiance[i].c));
					}
				}
				return MK_SPLAT_SAMPLE;
			}
			case PGIC_DEBUG_SHOWCAUSTIC: {
				if (isPhotonGIEnabled) {
					PhotonGICache_ConnectWithCausticPaths(bsdf,
							pgicCausticPhotons, pgicCausticPhotonsBVHNodes,
							taskConfig->pathTracer.pgic.causticPhotonTracedCount,
							taskConfig->pathTracer.pgic.causticLookUpRadius,
							taskConfig->pathTracer.pgic.causticLookUpNormalCosAngle,
							WHITE,
							&sampleResult->radiancePerPixelNormalized[0]
							MATERIALS_PARAM);
				}
				return MK_SPLAT_SAMPLE;
			}
			case PGIC_DEBUG_SHOWINDIRECTPATHMIX: {
				if (isPhotonGIEnabled) {
					Seed seedPassThroughEvent = taskState->seedPassThroughEvent;
					const float passThroughEvent = Rnd_FloatValue(&seedPassThroughEvent);

					if (taskState->photonGICacheEnabledOnLastHit &&
							(rayHit->t > PhotonGICache_GetIndirectUsageThreshold(
								pathInfo->lastBSDFEvent,
								pathInfo->lastGlossiness,
								// I hope to not introduce strange sample correlations
								// by using passThrough here
								passThroughEvent,
								taskConfig->pathTracer.pgic.glossinessUsageThreshold,
								taskConfig->pathTracer.pgic.indirectUsageThresholdScale,
								taskConfig->pathTracer.pgic.indirectLookUpRadius))) {
						VSTORE3F(MAKE_FLOAT3(0.f, 0.f, 1.f), sampleResult->radiancePerPixelNormalized[0].c);
						taskState->photonGIShowIndirectPathMixUsed = true;

						return MK_SPLAT_SAMPLE;
					}

					taskState->photonGICacheEnabledOnLastHit = true;
				} else
					taskState->photonGICacheEnabledOnLastHit = false;

				break;
			}
			case PGIC_DEBUG_NONE:
			default: {
				if (isPhotonGIEnabled) {
					if (taskConfig->pathTracer.pgic.causticEnabled &&
							(!taskConfig->pathTracer.hybridBackForward.enabled || (pathInfo->depth.depth != 0))) {
						const bool isEmpty = PhotonGICache_ConnectWithCausticPaths(bsdf,
								pgicCausticPhotons, pgicCausticPhotonsBVHNodes,
								taskConfig->pathTracer.pgic.causticPhotonTracedCount,
								taskConfig->pathTracer.pgic.causticLookUpRadius,
								taskConfig->pathTracer.pgic.causticLookUpNormalCosAngle,
								VLOAD3F(taskState->throughput.c),
								&sampleResult->radiancePerPixelNormalized[0]
								MATERIALS_PARAM);

						if (!isEmpty)
							taskState->photonGICausticCacheUsed = true;
					}

					if (taskConfig->pathTracer.pgic.indirectEnabled) {
						Seed seedPassThroughEvent = taskState->seedPassThroughEvent;
						const float passThroughEvent = Rnd_FloatValue(&seedPassThroughEvent);

						if (taskState->photonGICacheEnabledOnLastHit &&
								(rayHit->t > PhotonGICache_GetIndirectUsageThreshold(
									pathInfo->lastBSDFEvent,
									pathInfo->lastGlossiness,
									// I hope to not introduce strange sample correlations
									// by using passThrough here
									passThroughEvent,
									taskConfig->pathTracer.pgic.glossinessUsageThreshold,
									taskConfig->pathTracer.pgic.indirectUsageThresholdScale,
									taskConfig->pathTracer.pgic.indirectLookUpRadius))) {
							__global const Spectrum* restrict radiance = PhotonGICache_GetIndirectRadiance(bsdf,
								pgicRadiancePhotons, pgicLightGroupCounts, pgicRadiancePhotonsValues, pgicRadiancePhotonsBVHNodes,
								taskConfig->pathTracer.pgic.indirectLookUpRadius * taskConfig->pathTracer.pgic.indirectLookUpRadius,
								taskConfig->pathTracer.pgic.indirectLookUpNormalCosAngle);

							if (radiance) {
								for (uint i = 0; i < pgicLightGroupCounts; ++i)
									VADD3F(sampleResult->radiancePerPixelNormalized[i].c, VLOAD3F(taskState->throughput.c) * VLOAD3F(radiance[i].c));
							}

							// I can terminate the path, all done
							return MK_SPLAT_SAMPLE;
						}
					}

					taskState->photonGICacheEnabledOnLastHit = true;
				} else
					taskState->photonGICacheEnabledOnLastHit = false;

				break;
			}
		}
	}

	//----------------------------------------------------------------------
	// Check if this is the last path vertex (but not also the first)
	//
	// I handle as a special case when the path vertex is both the first
	// and the last: I do direct light sampling without MIS.
	if (sampleResult->lastPathVertex && !sampleResult->firstPathVertex) {
		return MK_SPLAT_SAMPLE;
	}

	//--------------------------------------------------------------------------
	// Direct light sampling: pick a light source and a point on it, then
	// evaluate the BSDF in the direction of the light
	//--------------------------------------------------------------------------

	const uint sampleOffset = taskConfig->pathTracer.eyeSampleBootSize + pathInfo->depth.depth * taskConfig->pathTracer.eyeSampleStepSize;

	// Read the seed
	Seed seedValue = task->seed;
	// This trick is required by SAMPLER_PARAM macro
	Seed *seed = &seedValue;

	PathState nextState;

	// It will set eventually to true if the light is visible
	taskDirectLight->directLightResult = NOT_VISIBLE;

	if (!BSDF_IsDelta(bsdf
			MATERIALS_PARAM) &&
			DirectLight_Illuminate(
				bsdf,
				shadowRay,
				worldCenterX, worldCenterY, worldCenterZ, worldRadius,
				&taskTmp->tmpHitPoint,
				ray->time,
				Sampler_GetSample(taskConfig, sampleOffset + IDX_DIRECTLIGHT_X SAMPLER_PARAM),
				Sampler_GetSample(taskConfig, sampleOffset + IDX_DIRECTLIGHT_Y SAMPLER_PARAM),
				Sampler_GetSample(taskConfig, sampleOffset + IDX_DIRECTLIGHT_Z SAMPLER_PARAM),
				Sampler_GetSample(taskConfig, sampleOffset + IDX_DIRECTLIGHT_W SAMPLER_PARAM),
				&taskDirectLight->illumInfo
				LIGHTS_PARAM) &&
			DirectLight_BSDFSampling(
				taskConfig,
				&taskDirectLight->illumInfo,
				ray->time, sampleResult->lastPathVertex,
				pathInfo,
				&taskTmp->tmpPathDepthInfo,
				bsdf,
				VLOAD3F(&shadowRay->d.x)
				LIGHTS_PARAM)) {
		// Initialize the pass-through event for the shadow ray
		const float passThroughEvent = Sampler_GetSample(taskConfig, sampleOffset + IDX_DIRECTLIGHT_A SAMPLER_PARAM);
		Seed seedPassThroughEvent;
		Rnd_InitFloat(passThroughEvent, &seedPassThroughEvent);
		taskDirectLight->seedPassThroughEvent = seedPassThroughEvent;

		// Initialize the trough a shadow transparency flag used by Scene_Intersect()
		taskDirectLight->throughShadowTransparency = false;

		// Make a copy of current PathVolumeInfo for tracing the
		// shadow ray (an any-hit shadow ray doesn't use it, see
		// AdvancePaths_ShadowRayHit())
		if (!taskConfig->scene.shadowRayAnyHit)
			*directLightVolInfo = pathInfo->volume;

		// The shadow ray was initialized by Light_Illuminate(): flag it for
		// an any-hit trace if the scene allows it
		if (taskConfig->scene.shadowRayAnyHit)
			shadowRay->flags |= RAY_FLAGS_SHADOW;

		// I have to trace the shadow ray
		nextState = MK_RT_DL;
	} else {
		// No shadow ray to trace, move to the next vertex ray
		// however, I have to check if this is the last path vertex
		nextState = (sampleResult->lastPathVertex) ? MK_SPLAT_SAMPLE : MK_GENERATE_NEXT_VERTEX_RAY;
	}

	// Save the seed
	task->seed = seedValue;

	return nextState;
}

//------------------------------------------------------------------------------
// Evaluation of the Path finite state machine.
//
// From: the MK_RT_* states (waiting for a trace), or MK_GENERATE_CAMERA_RAY (a
//       sample Init() could not start)
// To: the MK_RT_* states (or MK_DONE with TILEPATHOCL/RTPATHOCL)
//
// Everything a path does with the results of a trace until it needs the next
// one, in one kernel so that the per-task state is loaded once:
// - the shadow ray (rays[taskCount + gid] when it was traced with the vertex
//   ray, else rays[gid]): the direct light if the light is visible;
// - the vertex ray (rays[gid]): the hit (or the miss), the shading of the
//   vertex, the direct light sampling and its shadow ray;
// - then the next vertex ray;
// - at the end of a path, the splat of the sample, the next sample and its
//   camera ray.
// The next vertex ray doesn't depend on the shadow ray, so both are traced
// together and a path vertex takes one iteration, except where the result of
// the shadow ray is needed first or the shadow ray may need more than one
// trace (shadow catchers, shadow rays crossing transparent surfaces or
// volumes): the next vertex ray is then sampled after the shadow ray (MK_RT_DL)
// and a vertex takes two iterations.
// It used to be ten kernels, one per state. The rays are traced before the
// kernel and a task reads its state once, at the start, so it never consumes
// a ray set up in the same iteration.
//------------------------------------------------------------------------------

__kernel void AdvancePaths_MK_RT(
		KERNEL_ARGS
		) {
	const size_t gid = get_global_id(0);

	// Read the path state
	__global GPUTaskState *taskState = &tasksState[gid];
	PathState pathState = pathStates[gid];
#if defined(DEBUG_PRINTF_KERNEL_NAME)
	if (gid == 0)
		printf("Kernel: AdvancePaths_MK_RT(state = %d)\n", pathState);
	else
		return;
#endif
	if (pathState == MK_DONE)
		return;

	//--------------------------------------------------------------------------
	// Start of variables setup
	//--------------------------------------------------------------------------

	__global GPUTask *task = &tasks[gid];
#if defined(LUXRAYS_CUDA_DEVICE)
	GPUTaskTmp taskTmpValue;
	GPUTaskTmp *taskTmp = &taskTmpValue;
	// The HitPoint used by Scene_Intersect() for the volumes stays in global
	// memory: a private one makes the ray marching of heterogeneous volumes
	// 7% slower (luxball-hdr-vol)
	__global HitPoint *intersectHitPoint = &tasksIntersectHitPoint[gid];
	// The BSDF of the vertex hit in this iteration. tasksBSDF[gid] keeps,
	// for the next iteration, the members of AdvancePaths_SaveVertexBSDF() or,
	// in the two-iteration flow (MK_RT_DL), the whole BSDF for the next vertex
	// ray
	BSDF bsdfValue;
	BSDF *bsdf = &bsdfValue;
	// Scene_Intersect() initializes the BSDF through a pointer the compiler
	// can't see is bsdf. Otherwise it forwards the values BSDF_Init() stores
	// to the reads of the shading code and keeps them in registers all along:
	// 14 spilled VGPRs instead of 2, manylights 3% slower.
	BSDF *initBsdf = bsdf;
#if defined(LUXRAYS_HIP_DEVICE)
	asm volatile("" : "+s"(initBsdf));
#endif
#else
	__global GPUTaskTmp *taskTmp = &tasksTmp[gid];
	__global HitPoint *intersectHitPoint = &taskTmp->tmpHitPoint;
	__global BSDF *bsdf = &tasksBSDF[gid];
	__global BSDF *initBsdf = bsdf;
#endif
	// The BSDF of the previous vertex (the vertex of the shadow ray, or the
	// last one before a miss)
	__global BSDF *vertexBsdf = &tasksBSDF[gid];
	__global const BSDF *prevBsdf = vertexBsdf;
	__global SampleResult *sampleResult = &sampleResultsBuff[gid];
	__global EyePathInfo *pathInfo = &eyePathInfos[gid];
	__constant const Scene* restrict scene = &taskConfig->scene;

	// Initialize image maps page pointer table
	INIT_IMAGEMAPS_PAGES

	//--------------------------------------------------------------------------
	// End of variables setup
	//--------------------------------------------------------------------------

	// A shadow ray traced with the vertex ray is in the second half of the ray
	// buffer, a shadow ray traced alone uses the slot of the vertex ray (so the
	// trace kernel doesn't run half empty waves)
	const size_t secondRayIndex = get_global_size(0) + gid;

	//--------------------------------------------------------------------------
	// The shadow ray
	//--------------------------------------------------------------------------

	if ((pathState == MK_RT_DL) || (pathState == MK_RT_NEXT_VERTEX_DL) ||
			(pathState == MK_RT_DL_SPLAT)) {
		const size_t shadowRayIndex = (pathState == MK_RT_NEXT_VERTEX_DL) ? secondRayIndex : gid;
		const PathState shadowRayState = AdvancePaths_ShadowRayHit(taskConfig, taskTmp, intersectHitPoint, &tasksDirectLight[gid], taskState,
				prevBsdf, &directLightVolInfos[gid], &rays[shadowRayIndex], &rayHits[shadowRayIndex], sampleResult,
				(pathState == MK_RT_DL) ? NULL : &tasksDirectLightVertex[gid]
				MATERIALS_PARAM);

		// The second slot is free again (the merged flow is used only when the
		// shadow ray is final after one trace)
		if (pathState == MK_RT_NEXT_VERTEX_DL)
			rays[secondRayIndex].flags = RAY_FLAGS_MASKED;

		if (pathState == MK_RT_DL) {
			pathState = shadowRayState;
#if defined(LUXRAYS_CUDA_DEVICE)
			// The next vertex ray samples the BSDF the previous iteration
			// stored
			if (pathState == MK_GENERATE_NEXT_VERTEX_RAY)
				*bsdf = *vertexBsdf;
#endif
		} else if (pathState == MK_RT_DL_SPLAT)
			pathState = MK_SPLAT_SAMPLE;
		else
			pathState = MK_RT_NEXT_VERTEX;
	}

	//--------------------------------------------------------------------------
	// The vertex ray
	//--------------------------------------------------------------------------

	bool shadowRayToTrace = false;
	if (pathState == MK_RT_NEXT_VERTEX) {
		// Count the ray traced by the RT kernel (the shadow rays are counted
		// by AdvancePaths_ShadowRayHit())
		if (taskConfig->film.hasChannelRayCount)
			sampleResult->rayCount += 1;

		float3 connectionThroughput;

		Seed seedPassThroughEvent = taskState->seedPassThroughEvent;
		const float passThroughEvent = Rnd_FloatValue(&seedPassThroughEvent);
		taskState->seedPassThroughEvent = seedPassThroughEvent;

#if defined(LUXRAYS_CUDA_DEVICE)
		// On a miss, Scene_Intersect() reads the light ID of the previous
		// vertex for the emission of a volume
		if (rayHits[gid].meshIndex == NULL_INDEX)
			bsdf->materialIndex = prevBsdf->materialIndex;
#endif

		int throughShadowTransparency = taskState->throughShadowTransparency;
		const bool continueToTrace = Scene_Intersect(taskConfig,
				EYE_RAY | ((pathInfo->depth.depth == 0) ? CAMERA_RAY : INDIRECT_RAY),
				&throughShadowTransparency,
				&pathInfo->volume,
				intersectHitPoint,
				passThroughEvent,
				&rays[gid], &rayHits[gid], initBsdf,
				&connectionThroughput, VLOAD3F(taskState->throughput.c),
				sampleResult,
				false
				MATERIALS_PARAM
				);
		taskState->throughShadowTransparency = throughShadowTransparency;
		VSTORE3F(connectionThroughput * VLOAD3F(taskState->throughput.c), taskState->throughput.c);

		// If continueToTrace, there is nothing to do, just keep the same state
		if (!continueToTrace) {
			if (rayHits[gid].meshIndex == NULL_INDEX) {
				AdvancePaths_HitNothing(taskConfig, taskState, prevBsdf, pathInfo, &rays[gid], sampleResult
						LIGHTS_PARAM);

				pathState = MK_SPLAT_SAMPLE;
			} else {
				const BSDFEvent eventTypes = BSDF_GetEventTypes(bsdf
						MATERIALS_PARAM);

				sampleResult->lastPathVertex = PathDepthInfo_IsLastPathVertex(&pathInfo->depth,
						&taskConfig->pathTracer.maxPathDepth, eventTypes);

				// The next vertex ray can be sampled without waiting for the
				// shadow ray if the shadow ray is final after one trace and
				// the BSDF sampling doesn't depend on it (shadow catchers).
				// The last vertex has no next vertex ray.
				const bool sampleNextVertexRayNow = scene->shadowRayAnyHit &&
						!sampleResult->lastPathVertex &&
						!BSDF_IsShadowCatcher(bsdf MATERIALS_PARAM);

				pathState = AdvancePaths_HitObject(taskConfig,
						task, taskTmp, &tasksDirectLight[gid], taskState, bsdf, pathInfo,
						&directLightVolInfos[gid], &rays[gid], &rayHits[gid],
						&rays[sampleNextVertexRayNow ? secondRayIndex : gid], sampleResult,
						samplerSharedDataBuff, samplesBuff, samplesDataBuff, sampleResultsBuff,
						worldCenterX, worldCenterY, worldCenterZ, worldRadius,
						pgicRadiancePhotons, pgicLightGroupCounts, pgicRadiancePhotonsValues,
						pgicRadiancePhotonsBVHNodes, pgicCausticPhotons, pgicCausticPhotonsBVHNodes
						LIGHTS_PARAM);

				// Otherwise the shadow ray took the place of the vertex ray, and
				// the next vertex ray is sampled after it (MK_RT_DL)
				if ((pathState == MK_RT_DL) && sampleNextVertexRayNow) {
					// Save the path values of this vertex for
					// AdvancePaths_ShadowRayHit()
					__global GPUTaskDirectLightVertex *taskDirectLightVertex = &tasksDirectLightVertex[gid];
					taskDirectLightVertex->throughput = taskState->throughput;
					if (taskConfig->film.hasChannelIrradiance)
						taskDirectLightVertex->irradiancePathThroughput = sampleResult->irradiancePathThroughput;
					taskDirectLightVertex->firstPathVertex = sampleResult->firstPathVertex;

					shadowRayToTrace = true;
					pathState = MK_GENERATE_NEXT_VERTEX_RAY;
				}
			}
		}

#if defined(LUXRAYS_CUDA_DEVICE)
		// Keep what the next iteration reads of the BSDF (there is one if
		// something was hit, or on a volume scatter event)
		if (rayHits[gid].meshIndex != NULL_INDEX) {
			if ((pathState == MK_RT_DL) && !sampleResult->lastPathVertex)
				*vertexBsdf = *bsdf;
			else
				AdvancePaths_SaveVertexBSDF(vertexBsdf, bsdf);
		}
#endif
	}

	if (pathState == MK_GENERATE_NEXT_VERTEX_RAY) {
		// Read the seed
		Seed seedValue = task->seed;
		// This trick is required by SAMPLER_PARAM macro
		Seed *seed = &seedValue;

		pathState = AdvancePaths_GenerateNextVertexRay(taskConfig, taskState, bsdf, pathInfo,
				&rays[gid], sampleResult, tasksDirectLight[gid].directLightResult
				SAMPLER_PARAM
				MATERIALS_PARAM);

		// Save the seed
		task->seed = seedValue;
	}

	if (shadowRayToTrace) {
		if (pathState == MK_RT_NEXT_VERTEX)
			pathState = MK_RT_NEXT_VERTEX_DL;
		else {
			// The path is over (MK_SPLAT_SAMPLE) but the direct light of its
			// last vertex is still to add: the sample is splat after the
			// shadow ray, which moves to the slot of the vertex ray
			pathState = MK_RT_DL_SPLAT;
			rays[gid] = rays[secondRayIndex];
			rays[secondRayIndex].flags = RAY_FLAGS_MASKED;
		}
	}

	//--------------------------------------------------------------------------
	// The end of a path: splat the sample, then start the next one
	//--------------------------------------------------------------------------

	if ((pathState == MK_SPLAT_SAMPLE) || (pathState == MK_NEXT_SAMPLE) ||
			(pathState == MK_GENERATE_CAMERA_RAY)) {
		// Read the seed
		Seed seedValue = task->seed;
		// This trick is required by SAMPLER_PARAM macro
		Seed *seed = &seedValue;

		if (pathState == MK_SPLAT_SAMPLE) {
			__constant const Film* restrict film = &taskConfig->film;

			// Initialize Film radiance group pointer table
			__global float *filmRadianceGroup[FILM_MAX_RADIANCE_GROUP_COUNT];
			filmRadianceGroup[0] = filmRadianceGroup0;
			filmRadianceGroup[1] = filmRadianceGroup1;
			filmRadianceGroup[2] = filmRadianceGroup2;
			filmRadianceGroup[3] = filmRadianceGroup3;
			filmRadianceGroup[4] = filmRadianceGroup4;
			filmRadianceGroup[5] = filmRadianceGroup5;
			filmRadianceGroup[6] = filmRadianceGroup6;
			filmRadianceGroup[7] = filmRadianceGroup7;

			// Initialize Film radiance group scale table
			float3 filmRadianceGroupScale[FILM_MAX_RADIANCE_GROUP_COUNT];
			filmRadianceGroupScale[0] = MAKE_FLOAT3(filmRadianceGroupScale0_R, filmRadianceGroupScale0_G, filmRadianceGroupScale0_B);
			filmRadianceGroupScale[1] = MAKE_FLOAT3(filmRadianceGroupScale1_R, filmRadianceGroupScale1_G, filmRadianceGroupScale1_B);
			filmRadianceGroupScale[2] = MAKE_FLOAT3(filmRadianceGroupScale2_R, filmRadianceGroupScale2_G, filmRadianceGroupScale2_B);
			filmRadianceGroupScale[3] = MAKE_FLOAT3(filmRadianceGroupScale3_R, filmRadianceGroupScale3_G, filmRadianceGroupScale3_B);
			filmRadianceGroupScale[4] = MAKE_FLOAT3(filmRadianceGroupScale4_R, filmRadianceGroupScale4_G, filmRadianceGroupScale4_B);
			filmRadianceGroupScale[5] = MAKE_FLOAT3(filmRadianceGroupScale5_R, filmRadianceGroupScale5_G, filmRadianceGroupScale5_B);
			filmRadianceGroupScale[6] = MAKE_FLOAT3(filmRadianceGroupScale6_R, filmRadianceGroupScale6_G, filmRadianceGroupScale6_B);
			filmRadianceGroupScale[7] = MAKE_FLOAT3(filmRadianceGroupScale7_R, filmRadianceGroupScale7_G, filmRadianceGroupScale7_B);

			if (sampleResult->isHoldout) {
				SampleResult_ClearRadiance(sampleResult);
				VSTORE3F(BLACK, sampleResult->albedo.c);
			}

			if (taskConfig->pathTracer.pgic.indirectEnabled &&
					(taskConfig->pathTracer.pgic.debugType == PGIC_DEBUG_SHOWINDIRECTPATHMIX) &&
					!taskState->photonGIShowIndirectPathMixUsed)
				VSTORE3F(MAKE_FLOAT3(1.f, 0.f, 0.f), sampleResult->radiancePerPixelNormalized[0].c);

			//--------------------------------------------------------------------------
			// Variance clamping
			//--------------------------------------------------------------------------

			const float sqrtVarianceClampMaxValue = taskConfig->pathTracer.sqrtVarianceClampMaxValue;
			if (sqrtVarianceClampMaxValue > 0.f) {
				// Radiance clamping
				VarianceClamping_Clamp(sampleResult, sqrtVarianceClampMaxValue
						FILM_PARAM);
			}

			//--------------------------------------------------------------------------
			// Sampler splat sample
			//--------------------------------------------------------------------------

			Sampler_SplatSample(taskConfig
					SAMPLER_PARAM
					FILM_PARAM);
			taskStats[gid].sampleCount += 1;

			pathState = MK_NEXT_SAMPLE;
		}

		if (pathState == MK_NEXT_SAMPLE) {
			Sampler_NextSample(taskConfig,
					filmNoise,
					filmUserImportance,
					filmWidth, filmHeight,
					filmSubRegion0, filmSubRegion1, filmSubRegion2, filmSubRegion3
					SAMPLER_PARAM);

			// Generate a new path and camera ray only it is not TILEPATHOCL
#if !defined(RENDER_ENGINE_TILEPATHOCL) && !defined(RENDER_ENGINE_RTPATHOCL)
			pathState = MK_GENERATE_CAMERA_RAY;
#else
			pathState = MK_DONE;
			// Mark the ray like like one to NOT trace
			rays[gid].flags = RAY_FLAGS_MASKED;
#endif
		}

		// Generate a new path and camera ray only it is not TILEPATHOCL: path
		// regeneration is not used in this case
#if !defined(RENDER_ENGINE_TILEPATHOCL) && !defined(RENDER_ENGINE_RTPATHOCL)
		if (pathState == MK_GENERATE_CAMERA_RAY) {
			// Re-initialize the volume information
			PathVolumeInfo_Init(&pathInfo->volume);

			GenerateEyePath(taskConfig,
					&tasksDirectLight[gid], taskState,
					camera,
					cameraBokehDistribution,
					filmWidth, filmHeight,
					filmSubRegion0, filmSubRegion1, filmSubRegion2, filmSubRegion3,
					pixelFilterDistribution,
					&rays[gid],
					pathInfo
					SAMPLER_PARAM);
			pathState = MK_RT_NEXT_VERTEX;
		}
#endif

		// Save the seed
		task->seed = seedValue;
	}

	// Save the state
	pathStates[gid] = pathState;
}

// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
