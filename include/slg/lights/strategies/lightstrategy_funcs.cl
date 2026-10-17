#line 2 "lightstrategy_funcs.cl"

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

// The light tree is in use when lightTreeLightIndex is set (the caller passes
// NULL for the code paths that keep the distribution, e.g. the shadow
// catchers that only see the infinite lights)
OPENCL_FORCE_INLINE uint LightStrategy_SampleLights(
		__global const float* restrict lightsDistribution1D,
		__global const DLSCacheEntry* restrict dlscAllEntries,
		__global const float* restrict dlscDistributions,
		__global const IndexBVHArrayNode* restrict dlscBVHNodes,
		const float dlscRadius2, const float dlscNormalCosAngle,
		__global const LightTreeNode* restrict lightTreeNodes,
		__global const LightTreeDistantLight* restrict lightTreeDistantLights,
		const uint lightTreeDistantLightCount,
		__global const uint* restrict lightTreeLightIndex,
		__global const LightTreeLearn* restrict lightTreeLearn,
		__global const float* restrict lightTreeLearnCells,
		const float3 p, const float3 n,
		const bool isVolume,
		const float u, float *pdf, float *learnBoundsProb) {
	if (lightTreeLightIndex) {
		// Light tree strategy
		const uint lightIndex = LightTree_SampleLights(lightTreeNodes,
				lightTreeDistantLights, lightTreeDistantLightCount,
				lightTreeLearn, lightTreeLearnCells,
				p, n, isVolume, u, pdf, learnBoundsProb);

		return ((lightIndex != NULL_INDEX) && (*pdf > 0.f)) ? lightIndex : NULL_INDEX;
	}

#if !defined(RENDER_ENGINE_RTPATHOCL)
	if (dlscAllEntries) {
		// DLSC strategy

		// Check if a cache entry is available for this point
		const uint lightsDistributionOffset = DirectLightSamplingCache_GetLightDistribution(
				dlscAllEntries,
				dlscDistributions, dlscBVHNodes,
				dlscRadius2, dlscNormalCosAngle,
				p, n, isVolume
		);

		if (lightsDistributionOffset != NULL_INDEX) {
			const uint lightIndex = Distribution1D_SampleDiscrete(
					&dlscDistributions[lightsDistributionOffset], u, pdf);

			if (*pdf > 0.f)
				return lightIndex;
			else
				return NULL_INDEX;
		} else
			return Distribution1D_SampleDiscrete(lightsDistribution1D, u, pdf);
	} else
#endif
	{
		// All other strategies
		if (lightsDistribution1D)
			return Distribution1D_SampleDiscrete(lightsDistribution1D, u, pdf);
		else
			return NULL_INDEX;
	}
}

OPENCL_FORCE_INLINE float LightStrategy_SampleLightPdf(
		__global const float* restrict lightsDistribution1D,
		__global const DLSCacheEntry* restrict dlscAllEntries,
		__global const float* restrict dlscDistributions,
		__global const IndexBVHArrayNode* restrict dlscBVHNodes,
		const float dlscRadius2, const float dlscNormalCosAngle,
		__global const LightTreeNode* restrict lightTreeNodes,
		__global const LightTreeDistantLight* restrict lightTreeDistantLights,
		const uint lightTreeDistantLightCount,
		__global const uint* restrict lightTreeLightIndex,
		__global const LightTreeLearn* restrict lightTreeLearn,
		__global const float* restrict lightTreeLearnCells,
		const float3 p, const float3 n,
		const bool isVolume,
		const uint lightIndex) {
	if (lightTreeLightIndex) {
		// Light tree strategy
		return LightTree_SampleLightPdf(lightTreeNodes,
				lightTreeDistantLights, lightTreeDistantLightCount,
				lightTreeLightIndex, lightTreeLearn, lightTreeLearnCells,
				p, n, isVolume, lightIndex);
	}

#if !defined(RENDER_ENGINE_RTPATHOCL)
	if (dlscAllEntries) {
		// DLSC strategy
		
		// Check if a cache entry is available for this point
		const uint lightsDistributionOffset = DirectLightSamplingCache_GetLightDistribution(
				dlscAllEntries,
				dlscDistributions, dlscBVHNodes,
				dlscRadius2, dlscNormalCosAngle,
				p, n, isVolume
		);

		if (lightsDistributionOffset != NULL_INDEX)
			return Distribution1D_PdfDiscrete(&dlscDistributions[lightsDistributionOffset], lightIndex);
		else
			return Distribution1D_PdfDiscrete(lightsDistribution1D, lightIndex);
	} else
#endif
	{
		// All other strategies
		if (lightsDistribution1D)
			return Distribution1D_PdfDiscrete(lightsDistribution1D, lightIndex);
		else
			return NULL_INDEX;
	}
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
