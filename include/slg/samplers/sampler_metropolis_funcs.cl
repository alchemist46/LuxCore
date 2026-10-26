#line 2 "sampler_metropolis_funcs.cl"

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
// Metropolis Sampler Kernel
//------------------------------------------------------------------------------

// The size of a sample vector, as allocated by the host
#define METROPOLISSAMPLER_TOTAL_U_SIZE (taskConfig->pathTracer.eyeSampleSize)

// Each task owns two sample vectors (the current sample and the proposed
// mutation), so a task's data is 2 * METROPOLISSAMPLER_TOTAL_U_SIZE floats
OPENCL_FORCE_INLINE __global float *MetropolisSampler_GetSampleData(
		__constant const GPUTaskConfiguration* restrict taskConfig,
		__global float *samplesDataBuff, const uint index) {
	const size_t gid = get_global_id(0);

	return &samplesDataBuff[(gid * 2 + index) * METROPOLISSAMPLER_TOTAL_U_SIZE];
}


OPENCL_FORCE_INLINE float MetropolisSampler_GetSample(
		__constant const GPUTaskConfiguration* restrict taskConfig,
		const uint index
		SAMPLER_PARAM_DECL) {
	const size_t gid = get_global_id(0);
	__global MetropolisSample *samples = (__global MetropolisSample *)samplesBuff;
	__global MetropolisSample *sample = &samples[gid];

	__global float *samplesData = MetropolisSampler_GetSampleData(taskConfig,
			samplesDataBuff, sample->proposed);

	return samplesData[index];
}

OPENCL_FORCE_INLINE void LargeStep(__constant const GPUTaskConfiguration* restrict taskConfig,
		Seed *seed, __global float *proposedU) {
	for (uint i = 0; i < METROPOLISSAMPLER_TOTAL_U_SIZE; ++i)
		proposedU[i] = Rnd_FloatValue(seed);
}

OPENCL_FORCE_INLINE float Mutate(const float x, const float randomValue) {
	const float s1 = 1.f / 512.f;
	const float s2 = 1.f / 16.f;

	const float dx = s1 / (s1 / s2 + fabs(2.f * randomValue - 1.f)) -
		s1 / (s1 / s2 + 1.f);

	float mutatedX = x;
	if (randomValue < .5f) {
		mutatedX += dx;
		mutatedX = (mutatedX < 1.f) ? mutatedX : (mutatedX - 1.f);
	} else {
		mutatedX -= dx;
		mutatedX = (mutatedX < 0.f) ? (mutatedX + 1.f) : mutatedX;
	}

	// mutatedX can still be 1.f due to numerical precision problems
	if (mutatedX == 1.f)
		mutatedX = 0.f;

	return mutatedX;
}

OPENCL_FORCE_INLINE float MutateScaled(const float x, const float range, const float randomValue) {
	const float s1 = 32.f;

	const float dx = range / (s1 / (1.f + s1) + (s1 * s1) / (1.f + s1) *
		fabs(2.f * randomValue - 1.f)) - range / s1;

	float mutatedX = x;
	if (randomValue < .5f) {
		mutatedX += dx;
		mutatedX = (mutatedX < 1.f) ? mutatedX : (mutatedX - 1.f);
	} else {
		mutatedX -= dx;
		mutatedX = (mutatedX < 0.f) ? (mutatedX + 1.f) : mutatedX;
	}

	// mutatedX can still be 1.f due to numerical precision problems
	if (mutatedX == 1.f)
		mutatedX = 0.f;

	return mutatedX;
}

OPENCL_FORCE_INLINE void SmallStep(__constant const GPUTaskConfiguration* restrict taskConfig,
		Seed *seed, __global float *currentU, __global float *proposedU) {
	// Metropolis return IDX_SCREEN_X and IDX_SCREEN_Y between [0.0, 1.0] instead
	// that in film pixels like the SOBOL sampler
	proposedU[IDX_SCREEN_X] = MutateScaled(currentU[IDX_SCREEN_X],
			taskConfig->sampler.metropolis.imageMutationRange, Rnd_FloatValue(seed));
	proposedU[IDX_SCREEN_Y] = MutateScaled(currentU[IDX_SCREEN_Y],
			taskConfig->sampler.metropolis.imageMutationRange, Rnd_FloatValue(seed));

	for (int i = IDX_SCREEN_Y + 1; i < METROPOLISSAMPLER_TOTAL_U_SIZE; ++i)
		proposedU[i] = Mutate(currentU[i], Rnd_FloatValue(seed));
}

OPENCL_FORCE_INLINE __global MetropolisTaskLuminance *MetropolisSampler_GetTaskLuminances(
		__global void *samplerSharedDataBuff) {
	return (__global MetropolisTaskLuminance *)((__global char *)samplerSharedDataBuff +
			sizeof(MetropolisSamplerSharedData));
}

OPENCL_FORCE_INLINE void MetropolisSampler_SplatSample(
		__constant const GPUTaskConfiguration* restrict taskConfig
		SAMPLER_PARAM_DECL
		FILM_PARAM_DECL
		) {
	const size_t gid = get_global_id(0);
	__global MetropolisSamplerSharedData *sharedData = (__global MetropolisSamplerSharedData *)samplerSharedDataBuff;
	__global MetropolisTaskLuminance *taskLuminance = &MetropolisSampler_GetTaskLuminances(samplerSharedDataBuff)[gid];
	__global MetropolisSample *samples = (__global MetropolisSample *)samplesBuff;
	__global MetropolisSample *sample = &samples[gid];
	__global SampleResult *sampleResult = &sampleResultsBuff[gid];

	//--------------------------------------------------------------------------
	// Accept/Reject the sample
	//--------------------------------------------------------------------------

	uint current = sample->current;
	uint proposed = sample->proposed;

	if (current == NULL_INDEX) {
		// It is the very first sample of the task, a uniform one (the
		// bootstrap, see PathOCLOpenCLRenderThread::MetropolisBootstrap()):
		// it is splat as such, it is the first large step of the luminance
		// estimate and a candidate for the start of the chains

		Film_AddSample(sampleResult->pixelX, sampleResult->pixelY,
				sampleResult, 1.f, false
				FILM_PARAM);

		sample->currentResult = *sampleResult;

		float firstI = SampleResult_GetRadianceY(&taskConfig->film, sampleResult);
		firstI = (isnan(firstI) || isinf(firstI)) ? 0.f : firstI;
		taskLuminance->totalI = firstI;
		taskLuminance->largeMutationCount = 1;

		current = proposed;
		proposed ^= 1;
	} else {
		const float currentI = SampleResult_GetRadianceY(&taskConfig->film, &sample->currentResult);

		float proposedI = SampleResult_GetRadianceY(&taskConfig->film, sampleResult);
		proposedI = (isnan(proposedI) || isinf(proposedI)) ? 0.f : proposedI;

		uint smallMutationCount = sample->smallMutationCount;
		if (smallMutationCount == 0) {
			// It is a large mutation
			taskLuminance->totalI += proposedI;
			taskLuminance->largeMutationCount += 1;
		}

		// The mean luminance of the image, from the large steps of all the
		// tasks: a chain's own estimate is poor when the chain is short
		const float invMeanI = sharedData->invMeanI;

		// Calculate accept probability from old and new image sample
		uint consecutiveRejects = sample->consecutiveRejects;

		float accProb;
		if ((currentI > 0.f) && (consecutiveRejects < taskConfig->sampler.metropolis.maxRejects))
			accProb = min(1.f, proposedI / currentI);
		else
			accProb = 1.f;

		const float newWeight = accProb + ((smallMutationCount == 0) ? 1.f : 0.f);
		float weight = sample->weight;
		weight += 1.f - accProb;

		const float rndVal = Rnd_FloatValue(seed);

		/*if (get_global_id(0) == 0)
			printf("[%d, %d][%f, %f][%d] Current: (%f, %f, %f) [%f/%f] Proposed: (%f, %f, %f) [%f] accProb: %f <%f>\n",
					current, proposed,
					currentI, proposedI,
					consecutiveRejects,
					sample->currentResult.radiancePerPixelNormalized[0].c[0], sample->currentResult.radiancePerPixelNormalized[0].c[1], sample->currentResult.radiancePerPixelNormalized[0].c[2], weight, sample->weight,
					sampleResult->radiancePerPixelNormalized[0].c[0], sampleResult->radiancePerPixelNormalized[0].c[1], sampleResult->radiancePerPixelNormalized[0].c[2], newWeight,
					accProb, rndVal);*/

		__global SampleResult *contrib;
		float norm;
		if ((accProb == 1.f) || (rndVal < accProb)) {
			/*if (get_global_id(0) == 0)
				printf("\t\tACCEPTED ! [%f]\n", currentI);*/

			// Add accumulated contribution of previous reference sample
			norm = weight / (currentI * invMeanI + taskConfig->sampler.metropolis.largeMutationProbability);
			contrib = &sample->currentResult;

			// Save new contributions for reference
			weight = newWeight;
			current ^= 1;
			proposed ^= 1;

			consecutiveRejects = 0;
		} else {
			/*if (get_global_id(0) == 0)
				printf("\t\tREJECTED ! [%f]\n", proposedI);*/

			// Add contribution of new sample before rejecting it
			norm = newWeight / (proposedI * invMeanI + taskConfig->sampler.metropolis.largeMutationProbability);
			contrib = sampleResult;

			++consecutiveRejects;
		}

		if (norm > 0.f) {
			/*if (get_global_id(0) == 0)
				printf("\t\tContrib: (%f, %f, %f) [%f] consecutiveRejects: %d\n",
						contrib->radiancePerPixelNormalized[0].c[0],
						contrib->radiancePerPixelNormalized[0].c[1],
						contrib->radiancePerPixelNormalized[0].c[2],
						norm, consecutiveRejects);*/

			Film_AddSample(contrib->pixelX, contrib->pixelY,
					contrib, norm, false
					FILM_PARAM);
		}

		if (sharedData->drain) {
			// The chain ends with this sample (the halt): the weight its
			// current sample has gathered is splat too. Left out, the
			// short chains of the GPU lost the bright samples most, as the
			// chains stay longer on them (-1.5% at 16 spp).
			const bool accepted = (consecutiveRejects == 0);
			const float lastNorm = weight / ((accepted ? proposedI : currentI) * invMeanI +
					taskConfig->sampler.metropolis.largeMutationProbability);
			if (lastNorm > 0.f) {
				__global SampleResult *last = accepted ? sampleResult : &sample->currentResult;
				Film_AddSample(last->pixelX, last->pixelY,
						last, lastNorm, false
						FILM_PARAM);
			}

			weight = 0.f;
		}

		// Check if it is an accepted mutation
		if (consecutiveRejects == 0) {
			// I can now (after Film_SplatSample()) overwrite sample->currentResult and sampleResult
			sample->currentResult = *sampleResult;
		}

		sample->weight = weight;
		sample->consecutiveRejects = consecutiveRejects;
	}

	sample->current = current;
	sample->proposed = proposed;
}

OPENCL_FORCE_INLINE void MetropolisSampler_NextSample(
		__constant const GPUTaskConfiguration* restrict taskConfig,
		__global float *filmNoise,
		__global float *filmUserImportance,
		const uint filmWidth, const uint filmHeight,
		const uint filmSubRegion0, const uint filmSubRegion1,
		const uint filmSubRegion2, const uint filmSubRegion3
		SAMPLER_PARAM_DECL) {
	const size_t gid = get_global_id(0);
	__global MetropolisSample *samples = (__global MetropolisSample *)samplesBuff;
	__global MetropolisSample *sample = &samples[gid];

	//--------------------------------------------------------------------------
	// Mutate the sample
	//--------------------------------------------------------------------------

	__global float *proposedU = MetropolisSampler_GetSampleData(taskConfig,
			samplesDataBuff, sample->proposed);
	if (Rnd_FloatValue(seed) < taskConfig->sampler.metropolis.largeMutationProbability) {
		LargeStep(taskConfig, seed, proposedU);
		sample->smallMutationCount = 0;
	} else {
		__global float *currentU = MetropolisSampler_GetSampleData(taskConfig,
			samplesDataBuff, sample->current);

		SmallStep(taskConfig, seed, currentU, proposedU);
		sample->smallMutationCount += 1;
	}
}

OPENCL_FORCE_INLINE bool MetropolisSampler_Init(
		__constant const GPUTaskConfiguration* restrict taskConfig,
		__global float *filmNoise,
		__global float *filmUserImportance,
		const uint filmWidth, const uint filmHeight,
		const uint filmSubRegion0, const uint filmSubRegion1,
		const uint filmSubRegion2, const uint filmSubRegion3
		SAMPLER_PARAM_DECL) {
	const size_t gid = get_global_id(0);
	__global MetropolisSample *samples = (__global MetropolisSample *)samplesBuff;
	__global MetropolisSample *sample = &samples[gid];

	sample->current = NULL_INDEX;
	sample->proposed = 1;

	sample->smallMutationCount = 0;
	sample->consecutiveRejects = 0;

	sample->weight = 0.f;

	__global float *samplesData = MetropolisSampler_GetSampleData(taskConfig,
			samplesDataBuff, sample->proposed);
	LargeStep(taskConfig, seed, samplesData);

	return true;
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
