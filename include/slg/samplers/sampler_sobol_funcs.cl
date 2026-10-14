#line 2 "sampler_sobol_funcs.cl"

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
// Sobol Sequence
//------------------------------------------------------------------------------

OPENCL_FORCE_INLINE uint SobolSequence_SobolDimension(
		__global const uint* restrict sobolDirections,
		const uint index, const uint dimension) {
	const uint offset = dimension * SOBOL_BITS;
	uint result = 0;
	uint i = index;

	for (uint j = 0; i; i >>= 1, j++) {
		if (i & 1)
			result ^= sobolDirections[offset + j];
	}

	return result;
}

// An Owen scrambled Sobol sequence (B. Burley, "Practical Hash-based Owen
// Scrambling", JCGT 2020). It is the same code of slg::SobolSequence: the two
// must be kept in sync.

OPENCL_FORCE_INLINE uint SobolSequence_Hash(uint x) {
	x ^= x >> 16;
	x *= 0x7feb352du;
	x ^= x >> 15;
	x *= 0x846ca68bu;
	x ^= x >> 16;
	return x;
}

OPENCL_FORCE_INLINE uint SobolSequence_HashCombine(const uint seed, const uint v) {
	return seed ^ (v + 0x9e3779b9u + (seed << 6) + (seed >> 2));
}

OPENCL_FORCE_INLINE uint SobolSequence_ReverseBits(uint x) {
	x = ((x >> 1) & 0x55555555u) | ((x & 0x55555555u) << 1);
	x = ((x >> 2) & 0x33333333u) | ((x & 0x33333333u) << 2);
	x = ((x >> 4) & 0x0f0f0f0fu) | ((x & 0x0f0f0f0fu) << 4);
	x = ((x >> 8) & 0x00ff00ffu) | ((x & 0x00ff00ffu) << 8);
	return (x >> 16) | (x << 16);
}

// The improved Laine-Karras permutation by N. Vegdahl ("Building a Better
// LK Hash", 2021): it is an Owen scrambling of the reversed bits
OPENCL_FORCE_INLINE uint SobolSequence_LaineKarrasPermutation(uint x, const uint seed) {
	x ^= x * 0x3d20adeau;
	x += seed;
	x *= (seed >> 16) | 1u;
	x ^= x * 0x05526c56u;
	x ^= x * 0x53a22864u;
	return x;
}

OPENCL_FORCE_INLINE uint SobolSequence_NestedUniformScramble(const uint x, const uint seed) {
	return SobolSequence_ReverseBits(SobolSequence_LaineKarrasPermutation(SobolSequence_ReverseBits(x), seed));
}

OPENCL_FORCE_INLINE float SobolSequence_Sample(
		__global const uint* restrict sobolDirections,
		const uint sobolIndex, const uint seed,
		const uint index) {
	// Owen scramble each dimension with a different seed
	const uint dimensionSeed = SobolSequence_Hash(SobolSequence_HashCombine(seed, index));
	const uint iResult = SobolSequence_NestedUniformScramble(
			SobolSequence_SobolDimension(sobolDirections, sobolIndex, index), dimensionSeed);

	// 24 bits to have a result in [0, 1)
	return (iResult >> 8) * (1.f / 16777216.f);
}

OPENCL_FORCE_INLINE float SobolSequence_GetSample(
		__global const uint* restrict sobolDirections,
		const uint pass, const uint seed,
		const uint index) {
	// Shuffle the order of the samples (it keeps the stratification of the
	// power of 2 sized prefixes of the sequence)
	const uint shuffledPass = SobolSequence_NestedUniformScramble(pass, seed);

	return SobolSequence_Sample(sobolDirections, shuffledPass, seed, index);
}

// The 24 permutations of the 4 values of a base-4 digit
__constant uchar SobolSequence_base4Permutations[24][4] = {
	{0, 1, 2, 3}, {0, 1, 3, 2}, {0, 2, 1, 3}, {0, 2, 3, 1}, {0, 3, 2, 1}, {0, 3, 1, 2},
	{1, 0, 2, 3}, {1, 0, 3, 2}, {1, 2, 0, 3}, {1, 2, 3, 0}, {1, 3, 2, 0}, {1, 3, 0, 2},
	{2, 1, 0, 3}, {2, 1, 3, 0}, {2, 0, 1, 3}, {2, 0, 3, 1}, {2, 3, 0, 1}, {2, 3, 1, 0},
	{3, 1, 2, 0}, {3, 1, 0, 2}, {3, 2, 1, 0}, {3, 2, 0, 1}, {3, 0, 2, 1}, {3, 0, 1, 2}
};

// The nested uniform scramble of the base-4 digits of x: the hierarchical
// pixel ordering of Ahmed & Wonka 2020 when x is the Morton code of the pixel
// (the same code as SobolSequence::NestedUniformScrambleBase4())
OPENCL_FORCE_INLINE uint SobolSequence_NestedUniformScrambleBase4(const uint x, const uint seed, const uint digits) {
	uint result = 0;
	for (int d = (int)digits - 1; d >= 0; --d) {
		const uint shift = 2 * d;
		const uint digit = (x >> shift) & 3u;
		const uint higher = (d < 15) ? (x >> (shift + 2)) : 0u;
		const uint p = (SobolSequence_Hash(SobolSequence_HashCombine(SobolSequence_HashCombine(seed, higher), d)) >> 24) % 24;
		result |= ((uint)SobolSequence_base4Permutations[p][digit]) << shift;
	}
	return result;
}

OPENCL_FORCE_INLINE uint SobolSequence_BlueNoiseMortonDigits(const uint width, const uint height) {
	const uint maxSize = max(width, height);
	uint bits = 0;
	while ((1u << bits) < maxSize)
		++bits;
	return clamp(bits, 1u, 15u);
}

//------------------------------------------------------------------------------
// Sobol Sampler Kernel
//------------------------------------------------------------------------------

#define SOBOLSAMPLER_TOTAL_U_SIZE 2

OPENCL_FORCE_INLINE __global uint *SobolSampler_GetPassesPtr(__global SobolSamplerSharedData *samplerSharedData) {
	// Sobol pass array is appended at the end of slg::ocl::SobolSamplerSharedData
	return (__global uint *)(
			(__global char *)samplerSharedData +
			sizeof(SobolSamplerSharedData));
}

OPENCL_FORCE_INLINE __global const uint* restrict SobolSampler_GetSobolDirectionsPtr(__global SobolSamplerSharedData *samplerSharedData) {
	// Sobol directions array is appended at the end of slg::ocl::SobolSamplerSharedData + all pass values
	return (__global uint *)(
			(__global char *)samplerSharedData +
			sizeof(SobolSamplerSharedData) +
			sizeof(uint) * samplerSharedData->filmRegionPixelCount);
}

OPENCL_FORCE_INLINE float SobolSampler_GetSample(
		__constant const GPUTaskConfiguration* restrict taskConfig,
		const uint index
		SAMPLER_PARAM_DECL) {
	const size_t gid = get_global_id(0);

	switch (index) {
		case IDX_SCREEN_X: {
			__global float *samplesData = &samplesDataBuff[gid * SOBOLSAMPLER_TOTAL_U_SIZE];
			return samplesData[IDX_SCREEN_X];
		}
		case IDX_SCREEN_Y: {
			__global float *samplesData = &samplesDataBuff[gid * SOBOLSAMPLER_TOTAL_U_SIZE];
			return samplesData[IDX_SCREEN_Y];
		}
		default: {
			__global SobolSamplerSharedData *samplerSharedData = (__global SobolSamplerSharedData *)samplerSharedDataBuff;
			__global const uint* restrict sobolDirections = SobolSampler_GetSobolDirectionsPtr(samplerSharedData);

			__global SobolSample *samples = (__global SobolSample *)samplesBuff;
			__global SobolSample *sample = &samples[gid];

			return taskConfig->sampler.sobol.blueNoise ?
				SobolSequence_Sample(sobolDirections, sample->pass, sample->seed, index) :
				SobolSequence_GetSample(sobolDirections, sample->pass, sample->seed, index);
		}
	}
}

OPENCL_FORCE_INLINE void SobolSampler_SplatSample(
		__constant const GPUTaskConfiguration* restrict taskConfig
		SAMPLER_PARAM_DECL
		FILM_PARAM_DECL
		) {
	const size_t gid = get_global_id(0);
	__global SampleResult *sampleResult = &sampleResultsBuff[gid];

	Film_AddSample(sampleResult->pixelX, sampleResult->pixelY,
			sampleResult, 1.f
			FILM_PARAM);
}

OPENCL_FORCE_INLINE void SobolSamplerSharedData_GetNewBucket(__global SobolSamplerSharedData *samplerSharedData,
		const uint bucketCount, uint *newBucketIndex, uint *seed) {
	*newBucketIndex = atomic_inc(&samplerSharedData->bucketIndex) % bucketCount;

    *seed = (samplerSharedData->seedBase + *newBucketIndex) % (0xFFFFFFFFu - 1u) + 1u;
}

OPENCL_FORCE_INLINE void SobolSampler_InitNewSample(
		__constant const GPUTaskConfiguration* restrict taskConfig,
		__global float *filmNoise,
		__global float * filmUserImportance,
		const uint filmWidth, const uint filmHeight,
		const uint filmSubRegion0, const uint filmSubRegion1,
		const uint filmSubRegion2, const uint filmSubRegion3
		SAMPLER_PARAM_DECL) {
	const size_t gid = get_global_id(0);
	__constant const Sampler *sampler = &taskConfig->sampler;
	__global SobolSamplerSharedData *samplerSharedData = (__global SobolSamplerSharedData *)samplerSharedDataBuff;
	__global SobolSample *samples = (__global SobolSample *)samplesBuff;
	__global SobolSample *sample = &samples[gid];
	__global float *samplesData = &samplesDataBuff[gid * SOBOLSAMPLER_TOTAL_U_SIZE];

	const uint bucketSize = sampler->sobol.bucketSize;
	const uint tileSize = sampler->sobol.tileSize;
	const uint superSampling = sampler->sobol.superSampling;
	const uint overlapping = sampler->sobol.overlapping;
	
	const uint subRegionWidth = filmSubRegion1 - filmSubRegion0 + 1;
	const uint subRegionHeight = filmSubRegion3 - filmSubRegion2 + 1;

	const uint tiletWidthCount = (subRegionWidth + tileSize - 1) / tileSize;
	const uint tileHeightCount = (subRegionHeight + tileSize - 1) / tileSize;

	const uint bucketCount = overlapping * (tiletWidthCount * tileSize * tileHeightCount * tileSize + bucketSize - 1) / bucketSize;

	// Pick the pixel to render

	uint bucketIndex = sample->bucketIndex;
	uint pixelOffset = sample->pixelOffset;
	uint passOffset = sample->passOffset;

	for (;;) {
		passOffset++;
		if (passOffset >= superSampling) {
			pixelOffset++;
			passOffset = 0;

			if (pixelOffset >= bucketSize) {
				// Ask for a new bucket
				uint bucketSeed;
				SobolSamplerSharedData_GetNewBucket(samplerSharedData, bucketCount,
						&bucketIndex, &bucketSeed);

				sample->bucketIndex = bucketIndex;
				pixelOffset = 0;
				passOffset = 0;
			}
		}

		// Transform the bucket index in a pixel coordinate

		const uint pixelBucketIndex = (bucketIndex / overlapping) * bucketSize + pixelOffset;
		const uint mortonCurveOffset = pixelBucketIndex % (tileSize * tileSize);
		const uint pixelTileIndex = pixelBucketIndex / (tileSize * tileSize);

		const uint subRegionPixelX = (pixelTileIndex % tiletWidthCount) * tileSize + DecodeMorton2X(mortonCurveOffset);
		const uint subRegionPixelY = (pixelTileIndex / tiletWidthCount) * tileSize + DecodeMorton2Y(mortonCurveOffset);
		if ((subRegionPixelX >= subRegionWidth) || (subRegionPixelY >= subRegionHeight)) {
			// Skip the pixels out of the film sub region
			continue;
		}

		const uint pixelX = filmSubRegion0 + subRegionPixelX;
		const uint pixelY = filmSubRegion2 + subRegionPixelY;

		if (filmNoise) {
			const float adaptiveStrength = sampler->sobol.adaptiveStrength;

			if (adaptiveStrength > 0.f) {
				// Pixels are sampled in accordance with how far from convergence they are
				const float noise = filmNoise[pixelX + pixelY * filmWidth];

				// Factor user driven importance sampling too
				float threshold;
				if (filmUserImportance) {
					const float userImportance = filmUserImportance[pixelX + pixelY * filmWidth];

					// Noise is initialized to INFINITY at start
					if (isinf(noise))
						threshold = userImportance;
					else
						threshold = (userImportance > 0.f) ? Lerp(sampler->sobol.adaptiveUserImportanceWeight, noise, userImportance) : 0.f;
				} else
					threshold = noise;

				// The floor for the pixel importance is given by the adaptiveness strength
				threshold = fmax(threshold, 1.f - adaptiveStrength);

				if (Rnd_FloatValue(seed) > threshold) {
					// Skip this pixel and try the next one
					continue;
				}
			}
		}

		//----------------------------------------------------------------------
		// This code crashes the AMD OpenCL compiler:
		//
		// The array of fields is attached to the SamplerSharedData structure
#if !defined(LUXCORE_AMD_OPENCL)
		__global uint *pixelPasses = SobolSampler_GetPassesPtr(samplerSharedData);
		// Get the pass to do
		sample->pass = atomic_inc(&pixelPasses[subRegionPixelX + subRegionPixelY * subRegionWidth]);
#else   //----------------------------------------------------------------------
		// This one works:
		//
		// The array of fields is attached to the SamplerSharedData structure
		__global uint *pixelPass = SobolSampler_GetPassesPtr(samplerSharedData) + (subRegionPixelX + subRegionPixelY * subRegionWidth);
		// Get the pass to do
		uint oldVal, newVal;
		do {
				oldVal = *pixelPass;
				newVal = oldVal + 1;
		} while (atomic_cmpxchg(pixelPass, oldVal, newVal) != oldVal);
		sample->pass = oldVal;
#endif
		//----------------------------------------------------------------------

		__global const uint* restrict sobolDirections = SobolSampler_GetSobolDirectionsPtr(samplerSharedData);
		const uint seedBase = samplerSharedData->seedBase;
		if (taskConfig->sampler.sobol.blueNoise) {
			// Blue-noise pixel ordering (see SobolSampler::InitNewSample() on
			// the CPU): from here on sample->pass is the index in the sequence
			// and sample->seed the seed of the sequence
			const uint mortonDigits = SobolSequence_BlueNoiseMortonDigits(subRegionWidth, subRegionHeight);
			const uint chunkBits = clamp(taskConfig->sampler.sobol.blueNoiseChunkBits, 1u, 32u - 2u * mortonDigits);
			const uint morton = EncodeMorton2(subRegionPixelX, subRegionPixelY);
			const uint pass = sample->pass;
			const bool first = taskConfig->sampler.sobol.blueNoiseFirst;
			if (first && (pass == 0)) {
				sample->pass = SobolSequence_NestedUniformScrambleBase4(morton,
						SobolSequence_Hash(SobolSequence_HashCombine(seedBase, 0xb1u)), mortonDigits);
				sample->seed = SobolSequence_Hash(SobolSequence_HashCombine(seedBase, 0xb2u));
			} else {
				const uint p = first ? (pass - 1) : pass;
				const uint block = p >> chunkBits;
				const uint blockSeed = SobolSequence_Hash(SobolSequence_HashCombine(seedBase, 0xb3u + block));
				sample->pass = SobolSequence_NestedUniformScrambleBase4(
						(morton << chunkBits) | (p & ((1u << chunkBits) - 1u)),
						blockSeed, mortonDigits + (chunkBits + 1u) / 2u);
				sample->seed = SobolSequence_Hash(SobolSequence_HashCombine(blockSeed, 0xb4u));
			}

			samplesData[IDX_SCREEN_X] = pixelX + SobolSequence_Sample(sobolDirections, sample->pass, sample->seed, IDX_SCREEN_X);
			samplesData[IDX_SCREEN_Y] = pixelY + SobolSequence_Sample(sobolDirections, sample->pass, sample->seed, IDX_SCREEN_Y);
		} else {
			// Each pixel has its own scrambling of the Sobol sequence
			sample->seed = SobolSequence_Hash(SobolSequence_HashCombine(seedBase,
					subRegionPixelX + subRegionPixelY * subRegionWidth));

			// Initialize IDX_SCREEN_X and IDX_SCREEN_Y sample
			samplesData[IDX_SCREEN_X] = pixelX + SobolSequence_GetSample(sobolDirections, sample->pass, sample->seed, IDX_SCREEN_X);
			samplesData[IDX_SCREEN_Y] = pixelY + SobolSequence_GetSample(sobolDirections, sample->pass, sample->seed, IDX_SCREEN_Y);
		}
		break;
	}

	// Save the new values
	sample->pixelOffset = pixelOffset;
	sample->passOffset = passOffset;
}

OPENCL_FORCE_INLINE void SobolSampler_NextSample(
		__constant const GPUTaskConfiguration* restrict taskConfig,
		__global float *filmNoise,
		__global float *filmUserImportance,
		const uint filmWidth, const uint filmHeight,
		const uint filmSubRegion0, const uint filmSubRegion1,
		const uint filmSubRegion2, const uint filmSubRegion3
		SAMPLER_PARAM_DECL) {
	SobolSampler_InitNewSample(taskConfig,
			filmNoise,
			filmUserImportance,
			filmWidth, filmHeight,
			filmSubRegion0, filmSubRegion1, filmSubRegion2, filmSubRegion3
			SAMPLER_PARAM);
}

OPENCL_FORCE_INLINE bool SobolSampler_Init(__constant const GPUTaskConfiguration* restrict taskConfig,
		__global float *filmNoise,
		__global float *filmUserImportance,
		const uint filmWidth, const uint filmHeight,
		const uint filmSubRegion0, const uint filmSubRegion1,
		const uint filmSubRegion2, const uint filmSubRegion3
		SAMPLER_PARAM_DECL) {
	const size_t gid = get_global_id(0);
	__constant const Sampler *sampler = &taskConfig->sampler;
	__global SobolSample *samples = (__global SobolSample *)samplesBuff;
	__global SobolSample *sample = &samples[gid];

	const uint bucketSize = sampler->sobol.bucketSize;
	sample->pixelOffset = bucketSize * bucketSize;
	sample->passOffset = sampler->sobol.superSampling;

	SobolSampler_NextSample(taskConfig,
			filmNoise,
			filmUserImportance,
			filmWidth, filmHeight,
			filmSubRegion0, filmSubRegion1, filmSubRegion2, filmSubRegion3
			SAMPLER_PARAM);

	return true;
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
