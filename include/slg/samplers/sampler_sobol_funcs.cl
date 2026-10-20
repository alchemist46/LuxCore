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

#if defined(LUXRAYS_CUDA_DEVICE)

// CUDA and HIP: the compiler turns it into a single bit reverse instruction
OPENCL_FORCE_INLINE uint SobolSequence_ReverseBits(uint x) {
	x = ((x >> 1) & 0x55555555u) | ((x & 0x55555555u) << 1);
	x = ((x >> 2) & 0x33333333u) | ((x & 0x33333333u) << 2);
	x = ((x >> 4) & 0x0f0f0f0fu) | ((x & 0x0f0f0f0fu) << 4);
	x = ((x >> 8) & 0x00ff00ffu) | ((x & 0x00ff00ffu) << 8);
	return (x >> 16) | (x << 16);
}

#else

// OpenCL: the shifts above become the llvm.bitreverse intrinsic, which the
// SPIR-V translation (llvm-spirv up to at least version 18) supports only with
// the SPV_KHR_bit_instructions extension, i.e. the SPIR-V bake fails. A table
// of the reversed bytes gives the same result without the intrinsic.
__constant uchar SobolSequence_reversedBytes[256] = {
	0x00, 0x80, 0x40, 0xc0, 0x20, 0xa0, 0x60, 0xe0, 0x10, 0x90, 0x50, 0xd0,
	0x30, 0xb0, 0x70, 0xf0, 0x08, 0x88, 0x48, 0xc8, 0x28, 0xa8, 0x68, 0xe8,
	0x18, 0x98, 0x58, 0xd8, 0x38, 0xb8, 0x78, 0xf8, 0x04, 0x84, 0x44, 0xc4,
	0x24, 0xa4, 0x64, 0xe4, 0x14, 0x94, 0x54, 0xd4, 0x34, 0xb4, 0x74, 0xf4,
	0x0c, 0x8c, 0x4c, 0xcc, 0x2c, 0xac, 0x6c, 0xec, 0x1c, 0x9c, 0x5c, 0xdc,
	0x3c, 0xbc, 0x7c, 0xfc, 0x02, 0x82, 0x42, 0xc2, 0x22, 0xa2, 0x62, 0xe2,
	0x12, 0x92, 0x52, 0xd2, 0x32, 0xb2, 0x72, 0xf2, 0x0a, 0x8a, 0x4a, 0xca,
	0x2a, 0xaa, 0x6a, 0xea, 0x1a, 0x9a, 0x5a, 0xda, 0x3a, 0xba, 0x7a, 0xfa,
	0x06, 0x86, 0x46, 0xc6, 0x26, 0xa6, 0x66, 0xe6, 0x16, 0x96, 0x56, 0xd6,
	0x36, 0xb6, 0x76, 0xf6, 0x0e, 0x8e, 0x4e, 0xce, 0x2e, 0xae, 0x6e, 0xee,
	0x1e, 0x9e, 0x5e, 0xde, 0x3e, 0xbe, 0x7e, 0xfe, 0x01, 0x81, 0x41, 0xc1,
	0x21, 0xa1, 0x61, 0xe1, 0x11, 0x91, 0x51, 0xd1, 0x31, 0xb1, 0x71, 0xf1,
	0x09, 0x89, 0x49, 0xc9, 0x29, 0xa9, 0x69, 0xe9, 0x19, 0x99, 0x59, 0xd9,
	0x39, 0xb9, 0x79, 0xf9, 0x05, 0x85, 0x45, 0xc5, 0x25, 0xa5, 0x65, 0xe5,
	0x15, 0x95, 0x55, 0xd5, 0x35, 0xb5, 0x75, 0xf5, 0x0d, 0x8d, 0x4d, 0xcd,
	0x2d, 0xad, 0x6d, 0xed, 0x1d, 0x9d, 0x5d, 0xdd, 0x3d, 0xbd, 0x7d, 0xfd,
	0x03, 0x83, 0x43, 0xc3, 0x23, 0xa3, 0x63, 0xe3, 0x13, 0x93, 0x53, 0xd3,
	0x33, 0xb3, 0x73, 0xf3, 0x0b, 0x8b, 0x4b, 0xcb, 0x2b, 0xab, 0x6b, 0xeb,
	0x1b, 0x9b, 0x5b, 0xdb, 0x3b, 0xbb, 0x7b, 0xfb, 0x07, 0x87, 0x47, 0xc7,
	0x27, 0xa7, 0x67, 0xe7, 0x17, 0x97, 0x57, 0xd7, 0x37, 0xb7, 0x77, 0xf7,
	0x0f, 0x8f, 0x4f, 0xcf, 0x2f, 0xaf, 0x6f, 0xef, 0x1f, 0x9f, 0x5f, 0xdf,
	0x3f, 0xbf, 0x7f, 0xff
};

OPENCL_FORCE_INLINE uint SobolSequence_ReverseBits(const uint x) {
	return ((uint)SobolSequence_reversedBytes[x & 0xffu] << 24) |
			((uint)SobolSequence_reversedBytes[(x >> 8) & 0xffu] << 16) |
			((uint)SobolSequence_reversedBytes[(x >> 16) & 0xffu] << 8) |
			(uint)SobolSequence_reversedBytes[x >> 24];
}

#endif

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
	__global SobolSample *samples = (__global SobolSample *)samplesBuff;
	__global SobolSample *sample = &samples[gid];

	// sample->isHalf is set by SobolSampler_InitNewSample()
	Film_AddSample(sampleResult->pixelX, sampleResult->pixelY,
			sampleResult, 1.f, sample->isHalf != 0u
			FILM_PARAM);
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
	uint bucketPixelStart = sample->bucketPixelStart;
	uint pixelOffset = sample->pixelOffset;
	uint passOffset = sample->passOffset;

	// The pixels skipped by the adaptive sampling: when the noise threshold
	// halt test has converged every pixel the loop must not spin forever (the
	// engine stops the rendering after the next test)
	uint skippedPixels = 0;

	// The per-pixel noise threshold halt test lists the pixels still to
	// converge: the buckets are taken from the list instead of skipping the
	// converged pixels (a GPU task can otherwise loop over thousands of
	// pixels, with an atomic per bucket, for one sample). Past the buckets of
	// the list come the buckets of a sweep of the whole film (no pixel is
	// skipped), so the few pixels left at the end don't get every sample of
	// every task until the host runs the next test. The buckets of the list
	// have the 0x80000000 flag.
	const uint todoPixelCount = sampler->sobol.halfReplicas ? samplerSharedData->todoPixelCount : 0u;
	const uint todoListBucketCount = (todoPixelCount + bucketSize - 1) / bucketSize;
	const uint todoBucketCount = todoPixelCount ? max(samplerSharedData->todoBucketCount, todoListBucketCount) : 0u;
	__global const uint *todoPixels = SobolSampler_GetPassesPtr(samplerSharedData) + samplerSharedData->todoPixelsOffset;

	for (;;) {
		passOffset++;
		if (passOffset >= superSampling) {
			pixelOffset++;
			passOffset = 0;

			if (pixelOffset >= bucketSize) {
				// Ask for a new bucket
				const uint counter = atomic_inc(&samplerSharedData->bucketIndex);
				uint bucketRound;
				if (todoBucketCount) {
					const uint b = counter % todoBucketCount;
					bucketRound = counter / todoBucketCount;
					if (b < todoListBucketCount)
						bucketIndex = b | 0x80000000u;
					else {
						// The sweep of the whole film
						const uint sweepBucketCount = todoBucketCount - todoListBucketCount;
						bucketIndex = (bucketRound * sweepBucketCount + (b - todoListBucketCount)) % bucketCount;
					}
				} else {
					bucketIndex = counter % bucketCount;
					bucketRound = counter / bucketCount;
				}

				// With more tasks than buckets (a small film, or the last
				// pixels of the noise threshold halt), many tasks get the same
				// bucket: each one starts at a different pixel of it. Walking
				// it from the first pixel, they all sampled the same pixel at
				// the same time, so a low spp render of a small film left most
				// of its pixels without a sample (96x96 at 16 spp: 78%).
				bucketPixelStart = bucketRound % bucketSize;

				sample->bucketIndex = bucketIndex;
				sample->bucketPixelStart = bucketPixelStart;
				pixelOffset = 0;
				passOffset = 0;
			}
		}

		// Transform the bucket index in a pixel coordinate

		const uint bucketPixel = (bucketPixelStart + pixelOffset) % bucketSize;
		uint subRegionPixelX, subRegionPixelY;
		if (bucketIndex & 0x80000000u) {
			const uint listIndex = (bucketIndex & 0x7fffffffu) * bucketSize + bucketPixel;
			if (listIndex >= todoPixelCount) {
				// The end of the last bucket (or of an older list)
				continue;
			}

			const uint p = todoPixels[listIndex];
			subRegionPixelX = p % subRegionWidth;
			subRegionPixelY = p / subRegionWidth;
		} else {
			const uint pixelBucketIndex = (bucketIndex / overlapping) * bucketSize + bucketPixel;
			const uint mortonCurveOffset = pixelBucketIndex % (tileSize * tileSize);
			const uint pixelTileIndex = pixelBucketIndex / (tileSize * tileSize);

			subRegionPixelX = (pixelTileIndex % tiletWidthCount) * tileSize + DecodeMorton2X(mortonCurveOffset);
			subRegionPixelY = (pixelTileIndex / tiletWidthCount) * tileSize + DecodeMorton2Y(mortonCurveOffset);
			if ((subRegionPixelX >= subRegionWidth) || (subRegionPixelY >= subRegionHeight)) {
				// Skip the pixels out of the film sub region
				continue;
			}
		}

		const uint pixelX = filmSubRegion0 + subRegionPixelX;
		const uint pixelY = filmSubRegion2 + subRegionPixelY;

		if (filmNoise) {
			const float adaptiveStrength = sampler->sobol.adaptiveStrength;

			if (adaptiveStrength > 0.f) {
				// Pixels are sampled in accordance with how far from
				// convergence they are (with the list, every pixel is to be
				// sampled)
				const float noise = todoBucketCount ? 1.f : filmNoise[pixelX + pixelY * filmWidth];

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

				if ((Rnd_FloatValue(seed) > threshold) && (++skippedPixels < SAMPLER_MAX_SKIPPED_PIXELS)) {
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

		// The two replicas of the per-pixel error estimate of the noise
		// threshold halt: the samples of a pixel alternate between two
		// independently scrambled sequences, the half is the second one (see
		// SobolSampler::InitNewSample() on the CPU)
		uint seedBase = samplerSharedData->seedBase;
		uint replica = 0u;
		if (taskConfig->sampler.sobol.halfReplicas) {
			replica = sample->pass & 1u;
			sample->pass >>= 1;
			if (replica)
				seedBase = SobolSequence_Hash(SobolSequence_HashCombine(seedBase, 0x4a1fu));
		}

		__global const uint* restrict sobolDirections = SobolSampler_GetSobolDirectionsPtr(samplerSharedData);
		if (taskConfig->sampler.sobol.blueNoise) {
			// Blue-noise pixel ordering (see SobolSampler::InitNewSample() on
			// the CPU): from here on sample->pass is the index in the sequence
			// and sample->seed the seed of the sequence
			const uint mortonDigits = SobolSequence_BlueNoiseMortonDigits(subRegionWidth, subRegionHeight);
			// No chunk bits: the pixels use their own sequence from the first
			// sample (a halt spp larger than the chunks can be)
			const uint chunkBits = clamp(taskConfig->sampler.sobol.blueNoiseChunkBits, 1u, 32u - 2u * mortonDigits);
			const uint chunkSize = taskConfig->sampler.sobol.blueNoiseChunkBits ? (1u << chunkBits) : 0u;
			const uint morton = EncodeMorton2(subRegionPixelX, subRegionPixelY);
			const uint pass = sample->pass;
			const bool first = taskConfig->sampler.sobol.blueNoiseFirst;
			if (first && (pass == 0)) {
				sample->pass = SobolSequence_NestedUniformScrambleBase4(morton,
						SobolSequence_Hash(SobolSequence_HashCombine(seedBase, 0xb1u)), mortonDigits);
				sample->seed = SobolSequence_Hash(SobolSequence_HashCombine(seedBase, 0xb2u));
			} else {
				const uint p = first ? (pass - 1) : pass;
				if (p < chunkSize) {
					// The pixels ordered by the base-4 scramble of the Morton
					// code, the samples shuffled by the base-2 nested scramble
					// of the whole index (see SobolSampler::InitNewSample() on
					// the CPU)
					const uint chunkSeed = SobolSequence_Hash(SobolSequence_HashCombine(seedBase, 0xb3u));
					const uint pixelOrder = SobolSequence_NestedUniformScrambleBase4(morton, chunkSeed, mortonDigits);
					sample->pass = SobolSequence_NestedUniformScramble((pixelOrder << chunkBits) | p,
							SobolSequence_Hash(SobolSequence_HashCombine(chunkSeed, 0xb6u)));
					sample->seed = SobolSequence_Hash(SobolSequence_HashCombine(chunkSeed, 0xb4u));
				} else {
					// Past its chunk, a pixel continues with one sequence of
					// its own (see SobolSampler::InitNewSample() on the CPU)
					sample->seed = SobolSequence_Hash(SobolSequence_HashCombine(
							SobolSequence_HashCombine(seedBase, 0xb5u),
							subRegionPixelX + subRegionPixelY * subRegionWidth));
					sample->pass = SobolSequence_NestedUniformScramble(p - chunkSize, sample->seed);
				}
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

		// Without the replicas, the half of the sample is given by its index
		// in its sequence (sample->pass)
		sample->isHalf = taskConfig->sampler.sobol.halfReplicas ? replica : (Sampler_IsHalfSample(sample->pass) ? 1u : 0u);
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
	sample->bucketPixelStart = 0;
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
