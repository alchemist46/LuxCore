#line 2 "sampler_types.cl"

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
// Indices of Sample related u[] array
//------------------------------------------------------------------------------

#if defined(SLG_OPENCL_KERNEL)

#define IDX_SCREEN_X 0
#define IDX_SCREEN_Y 1
#define IDX_EYE_TIME 2
#define IDX_DOF_X 3
#define IDX_DOF_Y 4
#define IDX_BSDF_OFFSET 5

// Relative to IDX_BSDF_OFFSET + PathDepth * VERTEX_SAMPLE_SIZE
#define IDX_PASSTHROUGH 0
#define IDX_BSDF_X 1
#define IDX_BSDF_Y 2
#define IDX_DIRECTLIGHT_X 3
#define IDX_DIRECTLIGHT_Y 4
#define IDX_DIRECTLIGHT_Z 5
#define IDX_DIRECTLIGHT_W 6
#define IDX_DIRECTLIGHT_A 7
#define IDX_RR 8

#define VERTEX_SAMPLE_SIZE 9

// The samples of a pixel are split in two halves for the per-pixel error
// estimate of the noise threshold halt test (see FilmConvTest): the half is
// the parity of the odd bits of the sample index (the class A/B split of the
// (0, 2)-sequence, Christensen et al. 2018 section 10.2.1, as in Cycles), so
// both halves stay well stratified. The parity is computed without popcount
// because the function is shared with the CPU (see SobolSequence::IsHalfSample())
// The pixels a task may skip in a row before sampling one anyway (see
// SobolSampler_InitNewSample())
#define SAMPLER_MAX_SKIPPED_PIXELS 4096u

OPENCL_FORCE_INLINE bool Sampler_IsHalfSample(const uint sampleIndex) {
	uint x = (sampleIndex >> 1) & 0x55555555u;
	x ^= x >> 16;
	x ^= x >> 8;
	x ^= x >> 4;
	x ^= x >> 2;
	return (x & 1u) != 0u;
}

#endif

//------------------------------------------------------------------------------
// Sample data types
//------------------------------------------------------------------------------

typedef struct {
	unsigned int bucketIndex, pixelOffset, passOffset, pass;
} RandomSample;

typedef struct {
	float totalI;

	// Using ushort here totally freeze the ATI driver
	unsigned int largeMutationCount, smallMutationCount;
	unsigned int current, proposed, consecutiveRejects;

	float weight;

	SampleResult currentResult;
} MetropolisSample;

typedef struct {
	unsigned int bucketIndex, pixelOffset, passOffset, pass;

	// The Owen scrambling seed of the Sobol sequence of the pixel
	unsigned int seed;
	// The sample belongs to the half accumulated in the *_HALF film channels
	unsigned int half;
} SobolSample;

typedef struct {
	// The Owen scrambling seed of the Sobol sequence of the pixel
	unsigned int seed;

	unsigned int pass;
} TilePathSample;

//------------------------------------------------------------------------------
// Sampler shared data types
//------------------------------------------------------------------------------

typedef struct {
	unsigned int bucketIndex;
} RandomSamplerSharedData;

typedef struct {
	unsigned int seedBase;
	unsigned int bucketIndex;

	// This is used to compute the size of appended data at the end
	// of SobolSamplerSharedData
	unsigned int filmRegionPixelCount;

	// The per-pixel noise threshold halt test: the pixels still to converge
	// (0: no list) and the buckets to draw from: the ones of the list plus a
	// sweep of the whole film, so a few pixels left don't get every sample
	// (see SobolSampler_InitNewSample()). The list is at todoPixelsOffset
	// uints after the end of SobolSamplerSharedData.
	unsigned int todoPixelCount, todoBucketCount, todoPixelsOffset;

	// Plus the a pass field for each pixel
	// Plus Sobol directions array
	// Plus the list of the pixels to converge (with the per-pixel halt test)
} SobolSamplerSharedData;

typedef struct {
	// cameraFilmWidth/cameraFilmHeight and filmWidth/filmHeight are usually
	// the same. They are different when doing tile rendering
	unsigned int cameraFilmWidth, cameraFilmHeight;
	unsigned int tileStartX, tileStartY;
	unsigned int tileWidth, tileHeight;
	unsigned int tilePass, aaSamples;
	unsigned int multipassIndexToRender;

	// Plus Sobol directions array
} TilePathSamplerSharedData;

//------------------------------------------------------------------------------
// Sampler data types
//------------------------------------------------------------------------------

typedef enum {
	RANDOM = 0,
	METROPOLIS = 1,
	SOBOL = 2,
	TILEPATHSAMPLER = 3
} SamplerType;

typedef struct {
	SamplerType type;
	union {
		struct {
			float adaptiveStrength, adaptiveUserImportanceWeight;
			unsigned int bucketSize, tileSize, superSampling, overlapping;
		} random;
		struct {
			float adaptiveStrength, adaptiveUserImportanceWeight;
			unsigned int bucketSize, tileSize, superSampling, overlapping;
			// Blue-noise pixel ordering (see SobolSampler_NextSample())
			int blueNoise, blueNoiseFirst;
			unsigned int blueNoiseChunkBits;
			// The two replicas of the per-pixel error estimate (see
			// SobolSampler_InitNewSample())
			int halfReplicas;
		} sobol;
		struct {
			float largeMutationProbability, imageMutationRange;
			unsigned int maxRejects;
		} metropolis;
	};
} Sampler;

#define SOBOL_BITS 32
#define SOBOL_MAX_DIMENSIONS 21201
// With the Owen scrambling, the sequence can start from the first sample (and
// the power of 2 sized prefixes of the samples of a pixel are stratified)
#define SOBOL_STARTOFFSET 0

#define SAMPLER_PARAM_DECL \
		, Seed *seed \
		, __global void *samplerSharedDataBuff \
		, __global void *samplesBuff \
		, __global float *samplesDataBuff \
		, __global SampleResult *sampleResultsBuff
#define SAMPLER_PARAM \
		, seed \
		, samplerSharedDataBuff \
		, samplesBuff \
		, samplesDataBuff \
		, sampleResultsBuff
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
