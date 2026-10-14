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

#include <math.h>

#include "luxrays/utils/utils.h"
#include "slg/samplers/sobolsequence.h"

using namespace std;
using namespace luxrays;
using namespace slg;

//------------------------------------------------------------------------------
// SobolSequence
//------------------------------------------------------------------------------

static u_int NestedUniformScramble(const u_int x, const u_int seed);

SobolSequence::SobolSequence() : directions(NULL) {
	seed = 0;

	cachedPass = 0;
	cachedSeed = 0;
	cachedShuffledPass = NestedUniformScramble(cachedPass, cachedSeed);
}

SobolSequence::~SobolSequence() {
	delete[] directions;
}

void SobolSequence::RequestSamples(const u_int size) {
	directions = new u_int[size * SOBOL_BITS];
	GenerateDirectionVectors(directions, size);
}

u_int SobolSequence::SobolDimension(const u_int index, const u_int dimension) const {
	const u_int offset = dimension * SOBOL_BITS;
	u_int result = 0;
	u_int i = index;

	for (u_int j = 0; i; i >>= 1, j++) {
		if (i & 1)
			result ^= directions[offset + j];
	}

	return result;
}

static inline u_int ReverseBits(u_int x) {
	x = ((x >> 1) & 0x55555555u) | ((x & 0x55555555u) << 1);
	x = ((x >> 2) & 0x33333333u) | ((x & 0x33333333u) << 2);
	x = ((x >> 4) & 0x0f0f0f0fu) | ((x & 0x0f0f0f0fu) << 4);
	x = ((x >> 8) & 0x00ff00ffu) | ((x & 0x00ff00ffu) << 8);
	return (x >> 16) | (x << 16);
}

// The improved Laine-Karras permutation by N. Vegdahl ("Building a Better
// LK Hash", 2021): it is an Owen scrambling of the reversed bits
static inline u_int LaineKarrasPermutation(u_int x, const u_int seed) {
	x ^= x * 0x3d20adeau;
	x += seed;
	x *= (seed >> 16) | 1u;
	x ^= x * 0x05526c56u;
	x ^= x * 0x53a22864u;
	return x;
}

static u_int NestedUniformScramble(const u_int x, const u_int seed) {
	return ReverseBits(LaineKarrasPermutation(ReverseBits(x), seed));
}

float SobolSequence::Sample(const u_int sobolIndex, const u_int index) const {
	// Owen scramble each dimension with a different seed
	const u_int dimensionSeed = Hash(HashCombine(seed, index));
	const u_int iResult = NestedUniformScramble(SobolDimension(sobolIndex, index), dimensionSeed);

	// 24 bits to have a result in [0, 1)
	return (iResult >> 8) * (1.f / 16777216.f);
}

float SobolSequence::GetSample(const u_int pass, const u_int index) const {
	// Shuffle the order of the samples (it keeps the stratification of the
	// power of 2 sized prefixes of the sequence). The scramble of the pass
	// is the same for all the dimensions of a sample: cache it.
	if ((pass != cachedPass) || (seed != cachedSeed)) {
		cachedPass = pass;
		cachedSeed = seed;
		cachedShuffledPass = NestedUniformScramble(pass, seed);
	}

	return Sample(cachedShuffledPass, index);
}

float SobolSequence::GetSampleAtIndex(const u_int sobolIndex, const u_int index) const {
	return Sample(sobolIndex, index);
}

// The 24 permutations of the 4 values of a base-4 digit
static const unsigned char base4Permutations[24][4] = {
	{0, 1, 2, 3}, {0, 1, 3, 2}, {0, 2, 1, 3}, {0, 2, 3, 1}, {0, 3, 2, 1}, {0, 3, 1, 2},
	{1, 0, 2, 3}, {1, 0, 3, 2}, {1, 2, 0, 3}, {1, 2, 3, 0}, {1, 3, 2, 0}, {1, 3, 0, 2},
	{2, 1, 0, 3}, {2, 1, 3, 0}, {2, 0, 1, 3}, {2, 0, 3, 1}, {2, 3, 0, 1}, {2, 3, 1, 0},
	{3, 1, 2, 0}, {3, 1, 0, 2}, {3, 2, 1, 0}, {3, 2, 0, 1}, {3, 0, 2, 1}, {3, 0, 1, 2}
};

u_int SobolSequence::NestedUniformScrambleBase4(const u_int x, const u_int seed, const u_int digits) {
	u_int result = 0;
	for (int d = (int)digits - 1; d >= 0; --d) {
		const u_int shift = 2 * d;
		const u_int digit = (x >> shift) & 3u;
		// The permutation depends on the higher digits (the node of the tree)
		const u_int higher = (d < 15) ? (x >> (shift + 2)) : 0u;
		const u_int p = (Hash(HashCombine(HashCombine(seed, higher), d)) >> 24) % 24;
		result |= u_int(base4Permutations[p][digit]) << shift;
	}
	return result;
}

u_int SobolSequence::BlueNoiseMortonDigits(const u_int width, const u_int height) {
	const u_int maxSize = Max(width, height);
	u_int bits = 0;
	while ((1u << bits) < maxSize)
		++bits;
	// At least 1 bit for the pixel samples
	return Clamp(bits, 1u, 15u);
}

// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
