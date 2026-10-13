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

#include "slg/samplers/sobolsequence.h"

using namespace std;
using namespace luxrays;
using namespace slg;

//------------------------------------------------------------------------------
// SobolSequence
//------------------------------------------------------------------------------

SobolSequence::SobolSequence() : directions(NULL) {
	seed = 0;
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

static inline u_int NestedUniformScramble(const u_int x, const u_int seed) {
	return ReverseBits(LaineKarrasPermutation(ReverseBits(x), seed));
}

float SobolSequence::GetSample(const u_int pass, const u_int index) const {
	// Shuffle the order of the samples (it keeps the stratification of the
	// power of 2 sized prefixes of the sequence)
	const u_int shuffledPass = NestedUniformScramble(pass, seed);

	// Owen scramble each dimension with a different seed
	const u_int dimensionSeed = Hash(HashCombine(seed, index));
	const u_int iResult = NestedUniformScramble(SobolDimension(shuffledPass, index), dimensionSeed);

	// 24 bits to have a result in [0, 1)
	return (iResult >> 8) * (1.f / 16777216.f);
}

// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
