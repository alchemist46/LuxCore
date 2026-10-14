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

#ifndef _SLG_SOBOL_SEQUENCE_H
#define	_SLG_SOBOL_SEQUENCE_H

#include "luxrays/core/randomgen.h"

#include "slg/slg.h"
#include "slg/samplers/sampler.h"

namespace slg {

//------------------------------------------------------------------------------
// SobolSequence
//------------------------------------------------------------------------------

// An Owen scrambled Sobol sequence (B. Burley, "Practical Hash-based Owen
// Scrambling", JCGT 2020): each seed (i.e. one per pixel) has its own
// randomization of the sequence, so the samples of a pixel are stratified
// while the pixels are decorrelated.
//
// The same code is in sampler_sobol_funcs.cl: the two must be kept in sync.

class SobolSequence {
public:
	SobolSequence();
	~SobolSequence();
	
	void RequestSamples(const u_int size);
	// pass is the index of the sample in the sequence of the seed
	float GetSample(const u_int pass, const u_int index) const;
	// The sample at an index of the sequence (the index is used as it is,
	// without the shuffling of GetSample())
	float GetSampleAtIndex(const u_int sobolIndex, const u_int index) const;

	// The nested uniform scramble of the base-4 digits of x (one random
	// permutation of the 4 digit values per node of the digit tree): the
	// hierarchical pixel ordering of Ahmed & Wonka 2020 (pbrt-v4's
	// ZSobolSampler, Cycles 4.2), when x is the Morton code of the pixel
	// The scramble permutes the `digits` low base-4 digits of x only: the
	// result stays below 4^digits
	static u_int NestedUniformScrambleBase4(const u_int x, const u_int seed, const u_int digits);
	// The number of base-4 digits of the Morton codes of a film of this size
	// (the bits per axis); the pixel sample index gets the remaining
	// 32 - 2 * digits bits
	static u_int BlueNoiseMortonDigits(const u_int width, const u_int height);

	// The scrambling seed of the sequence
	u_int seed;

	static void GenerateDirectionVectors(u_int *vectors, const u_int dimensions);

	// A good quality integer hash (to build seeds)
	static u_int Hash(u_int x) {
		x ^= x >> 16;
		x *= 0x7feb352du;
		x ^= x >> 15;
		x *= 0x846ca68bu;
		x ^= x >> 16;
		return x;
	}

	static u_int HashCombine(const u_int seed, const u_int v) {
		return seed ^ (v + 0x9e3779b9u + (seed << 6) + (seed >> 2));
	}

private:
	u_int SobolDimension(const u_int index, const u_int dimension) const;
	float Sample(const u_int sobolIndex, const u_int index) const;

	u_int *directions;

	// The scrambled pass is the same for all the dimensions of a sample
	mutable u_int cachedPass, cachedSeed, cachedShuffledPass;
};

}

#endif	/* _SLG_SOBOL_SEQUENCE_H */
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
