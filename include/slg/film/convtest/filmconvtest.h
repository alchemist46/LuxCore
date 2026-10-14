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

#ifndef _SLG_FILMCONVTEST_H
#define	_SLG_FILMCONVTEST_H

#include <atomic>
#include <memory>
#include <vector>

#include "luxrays/usings.h"
#include "luxrays/utils/properties.h"
#include "luxrays/utils/serializationutils.h"
#include "slg/usings.h"
#include "slg/film/film.h"
#include "slg/film/framebuffer.h"

namespace slg {


//------------------------------------------------------------------------------
// FilmConvTest
//
// The noise threshold halt test. Two error estimates:
//
// - the per-pixel estimate (halfBuffer, an option; Cycles' model after
//   Dammertz et al. 2010 and Christensen et al. 2018 section 10.2.1): the
//   samplers accumulate one half of the samples of each pixel in the *_HALF
//   film channels too (with SOBOL, the samples of a pixel alternate between
//   two independently scrambled sequences and the half is the second one),
//   and the error of a pixel is the difference between the image and the
//   image of that half, on the linear radiance: (|dR| + |dG| + |dB|) /
//   max(sqrt(I), I) with I = R + G + B. The standard deviation of a sample
//   (error * sqrt(N)) is averaged over the 5x5 neighbours with useFilter (the
//   estimate of a single pixel is very noisy). A pixel is converged when its
//   error is under threshold * sqrt(sigma / mean(sigma)) (at most twice the
//   threshold): each pixel then gets a sample count proportional to its
//   standard deviation, the allocation with the lowest mean squared error,
//   and the threshold is the RMS error of the image. A converged pixel stays
//   converged until the film is reset, the SOBOL sampler draws its pixels
//   from the list of the unconverged ones (GetTodoPixels(); RANDOM skips the
//   converged ones) and the rendering stops when every pixel is converged
//   (minus the tolerance). It reads the
//   radiance channels, not the image pipeline output. Used when the film has
//   the half channel (the SOBOL and RANDOM samplers add it), the classic
//   test otherwise. Measured on 2026-10-14 (see MODERNIZATION_PLAN.md): about
//   the classic test's quality at the same render time where the classic
//   test does well, much better where its halt time explodes, and always
//   bounded and predictable; it is not the default (the threshold has a
//   different meaning).
//
// - the classic test (the default): the difference between two snapshots of the image
//   pipeline output `testStep` samples apart, the rendering stops when no
//   pixel changed by more than the threshold (see `passes` and `tolerance`).
//------------------------------------------------------------------------------


class FilmConvTest {
public:
	FilmConvTest(
		FilmConstPtr flm,
		const float threshold,
		const u_int warmup,
		const u_int testStep,
		const bool useFilter,
		const u_int imagePipelineIndex,
		const float tolerance = 0.f,
		const u_int passes = 0,
		const bool halfBuffer = false
	);
	~FilmConvTest();

	bool IsTestUpdateRequired() const;
	// The per-pixel error estimate from the *_HALF channels is in use
	bool UsesHalfBuffer() const;

	void Reset();
	u_int Test();

	// The pixels still above the threshold (a pixel that passed once stays
	// converged until the film is reset)
	u_int todoPixelsCount;
	float maxError;

	// The halt tolerates this fraction of unconverged pixels
	bool IsConverged() const;

	// The per-pixel test only: the pixels still to converge, as indices
	// x + y * width relative to the film sub-region, in the order of the
	// samplers' buckets (16x16 tiles in rows, Morton order inside a tile).
	// The samplers take their pixels from it instead of skipping the
	// converged ones. nullptr until the first test (and after a reset).
	std::shared_ptr<const std::vector<u_int>> GetTodoPixels() const { return todoPixels.load(); }

	FilmConstRef GetFilm() const { return *film; }

	friend class boost::serialization::access;

private:
	// Used by serialization
	FilmConvTest() = default;

	template<class Archive> void serialize(Archive &ar, const u_int version);

	void TestHalfBuffer();
	void TestImageDifference();

	float threshold;
	u_int warmup;
	u_int testStep;
	bool useFilter;
	u_int imagePipelineIndex;
	float tolerance;
	// The consecutive tests a pixel has to pass to stay converged (0: every
	// pixel has to pass in the same test, the classic behaviour)
	u_int passes;
	bool halfBuffer;

	FilmConstPtr film;  // This could be a const ref, but due to
								   // boost serialization, it isn't...

	GenericFrameBuffer<3, 0, float> *referenceImage;
	double lastSamplesCount;
	bool firstTest;
	// Classic test: the consecutive passes of each pixel (>= passes:
	// converged, sticky). Per-pixel test: 1 for the converged pixels.
	std::vector<unsigned char> convergedPixels;
	// Written by the test, read by the rendering threads (not serialized)
	std::atomic<std::shared_ptr<const std::vector<u_int>>> todoPixels;
};

}

BOOST_CLASS_VERSION(slg::FilmConvTest, 4)

BOOST_CLASS_EXPORT_KEY(slg::FilmConvTest)

#endif	/* _SLG_FILMCONVTEST_H */
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
