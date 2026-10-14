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

#include <algorithm>
#include <limits>

#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>

#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/vector.hpp>

#include "slg/film/film.h"
#include "slg/film/convtest/filmconvtest.h"
#include "slg/film/imagepipeline/plugins/gaussianblur3x3.h"

using namespace std;
using namespace luxrays;
using namespace slg;

//------------------------------------------------------------------------------
// FilmConvTest
//------------------------------------------------------------------------------

BOOST_CLASS_EXPORT_IMPLEMENT(slg::FilmConvTest)

FilmConvTest::FilmConvTest(
	FilmConstPtr flm,
	const float thresholdVal,
	const u_int warmupVal,
	const u_int testStepVal,
	const bool useFilt,
	const u_int idx,
	const float tol,
	const u_int passesVal
) :
	film(flm),
	threshold(thresholdVal),
	warmup(warmupVal),
	testStep(testStepVal),
	useFilter(useFilt),
	imagePipelineIndex(idx),
	tolerance(tol),
	passes(passesVal),
	referenceImage(nullptr)
{
	Reset();
}

//FilmConvTest::FilmConvTest() {
	//referenceImage = NULL;
//}

FilmConvTest::~FilmConvTest() {
	delete referenceImage;
}

void FilmConvTest::Reset() {
	todoPixelsCount = GetFilm().GetWidth() * GetFilm().GetHeight();
	maxError = numeric_limits<float>::infinity();

	delete referenceImage;
	referenceImage = new GenericFrameBuffer<3, 0, float>(GetFilm().GetWidth(), GetFilm().GetHeight());

	lastSamplesCount = 0.0;
	firstTest = true;
	convergedPixels.assign(todoPixelsCount, 0);
}

bool FilmConvTest::IsConverged() const {
	const u_int pixelsCount = GetFilm().GetWidth() * GetFilm().GetHeight();

	return todoPixelsCount <= tolerance * pixelsCount;
}

bool FilmConvTest::IsTestUpdateRequired() const {
	const u_int pixelsCount = GetFilm().GetWidth() * GetFilm().GetHeight();

	// Run the test only after a initial warmup
	if (GetFilm().GetTotalSampleCount() / pixelsCount <= warmup)
		return false;

	// Do not run the test if we don't have at least testStep new samples per pixel
	if (GetFilm().GetTotalSampleCount() - lastSamplesCount <= pixelsCount * static_cast<double>(testStep))
		return false;

	return true;
}

u_int FilmConvTest::Test() {
	const u_int pixelsCount = GetFilm().GetWidth() * GetFilm().GetHeight();

	if (IsTestUpdateRequired()) {
		lastSamplesCount = GetFilm().GetTotalSampleCount();

		const u_int index = (imagePipelineIndex <= (GetFilm().GetImagePipelineCount() - 1)) ? imagePipelineIndex : 0;

		if (firstTest) {
			SLG_LOG("Convergence test first pass");

			// Copy the current image
			referenceImage->Copy(GetFilm().channel_IMAGEPIPELINEs[index]);
			firstTest = false;
		} else {
			// Check the number of pixels over the threshold
			const float *ref = referenceImage->GetPixels();
			const float *img = GetFilm().channel_IMAGEPIPELINEs[index]->GetPixels();
			const bool hasConvChannel = GetFilm().HasChannel(Film::CONVERGENCE);
			unsigned char *converged = &convergedPixels[0];
			// passes == 0: every pixel has to pass in the same test (no memory)
			const bool sticky = (passes > 0);

			// The count and the maximum don't depend on the order, so the
			// parallel result is the same as the serial one
			struct TestResult { u_int todoPixelsCount; float maxError; };
			const TestResult result = tbb::parallel_reduce(
					tbb::blocked_range<u_int>(0, pixelsCount), TestResult{0, 0.f},
					[&](const tbb::blocked_range<u_int> &range, TestResult res) {
						for (u_int i = range.begin(); i < range.end(); ++i) {
							float diff = 0.f;
							if (sticky && (converged[i] >= passes)) {
								// Converged for good
							} else {
								const float *imgPixel = &img[i * 3];
								const float *refPixel = &ref[i * 3];
								const float dr = fabsf(imgPixel[0] - refPixel[0]);
								const float dg = fabsf(imgPixel[1] - refPixel[1]);
								const float db = fabsf(imgPixel[2] - refPixel[2]);
								diff = Max(Max(dr, dg), db);

								res.maxError = Max(res.maxError, diff);

								if (diff > threshold) {
									++res.todoPixelsCount;
									converged[i] = 0;
								} else if (sticky) {
									// A pixel that passed the test `passes` times
									// in a row stays converged
									++converged[i];
									if (converged[i] >= passes)
										diff = 0.f;
									else
										++res.todoPixelsCount;
								}
							}

							if (hasConvChannel)
								*(GetFilm().channel_CONVERGENCE->GetPixel(i)) = Max(diff - threshold, 0.f);
						}
						return res;
					},
					[](const TestResult &a, const TestResult &b) {
						return TestResult{a.todoPixelsCount + b.todoPixelsCount, Max(a.maxError, b.maxError)};
					});
			todoPixelsCount = result.todoPixelsCount;
			maxError = result.maxError;

			if (hasConvChannel && useFilter) {
				// Apply a 3x3 blur filter to the CONVERGENCE channel
				GaussianBlur3x3FilterPlugin::ApplyBlurFilter(GetFilm().GetWidth(), GetFilm().GetHeight(),
						GetFilm().channel_CONVERGENCE->GetPixels(), referenceImage->GetPixels(),
						1.f, 1.f, 1.f);
			}
			referenceImage->Copy(GetFilm().channel_IMAGEPIPELINEs[index]);

			SLG_LOG("Convergence test: ToDo Pixels = " << todoPixelsCount << ", Max. Error = " << maxError << " [" << (256.f * maxError) << "/256]");

			if ((threshold > 0.f) && IsConverged())
				SLG_LOG("Convergence 100%, rendering done.");
		}
	}

	return (threshold == 0.f) ? pixelsCount : todoPixelsCount;
}

template<class Archive> void FilmConvTest::serialize(Archive &ar, const u_int version) {
	ar & todoPixelsCount;
	ar & maxError;

	ar & threshold;
	ar & warmup;
	ar & testStep;
	ar & film;
	ar & referenceImage;
	ar & lastSamplesCount;
	ar & firstTest;
	if (version >= 3) {
		ar & tolerance;
		ar & passes;
		ar & convergedPixels;
	} else if (Archive::is_loading::value) {
		tolerance = 0.f;
		passes = 0;
		convergedPixels.assign(GetFilm().GetWidth() * GetFilm().GetHeight(), 0);
	}
}

namespace slg {
// Explicit instantiations for portable archives
template void FilmConvTest::serialize(LuxOutputArchive &ar, const u_int version);
template void FilmConvTest::serialize(LuxInputArchive &ar, const u_int version);
template void FilmConvTest::serialize(LuxOutputArchiveText &ar, const u_int version);
template void FilmConvTest::serialize(LuxInputArchiveText &ar, const u_int version);
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
