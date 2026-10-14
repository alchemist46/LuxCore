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

#include <tbb/parallel_for.h>
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
	const u_int passesVal,
	const bool halfBuf
) :
	film(flm),
	threshold(thresholdVal),
	warmup(warmupVal),
	testStep(testStepVal),
	useFilter(useFilt),
	imagePipelineIndex(idx),
	tolerance(tol),
	passes(passesVal),
	halfBuffer(halfBuf),
	referenceImage(nullptr)
{
	Reset();
}

FilmConvTest::~FilmConvTest() {
	delete referenceImage;
}

bool FilmConvTest::UsesHalfBuffer() const {
	return halfBuffer && GetFilm().HasChannel(Film::RADIANCE_PER_PIXEL_NORMALIZED_HALF);
}

void FilmConvTest::Reset() {
	todoPixelsCount = GetFilm().GetWidth() * GetFilm().GetHeight();
	maxError = numeric_limits<float>::infinity();

	delete referenceImage;
	referenceImage = new GenericFrameBuffer<3, 0, float>(GetFilm().GetWidth(), GetFilm().GetHeight());

	lastSamplesCount = 0.0;
	firstTest = true;
	convergedPixels.assign(todoPixelsCount, 0);

	// The samplers skip the pixels with NOISE 0: after a reset every pixel
	// is to be sampled again (the film could have not been initialized yet)
	if (UsesHalfBuffer() && GetFilm().HasChannel(Film::NOISE) && GetFilm().channel_NOISE)
		GetFilm().channel_NOISE->Clear(numeric_limits<float>::infinity());
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

	// Do not run the test if we don't have at least testStep new samples per
	// pixel. With the per-pixel error estimate the samplers skip the converged
	// pixels, so the new samples go to the pixels still to converge
	const u_int pixelsToSample = UsesHalfBuffer() ? Max(todoPixelsCount, 1u) : pixelsCount;
	if (GetFilm().GetTotalSampleCount() - lastSamplesCount <= pixelsToSample * static_cast<double>(testStep))
		return false;

	return true;
}

u_int FilmConvTest::Test() {
	const u_int pixelsCount = GetFilm().GetWidth() * GetFilm().GetHeight();

	if (IsTestUpdateRequired()) {
		lastSamplesCount = GetFilm().GetTotalSampleCount();

		if (UsesHalfBuffer())
			TestHalfBuffer();
		else
			TestImageDifference();
	}

	return (threshold == 0.f) ? pixelsCount : todoPixelsCount;
}

//------------------------------------------------------------------------------
// The per-pixel error estimate
//------------------------------------------------------------------------------

void FilmConvTest::TestHalfBuffer() {
	const Film &film = GetFilm();
	const int width = static_cast<int>(film.GetWidth());
	const int height = static_cast<int>(film.GetHeight());
	const u_int pixelsCount = width * height;
	const u_int *subRegion = film.GetSubRegion();

	const u_int groupCount = film.channel_RADIANCE_PER_PIXEL_NORMALIZEDs.size();
	const float *half = film.channel_RADIANCE_PER_PIXEL_NORMALIZED_HALF->GetPixels();

	// The light traced samples (BiDir, hybrid back/forward path tracing) are
	// normalized by the light sample count; their half holds about half of
	// those samples
	const bool hasSN = (film.channel_RADIANCE_PER_SCREEN_NORMALIZEDs.size() > 0) &&
			film.HasChannel(Film::RADIANCE_PER_SCREEN_NORMALIZED_HALF) &&
			film.channel_RADIANCE_PER_SCREEN_NORMALIZED_HALF;
	const double lightSampleCount = film.GetTotalLightSampleCount();
	const float snScale = (hasSN && (lightSampleCount > 0.0)) ? static_cast<float>(1.0 / lightSampleCount) : 0.f;
	const float *halfSN = hasSN ? film.channel_RADIANCE_PER_SCREEN_NORMALIZED_HALF->GetPixels() : nullptr;

	const bool hasConvChannel = film.HasChannel(Film::CONVERGENCE);
	const bool hasNoiseChannel = film.HasChannel(Film::NOISE);
	unsigned char *converged = &convergedPixels[0];
	const float thr = threshold;

	// The error of each pixel and the pixels under the threshold
	vector<float> pixelError(pixelsCount);
	vector<unsigned char> passed(pixelsCount);

	tbb::parallel_for(tbb::blocked_range<u_int>(0, pixelsCount),
			[&](const tbb::blocked_range<u_int> &range) {
		for (u_int i = range.begin(); i < range.end(); ++i) {
			const u_int x = i % width;
			const u_int y = i / width;

			if ((x < subRegion[0]) || (x > subRegion[1]) || (y < subRegion[2]) || (y > subRegion[3])) {
				// Outside the rendered region: nothing to converge
				pixelError[i] = 0.f;
				passed[i] = 1;
				continue;
			}

			if (converged[i]) {
				// Converged for good (unless a neighbour reopens it below)
				pixelError[i] = 0.f;
				passed[i] = 1;
				continue;
			}

			// The image: the sum of the radiance groups
			Spectrum full, halfImg;
			bool hasSamples = false;
			for (u_int g = 0; g < groupCount; ++g) {
				const float *p = film.channel_RADIANCE_PER_PIXEL_NORMALIZEDs[g]->GetPixel(i);
				if (p[3] > 0.f) {
					full += Spectrum(p[0], p[1], p[2]) / p[3];
					hasSamples = true;
				}
			}

			// The image of the half
			const float *hp = &half[i * 4];
			bool hasHalfSamples = false;
			if (hp[3] > 0.f) {
				halfImg = Spectrum(hp[0], hp[1], hp[2]) / hp[3];
				hasHalfSamples = true;
			}

			if (hasSN) {
				for (u_int g = 0; g < groupCount; ++g) {
					const float *p = film.channel_RADIANCE_PER_SCREEN_NORMALIZEDs[g]->GetPixel(i);
					full += Spectrum(p[0], p[1], p[2]) * snScale;
				}
				const float *hs = &halfSN[i * 3];
				halfImg += Spectrum(hs[0], hs[1], hs[2]) * (2.f * snScale);
			}

			float error;
			if (!hasSamples || !hasHalfSamples) {
				// Not enough samples to estimate anything
				error = numeric_limits<float>::infinity();
			} else {
				const float diff = fabsf(full.c[0] - halfImg.c[0]) +
						fabsf(full.c[1] - halfImg.c[1]) +
						fabsf(full.c[2] - halfImg.c[2]);
				if (diff == 0.f)
					error = 0.f;
				else {
					// The noise of a dark pixel grows like the square root of
					// its brightness, the noise of a bright one with it
					const float I = Max(full.c[0] + full.c[1] + full.c[2], 0.f);
					const float norm = Max(sqrtf(I), I);
					error = (norm > 0.f) ? (diff / norm) : numeric_limits<float>::infinity();
				}
			}

			pixelError[i] = error;
			passed[i] = (error < thr) ? 1 : 0;
		}
	});

	if (useFilter) {
		// The difference of two halves of a single pixel is a very noisy
		// estimate of its error (pixels would stop anywhere between half and
		// twice the threshold): the average over the 3x3 neighbours (smaller
		// at the borders) is 3 times less noisy, and a pixel that passed by
		// chance next to noisy ones keeps getting samples. The converged
		// pixels (a sticky state) keep their 0 error.
		vector<float> rowSums(pixelsCount);
		tbb::parallel_for(tbb::blocked_range<int>(0, height),
				[&](const tbb::blocked_range<int> &range) {
			for (int y = range.begin(); y < range.end(); ++y) {
				const float *row = &pixelError[y * width];
				float *out = &rowSums[y * width];
				for (int x = 0; x < width; ++x) {
					float acc = row[x];
					if (x > 0)
						acc += row[x - 1];
					if (x < width - 1)
						acc += row[x + 1];
					out[x] = acc;
				}
			}
		});
		tbb::parallel_for(tbb::blocked_range<int>(0, height),
				[&](const tbb::blocked_range<int> &range) {
			for (int y = range.begin(); y < range.end(); ++y) {
				const int y0 = Max(y - 1, 0);
				const int y1 = Min(y + 1, height - 1);
				for (int x = 0; x < width; ++x) {
					const u_int i = x + y * width;
					if (converged[i])
						continue;

					float acc = 0.f;
					for (int ny = y0; ny <= y1; ++ny)
						acc += rowSums[x + ny * width];
					const int x0 = Max(x - 1, 0);
					const int x1 = Min(x + 1, width - 1);
					passed[i] = (acc / ((y1 - y0 + 1) * (x1 - x0 + 1)) < thr) ? 1 : 0;
				}
			}
		});
	}

	// The converged pixels are skipped by the samplers
	struct TestResult { u_int todoPixelsCount; float maxError; };
	const TestResult result = tbb::parallel_reduce(
			tbb::blocked_range<int>(0, height), TestResult{0, 0.f},
			[&](const tbb::blocked_range<int> &range, TestResult res) {
				for (int y = range.begin(); y < range.end(); ++y) {
					for (int x = 0; x < width; ++x) {
						const u_int i = x + y * width;

						const bool conv = passed[i];
						converged[i] = conv ? 1 : 0;
						const float error = pixelError[i];
						if (!conv) {
							++res.todoPixelsCount;
							res.maxError = Max(res.maxError, error);
						}

						if (hasConvChannel)
							*(film.channel_CONVERGENCE->GetPixel(i)) = conv ? 0.f : Max(error - thr, 0.f);
						if (hasNoiseChannel)
							*(film.channel_NOISE->GetPixel(i)) = conv ? 0.f : 1.f;
					}
				}
				return res;
			},
			[](const TestResult &a, const TestResult &b) {
				return TestResult{a.todoPixelsCount + b.todoPixelsCount, Max(a.maxError, b.maxError)};
			});
	todoPixelsCount = result.todoPixelsCount;
	maxError = result.maxError;

	SLG_LOG("Convergence test: ToDo Pixels = " << todoPixelsCount << ", Max. Error = " << maxError);

	if ((threshold > 0.f) && IsConverged())
		SLG_LOG("Convergence 100%, rendering done.");
}

//------------------------------------------------------------------------------
// The classic test
//------------------------------------------------------------------------------

void FilmConvTest::TestImageDifference() {
	const u_int pixelsCount = GetFilm().GetWidth() * GetFilm().GetHeight();

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
	if (version >= 4)
		ar & halfBuffer;
	else if (Archive::is_loading::value)
		halfBuffer = false;
}

namespace slg {
// Explicit instantiations for portable archives
template void FilmConvTest::serialize(LuxOutputArchive &ar, const u_int version);
template void FilmConvTest::serialize(LuxInputArchive &ar, const u_int version);
template void FilmConvTest::serialize(LuxOutputArchiveText &ar, const u_int version);
template void FilmConvTest::serialize(LuxInputArchiveText &ar, const u_int version);
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
