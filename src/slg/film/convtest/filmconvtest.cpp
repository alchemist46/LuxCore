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
#include "slg/utils/mortoncurve.h"

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
	todoPixels.store(nullptr);

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

	auto inRegion = [&](const int x, const int y) {
		return (x >= static_cast<int>(subRegion[0])) && (x <= static_cast<int>(subRegion[1])) &&
				(y >= static_cast<int>(subRegion[2])) && (y <= static_cast<int>(subRegion[3]));
	};

	// The error of each pixel and its sample weight (the filter weights of the
	// samples, proportional to the sample count). The converged pixels get no
	// new samples, their error is computed again from the same sums.
	vector<float> pixelError(pixelsCount);
	vector<float> pixelWeight(pixelsCount);

	tbb::parallel_for(tbb::blocked_range<u_int>(0, pixelsCount),
			[&](const tbb::blocked_range<u_int> &range) {
		for (u_int i = range.begin(); i < range.end(); ++i) {
			if (!inRegion(i % width, i / width)) {
				pixelError[i] = 0.f;
				pixelWeight[i] = 0.f;
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
			pixelWeight[i] = (groupCount > 0) ? film.channel_RADIANCE_PER_PIXEL_NORMALIZEDs[0]->GetPixel(i)[3] : 0.f;

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
		}
	});

	// The standard deviation of a sample of each pixel, in the same units:
	// sigma_i = error_i * sqrt(N_i), with the sample weight for N_i (the
	// scale doesn't matter, only the ratios below). Negative: no estimate.
	vector<float> pixelSigma(pixelsCount);
	tbb::parallel_for(tbb::blocked_range<u_int>(0, pixelsCount),
			[&](const tbb::blocked_range<u_int> &range) {
		for (u_int i = range.begin(); i < range.end(); ++i) {
			const float error = pixelError[i];
			pixelSigma[i] = (inRegion(i % width, i / width) && !isinf(error) && !isnan(error)) ?
				(error * sqrtf(Max(pixelWeight[i], 0.f))) : -1.f;
		}
	});

	if (useFilter) {
		// The difference of two halves of a single pixel is a very noisy
		// estimate of its error: the standard deviation is averaged over the
		// 5x5 neighbours inside the rendered region (the ones with an
		// estimate), and the error of the pixel is that average divided by
		// the square root of its own sample count. Averaging the error itself
		// would let the frozen errors of the converged neighbours hold a
		// pixel above its threshold whatever its sample count. Simulated
		// against 1024 independent 2 spp renders: 5x5 is a few % more
		// efficient than 3x3, unfiltered loses 20-35%.
		const int r = 2;
		vector<float> rowSums(pixelsCount);
		vector<float> rowCounts(pixelsCount);
		vector<float> filtered(pixelsCount);
		tbb::parallel_for(tbb::blocked_range<int>(0, height),
				[&](const tbb::blocked_range<int> &range) {
			for (int y = range.begin(); y < range.end(); ++y) {
				const float *row = &pixelSigma[y * width];
				for (int x = 0; x < width; ++x) {
					const int x0 = Max(x - r, 0);
					const int x1 = Min(x + r, width - 1);
					float acc = 0.f, count = 0.f;
					for (int nx = x0; nx <= x1; ++nx) {
						if (row[nx] >= 0.f) {
							acc += row[nx];
							count += 1.f;
						}
					}
					rowSums[x + y * width] = acc;
					rowCounts[x + y * width] = count;
				}
			}
		});
		tbb::parallel_for(tbb::blocked_range<int>(0, height),
				[&](const tbb::blocked_range<int> &range) {
			for (int y = range.begin(); y < range.end(); ++y) {
				const int y0 = Max(y - r, 0);
				const int y1 = Min(y + r, height - 1);
				for (int x = 0; x < width; ++x) {
					const u_int i = x + y * width;
					if (pixelSigma[i] < 0.f) {
						filtered[i] = -1.f;
						continue;
					}

					float acc = 0.f, count = 0.f;
					for (int ny = y0; ny <= y1; ++ny) {
						acc += rowSums[x + ny * width];
						count += rowCounts[x + ny * width];
					}
					filtered[i] = acc / count;
				}
			}
		});
		pixelSigma.swap(filtered);

		tbb::parallel_for(tbb::blocked_range<u_int>(0, pixelsCount),
				[&](const tbb::blocked_range<u_int> &range) {
			for (u_int i = range.begin(); i < range.end(); ++i) {
				if (pixelSigma[i] >= 0.f) {
					const float w = pixelWeight[i];
					pixelError[i] = (w > 0.f) ? (pixelSigma[i] / sqrtf(w)) : numeric_limits<float>::infinity();
				}
			}
		});
	}

	// The stop rule. Stopping every pixel at the same error gives each pixel
	// a sample count proportional to its variance, and the image the same
	// mean squared error as uniform sampling at the same total sample count.
	// The allocation that minimizes the mean squared error gives each pixel a
	// sample count proportional to its standard deviation sigma_i instead: a
	// pixel is converged when
	//
	//   error_i <= threshold * sqrt(sigma_i / mean(sigma))
	//
	// i.e. when N_i >= sigma_i * mean(sigma) / threshold^2. The threshold is
	// then the RMS error of the image (in the error metric above). The
	// allowed error of a pixel is capped at 2 * threshold (it changed no
	// result in the simulations nor in the renders: it only bounds the worst
	// pixel). Simulated against 1024 independent 2 spp renders of cornell
	// and luxball-hdr, at equal sample count: 1.31-1.40x more efficient than
	// uniform sampling, where the same error for every pixel is 1.15-1.25x
	// (the bound, the optimal allocation with the exact standard deviations,
	// is 1.36-1.57x). Measured in renders: see docs/history/MODERNIZATION_PLAN.md.
	struct MeanAcc { double sum; double count; };
	const MeanAcc meanAcc = tbb::parallel_reduce(
			tbb::blocked_range<u_int>(0, pixelsCount), MeanAcc{0.0, 0.0},
			[&](const tbb::blocked_range<u_int> &range, MeanAcc acc) {
				for (u_int i = range.begin(); i < range.end(); ++i) {
					if (pixelSigma[i] >= 0.f) {
						acc.sum += pixelSigma[i];
						acc.count += 1.0;
					}
				}
				return acc;
			},
			[](const MeanAcc &a, const MeanAcc &b) {
				return MeanAcc{a.sum + b.sum, a.count + b.count};
			});
	const float meanSigma = (meanAcc.count > 0.0) ? static_cast<float>(meanAcc.sum / meanAcc.count) : 0.f;
	const float thr = threshold;
	const float maxScale = 2.f;

	// The converged pixels are skipped by the samplers
	struct TestResult { u_int todoPixelsCount; float maxError; };
	const TestResult result = tbb::parallel_reduce(
			tbb::blocked_range<int>(0, height), TestResult{0, 0.f},
			[&](const tbb::blocked_range<int> &range, TestResult res) {
				for (int y = range.begin(); y < range.end(); ++y) {
					for (int x = 0; x < width; ++x) {
						const u_int i = x + y * width;

						const float error = pixelError[i];
						float pixelThr = 0.f;
						if (!inRegion(x, y)) {
							// Outside the rendered region: nothing to converge
							converged[i] = 1;
						} else if (!converged[i] && (pixelSigma[i] >= 0.f) && !isinf(error)) {
							// A pixel that passed stays converged until the film is reset
							pixelThr = thr * ((meanSigma > 0.f) ? Min(sqrtf(pixelSigma[i] / meanSigma), maxScale) : 1.f);
							if (error <= pixelThr)
								converged[i] = 1;
						}

						const bool conv = converged[i];
						if (!conv) {
							++res.todoPixelsCount;
							res.maxError = Max(res.maxError, error);
						}

						if (hasConvChannel)
							*(film.channel_CONVERGENCE->GetPixel(i)) = conv ? 0.f : Max(error - pixelThr, 0.f);
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

	// The list of the pixels still to converge for the samplers, in the order
	// of their buckets (16x16 tiles, Morton order inside a tile, the default
	// sampler.sobol.tilesize) so neighbouring samples hit neighbouring pixels
	{
		const u_int regionWidth = subRegion[1] - subRegion[0] + 1;
		const u_int regionHeight = subRegion[3] - subRegion[2] + 1;
		const u_int tileSize = 16;
		const u_int tileWidthCount = (regionWidth + tileSize - 1) / tileSize;
		const u_int tileHeightCount = (regionHeight + tileSize - 1) / tileSize;

		auto list = std::make_shared<vector<u_int>>();
		list->reserve(todoPixelsCount);
		for (u_int tile = 0; tile < tileWidthCount * tileHeightCount; ++tile) {
			const u_int tileX = (tile % tileWidthCount) * tileSize;
			const u_int tileY = (tile / tileWidthCount) * tileSize;
			for (u_int m = 0; m < tileSize * tileSize; ++m) {
				const u_int x = tileX + DecodeMorton2X(m);
				const u_int y = tileY + DecodeMorton2Y(m);
				if ((x < regionWidth) && (y < regionHeight) &&
						!converged[(subRegion[0] + x) + (subRegion[2] + y) * width])
					list->push_back(x + y * regionWidth);
			}
		}
		todoPixels.store(std::move(list));
	}

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
