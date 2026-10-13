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

#include <limits>
#include <functional>

#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>

#include <boost/serialization/shared_ptr.hpp>

#include "slg/film/film.h"
#include "slg/film/noiseestimation/filmnoiseestimation.h"

using namespace std;
using namespace luxrays;
using namespace slg;

//------------------------------------------------------------------------------
// FilmNoiseEstimation
//------------------------------------------------------------------------------

BOOST_CLASS_EXPORT_IMPLEMENT(slg::FilmNoiseEstimation)

FilmNoiseEstimation::FilmNoiseEstimation(
	FilmConstRef flm, const u_int warmupVal,
	const u_int testStepVal, const u_int filtScale, const u_int index
) :
	warmup(warmupVal),
	testStep(testStepVal),
	filterScale(filtScale),
	imagePipelineIndex(index),
	film(&flm),
	referenceImage(NULL)
{
	Reset();
}

FilmNoiseEstimation::FilmNoiseEstimation() {
	referenceImage = NULL;
}

FilmNoiseEstimation::~FilmNoiseEstimation() {
	delete referenceImage;
}

void FilmNoiseEstimation::Reset() {
	todoPixelsCount = GetFilm().GetWidth() * GetFilm().GetHeight();
	maxDiff = numeric_limits<float>::infinity();

	delete referenceImage;
	referenceImage = new GenericFrameBuffer<3, 0, float>(GetFilm().GetWidth(), GetFilm().GetHeight());

	// Allocate new framebuffers according to film size
	u_int pixelsCount = GetFilm().GetWidth() * GetFilm().GetHeight();
	const bool hasNoiseChannel = GetFilm().HasChannel(Film::NOISE);

	if (hasNoiseChannel) {
		// Resize vectors according to film size. Start at zero
		errorVector.resize(pixelsCount, 0);

		// The film could have not been yet initialized
		if (GetFilm().channel_NOISE)
			GetFilm().channel_NOISE->Clear(numeric_limits<float>::infinity());
	}

	lastSamplesCount = 0.0;
	firstTest = true;
}

bool FilmNoiseEstimation::IsTestUpdateRequired() const {
	if (!GetFilm().HasChannel(Film::NOISE))
		return false;

	const u_int pixelsCount = GetFilm().GetWidth() * GetFilm().GetHeight();

	// Run the test only after a initial warmup
	if (GetFilm().GetTotalSampleCount() / pixelsCount <= warmup)
		return false;

	// Do not run the test if we don't have at least testStep new samples per pixel
	if (GetFilm().GetTotalSampleCount() - lastSamplesCount <= pixelsCount * static_cast<double>(testStep))
		return false;

	return true;
}

void FilmNoiseEstimation::Test() {
	if (!IsTestUpdateRequired())
		return;
	
	lastSamplesCount = GetFilm().GetTotalSampleCount();

	const u_int index = (imagePipelineIndex <= (GetFilm().GetImagePipelineCount() - 1)) ? imagePipelineIndex : 0;

	if (firstTest) {
		SLG_LOG("Noise estimation: first pass");

		// Copy the current image
		referenceImage->Copy(GetFilm().channel_IMAGEPIPELINEs[index]);
		firstTest = false;
	} else {
		// Check if all pixels has been sampled

		const u_int width = GetFilm().GetWidth();
		const u_int height = GetFilm().GetHeight();
		const u_int pixelsCount = width * height;
		const bool hasPN = GetFilm().HasChannel(Film::RADIANCE_PER_PIXEL_NORMALIZED);
		const bool hasSN = GetFilm().HasChannel(Film::RADIANCE_PER_SCREEN_NORMALIZED);

		const bool hasEnoughSamples = tbb::parallel_reduce(
				tbb::blocked_range<u_int>(0, pixelsCount), true,
				[&](const tbb::blocked_range<u_int> &range, bool enough) {
					for (u_int i = range.begin(); enough && (i < range.end()); ++i)
						enough = GetFilm().HasThresholdSamples(hasPN, hasSN, i, testStep);
					return enough;
				},
				[](const bool a, const bool b) { return a && b; });

		if (hasEnoughSamples) {
			const float *ref = referenceImage->GetPixels();
			const float *img = GetFilm().channel_IMAGEPIPELINEs[index]->GetPixels();

			vector<float> pixelErrorVector(pixelsCount, 0);

			// Calculate difference per pixel between images 
			tbb::parallel_for(tbb::blocked_range<u_int>(0, pixelsCount),
					[&](const tbb::blocked_range<u_int> &range) {
				for (u_int i = range.begin(); i < range.end(); ++i) {
					const float *refPixel = &ref[i * 3];
					const float *imgPixel = &img[i * 3];
					const float imgR = imgPixel[0];
					const float imgG = imgPixel[1];
					const float imgB = imgPixel[2];

					const float dr = fabsf(imgR - refPixel[0]);
					const float dg = fabsf(imgG - refPixel[1]);
					const float db = fabsf(imgB - refPixel[2]);

					const float imgSum = imgR + imgG + imgB;
					pixelErrorVector[i] = (imgSum != 0.f) ?
						((dr + dg + db) / sqrt(imgSum)) : 0.f;
				}
			});

			if (filterScale > 0) {
				// Filter the noise with the average of a box window of
				// [-filterScale, filterScale) pixels around each pixel (smaller
				// at the borders). The box is separable: a horizontal pass and
				// a vertical pass on its sums, with the same bounds and divisor
				// as the 2D window.

				const int fs = static_cast<int>(filterScale);
				const int w = static_cast<int>(width);
				const int h = static_cast<int>(height);
				vector<float> rowSums(pixelsCount);
				tbb::parallel_for(tbb::blocked_range<int>(0, h),
						[&](const tbb::blocked_range<int> &range) {
					for (int i = range.begin(); i < range.end(); ++i) {
						const float *row = &pixelErrorVector[i * w];
						float *out = &rowSums[i * w];
						for (int j = 0; j < w; ++j) {
							const int minWidth = Max(0, j - fs);
							const int maxWidth = Min(w, j + fs);
							float acc = 0.f;
							for (int c = minWidth; c < maxWidth; ++c)
								acc += row[c];
							out[j] = acc;
						}
					}
				});
				tbb::parallel_for(tbb::blocked_range<int>(0, h),
						[&](const tbb::blocked_range<int> &range) {
					for (int i = range.begin(); i < range.end(); ++i) {
						const int minHeight = Max(0, i - fs);
						const int maxHeight = Min(h, i + fs);
						for (int j = 0; j < w; ++j) {
							const int minWidth = Max(0, j - fs);
							const int maxWidth = Min(w, j + fs);
							float acc = 0.f;
							for (int r = minHeight; r < maxHeight; ++r)
								acc += rowSums[r * w + j];
							const u_int windowSize = (maxHeight - minHeight) * (maxWidth - minWidth);
							errorVector[i * w + j] = acc / windowSize;
						}
					}
				});

				// The error mean and standard deviation after the filtering
				// (invalid values are skipped)
				const double errorSum = tbb::parallel_reduce(
						tbb::blocked_range<u_int>(0, pixelsCount), 0.0,
						[&](const tbb::blocked_range<u_int> &range, double acc) {
							for (u_int j = range.begin(); j < range.end(); ++j) {
								const float pixelVal = errorVector[j];
								if (!isnan(pixelVal) && !isinf(pixelVal))
									acc += pixelVal;
							}
							return acc;
						}, std::plus<double>());
				const float errorMean = static_cast<float>(errorSum / pixelsCount);

				const double errorSqrSum = tbb::parallel_reduce(
						tbb::blocked_range<u_int>(0, pixelsCount), 0.0,
						[&](const tbb::blocked_range<u_int> &range, double acc) {
							for (u_int j = range.begin(); j < range.end(); ++j) {
								const float pixelVal = errorVector[j];
								if (!isnan(pixelVal) && !isinf(pixelVal)) {
									const double delta = pixelVal - errorMean;
									acc += delta * delta;
								}
							}
							return acc;
						}, std::plus<double>());
				const float errorStd = static_cast<float>(sqrt(errorSqrSum / pixelsCount));
				// With an unchanged image (no noise left) the deviation is
				// 0: the scores below are 0 instead of NaN
				const float errorStdSafe = (errorStd > 0.f) ? errorStd : 1.f;

				// Remove outliers: the standard score, clamped at 6 standard
				// deviations from the mean, and its range
				struct ScoreRange { float min, max; };
				const ScoreRange scoreRange = tbb::parallel_reduce(
						tbb::blocked_range<u_int>(0, pixelsCount),
						ScoreRange{numeric_limits<float>::infinity(), -numeric_limits<float>::infinity()},
						[&](const tbb::blocked_range<u_int> &range, ScoreRange sr) {
							for (u_int j = range.begin(); j < range.end(); ++j) {
								const float score = Clamp((errorVector[j] - errorMean) / errorStdSafe, -6.f, 6.f);
								errorVector[j] = score;

								sr.max = Max(score, sr.max);
								sr.min = Min(score, sr.min);
							}
							return sr;
						},
						[](const ScoreRange &a, const ScoreRange &b) {
							return ScoreRange{Min(a.min, b.min), Max(a.max, b.max)};
						});

				// Normalize error values
				const float scoreRangeSize = scoreRange.max - scoreRange.min;
				tbb::parallel_for(tbb::blocked_range<u_int>(0, pixelsCount),
						[&](const tbb::blocked_range<u_int> &range) {
					for (u_int i = range.begin(); i < range.end(); ++i) {
						// Update NOISE channel
						const float noise = (scoreRangeSize > 0.f) ?
							((errorVector[i] - scoreRange.min) / scoreRangeSize) : 0.f;
						*(GetFilm().channel_NOISE->GetPixel(i)) = noise;
					}
				});

				SLG_LOG("Noise estimation: Error mean = " << errorMean);
			}
		}

		// Copy the current image
		referenceImage->Copy(GetFilm().channel_IMAGEPIPELINEs[index]);
	}
}


template<class Archive> void FilmNoiseEstimation::serialize(Archive &ar, const u_int version) {
	ar & warmup;
	ar & testStep;
	ar & film;
	ar & referenceImage;
	ar & lastSamplesCount;
	ar & firstTest;
}

namespace slg {
// Explicit instantiations for portable archives
template void FilmNoiseEstimation::serialize(LuxOutputArchive &ar, const u_int version);
template void FilmNoiseEstimation::serialize(LuxInputArchive &ar, const u_int version);
template void FilmNoiseEstimation::serialize(LuxOutputArchiveText &ar, const u_int version);
template void FilmNoiseEstimation::serialize(LuxInputArchiveText &ar, const u_int version);
}// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
