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
#include <stdexcept>

#if defined(__AVX__)
#include <immintrin.h>
#endif

#include "luxrays/kernels/kernels.h"
#include "slg/kernels/kernels.h"
#include "slg/film/film.h"
#include "slg/film/imagepipeline/plugins/bloom.h"

using namespace std;
using namespace luxrays;
using namespace slg;

//------------------------------------------------------------------------------
// Bloom filter plugin
//------------------------------------------------------------------------------

BOOST_CLASS_EXPORT_IMPLEMENT(slg::BloomFilterPlugin)

BloomFilterPlugin::BloomFilterPlugin(const float r, const float w) :
		radius(r), weight(w), bloomBuffer(nullptr), bloomBufferTmp(nullptr),
		bloomBufferSize(0), bloomFilterSize(0) {
	hardwareDevice = nullptr;
	hwBloomBuffer = nullptr;
	hwBloomBufferTmp = nullptr;
	hwBloomFilter = nullptr;

	bloomFilterXKernel = nullptr;
	bloomFilterYKernel = nullptr;
	bloomFilterMergeKernel = nullptr;
}

BloomFilterPlugin::BloomFilterPlugin() {
	bloomBuffer = nullptr;
	bloomBufferTmp = nullptr;

	hardwareDevice = nullptr;
	hwBloomBuffer = nullptr;
	hwBloomBufferTmp = nullptr;
	hwBloomFilter = nullptr;

	bloomFilterXKernel = nullptr;
	bloomFilterYKernel = nullptr;
	bloomFilterMergeKernel = nullptr;
}

BloomFilterPlugin::~BloomFilterPlugin() {
	delete[] bloomBuffer;
	delete[] bloomBufferTmp;

	if (hardwareDevice) {
		hardwareDevice->FreeBuffer(&hwBloomBuffer);
		hardwareDevice->FreeBuffer(&hwBloomBufferTmp);
		hardwareDevice->FreeBuffer(&hwBloomFilter);
	}
}

ImagePipelinePlugin *BloomFilterPlugin::Copy() const {
	return new BloomFilterPlugin(radius, weight);
}

void BloomFilterPlugin::InitFilterTable(const Film &film) {
	const u_int width = film.GetWidth();
	const u_int height = film.GetHeight();

	// Compute image-space extent of bloom effect
	const u_int bloomSupport = Float2UInt(radius * Max(width, height));
	bloomWidth = bloomSupport / 2;

	bloomFilterSize = 2 * bloomWidth * bloomWidth + 1;
	bloomFilter.clear();
	bloomFilter.resize(bloomFilterSize);
	for (u_int i = 0; i < bloomFilterSize; ++i)
		bloomFilter[i] = 0.f;

	for (u_int i = 0; i < bloomWidth * bloomWidth; ++i) {
		const float z0 = 3.8317f;
		const float dist = z0 * sqrtf(i) / bloomWidth;
		if (dist == 0.f)
			bloomFilter[i] = 1.f;
		else if (dist >= z0)
			bloomFilter[i] = 0.f;
		else {
			// Airy function
			//const float b = boost::math::cyl_bessel_j(1, dist);
			//bloomFilter[i] = powf(2*b/dist, 2.f);

			// Gaussian approximation
			// best-fit sigma^2 for above airy function, based on RMSE
			// depends on choice of zero
			const float sigma2 = 1.698022698724f; 
			bloomFilter[i] = expf(-dist * dist / sigma2);
		}
	}
}

//------------------------------------------------------------------------------
// CPU version
//------------------------------------------------------------------------------

// The CPU version has a fast path for the parts of the image where all the
// pixels have samples (usually all of it): the pixels are processed in blocks
// whose sums stay in registers through all the taps (WeightedSum()). Each
// pixel still accumulates its taps in the same order as the per pixel path,
// one multiply and one add per tap and component, so the result is the same
// (the Spectrum operations are per component, so the rows are processed as
// float arrays). The taps at distance bloomWidth are skipped: their weight is
// 0 (see InitFilterTable()).

static_assert(sizeof(Spectrum) == 3 * sizeof(float), "Spectrum must be 3 packed floats");

namespace {

// The floats summed by one WeightedSum() call: 16 pixels
const u_int SUM_BLOCK_FLOATS = 48;

// dst[i] = (sum of weights[t] * src[offsets[t] + i] over the taps t, in their
// order) * scale for the SUM_BLOCK_FLOATS floats of a block
inline void WeightedSum(float *dst, const float *src, const ptrdiff_t *offsets,
		const float *weights, const u_int tapCount, const float scale) {
#if defined(__AVX__)
	__m256 sum0 = _mm256_setzero_ps();
	__m256 sum1 = _mm256_setzero_ps();
	__m256 sum2 = _mm256_setzero_ps();
	__m256 sum3 = _mm256_setzero_ps();
	__m256 sum4 = _mm256_setzero_ps();
	__m256 sum5 = _mm256_setzero_ps();
	for (u_int t = 0; t < tapCount; ++t) {
		const float *s = src + offsets[t];
		const __m256 wt = _mm256_set1_ps(weights[t]);
		sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(wt, _mm256_loadu_ps(s)));
		sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(wt, _mm256_loadu_ps(s + 8)));
		sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(wt, _mm256_loadu_ps(s + 16)));
		sum3 = _mm256_add_ps(sum3, _mm256_mul_ps(wt, _mm256_loadu_ps(s + 24)));
		sum4 = _mm256_add_ps(sum4, _mm256_mul_ps(wt, _mm256_loadu_ps(s + 32)));
		sum5 = _mm256_add_ps(sum5, _mm256_mul_ps(wt, _mm256_loadu_ps(s + 40)));
	}
	const __m256 sc = _mm256_set1_ps(scale);
	_mm256_storeu_ps(dst, _mm256_mul_ps(sum0, sc));
	_mm256_storeu_ps(dst + 8, _mm256_mul_ps(sum1, sc));
	_mm256_storeu_ps(dst + 16, _mm256_mul_ps(sum2, sc));
	_mm256_storeu_ps(dst + 24, _mm256_mul_ps(sum3, sc));
	_mm256_storeu_ps(dst + 32, _mm256_mul_ps(sum4, sc));
	_mm256_storeu_ps(dst + 40, _mm256_mul_ps(sum5, sc));
#else
	float sum[SUM_BLOCK_FLOATS] = {};
	for (u_int t = 0; t < tapCount; ++t) {
		const float *s = src + offsets[t];
		const float wt = weights[t];
		for (u_int i = 0; i < SUM_BLOCK_FLOATS; ++i)
			sum[i] += wt * s[i];
	}
	for (u_int i = 0; i < SUM_BLOCK_FLOATS; ++i)
		dst[i] = sum[i] * scale;
#endif
}

// The same for one pixel (the offsets are in floats)
inline void WeightedSum(Spectrum &dst, const Spectrum *src, const ptrdiff_t *offsets,
		const float *weights, const u_int tapCount, const float scale) {
	Spectrum sum;
	for (u_int t = 0; t < tapCount; ++t)
		sum += weights[t] * src[offsets[t] / 3];
	dst = sum * scale;
}

}

void BloomFilterPlugin::BloomFilterX(const Film &film, const Spectrum *pixels,
		const vector<char> &hasSamples, const vector<char> &rowHasSamples) {
	const u_int width = film.GetWidth();
	const u_int height = film.GetHeight();
	const int fastWidth = (int)bloomWidth - 1;

	// The taps of the pixels with all of them inside the row
	vector<ptrdiff_t> offsets;
	vector<float> weights;
	float sumWt = 0.f;
	for (int d = -fastWidth; d <= fastWidth; ++d) {
		const float wt = bloomFilter[d * d];
		if (wt == 0.f)
			continue;

		offsets.push_back(3 * d);
		weights.push_back(wt);
		sumWt += wt;
	}
	const float scale = (sumWt > 0.f) ? (1.f / sumWt) : 1.f;
	const u_int xInnerBegin = Min<u_int>(Max(fastWidth, 0), width);
	const u_int xInnerEnd = Max<int>(xInnerBegin, (int)width - Max(fastWidth, 0));
	const u_int blockPixels = SUM_BLOCK_FLOATS / 3;
	const u_int xBlocksEnd = xInnerBegin + (xInnerEnd - xInnerBegin) / blockPixels * blockPixels;

	// The pixels near the borders and the ones left over by the blocks: the
	// taps inside the row are a range of the ones above
	struct PixelTaps {
		u_int x, tapBegin, tapCount;
		float scale;
	};
	vector<PixelTaps> pixelTaps;
	for (u_int x = 0; x < width; ++x) {
		if (x == xInnerBegin)
			x = xBlocksEnd;
		if (x == width)
			break;

		PixelTaps taps = { x, 0, 0, 1.f };
		float pixelSumWt = 0.f;
		for (u_int t = 0; t < offsets.size(); ++t) {
			const int bx = (int)x + (int)offsets[t] / 3;
			if ((bx < 0) || (bx >= (int)width))
				continue;

			if (taps.tapCount == 0)
				taps.tapBegin = t;
			++taps.tapCount;
			pixelSumWt += weights[t];
		}
		if (pixelSumWt > 0.f)
			taps.scale = 1.f / pixelSumWt;
		pixelTaps.push_back(taps);
	}

	// Apply bloom filter to image pixels
	#pragma omp parallel for
	for (
		// Visual C++ 2013 supports only OpenMP 2.5
#if _OPENMP >= 200805
		unsigned
#endif
		int y = 0; y < height; ++y) {
		if (rowHasSamples[y]) {
			// Fast path: all the pixels of the row have samples
			const Spectrum *src = &pixels[y * width];
			Spectrum *dst = &bloomBufferTmp[y * width];

			for (u_int x = xInnerBegin; x < xBlocksEnd; x += blockPixels)
				WeightedSum(&dst[x].c[0], &src[x].c[0], offsets.data(), weights.data(),
						(u_int)weights.size(), scale);

			for (const PixelTaps &taps : pixelTaps)
				WeightedSum(dst[taps.x], &src[taps.x], &offsets[taps.tapBegin],
						&weights[taps.tapBegin], taps.tapCount, taps.scale);
			continue;
		}

		for (u_int x = 0; x < width; ++x) {
			if (hasSamples[x + y * width]) {
				// Compute bloom for pixel (x, y)
				// Compute extent of pixels contributing bloom
				const u_int x0 = Max<u_int>(x, bloomWidth) - bloomWidth;
				const u_int x1 = Min<u_int>(x + bloomWidth, width - 1);

				float sumWt = 0.f;
				Spectrum &pixel(bloomBufferTmp[x + y * width]);
				pixel = Spectrum();
				for (u_int bx = x0; bx <= x1; ++bx) {
					const u_int bloomOffset = bx + y * width;
					if (hasSamples[bloomOffset]) {
						// Accumulate bloom from pixel (bx, y)
						const u_int dist2 = (x - bx) * (x - bx);
						const float wt = bloomFilter[dist2];
						if (wt == 0.f)
							continue;

						sumWt += wt;
						pixel += wt * pixels[bloomOffset];
					}
				}
				if (sumWt > 0.f)
					pixel /= sumWt;
			}
		}
	}
}

void BloomFilterPlugin::BloomFilterY(const Film &film, const vector<char> &hasSamples,
		const vector<char> &rowHasSamples) {
	const u_int width = film.GetWidth();
	const u_int height = film.GetHeight();

	// The number of rows with some pixels without samples, before each row
	vector<u_int> rowsWithoutSamples(height + 1, 0);
	for (u_int y = 0; y < height; ++y)
		rowsWithoutSamples[y + 1] = rowsWithoutSamples[y] + (rowHasSamples[y] ? 0 : 1);

	// The image is filtered in blocks of columns and rows: the part of the
	// 2 * bloomWidth rows read by an output row that is in the block stays in
	// the cache for the next output rows of the block (the whole rows are
	// megabytes, so each output row read them from memory again). Measured on
	// a Ryzen 5 3600 at 1600x1000: 5.3 ms instead of 10 ms; narrower strips
	// are slower (16 pixels: 9.3 ms)
	const u_int blockPixels = SUM_BLOCK_FLOATS / 3;
	const u_int stripWidth = 8 * blockPixels;
	const u_int blockHeight = 128;
	const int stripCount = (width + stripWidth - 1) / stripWidth;
	const int taskCount = stripCount * ((height + blockHeight - 1) / blockHeight);

	// Apply bloom filter to image pixels
	#pragma omp parallel for schedule(dynamic)
	for (int task = 0; task < taskCount; ++task) {
		const u_int xBegin = (task % stripCount) * stripWidth;
		const u_int xEnd = Min(xBegin + stripWidth, width);
		const u_int yBegin = (task / stripCount) * blockHeight;
		const u_int yEnd = Min(yBegin + blockHeight, height);

		vector<ptrdiff_t> offsets;
		vector<float> weights;
		for (u_int y = yBegin; y < yEnd; ++y) {
			// Compute extent of pixels contributing bloom
			const u_int y0 = Max<u_int>(y, bloomWidth) - bloomWidth;
			const u_int y1 = Min<u_int>(y + bloomWidth, height - 1);

			if (rowsWithoutSamples[y1 + 1] == rowsWithoutSamples[y0]) {
				// Fast path: all the pixels of the rows y0 to y1 have samples
				offsets.clear();
				weights.clear();
				float sumWt = 0.f;
				for (u_int by = y0; by <= y1; ++by) {
					const u_int dist2 = (y - by) * (y - by);
					const float wt = bloomFilter[dist2];
					if (wt == 0.f)
						continue;

					offsets.push_back(3 * ((ptrdiff_t)by - (ptrdiff_t)y) * (ptrdiff_t)width);
					weights.push_back(wt);
					sumWt += wt;
				}
				const float scale = (sumWt > 0.f) ? (1.f / sumWt) : 1.f;

				const Spectrum *src = &bloomBufferTmp[y * width];
				Spectrum *dst = &bloomBuffer[y * width];
				u_int x = xBegin;
				for (; x + blockPixels <= xEnd; x += blockPixels)
					WeightedSum(&dst[x].c[0], &src[x].c[0], offsets.data(), weights.data(),
							(u_int)weights.size(), scale);
				for (; x < xEnd; ++x)
					WeightedSum(dst[x], &src[x], offsets.data(), weights.data(),
							(u_int)weights.size(), scale);
				continue;
			}

			for (u_int x = xBegin; x < xEnd; ++x) {
				if (hasSamples[x + y * width]) {
					// Compute bloom for pixel (x, y)
					float sumWt = 0.f;
					Spectrum &pixel(bloomBuffer[x + y * width]);
					pixel = Spectrum();
					for (u_int by = y0; by <= y1; ++by) {
						const u_int bloomOffset = x + by * width;
						if (hasSamples[bloomOffset]) {
							// Accumulate bloom from pixel (x, by)
							const u_int dist2 = (y - by) * (y - by);
							const float wt = bloomFilter[dist2];
							if (wt == 0.f)
								continue;

							sumWt += wt;
							pixel += wt * bloomBufferTmp[bloomOffset];
						}
					}

					if (sumWt > 0.f)
						pixel /= sumWt;
				}
			}
		}
	}
}

void BloomFilterPlugin::BloomFilter(const Film &film, const Spectrum *pixels,
		const vector<char> &hasSamples) {
	const u_int width = film.GetWidth();
	const u_int height = film.GetHeight();

	// Rows where all the pixels have samples
	vector<char> rowHasSamples(height);
	#pragma omp parallel for
	for (
		// Visual C++ 2013 supports only OpenMP 2.5
#if _OPENMP >= 200805
		unsigned
#endif
		int y = 0; y < height; ++y)
		rowHasSamples[y] = all_of(hasSamples.begin() + y * width,
				hasSamples.begin() + (y + 1) * width, [](const char v) { return v != 0; });

	BloomFilterX(film, pixels, hasSamples, rowHasSamples);
	BloomFilterY(film, hasSamples, rowHasSamples);
}

void BloomFilterPlugin::Apply(Film &film, const u_int index) {
	//const double t1 = WallClockTime();

	Spectrum *pixels = (Spectrum *)film.channel_IMAGEPIPELINEs[index]->GetPixels();
	const u_int width = film.GetWidth();
	const u_int height = film.GetHeight();

	const bool hasPN = film.HasChannel(Film::RADIANCE_PER_PIXEL_NORMALIZED);
	const bool hasSN = film.HasChannel(Film::RADIANCE_PER_SCREEN_NORMALIZED);

	// Allocate the temporary buffer if required
	if ((!bloomBuffer) || (width * height != bloomBufferSize)) {
		delete[] bloomBuffer;
		delete[] bloomBufferTmp;

		bloomBufferSize = width * height;
		bloomBuffer = new Spectrum[bloomBufferSize];
		bloomBufferTmp = new Spectrum[bloomBufferSize];

		InitFilterTable(film);
	}

	// Film::HasSamples() is evaluated once for each pixel instead of once for
	// each filter tap
	vector<char> hasSamples(bloomBufferSize);
	#pragma omp parallel for
	for (
		// Visual C++ 2013 supports only OpenMP 2.5
#if _OPENMP >= 200805
		unsigned
#endif
		int i = 0; i < bloomBufferSize; ++i)
		hasSamples[i] = film.HasSamples(hasPN, hasSN, i);

	// Apply separable filter
	BloomFilter(film, pixels, hasSamples);

	#pragma omp parallel for
	for (
		// Visual C++ 2013 supports only OpenMP 2.5
#if _OPENMP >= 200805
		unsigned
#endif
		int i = 0; i < bloomBufferSize; ++i) {
		if (hasSamples[i])
			pixels[i] = Lerp(weight, pixels[i], bloomBuffer[i]);
	}

	//const double t2 = WallClockTime();
	//SLG_LOG("Bloom time: " << int((t2 - t1) * 1000.0) << "ms");
}

//------------------------------------------------------------------------------
// HardwareDevice version
//------------------------------------------------------------------------------

void BloomFilterPlugin::AddHWChannelsUsed(unordered_set<Film::FilmChannelType, hash<int> > &hwChannelsUsed) const {
	hwChannelsUsed.insert(Film::IMAGEPIPELINE);
}

void BloomFilterPlugin::ApplyHW(Film &film, const u_int index) {
	const u_int width = film.GetWidth();
	const u_int height = film.GetHeight();

	if ((bloomFilter.empty()) || (width * height != bloomBufferSize)) {
		bloomBufferSize = width * height;
		InitFilterTable(film);
	}

	if (!bloomFilterXKernel) {
		film.ctx->SetVerbose(true);

		hardwareDevice = film.hardwareDevice;

		// Allocate OpenCL buffers
		hardwareDevice->AllocBufferRW(&hwBloomBuffer, nullptr, bloomBufferSize * sizeof(Spectrum), "Bloom buffer");
		hardwareDevice->AllocBufferRW(&hwBloomBufferTmp, nullptr, bloomBufferSize * sizeof(Spectrum), "Bloom temporary buffer");
		hardwareDevice->AllocBufferRO(&hwBloomFilter, bloomFilter.data(), bloomFilterSize * sizeof(float), "Bloom filter table");

		// Compile sources
		const double tStart = WallClockTime();

		vector<string> opts;
		opts.push_back("-D LUXRAYS_OPENCL_KERNEL");
		opts.push_back("-D SLG_OPENCL_KERNEL");

		auto program = hardwareDevice->CompileProgram(
				opts,
				luxrays::ocl::KernelSource_color_types +
				slg::ocl::KernelSource_plugin_bloom_funcs,
				"BloomFilterPlugin"
		);

		//----------------------------------------------------------------------
		// BloomFilterPlugin_FilterX kernel
		//----------------------------------------------------------------------

		SLG_LOG("[BloomFilterPlugin] Compiling BloomFilterPlugin_FilterX Kernel");
		bloomFilterXKernel = hardwareDevice->GetKernel(*program, "BloomFilterPlugin_FilterX");

		// Set kernel arguments
		u_int argIndex = 0;
		hardwareDevice->SetKernelArg(bloomFilterXKernel, argIndex++, width);
		hardwareDevice->SetKernelArg(bloomFilterXKernel, argIndex++, height);
		hardwareDevice->SetKernelArg(bloomFilterXKernel, argIndex++, film.hw_IMAGEPIPELINE);
		hardwareDevice->SetKernelArg(bloomFilterXKernel, argIndex++, hwBloomBuffer);
		hardwareDevice->SetKernelArg(bloomFilterXKernel, argIndex++, hwBloomBufferTmp);
		hardwareDevice->SetKernelArg(bloomFilterXKernel, argIndex++, hwBloomFilter);
		hardwareDevice->SetKernelArg(bloomFilterXKernel, argIndex++, bloomWidth);

		//----------------------------------------------------------------------
		// BloomFilterPlugin_FilterY kernel
		//----------------------------------------------------------------------

		SLG_LOG("[BloomFilterPlugin] Compiling BloomFilterPlugin_FilterY Kernel");
		bloomFilterYKernel = hardwareDevice->GetKernel(*program, "BloomFilterPlugin_FilterY");

		// Set kernel arguments
		argIndex = 0;
		hardwareDevice->SetKernelArg(bloomFilterYKernel, argIndex++, width);
		hardwareDevice->SetKernelArg(bloomFilterYKernel, argIndex++, height);
		hardwareDevice->SetKernelArg(bloomFilterYKernel, argIndex++, film.hw_IMAGEPIPELINE);
		hardwareDevice->SetKernelArg(bloomFilterYKernel, argIndex++, hwBloomBuffer);
		hardwareDevice->SetKernelArg(bloomFilterYKernel, argIndex++, hwBloomBufferTmp);
		hardwareDevice->SetKernelArg(bloomFilterYKernel, argIndex++, hwBloomFilter);
		hardwareDevice->SetKernelArg(bloomFilterYKernel, argIndex++, bloomWidth);

		//----------------------------------------------------------------------
		// BloomFilterPlugin_Merge kernel
		//----------------------------------------------------------------------

		SLG_LOG("[BloomFilterPlugin] Compiling BloomFilterPlugin_Merge Kernel");
		bloomFilterMergeKernel = hardwareDevice->GetKernel(*program, "BloomFilterPlugin_Merge");

		// Set kernel arguments
		argIndex = 0;
		hardwareDevice->SetKernelArg(bloomFilterMergeKernel, argIndex++, width);
		hardwareDevice->SetKernelArg(bloomFilterMergeKernel, argIndex++, height);
		hardwareDevice->SetKernelArg(bloomFilterMergeKernel, argIndex++, film.hw_IMAGEPIPELINE);
		hardwareDevice->SetKernelArg(bloomFilterMergeKernel, argIndex++, hwBloomBuffer);
		hardwareDevice->SetKernelArg(bloomFilterMergeKernel, argIndex++, weight);

		//----------------------------------------------------------------------

		const double tEnd = WallClockTime();
		SLG_LOG("[BloomFilterPlugin] Kernels compilation time: " << int((tEnd - tStart) * 1000.0) << "ms");

		film.ctx->SetVerbose(false);
	}

	hardwareDevice->EnqueueKernel(bloomFilterXKernel, HardwareDeviceRange(RoundUp(width * height, 256u)),
			HardwareDeviceRange(256));
	hardwareDevice->EnqueueKernel(bloomFilterYKernel, HardwareDeviceRange(RoundUp(width * height, 256u)),
			HardwareDeviceRange(256));
	hardwareDevice->EnqueueKernel(bloomFilterMergeKernel, HardwareDeviceRange(RoundUp(width * height, 256u)),
			HardwareDeviceRange(256));
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
