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
// pixels have samples (usually all of it): it accumulates whole rows, one tap
// offset at a time, so the pixels are independent of each other and the memory
// is accessed sequentially. Each pixel still accumulates its taps in the same
// order as the per pixel path, so the result is the same (the Spectrum
// operations are per component, so the rows are processed as float arrays,
// which the compiler vectorizes). The taps at distance bloomWidth are skipped:
// their weight is 0 (see InitFilterTable()).

static_assert(sizeof(Spectrum) == 3 * sizeof(float), "Spectrum must be 3 packed floats");

void BloomFilterPlugin::BloomFilterX(const Film &film, const Spectrum *pixels,
		const vector<char> &hasSamples, const vector<char> &rowHasSamples) {
	const u_int width = film.GetWidth();
	const u_int height = film.GetHeight();
	const int fastWidth = (int)bloomWidth - 1;

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
			vector<float> sumWt(width, 0.f);
			for (u_int x = 0; x < width; ++x)
				dst[x] = Spectrum();

			for (int d = -fastWidth; d <= fastWidth; ++d) {
				const float wt = bloomFilter[d * d];
				if (wt == 0.f)
					continue;

				const int xBegin = Max(0, -d);
				const int xEnd = Min((int)width, (int)width - d);
				for (int x = xBegin; x < xEnd; ++x)
					sumWt[x] += wt;

				float *dstF = &dst[xBegin].c[0];
				const float *srcF = &src[xBegin + d].c[0];
				const int count = 3 * (xEnd - xBegin);
				for (int i = 0; i < count; ++i)
					dstF[i] += wt * srcF[i];
			}

			for (u_int x = 0; x < width; ++x) {
				if (sumWt[x] > 0.f)
					dst[x] /= sumWt[x];
			}
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

	// Apply bloom filter to image pixels
	#pragma omp parallel for
	for (
		// Visual C++ 2013 supports only OpenMP 2.5
#if _OPENMP >= 200805
		unsigned
#endif
		int y = 0; y < height; ++y) {
		// Compute extent of pixels contributing bloom
		const u_int y0 = Max<u_int>(y, bloomWidth) - bloomWidth;
		const u_int y1 = Min<u_int>(y + bloomWidth, height - 1);

		if (rowsWithoutSamples[y1 + 1] == rowsWithoutSamples[y0]) {
			// Fast path: all the pixels of the rows y0 to y1 have samples
			Spectrum *dst = &bloomBuffer[y * width];
			for (u_int x = 0; x < width; ++x)
				dst[x] = Spectrum();

			float sumWt = 0.f;
			for (u_int by = y0; by <= y1; ++by) {
				const u_int dist2 = (y - by) * (y - by);
				const float wt = bloomFilter[dist2];
				if (wt == 0.f)
					continue;

				sumWt += wt;
				float *dstF = &dst[0].c[0];
				const float *srcF = &bloomBufferTmp[by * width].c[0];
				const u_int count = 3 * width;
				for (u_int i = 0; i < count; ++i)
					dstF[i] += wt * srcF[i];
			}

			if (sumWt > 0.f) {
				for (u_int x = 0; x < width; ++x)
					dst[x] /= sumWt;
			}
			continue;
		}

		for (u_int x = 0; x < width; ++x) {
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

void BloomFilterPlugin::BloomFilter(const Film &film, const Spectrum *pixels,
		const vector<char> &hasSamples) {
	const u_int width = film.GetWidth();
	const u_int height = film.GetHeight();

	// Rows where all the pixels have samples
	vector<char> rowHasSamples(height);
	for (u_int y = 0; y < height; ++y)
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

	for (u_int i = 0; i < bloomBufferSize; ++i) {
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
