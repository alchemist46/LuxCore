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

#include <stdexcept>

#if defined(__AVX__)
#include <immintrin.h>
#endif

#include "luxrays/kernels/kernels.h"
#include "slg/kernels/kernels.h"
#include "slg/film/film.h"
#include "slg/film/imagepipeline/plugins/coloraberration.h"

using namespace std;
using namespace luxrays;
using namespace slg;

//------------------------------------------------------------------------------
// Color aberration filter plugin
//------------------------------------------------------------------------------

BOOST_CLASS_EXPORT_IMPLEMENT(slg::ColorAberrationPlugin)

ColorAberrationPlugin::ColorAberrationPlugin(const float ax, const float ay) :
		amountX(ax), amountY(ay),
		tmpBuffer(nullptr), tmpBufferSize(0) {
	hardwareDevice = nullptr;
	hwTmpBuffer = nullptr;

	applyKernel = nullptr;
	copyKernel = nullptr;
}

ColorAberrationPlugin::~ColorAberrationPlugin() {
	delete[] tmpBuffer;

	if (hardwareDevice)
		hardwareDevice->FreeBuffer(&hwTmpBuffer);
}

ImagePipelinePlugin *ColorAberrationPlugin::Copy() const {
	return new ColorAberrationPlugin(amountX, amountY);
}

//------------------------------------------------------------------------------
// CPU version
//------------------------------------------------------------------------------

// The pixels are processed 8 at a time: the coordinates, the pixel indices
// and the weights of their bilinear samples are computed with 8 wide vectors
// when AVX is available, with the same IEEE operations in the same order as
// the scalar code of the #else branches, so the result is the same (the
// scalar per pixel loop took 60 ms at 1600x1000 on one Ryzen 5 3600 thread,
// the vectors 28 ms)

namespace {

const u_int LANES = 8;

// The 4 pixels and weights of the bilinear samples of LANES pixels
struct BilinearSamples {
	u_int x1[LANES], x2[LANES], y1[LANES], y2[LANES];
	float w11[LANES], w21[LANES], w12[LANES], w22[LANES];

	Spectrum Get(const Spectrum *pixels, const u_int width, const u_int i) const {
		Spectrum c;
		c.AddWeighted(w11[i], pixels[y1[i] * width + x1[i]]);
		c.AddWeighted(w21[i], pixels[y1[i] * width + x2[i]]);
		c.AddWeighted(w12[i], pixels[y2[i] * width + x1[i]]);
		c.AddWeighted(w22[i], pixels[y2[i] * width + x2[i]]);
		return c;
	}
};

#if defined(__AVX__)
// Clamp() of luxrays/utils/utils.h
inline __m256 Clamp8(const __m256 val, const __m256 low, const __m256 high) {
	const __m256 r = _mm256_blendv_ps(high, val, _mm256_cmp_ps(val, high, _CMP_LT_OQ));
	return _mm256_blendv_ps(low, r, _mm256_cmp_ps(val, low, _CMP_GT_OQ));
}

// Clamp(Floor2UInt(v), 0u, max) and Clamp(that + 1, 0u, max), as floats (the
// pixel coordinates are integers below 2^24)
inline void Floor8(const __m256 v, const __m256 max, __m256 &v1, __m256 &v2) {
	const __m256 zero = _mm256_setzero_ps();
	v1 = _mm256_blendv_ps(zero, _mm256_floor_ps(v), _mm256_cmp_ps(v, zero, _CMP_GT_OQ));
	v1 = _mm256_blendv_ps(max, v1, _mm256_cmp_ps(v1, max, _CMP_LT_OQ));
	v2 = _mm256_add_ps(v1, _mm256_set1_ps(1.f));
	v2 = _mm256_blendv_ps(max, v2, _mm256_cmp_ps(v2, max, _CMP_LT_OQ));
}
#endif

// The bilinear samples of the image at (x[i], y[i])
inline void GetBilinearSamples(BilinearSamples &s, const float *x, const float *y,
		const u_int width, const u_int height) {
#if defined(__AVX__)
	const __m256 zero = _mm256_setzero_ps();
	const __m256 one = _mm256_set1_ps(1.f);
	const __m256 sx = _mm256_loadu_ps(x);
	const __m256 sy = _mm256_loadu_ps(y);

	__m256 x1, x2, y1, y2;
	Floor8(sx, _mm256_set1_ps((float)(width - 1)), x1, x2);
	Floor8(sy, _mm256_set1_ps((float)(height - 1)), y1, y2);
	const __m256 tx = Clamp8(_mm256_sub_ps(sx, x1), zero, one);
	const __m256 ty = Clamp8(_mm256_sub_ps(sy, y1), zero, one);
	const __m256 txInv = _mm256_sub_ps(one, tx);
	const __m256 tyInv = _mm256_sub_ps(one, ty);

	_mm256_storeu_si256((__m256i *)s.x1, _mm256_cvttps_epi32(x1));
	_mm256_storeu_si256((__m256i *)s.x2, _mm256_cvttps_epi32(x2));
	_mm256_storeu_si256((__m256i *)s.y1, _mm256_cvttps_epi32(y1));
	_mm256_storeu_si256((__m256i *)s.y2, _mm256_cvttps_epi32(y2));
	_mm256_storeu_ps(s.w11, _mm256_mul_ps(txInv, tyInv));
	_mm256_storeu_ps(s.w21, _mm256_mul_ps(tx, tyInv));
	_mm256_storeu_ps(s.w12, _mm256_mul_ps(txInv, ty));
	_mm256_storeu_ps(s.w22, _mm256_mul_ps(tx, ty));
#else
	for (u_int i = 0; i < LANES; ++i) {
		s.x1[i] = Clamp(Floor2UInt(x[i]), 0u, width - 1);
		s.y1[i] = Clamp(Floor2UInt(y[i]), 0u, height - 1);
		s.x2[i] = Clamp(s.x1[i] + 1, 0u, width - 1);
		s.y2[i] = Clamp(s.y1[i] + 1, 0u, height - 1);
		const float tx = Clamp(x[i] - s.x1[i], 0.f, 1.f);
		const float ty = Clamp(y[i] - s.y1[i], 0.f, 1.f);

		s.w11[i] = (1.f - tx) * (1.f - ty);
		s.w21[i] = tx * (1.f - ty);
		s.w12[i] = (1.f - tx) * ty;
		s.w22[i] = tx * ty;
	}
#endif
}

}

void ColorAberrationPlugin::Apply(Film &film, const u_int index) {
	Spectrum *pixels = (Spectrum *)film.channel_IMAGEPIPELINEs[index]->GetPixels();

	const u_int width = film.GetWidth();
	const u_int height = film.GetHeight();
	const float invWidth = 1.f / width;
	const float invHeight = 1.f / height;

	// Allocate the temporary buffer if required
	if ((!tmpBuffer) || (width * height != tmpBufferSize)) {
		delete[] tmpBuffer;

		tmpBufferSize = width * height;
		tmpBuffer = new Spectrum[tmpBufferSize];
	}

	const bool hasPN = film.HasChannel(Film::RADIANCE_PER_PIXEL_NORMALIZED);
	const bool hasSN = film.HasChannel(Film::RADIANCE_PER_SCREEN_NORMALIZED);

	const Spectrum redblue(1.f, 0.f, 1.f);
	const Spectrum green(0.f, 1.f, 0.f);

	#pragma omp parallel for
	for (
			// Visual C++ 2013 supports only OpenMP 2.5
#if _OPENMP >= 200805
			unsigned
#endif
			int y = 0; y < height; ++y) {
		const float ny = y * invHeight;
		const float yOffset = ny - .5f;

		for (u_int xBegin = 0; xBegin < width; xBegin += LANES) {
			// The coordinates of the samples (the lanes past the end of the
			// row are computed and not used)
			float rbX[LANES], rbY[LANES], gX[LANES], gY[LANES];
#if defined(__AVX__)
			const __m256 one = _mm256_set1_ps(1.f);
			const __m256 half = _mm256_set1_ps(.5f);
			const __m256 xf = _mm256_add_ps(_mm256_set1_ps((float)xBegin),
					_mm256_setr_ps(0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f));
			const __m256 xOffset = _mm256_sub_ps(_mm256_mul_ps(xf, _mm256_set1_ps(invWidth)), half);
			const __m256 yOffset8 = _mm256_set1_ps(yOffset);
			const __m256 tOffset = _mm256_sqrt_ps(_mm256_add_ps(_mm256_mul_ps(xOffset, xOffset),
					_mm256_mul_ps(yOffset8, yOffset8)));
			const __m256 tX = _mm256_mul_ps(tOffset, _mm256_set1_ps(amountX));
			const __m256 tY = _mm256_mul_ps(tOffset, _mm256_set1_ps(amountY));
			const __m256 w = _mm256_set1_ps((float)width);
			const __m256 h = _mm256_set1_ps((float)height);

			_mm256_storeu_ps(rbX, _mm256_mul_ps(_mm256_add_ps(half, _mm256_mul_ps(xOffset, _mm256_add_ps(one, tX))), w));
			_mm256_storeu_ps(rbY, _mm256_mul_ps(_mm256_add_ps(half, _mm256_mul_ps(yOffset8, _mm256_add_ps(one, tY))), h));
			_mm256_storeu_ps(gX, _mm256_mul_ps(_mm256_add_ps(half, _mm256_mul_ps(xOffset, _mm256_sub_ps(one, tX))), w));
			_mm256_storeu_ps(gY, _mm256_mul_ps(_mm256_add_ps(half, _mm256_mul_ps(yOffset8, _mm256_sub_ps(one, tY))), h));
#else
			for (u_int i = 0; i < LANES; ++i) {
				const float nx = (xBegin + i) * invWidth;
				const float xOffset = nx - .5f;
				const float tOffset = sqrtf(xOffset * xOffset + yOffset * yOffset);

				rbX[i] = (.5f + xOffset * (1.f + tOffset * amountX)) * width;
				rbY[i] = (.5f + yOffset * (1.f + tOffset * amountY)) * height;
				gX[i] =  (.5f + xOffset * (1.f - tOffset * amountX)) * width;
				gY[i] =  (.5f + yOffset * (1.f - tOffset * amountY)) * height;
			}
#endif

			BilinearSamples rbSamples, gSamples;
			GetBilinearSamples(rbSamples, rbX, rbY, width, height);
			GetBilinearSamples(gSamples, gX, gY, width, height);

			const u_int lanes = Min(LANES, width - xBegin);
			for (u_int i = 0; i < lanes; ++i) {
				const u_int index = xBegin + i + y * width;

				if (film.HasSamples(hasPN, hasSN, index)) {
					Spectrum value = pixels[index];
					value += redblue * rbSamples.Get(pixels, width, i);
					value += green * gSamples.Get(pixels, width, i);
					// I added redblue+green luminance so I divide by 2.0 to go back
					// to original luminance
					value *= .5f;
					tmpBuffer[index] = value;
				} else
					tmpBuffer[index] = pixels[index];
			}
		}
	}

	#pragma omp parallel for
	for (
			// Visual C++ 2013 supports only OpenMP 2.5
#if _OPENMP >= 200805
			unsigned
#endif
			int y = 0; y < height; ++y)
		copy(&tmpBuffer[y * width], &tmpBuffer[(y + 1) * width], &pixels[y * width]);
}

//------------------------------------------------------------------------------
// HardwareDevice version
//------------------------------------------------------------------------------

void ColorAberrationPlugin::AddHWChannelsUsed(unordered_set<Film::FilmChannelType, hash<int> > &hwChannelsUsed) const {
	hwChannelsUsed.insert(Film::IMAGEPIPELINE);
}

void ColorAberrationPlugin::ApplyHW(Film &film, const u_int index) {
	const u_int width = film.GetWidth();
	const u_int height = film.GetHeight();

	if (!applyKernel) {
		film.ctx->SetVerbose(true);

		hardwareDevice = film.hardwareDevice;

		// Allocate OpenCL buffers
		hardwareDevice->AllocBufferRW(&hwTmpBuffer, nullptr, width * height * sizeof(Spectrum), "ColorAberration");

		// Compile sources
		const double tStart = WallClockTime();

		vector<string> opts;
		opts.push_back("-D LUXRAYS_OPENCL_KERNEL");
		opts.push_back("-D SLG_OPENCL_KERNEL");

		auto program = hardwareDevice->CompileProgram(
				opts,
				luxrays::ocl::KernelSource_luxrays_types +
				luxrays::ocl::KernelSource_utils_funcs +
				slg::ocl::KernelSource_plugin_coloraberration_funcs,
				"ColorAberrationPlugin");

		//----------------------------------------------------------------------
		// ColorAberrationPlugin_Apply kernel
		//----------------------------------------------------------------------

		SLG_LOG("[ColorAberrationPlugin] Compiling ColorAberrationPlugin_Apply Kernel");
		applyKernel = hardwareDevice->GetKernel(*program, "ColorAberrationPlugin_Apply");

		// Set kernel arguments
		u_int argIndex = 0;
		hardwareDevice->SetKernelArg(applyKernel, argIndex++, width);
		hardwareDevice->SetKernelArg(applyKernel, argIndex++, height);
		hardwareDevice->SetKernelArg(applyKernel, argIndex++, film.hw_IMAGEPIPELINE);
		hardwareDevice->SetKernelArg(applyKernel, argIndex++, hwTmpBuffer);
		hardwareDevice->SetKernelArg(applyKernel, argIndex++, amountX);
		hardwareDevice->SetKernelArg(applyKernel, argIndex++, amountY);

		//----------------------------------------------------------------------
		// ColorAberrationPlugin_Copy kernel
		//----------------------------------------------------------------------

		SLG_LOG("[ColorAberrationPlugin] Compiling ColorAberrationPlugin_Copy Kernel");
		copyKernel = hardwareDevice->GetKernel(*program, "ColorAberrationPlugin_Copy");

		// Set kernel arguments
		argIndex = 0;
		hardwareDevice->SetKernelArg(copyKernel, argIndex++, width);
		hardwareDevice->SetKernelArg(copyKernel, argIndex++, height);
		hardwareDevice->SetKernelArg(copyKernel, argIndex++, film.hw_IMAGEPIPELINE);
		hardwareDevice->SetKernelArg(copyKernel, argIndex++, hwTmpBuffer);

		//----------------------------------------------------------------------

		const double tEnd = WallClockTime();
		SLG_LOG("[ColorAberrationPlugin] Kernels compilation time: " << int((tEnd - tStart) * 1000.0) << "ms");

		film.ctx->SetVerbose(false);
	}

	hardwareDevice->EnqueueKernel(applyKernel, HardwareDeviceRange(RoundUp(width * height, 256u)),
			HardwareDeviceRange(256));
	hardwareDevice->EnqueueKernel(copyKernel, HardwareDeviceRange(RoundUp(width * height, 256u)),
			HardwareDeviceRange(256));
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
