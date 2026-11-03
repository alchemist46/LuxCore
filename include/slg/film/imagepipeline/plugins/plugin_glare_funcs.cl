#line 2 "plugin_glare_funcs.cl"

/***************************************************************************
 * Copyright 1998-2026 by authors (see AUTHORS.txt)                        *
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

// The GPU version of GlarePlugin::Apply(): the convolution of two channels at
// once by a complex FFT (one in the real part, the other in the imaginary
// part), the spectra separated by the symmetry of the FFT of a real image. The
// complex values are float pairs.

//------------------------------------------------------------------------------
// GlarePlugin_Pad
//------------------------------------------------------------------------------

// The same as PaddedToImage() of glare.cpp: the edge pixels repeated around
// the image, the right (bottom) edge in the first half of the padding, the left
// (top) edge in the second half, which the FFT wraps to the left of 0
OPENCL_FORCE_INLINE uint GlarePlugin_PaddedToImage(const uint p, const uint size, const uint paddedSize) {
	if (p < size)
		return p;
	else if (p < size + (paddedSize - size) / 2)
		return size - 1;
	else
		return 0;
}

// channel1 is 3 for none (its part is 0)
__kernel void GlarePlugin_Pad(
		const uint filmWidth, const uint filmHeight,
		const uint paddedWidth, const uint paddedHeight,
		const uint channel0, const uint channel1,
		__global const float *channel_IMAGEPIPELINE,
		__global float *data) {
	const size_t gid = get_global_id(0);
	if (gid >= paddedWidth * paddedHeight)
		return;

	const uint x = GlarePlugin_PaddedToImage(gid % paddedWidth, filmWidth, paddedWidth);
	const uint y = GlarePlugin_PaddedToImage(gid / paddedWidth, filmHeight, paddedHeight);
	__global const float *pixel = &channel_IMAGEPIPELINE[(y * filmWidth + x) * 3];

	data[gid * 2] = pixel[channel0];
	data[gid * 2 + 1] = (channel1 < 3) ? pixel[channel1] : 0.f;
}

//------------------------------------------------------------------------------
// GlarePlugin_FFTPass
//------------------------------------------------------------------------------

// A pass of radix 2, 3, 4 or 5 of a mixed radix Stockham FFT (self sorting,
// out of place) of count transforms of length n: element j of transform b at
// b * batchStride + j * elemStride. p is the product of the radices of the
// passes before, sign -1 for the forward transform, 1 for the inverse (not
// normalized).
__kernel void GlarePlugin_FFTPass(
		const uint n, const uint count,
		const uint elemStride, const uint batchStride,
		const uint radix, const uint p, const float sign,
		__global const float *in, __global float *out) {
	const size_t gid = get_global_id(0);
	const uint butterflies = n / radix;
	if (gid >= count * butterflies)
		return;

	// The consecutive threads read consecutive addresses: along the
	// transform for the rows, across the transforms for the columns
	uint i, b;
	if (elemStride == 1) {
		i = gid % butterflies;
		b = gid / butterflies;
	} else {
		b = gid % count;
		i = gid / count;
	}
	const uint base = b * batchStride;
	const uint k = i % p;

	// The inputs, times the twiddle factors
	float xr[5], xi[5];
	const float angle = sign * (2.f * M_PI_F) * k / (p * radix);
	for (uint j = 0; j < radix; ++j) {
		const uint index = (base + (i + j * butterflies) * elemStride) * 2;
		const float vr = in[index];
		const float vi = in[index + 1];
		if (j == 0) {
			xr[0] = vr;
			xi[0] = vi;
		} else {
			const float c = cos(angle * j);
			const float s = sin(angle * j);
			xr[j] = vr * c - vi * s;
			xi[j] = vr * s + vi * c;
		}
	}

	// The DFT of the radix
	float yr[5], yi[5];
	if (radix == 2) {
		yr[0] = xr[0] + xr[1];
		yi[0] = xi[0] + xi[1];
		yr[1] = xr[0] - xr[1];
		yi[1] = xi[0] - xi[1];
	} else if (radix == 4) {
		const float t0r = xr[0] + xr[2], t0i = xi[0] + xi[2];
		const float t1r = xr[0] - xr[2], t1i = xi[0] - xi[2];
		const float t2r = xr[1] + xr[3], t2i = xi[1] + xi[3];
		// (x1 - x3) times sign * i
		const float t3r = -sign * (xi[1] - xi[3]), t3i = sign * (xr[1] - xr[3]);
		yr[0] = t0r + t2r;
		yi[0] = t0i + t2i;
		yr[1] = t1r + t3r;
		yi[1] = t1i + t3i;
		yr[2] = t0r - t2r;
		yi[2] = t0i - t2i;
		yr[3] = t1r - t3r;
		yi[3] = t1i - t3i;
	} else if (radix == 3) {
		const float tr = xr[1] + xr[2], ti = xi[1] + xi[2];
		// (x1 - x2) times sign * i * sqrt(3) / 2
		const float h = sign * 0.86602540378f;
		const float dr = -h * (xi[1] - xi[2]), di = h * (xr[1] - xr[2]);
		yr[0] = xr[0] + tr;
		yi[0] = xi[0] + ti;
		yr[1] = xr[0] - .5f * tr + dr;
		yi[1] = xi[0] - .5f * ti + di;
		yr[2] = xr[0] - .5f * tr - dr;
		yi[2] = xi[0] - .5f * ti - di;
	} else {
		// Radix 5: cos and sin of 2 pi m / 5
		const float cs[5] = { 1.f, 0.30901699437f, -0.80901699437f, -0.80901699437f, 0.30901699437f };
		const float sn[5] = { 0.f, 0.95105651630f, 0.58778525229f, -0.58778525229f, -0.95105651630f };
		for (uint q = 0; q < 5; ++q) {
			float sr = 0.f, si = 0.f;
			for (uint m = 0; m < 5; ++m) {
				const uint w = (m * q) % 5;
				const float c = cs[w];
				const float s = sign * sn[w];
				sr += xr[m] * c - xi[m] * s;
				si += xr[m] * s + xi[m] * c;
			}
			yr[q] = sr;
			yi[q] = si;
		}
	}

	const uint outBase = (i - k) * radix + k;
	for (uint q = 0; q < radix; ++q) {
		const uint index = (base + (outBase + q * p) * elemStride) * 2;
		out[index] = yr[q];
		out[index + 1] = yi[q];
	}
}

//------------------------------------------------------------------------------
// GlarePlugin_Multiply
//------------------------------------------------------------------------------

// The transfer function at the frequency (x, y) from its half spectrum
// (paddedHeight x (paddedWidth / 2 + 1), see GlarePlugin::otf): it is real and
// even
OPENCL_FORCE_INLINE float GlarePlugin_OTF(__global const float *otf,
		const uint x, const uint y, const uint paddedWidth, const uint paddedHeight) {
	const uint halfWidth = paddedWidth / 2 + 1;
	if (x < halfWidth)
		return otf[y * halfWidth + x];
	else
		return otf[((paddedHeight - y) % paddedHeight) * halfWidth + (paddedWidth - x)];
}

// The spectrum Z of channel0 + i channel1: the spectrum of channel0 is
// (Z(k) + conj(Z(-k))) / 2, i times the one of channel1 (Z(k) - conj(Z(-k))) / 2.
// Each one is multiplied by its transfer function (out of place: Z(-k) is read
// by an other thread).
__kernel void GlarePlugin_Multiply(
		const uint paddedWidth, const uint paddedHeight,
		__global const float *otf0, __global const float *otf1,
		__global const float *in, __global float *out) {
	const size_t gid = get_global_id(0);
	if (gid >= paddedWidth * paddedHeight)
		return;

	const uint x = gid % paddedWidth;
	const uint y = gid / paddedWidth;
	const uint negIndex = ((paddedHeight - y) % paddedHeight) * paddedWidth + (paddedWidth - x) % paddedWidth;

	const float zr = in[gid * 2];
	const float zi = in[gid * 2 + 1];
	const float nr = in[negIndex * 2];
	const float ni = in[negIndex * 2 + 1];

	const float a = .5f * GlarePlugin_OTF(otf0, x, y, paddedWidth, paddedHeight);
	const float b = .5f * GlarePlugin_OTF(otf1, x, y, paddedWidth, paddedHeight);

	out[gid * 2] = a * (zr + nr) + b * (zr - nr);
	out[gid * 2 + 1] = a * (zi - ni) + b * (zi + ni);
}

//------------------------------------------------------------------------------
// GlarePlugin_Store
//------------------------------------------------------------------------------

// The rounding of the FFTs leaves tiny negative values around bright pixels:
// the point spread function is not negative. channel1 is 3 for none.
__kernel void GlarePlugin_Store(
		const uint filmWidth, const uint filmHeight,
		const uint paddedWidth,
		const uint channel0, const uint channel1,
		__global const float *data,
		__global float *channel_IMAGEPIPELINE) {
	const size_t gid = get_global_id(0);
	if (gid >= filmWidth * filmHeight)
		return;

	const uint x = gid % filmWidth;
	const uint y = gid / filmWidth;
	__global const float *value = &data[(y * paddedWidth + x) * 2];
	__global float *pixel = &channel_IMAGEPIPELINE[gid * 3];

	pixel[channel0] = fmax(value[0], 0.f);
	if (channel1 < 3)
		pixel[channel1] = fmax(value[1], 0.f);
}

// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
