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

#include <thread>

#include "pocketfft_hdronly.h"

#include "slg/film/film.h"
#include "slg/film/imagepipeline/plugins/glare.h"
#include "slg/film/imagepipeline/plugins/coloraberration.h"

using namespace std;
using namespace luxrays;
using namespace slg;

//------------------------------------------------------------------------------
// Glare plugin
//------------------------------------------------------------------------------

BOOST_CLASS_EXPORT_IMPLEMENT(slg::GlarePlugin)

// The autocorrelation of the aperture is tabulated on a APERTURE_GRID^2 grid,
// the aperture being APERTURE_DIAMETER samples wide: its support (a disk of
// radius APERTURE_DIAMETER) fits in the grid without wrapping onto itself
static const u_int APERTURE_GRID = 1024;
static const float APERTURE_DIAMETER = 500.f;
// The supersampling of the rasterized aperture, per axis
static const u_int APERTURE_SUPERSAMPLING = 4;
// The kernel of the diffraction is computed on a grid at most MAX_FINE_GRID
// samples wide and at most MAX_SUPERSAMPLING times finer than the pixels, its
// transfer function tapered to 0 from TAPER_START times its Nyquist frequency
static const u_int MAX_FINE_GRID = 4096;
static const u_int MAX_SUPERSAMPLING = 4;
static const float TAPER_START = .75f;

GlarePlugin::GlarePlugin(const float fs, const float sw,
		const u_int b, const float rot, const float round,
		const float obs, const float sc,
		const float scR, const float scF,
		const float s, const float r, const u_int ws,
		const WorkingColorSpace &colorSpace) :
		fstop(fs), sensorWidth(sw), blades(b), rotation(rot), roundness(round),
		obstruction(obs), scatter(sc), scatterRadius(scR), scatterFalloff(scF),
		scale(s), radius(r), samples(ws),
		otfWidth(0), otfHeight(0), paddedWidth(0), paddedHeight(0) {
	InitWavelengthTable(colorSpace);
}

GlarePlugin::GlarePlugin() :
		otfWidth(0), otfHeight(0), paddedWidth(0), paddedHeight(0) {
}

GlarePlugin::~GlarePlugin() {
}

ImagePipelinePlugin *GlarePlugin::Copy() const {
	GlarePlugin *plugin = new GlarePlugin();
	plugin->fstop = fstop;
	plugin->sensorWidth = sensorWidth;
	plugin->blades = blades;
	plugin->rotation = rotation;
	plugin->roundness = roundness;
	plugin->obstruction = obstruction;
	plugin->scatter = scatter;
	plugin->scatterRadius = scatterRadius;
	plugin->scatterFalloff = scatterFalloff;
	plugin->scale = scale;
	plugin->radius = radius;
	plugin->samples = samples;
	plugin->wavelengthTable = wavelengthTable;

	return plugin;
}

void GlarePlugin::InitWavelengthTable(const WorkingColorSpace &colorSpace) {
	wavelengthTable = ColorAberrationPlugin::WavelengthWeights(samples, colorSpace);
}

//------------------------------------------------------------------------------
// The aperture
//------------------------------------------------------------------------------

// (x, y) in units of the aperture radius, the vertices of the blade polygon on
// the unit circle
bool GlarePlugin::InsideAperture(const float x, const float y) const {
	const float r = sqrtf(x * x + y * y);
	if ((r > 1.f) || (r < obstruction))
		return false;
	if (blades < 3)
		return true;

	// The angle from the middle of the polygon side, in [-segment / 2, segment / 2]
	const float segment = 2.f * M_PI / blades;
	const float phi = atan2f(y, x) - Radians(rotation);
	const float phiSide = phi - segment * floorf(phi / segment) - .5f * segment;
	const float sideRadius = cosf(.5f * segment) / cosf(phiSide);

	return r <= Lerp(roundness, sideRadius, 1.f);
}

void GlarePlugin::InitApertureOTF() {
	const u_int size = APERTURE_GRID;
	const u_int halfSize = size / 2 + 1;
	const size_t nThreads = Max(1u, thread::hardware_concurrency());

	// Rasterize the aperture centered in the grid (the autocorrelation does
	// not depend on its position)
	vector<float> aperture(size * size, 0.f);
	const float invRadius = 2.f / APERTURE_DIAMETER;
	const float invSupersampling2 = 1.f / (APERTURE_SUPERSAMPLING * APERTURE_SUPERSAMPLING);
	const int first = (int)floorf(.5f * (size - APERTURE_DIAMETER)) - 1;
	const int last = (int)ceilf(.5f * (size + APERTURE_DIAMETER)) + 1;

	#pragma omp parallel for
	for (int y = first; y < last; ++y) {
		for (int x = first; x < last; ++x) {
			u_int inside = 0;
			for (u_int sy = 0; sy < APERTURE_SUPERSAMPLING; ++sy) {
				for (u_int sx = 0; sx < APERTURE_SUPERSAMPLING; ++sx) {
					const float px = (x + (sx + .5f) / APERTURE_SUPERSAMPLING - .5f * size) * invRadius;
					const float py = (y + (sy + .5f) / APERTURE_SUPERSAMPLING - .5f * size) * invRadius;
					if (InsideAperture(px, py))
						++inside;
				}
			}
			aperture[y * size + x] = inside * invSupersampling2;
		}
	}

	// The autocorrelation is the inverse FFT of the power spectrum
	const pocketfft::shape_t shape = { size, size };
	const pocketfft::stride_t strideReal = { (ptrdiff_t)(size * sizeof(float)), sizeof(float) };
	const pocketfft::stride_t strideComplex = { (ptrdiff_t)(halfSize * sizeof(complex<float>)), sizeof(complex<float>) };
	const pocketfft::shape_t axes = { 0, 1 };

	vector<complex<float> > spectrum(size * halfSize);
	pocketfft::r2c(shape, strideReal, strideComplex, axes, pocketfft::FORWARD,
			aperture.data(), spectrum.data(), 1.f, nThreads);
	for (auto &v : spectrum)
		v = norm(v);
	apertureOTF.resize(size * size);
	pocketfft::c2r(shape, strideComplex, strideReal, axes, pocketfft::BACKWARD,
			spectrum.data(), apertureOTF.data(), 1.f, nThreads);

	// Divided by the area: 1 at the origin
	const float invArea = 1.f / apertureOTF[0];
	for (auto &v : apertureOTF)
		v = Max(v * invArea, 0.f);
}

// The transfer function at the shift (sx, sy), in grid samples
float GlarePlugin::ApertureOTF(const float sx, const float sy) const {
	if (sx * sx + sy * sy >= APERTURE_DIAMETER * APERTURE_DIAMETER)
		return 0.f;

	// The origin is at (0, 0), the negative shifts are wrapped
	const u_int mask = APERTURE_GRID - 1;
	const float fx = (sx < 0.f) ? (sx + APERTURE_GRID) : sx;
	const float fy = (sy < 0.f) ? (sy + APERTURE_GRID) : sy;
	const u_int x0 = Floor2UInt(fx);
	const u_int y0 = Floor2UInt(fy);
	const float tx = fx - x0;
	const float ty = fy - y0;
	const u_int x1 = (x0 + 1) & mask;
	const u_int y1 = (y0 + 1) & mask;
	const u_int x0m = x0 & mask;
	const u_int y0m = y0 & mask;

	return (1.f - ty) * ((1.f - tx) * apertureOTF[y0m * APERTURE_GRID + x0m] + tx * apertureOTF[y0m * APERTURE_GRID + x1]) +
			ty * ((1.f - tx) * apertureOTF[y1 * APERTURE_GRID + x0m] + tx * apertureOTF[y1 * APERTURE_GRID + x1]);
}

//------------------------------------------------------------------------------
// The transfer function of the image
//------------------------------------------------------------------------------

void GlarePlugin::InitOTF(const u_int width, const u_int height) {
	if (apertureOTF.empty())
		InitApertureOTF();

	const double tStart = WallClockTime();

	// The image is padded by the radius of the point spread function on each
	// side (see Apply()), so nothing wraps from one side to the other
	const float diagonal = sqrtf((float)width * width + (float)height * height);
	const float r = Max(1.f, radius * diagonal);
	const u_int ri = Ceil2UInt(r);
	paddedWidth = (u_int)pocketfft::detail::util::good_size_real(width + 2 * ri);
	paddedHeight = (u_int)pocketfft::detail::util::good_size_real(height + 2 * ri);
	otfWidth = width;
	otfHeight = height;

	const u_int halfWidth = paddedWidth / 2 + 1;
	const size_t realSize = (size_t)paddedWidth * paddedHeight;
	const size_t complexSize = (size_t)halfWidth * paddedHeight;
	const float invRealSize = 1.f / realSize;
	const size_t nThreads = Max(1u, thread::hardware_concurrency());

	const pocketfft::shape_t shape = { paddedHeight, paddedWidth };
	const pocketfft::stride_t strideReal = { (ptrdiff_t)(paddedWidth * sizeof(float)), sizeof(float) };
	const pocketfft::stride_t strideComplex = { (ptrdiff_t)(halfWidth * sizeof(complex<float>)), sizeof(complex<float>) };
	const pocketfft::shape_t axes = { 0, 1 };

	realBuffer.resize(realSize);
	complexBuffer.resize(complexSize);

	//--------------------------------------------------------------------------
	// The diffraction of each channel, integrated over the pixel area as the
	// sensor does: the transfer function of each wavelength on a grid
	// supersampling times finer than the pixels (over the kernel only), summed
	// with the weights of the channel, back to space and box filtered into the
	// pixels. The transfer function goes on past the Nyquist frequency of the
	// pixels when the core is smaller than a pixel: cut there, it would ring
	// in streaks along the axes. It is cut at the Nyquist frequency of the
	// fine grid instead (with a taper), where the box filter of a pixel is 0
	//--------------------------------------------------------------------------

	// A frequency of f cycles per pixel is f / pitch cycles per mm, and the
	// shift of the autocorrelation scale * lambda * fstop * f / pitch in
	// aperture diameters
	const float pitch = sensorWidth / width;
	vector<float> shiftScale(samples);
	for (u_int k = 0; k < samples; ++k)
		shiftScale[k] = scale * (wavelengthTable[k * 4] * 1e-6f) * fstop / pitch * APERTURE_DIAMETER;

	// The fine grid covers the kernel with a margin for the tail its period
	// wraps
	const u_int kernelSize = 2 * ri + 1;
	const u_int fineSpan = kernelSize + kernelSize / 4;
	const u_int supersampling = Clamp(MAX_FINE_GRID / fineSpan, 1u, MAX_SUPERSAMPLING);
	const u_int fineSize = (u_int)pocketfft::detail::util::good_size_real(supersampling * fineSpan);
	const u_int fineHalf = fineSize / 2 + 1;
	const pocketfft::shape_t fineShape = { fineSize, fineSize };
	const pocketfft::stride_t fineStrideReal = { (ptrdiff_t)(fineSize * sizeof(float)), sizeof(float) };
	const pocketfft::stride_t fineStrideComplex = { (ptrdiff_t)(fineHalf * sizeof(complex<float>)), sizeof(complex<float>) };

	vector<float> fineOTF[3];
	for (u_int c = 0; c < 3; ++c)
		fineOTF[c].assign((size_t)fineSize * fineHalf, 0.f);

	#pragma omp parallel for
	for (
			// Visual C++ 2013 supports only OpenMP 2.5
#if _OPENMP >= 200805
			unsigned
#endif
			int ky = 0; ky < fineSize; ++ky) {
		// In cycles per fine sample
		const float fy = ((ky <= fineSize / 2) ? (float)ky : ((float)ky - fineSize)) / fineSize;
		for (u_int kx = 0; kx < fineHalf; ++kx) {
			const float fx = (float)kx / fineSize;

			// 1 at the Nyquist frequency of the fine grid on the axes
			const float rho = 2.f * sqrtf(fx * fx + fy * fy);
			if (rho >= 1.f)
				continue;
			const float taper = (rho < TAPER_START) ? 1.f :
				Sqr(cosf(.5f * M_PI * (rho - TAPER_START) / (1.f - TAPER_START)));

			const size_t i = (size_t)ky * fineHalf + kx;
			for (u_int k = 0; k < samples; ++k) {
				const float s = supersampling * shiftScale[k];
				const float t = taper * ApertureOTF(fx * s, fy * s);
				for (u_int c = 0; c < 3; ++c)
					fineOTF[c][i] += wavelengthTable[k * 4 + 1 + c] * t;
			}
		}
	}

	// The box filter of a pixel: the trapezoidal rule over the supersampling + 1
	// fine samples from -1/2 to 1/2 pixel for an even supersampling, the
	// supersampling samples in between for an odd one
	vector<int> boxOffsets;
	vector<float> boxWeights;
	if (supersampling % 2 == 0) {
		const int half = supersampling / 2;
		for (int j = -half; j <= half; ++j) {
			boxOffsets.push_back(j);
			boxWeights.push_back(((j == -half) || (j == half)) ? .5f : 1.f);
		}
	} else {
		const int half = (supersampling - 1) / 2;
		for (int j = -half; j <= half; ++j) {
			boxOffsets.push_back(j);
			boxWeights.push_back(1.f);
		}
	}

	vector<float> diffractionKernel[3];
	vector<float> fineKernel((size_t)fineSize * fineSize);
	vector<complex<float> > fineSpectrum((size_t)fineSize * fineHalf);
	for (u_int c = 0; c < 3; ++c) {
		for (size_t i = 0; i < fineSpectrum.size(); ++i)
			fineSpectrum[i] = complex<float>(fineOTF[c][i], 0.f);
		pocketfft::c2r(fineShape, fineStrideComplex, fineStrideReal, axes, pocketfft::BACKWARD,
				fineSpectrum.data(), fineKernel.data(), 1.f, nThreads);

		// The pixels of the kernel, cut at the radius, without the few
		// negative values the cut of the spectrum leaves
		diffractionKernel[c].assign((size_t)kernelSize * kernelSize, 0.f);
		#pragma omp parallel for
		for (int dy = -(int)ri; dy <= (int)ri; ++dy) {
			for (int dx = -(int)ri; dx <= (int)ri; ++dx) {
				if (dx * dx + dy * dy > r * r)
					continue;

				float v = 0.f;
				for (size_t l = 0; l < boxOffsets.size(); ++l) {
					const int fy = dy * (int)supersampling + boxOffsets[l];
					const u_int wy = (u_int)((fy < 0) ? (fy + (int)fineSize) : fy);
					for (size_t j = 0; j < boxOffsets.size(); ++j) {
						const int fx = dx * (int)supersampling + boxOffsets[j];
						const u_int wx = (u_int)((fx < 0) ? (fx + (int)fineSize) : fx);
						v += boxWeights[l] * boxWeights[j] * fineKernel[(size_t)wy * fineSize + wx];
					}
				}
				diffractionKernel[c][(dy + ri) * kernelSize + (dx + ri)] = Max(v, 0.f);
			}
		}
	}

	//--------------------------------------------------------------------------
	// The veiling glare, the same in the 3 channels
	//--------------------------------------------------------------------------

	const float r0 = Max(scatterRadius * diagonal, 1e-3f);
	vector<float> scatterKernel((size_t)kernelSize * kernelSize, 0.f);
	for (int dy = -(int)ri; dy <= (int)ri; ++dy) {
		for (int dx = -(int)ri; dx <= (int)ri; ++dx) {
			const float d2 = (float)(dx * dx + dy * dy);
			if (d2 <= r * r)
				scatterKernel[(dy + ri) * kernelSize + (dx + ri)] = powf(1.f + d2 / (r0 * r0), -.5f * scatterFalloff);
		}
	}

	//--------------------------------------------------------------------------
	// The point spread function of each channel, normalized to 1, wrapped
	// about the origin of the padded image, and its transform
	//--------------------------------------------------------------------------

	double scatterSum = 0.0;
	for (const float v : scatterKernel)
		scatterSum += v;

	for (u_int c = 0; c < 3; ++c) {
		double diffractionSum = 0.0;
		for (const float v : diffractionKernel[c])
			diffractionSum += v;

		const float diffractionScale = (diffractionSum > 0.0) ? ((1.f - scatter) / diffractionSum) : 0.f;
		const float scatterScale = (scatterSum > 0.0) ? (scatter / scatterSum) : 0.f;

		fill(realBuffer.begin(), realBuffer.end(), 0.f);
		for (int dy = -(int)ri; dy <= (int)ri; ++dy) {
			const u_int y = (u_int)((dy < 0) ? (dy + (int)paddedHeight) : dy);
			for (int dx = -(int)ri; dx <= (int)ri; ++dx) {
				const u_int x = (u_int)((dx < 0) ? (dx + (int)paddedWidth) : dx);
				const size_t k = (dy + ri) * kernelSize + (dx + ri);
				realBuffer[(size_t)y * paddedWidth + x] = diffractionScale * diffractionKernel[c][k] +
						scatterScale * scatterKernel[k];
			}
		}
		if (diffractionSum <= 0.0)
			realBuffer[0] += 1.f - scatter;

		// The kernel is symmetric: its transform is real. The 1 / size of the
		// inverse FFT of Apply() is folded in
		pocketfft::r2c(shape, strideReal, strideComplex, axes, pocketfft::FORWARD,
				realBuffer.data(), complexBuffer.data(), 1.f, nThreads);
		otf[c].resize(complexSize);
		for (size_t i = 0; i < complexSize; ++i)
			otf[c][i] = complexBuffer[i].real() * invRealSize;
	}

	const double tEnd = WallClockTime();
	SLG_LOG("[GlarePlugin] " << width << "x" << height << " padded to " <<
			paddedWidth << "x" << paddedHeight << ", " << supersampling <<
			"x supersampled kernel, transfer functions built in " <<
			int((tEnd - tStart) * 1000.0) << "ms");
}

//------------------------------------------------------------------------------
// CPU version
//------------------------------------------------------------------------------

// The padding repeats the edge pixels, as if the scene went on past the frame
// (a flat image stays flat): the image is in [0, size), its right (bottom)
// edge repeated in the first half of the padding, its left (top) edge in the
// second half, which the FFT wraps to the left of 0
static inline u_int PaddedToImage(const u_int p, const u_int size, const u_int paddedSize) {
	if (p < size)
		return p;
	else if (p < size + (paddedSize - size) / 2)
		return size - 1;
	else
		return 0;
}

void GlarePlugin::Apply(Film &film, const u_int index) {
	Spectrum *pixels = (Spectrum *)film.channel_IMAGEPIPELINEs[index]->GetPixels();

	const u_int width = film.GetWidth();
	const u_int height = film.GetHeight();

	if ((width != otfWidth) || (height != otfHeight))
		InitOTF(width, height);

	const bool hasPN = film.HasChannel(Film::RADIANCE_PER_PIXEL_NORMALIZED);
	const bool hasSN = film.HasChannel(Film::RADIANCE_PER_SCREEN_NORMALIZED);

	const u_int halfWidth = paddedWidth / 2 + 1;
	const size_t complexSize = (size_t)halfWidth * paddedHeight;
	const size_t nThreads = Max(1u, thread::hardware_concurrency());

	const pocketfft::shape_t shape = { paddedHeight, paddedWidth };
	const pocketfft::stride_t strideReal = { (ptrdiff_t)(paddedWidth * sizeof(float)), sizeof(float) };
	const pocketfft::stride_t strideComplex = { (ptrdiff_t)(halfWidth * sizeof(complex<float>)), sizeof(complex<float>) };
	const pocketfft::shape_t axes = { 0, 1 };

	for (u_int c = 0; c < 3; ++c) {
		// The channel in the top left corner, its edges repeated around it
		#pragma omp parallel for
		for (
				// Visual C++ 2013 supports only OpenMP 2.5
#if _OPENMP >= 200805
				unsigned
#endif
				int y = 0; y < paddedHeight; ++y) {
			float *row = &realBuffer[(size_t)y * paddedWidth];
			const Spectrum *imageRow = &pixels[PaddedToImage(y, height, paddedHeight) * width];
			for (u_int x = 0; x < width; ++x)
				row[x] = imageRow[x].c[c];
			const float right = imageRow[width - 1].c[c];
			const float left = imageRow[0].c[c];
			const u_int middle = width + (paddedWidth - width) / 2;
			for (u_int x = width; x < middle; ++x)
				row[x] = right;
			for (u_int x = middle; x < paddedWidth; ++x)
				row[x] = left;
		}

		pocketfft::r2c(shape, strideReal, strideComplex, axes, pocketfft::FORWARD,
				realBuffer.data(), complexBuffer.data(), 1.f, nThreads);
		const vector<float> &channelOTF = otf[c];
		#pragma omp parallel for
		for (int i = 0; i < (int)complexSize; ++i)
			complexBuffer[i] *= channelOTF[i];
		pocketfft::c2r(shape, strideComplex, strideReal, axes, pocketfft::BACKWARD,
				complexBuffer.data(), realBuffer.data(), 1.f, nThreads);

		// The rounding of the FFTs leaves tiny negative values around bright
		// pixels: the point spread function is not negative
		#pragma omp parallel for
		for (
				// Visual C++ 2013 supports only OpenMP 2.5
#if _OPENMP >= 200805
				unsigned
#endif
				int y = 0; y < height; ++y) {
			for (u_int x = 0; x < width; ++x) {
				const u_int pixelIndex = y * width + x;
				if (film.HasSamples(hasPN, hasSN, pixelIndex))
					pixels[pixelIndex].c[c] = Max(realBuffer[(size_t)y * paddedWidth + x], 0.f);
			}
		}
	}
}
