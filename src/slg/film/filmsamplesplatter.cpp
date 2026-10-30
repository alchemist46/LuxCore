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

#include "slg/film/filters/gaussian.h"
#include "slg/film/filmsamplesplatter.h"
#include "slg/film/sampleresult.h"

using namespace std;
using namespace luxrays;
using namespace slg;

//------------------------------------------------------------------------------
// FilmSampleSplatter
//------------------------------------------------------------------------------


const FilmSampleSplatterUPtr FilmSampleSplatter::Null{};  // Static

FilmSampleSplatter::FilmSampleSplatter(const FilterUPtr& flt) : filter(flt), borderSize(0) {
	if (filter) {
		// The offsets of a sample within its pixel: 1/17 of a pixel apart
		const u_int size = 16;
		filterLUTs = new FilterLUTs(*filter, size);

		if (filter->GetType() != FILTER_NONE)
			InitBorderScales();
	} else
		filterLUTs = NULL;
}

FilmSampleSplatter::~FilmSampleSplatter() {
	delete filterLUTs;
}

// A pixel near the border of the film gets the splats of the points within
// its filter's support inside the film only (the camera projects no point
// outside), while an eye sample's pixel is the average over its whole
// support (PathTracer's filter importance sampling) or over its part inside
// the film (the weights normalized per pixel of the splatted eye samples of
// BiDir). The splats of the samples normalized per screen (the light
// tracing) on such a pixel are scaled by the inverse of the part of its
// filter inside the film, so a uniform image stays uniform up to the border.
// The part is the integral of the LUT weights the pixel gets from the points
// of the film, by the pixel's distances from a corner (2 points in each LUT
// interval along each axis, so a pixel far from the border gets 1).
void FilmSampleSplatter::InitBorderScales() {
	borderSize = Ceil2Int(Max(filter->xWidth, filter->yWidth));
	const int n = borderSize + 1;
	borderScales.resize(n * n);

	const int pointsPerPixel = 2 * (int)filterLUTs->GetLUTsSize();
	const float invPointsPerPixel = 1.f / pointsPerPixel;
	// The pixels whose points can splat on a pixel
	const int reach = borderSize + 1;

	vector<float> parts(n * n);
	for (int jy = 0; jy < n; ++jy) {
		for (int jx = 0; jx < n; ++jx) {
			// The film starts at 0 along an axis where the pixel is nearer
			// than borderSize to the border
			const int qx0 = (jx < borderSize) ? Max(jx - reach, 0) : (jx - reach);
			const int qy0 = (jy < borderSize) ? Max(jy - reach, 0) : (jy - reach);

			double part = 0.0;
			for (int qy = qy0; qy <= jy + reach; ++qy) {
				for (int qx = qx0; qx <= jx + reach; ++qx) {
					for (int ky = 0; ky < pointsPerPixel; ++ky) {
						for (int kx = 0; kx < pointsPerPixel; ++kx) {
							const FilterLUT *filterLUT = filterLUTs->GetLUT(
									(kx + .5f) * invPointsPerPixel - .5f,
									(ky + .5f) * invPointsPerPixel - .5f);
							const int lx = jx - (qx + filterLUT->GetX0());
							const int ly = jy - (qy + filterLUT->GetY0());

							if ((lx >= 0) && (lx < (int)filterLUT->GetWidth()) &&
									(ly >= 0) && (ly < (int)filterLUT->GetHeight()))
								part += filterLUT->GetLUT()[lx + ly * filterLUT->GetWidth()];
						}
					}
				}
			}

			parts[jx + jy * n] = (float)(part * invPointsPerPixel * invPointsPerPixel);
		}
	}

	// Relative to the pixel far from the border, to cancel the integration's
	// own error
	const float interiorPart = parts[borderSize + borderSize * n];
	for (int i = 0; i < n * n; ++i)
		borderScales[i] = (parts[i] > 0.f) ? (interiorPart / parts[i]) : 1.f;
}

float FilmSampleSplatter::GetBorderScale(const u_int *subRegion, const int x, const int y) const {
	// The filters are symmetric: the distance from the nearest border
	const int jx = Min(Min(x - (int)subRegion[0], (int)subRegion[1] - x), borderSize);
	const int jy = Min(Min(y - (int)subRegion[2], (int)subRegion[3] - y), borderSize);

	return borderScales[jx + jy * (borderSize + 1)];
}

void FilmSampleSplatter::AtomicSplatSample(FilmConstRef film, const SampleResult &sampleResult,
		const float weight, const bool halfSample) const {
	const u_int *subRegion = film.GetSubRegion();

	if (!filter || (filter->GetType() == FILTER_NONE)) {
		const int x = Floor2Int(sampleResult.filmX);
		const int y = Floor2Int(sampleResult.filmY);

		if ((x >= (int)subRegion[0]) && (x <= (int)subRegion[1]) && (y >= (int)subRegion[2]) && (y <= (int)subRegion[3])) {
			film.AtomicAddSample(x, y, sampleResult, weight, halfSample);
		}
	} else {
		//----------------------------------------------------------------------
		// Add all data related information (not filtered)
		//----------------------------------------------------------------------

		if (film.HasDataChannel()) {
			const int x = Floor2Int(sampleResult.filmX);
			const int y = Floor2Int(sampleResult.filmY);

			if ((x >= (int)subRegion[0]) && (x <= (int)subRegion[1]) && (y >= (int)subRegion[2]) && (y <= (int)subRegion[3]))
				film.AtomicAddSampleResultData(x, y, sampleResult);
		}

		//----------------------------------------------------------------------
		// Add all color related information (filtered)
		//----------------------------------------------------------------------

		// Compute sample's raster extent: the LUT of the sample's offset from
		// the center of its pixel, placed from that pixel
		const int pixelX = Floor2Int(sampleResult.filmX);
		const int pixelY = Floor2Int(sampleResult.filmY);
		const FilterLUT *filterLUT = filterLUTs->GetLUT(sampleResult.filmX - .5f - pixelX,
				sampleResult.filmY - .5f - pixelY);
		const float *lut = filterLUT->GetLUT().data();
		const int lutWidth = filterLUT->GetWidth();

		const int x0 = pixelX + filterLUT->GetX0();
		const int y0 = pixelY + filterLUT->GetY0();
		const int ixStart = Max(x0, (int)subRegion[0]);
		const int ixEnd = Min(x0 + lutWidth - 1, (int)subRegion[1]);
		const int iyStart = Max(y0, (int)subRegion[2]);
		const int iyEnd = Min(y0 + (int)filterLUT->GetHeight() - 1, (int)subRegion[3]);

		const bool borderScale = !borderScales.empty() &&
				sampleResult.HasChannel(Film::RADIANCE_PER_SCREEN_NORMALIZED);

		for (int iy = iyStart; iy <= iyEnd; ++iy) {
			const float *lutRow = lut + (iy - y0) * lutWidth;

			for (int ix = ixStart; ix <= ixEnd; ++ix) {
				const float filteredWeight = weight * lutRow[ix - x0] *
						(borderScale ? GetBorderScale(subRegion, ix, iy) : 1.f);
				film.AtomicAddSampleResultColor(ix, iy, sampleResult, filteredWeight, halfSample);
			}
		}
	}
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
