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

#include "slg/film/filters/filterdistribution.h"
#include "slg/film/filters/filter.h"

using namespace std;
using namespace luxrays;
using namespace slg;

//------------------------------------------------------------------------------
// PrecomputedFilter
//------------------------------------------------------------------------------

FilterDistribution::FilterDistribution(const FilterUPtr& f, const u_int s) :
	filter(f)
{
	size = s;
	distrib = NULL;

	std::vector<float> data(size * size);

	const float isize = 1.f / (float)size;
	for (u_int y = 0; y < size; ++y) {
		for (u_int x = 0; x < size; ++x) {
			// Use an uniform distribution if the Filter is missing
			data[x + y * size] = (filter) ? filter->Evaluate(
					2.f * filter->xWidth * ((x + .5f) * isize - .5),
					2.f * filter->yWidth * ((y + .5f) * isize - .5)) : 1.f;
		}
	}

	distrib = std::make_unique<Distribution2D>(data, size, size);
}

FilterDistribution::~FilterDistribution() {}

void FilterDistribution::SampleContinuous(const float u0, const float u1, float *su0, float *su1) const {
	if (filter) {
		float uv[2];
		float pdf;
		distrib->SampleContinuous(u0, u1, uv, &pdf);

		*su0 = (uv[0] - .5f) * (2.f * filter->xWidth);
		*su1 = (uv[1] - .5f) * (2.f * filter->yWidth);
	} else {
		*su0 = u0 - .5f;
		*su1 = u1 - .5f;
	}
}

//------------------------------------------------------------------------------
// FilterLUT
//------------------------------------------------------------------------------

FilterLUT::FilterLUT(const Filter &filter, const float offsetX, const float offsetY) {
	// The pixels, relative to the one of the sample, whose center is within
	// the filter's support ([-xWidth, xWidth] x [-yWidth, yWidth], as the eye
	// samples' FilterDistribution), the sample's own pixel always
	lutX0 = luxrays::Min(luxrays::Ceil2Int(offsetX - filter.xWidth), 0);
	const int x1 = luxrays::Max(luxrays::Floor2Int(offsetX + filter.xWidth), 0);
	lutY0 = luxrays::Min(luxrays::Ceil2Int(offsetY - filter.yWidth), 0);
	const int y1 = luxrays::Max(luxrays::Floor2Int(offsetY + filter.yWidth), 0);
	lutWidth = x1 - lutX0 + 1;
	lutHeight = y1 - lutY0 + 1;
	lut.resize(lutWidth * lutHeight);

	float filterNorm = 0.f;
	unsigned int index = 0;
	for (int iy = lutY0; iy <= y1; ++iy) {
		for (int ix = lutX0; ix <= x1; ++ix) {
			const float filterVal = filter.Evaluate(fabsf(ix - offsetX), fabsf(iy - offsetY));
			filterNorm += filterVal;
			lut[index++] = filterVal;
		}
	}

	// Normalize LUT (all on the sample's own pixel if no pixel center is
	// within a narrow filter)
	if (filterNorm > 0.f) {
		filterNorm = 1.f / filterNorm;
		for (float &v : lut)
			v *= filterNorm;
	} else
		lut[-lutX0 - lutY0 * lutWidth] = 1.f;
}

//------------------------------------------------------------------------------
// FilterLUTs
//------------------------------------------------------------------------------

FilterLUTs::FilterLUTs(const Filter &filter, const size_t size) {
	lutsSize = size + 1;
	// The LUT of each interval of GetLUT() is built at the interval's center
	step = 1.f / float(lutsSize);

	luts.resize(lutsSize * lutsSize);

	for (size_t iy = 0; iy < lutsSize; ++iy) {
		for (size_t ix = 0; ix < lutsSize; ++ix) {
			const float x = (ix + .5f) * step - 0.5f;
			const float y = (iy + .5f) * step - 0.5f;

			luts[ix + iy * lutsSize] = std::make_unique<FilterLUT>(filter, x, y);
			/*cout << "===============================================\n";
			cout << ix << "," << iy << "\n";
			cout << x << "," << y << "\n";
			cout << *luts[ix + iy * lutsSize] << "\n";
			cout << "===============================================\n";*/
		}
	}
}

// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
