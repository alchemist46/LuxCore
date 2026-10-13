/***************************************************************************
 * Copyright 1998-2018 by authors (see AUTHORS.txt)                        *
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

#include "slg/film/film.h"

using namespace std;
using namespace luxrays;
using namespace slg;

//------------------------------------------------------------------------------
// FilmSamplesCounts
//------------------------------------------------------------------------------

FilmSamplesCounts::FilmSamplesCounts() {
	Init(1);
}

FilmSamplesCounts::~FilmSamplesCounts() {
}

void FilmSamplesCounts::Init(const u_int count) {
	assert (count > 0);

	threadCount = count;
	counts.assign(threadCount, ThreadSampleCounts());
}

void FilmSamplesCounts::Clear() {
	fill(counts.begin(), counts.end(), ThreadSampleCounts());
}

void FilmSamplesCounts::SetSampleCount(const double sampleCount,
		const double RADIANCE_PER_PIXEL_NORMALIZED_count,
		const double RADIANCE_PER_SCREEN_NORMALIZED_count) {
	Clear();

	counts[0].total = sampleCount;
	counts[0].RADIANCE_PER_PIXEL_NORMALIZED = RADIANCE_PER_PIXEL_NORMALIZED_count;
	counts[0].RADIANCE_PER_SCREEN_NORMALIZED = RADIANCE_PER_SCREEN_NORMALIZED_count;
}

void FilmSamplesCounts::AddSampleCount(const double sampleCount,
		const double RADIANCE_PER_PIXEL_NORMALIZED_count,
		const double RADIANCE_PER_SCREEN_NORMALIZED_count) {
	counts[0].total += sampleCount;
	counts[0].RADIANCE_PER_PIXEL_NORMALIZED += RADIANCE_PER_PIXEL_NORMALIZED_count;
	counts[0].RADIANCE_PER_SCREEN_NORMALIZED += RADIANCE_PER_SCREEN_NORMALIZED_count;
}

void FilmSamplesCounts::AddSampleCount(const u_int threadIndex,
		const double RADIANCE_PER_PIXEL_NORMALIZED_count,
		const double RADIANCE_PER_SCREEN_NORMALIZED_count) {
	assert (threadIndex < threadCount);

	ThreadSampleCounts &c = counts[threadIndex];
	c.total += Max(RADIANCE_PER_PIXEL_NORMALIZED_count, RADIANCE_PER_SCREEN_NORMALIZED_count);
	c.RADIANCE_PER_PIXEL_NORMALIZED += RADIANCE_PER_PIXEL_NORMALIZED_count;
	c.RADIANCE_PER_SCREEN_NORMALIZED += RADIANCE_PER_SCREEN_NORMALIZED_count;
}

double FilmSamplesCounts::GetSampleCount() const {
	double result = 0.0;
	for (u_int i = 0; i < threadCount; ++i)
		result += counts[i].total;
	return result;
}

double FilmSamplesCounts::GetSampleCount_RADIANCE_PER_PIXEL_NORMALIZED() const {
	double result = 0.0;
	for (u_int i = 0; i < threadCount; ++i)
		result += counts[i].RADIANCE_PER_PIXEL_NORMALIZED;
	return result;	
}

double FilmSamplesCounts::GetSampleCount_RADIANCE_PER_SCREEN_NORMALIZED() const {
	double result = 0.0;
	for (u_int i = 0; i < threadCount; ++i)
		result += counts[i].RADIANCE_PER_SCREEN_NORMALIZED;
	return result;	
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
