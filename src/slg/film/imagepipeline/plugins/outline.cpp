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

#include <cmath>
#include <limits>
#include <vector>

#include "slg/film/film.h"
#include "slg/film/imagepipeline/plugins/outline.h"

using namespace std;
using namespace luxrays;
using namespace slg;

//------------------------------------------------------------------------------
// Outline plugin
//------------------------------------------------------------------------------

BOOST_CLASS_EXPORT_IMPLEMENT(slg::OutlinePlugin)

OutlinePlugin::OutlinePlugin(const Spectrum &col, const float amnt, const float w,
		const bool objID, const float depthT, const float normalA) :
		color(col), amount(amnt), width(w), objectIDEnabled(objID),
		depthThreshold(depthT), normalAngle(normalA) {
}

OutlinePlugin::OutlinePlugin() : color(0.f), amount(1.f), width(1.f), objectIDEnabled(true),
		depthThreshold(.05f), normalAngle(45.f) {
}

ImagePipelinePlugin *OutlinePlugin::Copy() const {
	return new OutlinePlugin(color, amount, width, objectIDEnabled, depthThreshold, normalAngle);
}

//------------------------------------------------------------------------------
// CPU version
//------------------------------------------------------------------------------

void OutlinePlugin::Apply(Film &film, const u_int index) {
	const bool useObjectID = objectIDEnabled && film.HasChannel(Film::OBJECT_ID);
	const bool useDepth = (depthThreshold > 0.f) && film.HasChannel(Film::DEPTH);
	const bool useNormal = (normalAngle > 0.f) && film.HasChannel(Film::AVG_SHADING_NORMAL);
	if (!useObjectID && !useDepth && !useNormal)
		return;

	Spectrum *pixels = (Spectrum *)film.channel_IMAGEPIPELINEs[index]->GetPixels();
	const int filmWidth = film.GetWidth();
	const int filmHeight = film.GetHeight();
	const u_int pixelCount = filmWidth * filmHeight;

	const bool hasPN = film.HasChannel(Film::RADIANCE_PER_PIXEL_NORMALIZED);
	const bool hasSN = film.HasChannel(Film::RADIANCE_PER_SCREEN_NORMALIZED);

	// The values of each pixel: a pixel without samples is the background (no
	// object, infinitely far)
	vector<u_int> ids(useObjectID ? pixelCount : 0);
	vector<float> depths(pixelCount);
	vector<Normal> normals(useNormal ? pixelCount : 0);

	#pragma omp parallel for
	for (
			// Visual C++ 2013 supports only OpenMP 2.5
#if _OPENMP >= 200805
			unsigned
#endif
			int i = 0; i < pixelCount; ++i) {
		const bool hasSamples = film.HasSamples(hasPN, hasSN, i);
		if (useObjectID)
			ids[i] = hasSamples ? *(film.channel_OBJECT_ID->GetPixel(i)) : numeric_limits<u_int>::max();
		depths[i] = (hasSamples && film.HasChannel(Film::DEPTH)) ?
			*(film.channel_DEPTH->GetPixel(i)) : numeric_limits<float>::infinity();
		if (useNormal) {
			Normal n;
			if (hasSamples)
				film.channel_AVG_SHADING_NORMAL->GetWeightedPixel(i, &n.x);
			const float len = n.Length();
			normals[i] = (len > 0.f) ? (n / len) : Normal();
		}
	}

	const float cosAngle = cosf(Radians(normalAngle));
	const int radius = Max(1, static_cast<int>(ceilf(width)));
	const float radius2 = Max(1.f, width * width);

	#pragma omp parallel for
	for (int y = 0; y < filmHeight; ++y) {
		for (int x = 0; x < filmWidth; ++x) {
			const u_int p = x + y * filmWidth;
			const float depthP = depths[p];
			// The background is behind everything
			if (isinf(depthP) && film.HasChannel(Film::DEPTH))
				continue;

			bool onLine = false;
			for (int dy = -radius; (dy <= radius) && !onLine; ++dy) {
				const int qy = y + dy;
				if ((qy < 0) || (qy >= filmHeight))
					continue;

				for (int dx = -radius; dx <= radius; ++dx) {
					const int qx = x + dx;
					if ((qx < 0) || (qx >= filmWidth) || ((dx == 0) && (dy == 0)) ||
							(dx * dx + dy * dy > radius2))
						continue;

					const u_int q = qx + qy * filmWidth;
					const float depthQ = depths[q];

					// The line is on the near side of the edge
					if ((depthP > depthQ) || ((depthP == depthQ) && (p > q)))
						continue;

					if ((useObjectID && (ids[p] != ids[q])) ||
							(useDepth && (depthQ - depthP > depthThreshold * depthP)) ||
							// The background has no normal
							(useNormal && (normals[p].LengthSquared() > 0.f) && (normals[q].LengthSquared() > 0.f) &&
								(Dot(normals[p], normals[q]) < cosAngle))) {
						onLine = true;
						break;
					}
				}
			}

			if (onLine)
				pixels[p] = Lerp(amount, pixels[p], color);
		}
	}
}

// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
