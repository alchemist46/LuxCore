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

#include "luxrays/utils/atomic.h"
#include "slg/film/film.h"
#include "slg/film/imagepipeline/plugins/outline.h"

using namespace std;
using namespace luxrays;
using namespace slg;

//------------------------------------------------------------------------------
// Outline plugin
//------------------------------------------------------------------------------

BOOST_CLASS_EXPORT_IMPLEMENT(slg::OutlinePlugin)

OutlinePlugin::OutlinePlugin(const LineStyle &ext, const LineStyle &intern,
		const SeparateMeshes sepMeshes, const float depthT, const float normalA,
		const float s, const bool resRelative) :
		external(ext), internal(intern), separateMeshes(sepMeshes),
		depthThreshold(depthT), normalAngle(normalA), scale(s), resolutionRelative(resRelative) {
}

OutlinePlugin::OutlinePlugin() : separateMeshes(SEPARATE_MESHES_INTERSECTING),
		depthThreshold(.05f), normalAngle(45.f), scale(1.f), resolutionRelative(true) {
}

ImagePipelinePlugin *OutlinePlugin::Copy() const {
	return new OutlinePlugin(external, internal, separateMeshes, depthThreshold, normalAngle,
			scale, resolutionRelative);
}

//------------------------------------------------------------------------------
// CPU version
//------------------------------------------------------------------------------

namespace {

// The fraction of a pixel (a unit square around its centre) where the signed
// distance from a line is below s: the projection of the square on the normal
// of the line is the convolution of two boxes a and b wide (a <= b, the
// absolute values of the normal's components)
float SquareCDF(const float s, const float a, const float b) {
	const float h = .5f * (a + b);
	if (s <= -h)
		return 0.f;
	if (s >= h)
		return 1.f;

	const float u = s + h;
	if (a < 1e-4f)
		return u / b;
	if (u < a)
		return u * u / (2.f * a * b);
	if (u < b)
		return (u - .5f * a) / b;
	const float v = a + b - u;
	return 1.f - v * v / (2.f * a * b);
}

// The edges taken into account for the direction of an edge's line
const float normalRadius = 2.f;
// The edges taken into account for its position: along the line, and across it
const float tangentRadius = 3.f;
const float offsetSpread = 1.25f;
// The half length of the line drawn for each edge
const float segmentHalfLength = .75f;
// Covers the two radiuses above
const int fitRadius = 4;

// The edges (edgeSides in Apply()): their shape (sided or centred), whether
// they are internal lines, whether the line also covers the far side
const int8_t edgeSided = 1;
const int8_t edgeCentred = 2;
const int8_t edgeShapeMask = 3;
const int8_t edgeInternal = 4;
const int8_t edgeFarSide = 8;

}

void OutlinePlugin::Apply(Film &film, const u_int index) {
	const int filmWidth = film.GetWidth();
	const int filmHeight = film.GetHeight();
	const u_int pixelCount = filmWidth * filmHeight;

	// The widths of the external and the internal lines in pixels (0: not
	// drawn)
	const float resolutionScale = resolutionRelative ?
		sqrtf(static_cast<float>(pixelCount) / (1920.f * 1080.f)) : 1.f;
	auto lineWidth = [&](const LineStyle &style) {
		const float w = (style.enable && (style.opacity > 0.f)) ?
			Min(style.thickness * scale * resolutionScale, 32.f) : 0.f;
		return (w >= .01f) ? w : 0.f;
	};
	const float widths[2] = { lineWidth(external), lineWidth(internal) };

	const bool useObjectID = (widths[0] > 0.f) && film.HasChannel(Film::OBJECT_ID);
	const bool useDepth = (depthThreshold > 0.f) && film.HasChannel(Film::DEPTH);
	const bool useNormal = (widths[1] > 0.f) && (normalAngle > 0.f) && film.HasChannel(Film::AVG_SHADING_NORMAL);
	if (!useObjectID && !useDepth && !useNormal)
		return;

	Spectrum *pixels = (Spectrum *)film.channel_IMAGEPIPELINEs[index]->GetPixels();

	const bool hasPN = film.HasChannel(Film::RADIANCE_PER_PIXEL_NORMALIZED);
	const bool hasSN = film.HasChannel(Film::RADIANCE_PER_SCREEN_NORMALIZED);
	const bool hasDepth = film.HasChannel(Film::DEPTH);
	// The lines are drawn only over the two objects of their edge
	const bool hasIDs = film.HasChannel(Film::OBJECT_ID);

	// The values of each pixel: a pixel without samples is the background (no
	// object, infinitely far)
	vector<u_int> ids(hasIDs ? pixelCount : 0);
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
		if (hasIDs)
			ids[i] = hasSamples ? *(film.channel_OBJECT_ID->GetPixel(i)) : numeric_limits<u_int>::max();
		depths[i] = (hasSamples && hasDepth) ?
			*(film.channel_DEPTH->GetPixel(i)) : numeric_limits<float>::infinity();
		if (useNormal) {
			Normal n;
			if (hasSamples)
				film.channel_AVG_SHADING_NORMAL->GetWeightedPixel(i, &n.x);
			const float len = n.Length();
			normals[i] = (len > 0.f) ? (n / len) : Normal();
		}
	}

	const float maxAngle = Radians(normalAngle);
	// An object edge has a near side when its depth jumps by more than this
	const float sideThreshold = useDepth ? depthThreshold : .05f;

	const bool drawContacts = (separateMeshes == SEPARATE_MESHES_INTERSECTING) ||
			(separateMeshes == SEPARATE_MESHES_BOTH);
	const bool drawFarSide = (separateMeshes == SEPARATE_MESHES_OVERLAPPING) ||
			(separateMeshes == SEPARATE_MESHES_BOTH);

	// The edges are on the boundaries between 4-neighbour pixels: the one on
	// the right of pixel p is edgeSides[2 * p], the one below it
	// edgeSides[2 * p + 1]. Positive when p is the near side of the edge,
	// negative when the other pixel is, 0 without an edge. Its absolute value
	// is edgeSided, or edgeCentred for the edges without a near side, whose
	// line is centred on the edge: the creases (only the normal changes) and
	// where two objects touch (no depth jump, the nearer pixel changes from
	// one edge to the next); plus edgeInternal for the internal lines (the
	// same object on both sides) and edgeFarSide when the line also covers
	// the far object.
	vector<int8_t> edgeSides(2 * pixelCount, 0);

	#pragma omp parallel for
	for (int y = 0; y < filmHeight; ++y) {
		for (int x = 0; x < filmWidth; ++x) {
			const u_int p = x + y * filmWidth;

			for (u_int axis = 0; axis < 2; ++axis) {
				if ((axis == 0) ? (x + 1 >= filmWidth) : (y + 1 >= filmHeight))
					continue;
				const u_int q = (axis == 0) ? (p + 1) : (p + filmWidth);

				// The line is on the near side of the edge, p on a tie
				const bool pNear = (depths[p] <= depths[q]);
				const u_int n = pNear ? p : q;
				const u_int f = pNear ? q : p;
				// The background is behind everything
				if (isinf(depths[n]) && hasDepth)
					continue;

				// NaN without the DEPTH channel: no near side
				const float jump = depths[f] - depths[n];
				int8_t edge = 0;
				if (hasIDs && (ids[n] != ids[f])) {
					// Two objects, or an object and the background: an
					// external line
					if (useObjectID) {
						if (jump > sideThreshold * depths[n])
							edge = edgeSided | ((drawFarSide && !isinf(depths[f])) ? edgeFarSide : 0);
						else if (drawContacts)
							edge = edgeCentred;
					}
				} else if (useDepth && (jump > depthThreshold * depths[n])) {
					// Self-occlusion: an internal line (an external one
					// without the OBJECT_ID)
					if (!hasIDs && (widths[0] > 0.f))
						edge = edgeSided;
					else if (hasIDs && (widths[1] > 0.f))
						edge = edgeSided | edgeInternal;
				}
				// The background has no normal
				else if (useNormal && (normals[p].LengthSquared() > 0.f) && (normals[q].LengthSquared() > 0.f)) {
					const float angle = acosf(Clamp(Dot(normals[p], normals[q]), -1.f, 1.f));
					bool crease = (angle > maxAngle);

					// The filter of the film spreads a crease over two
					// boundaries, each turning by about half of its angle:
					// the pixels on the far side of p and q see all of it, and
					// the boundary turns more than 0.4 of it (a third on a
					// surface that turns evenly)
					const u_int step = (axis == 0) ? 1 : filmWidth;
					if (!crease && ((axis == 0) ? ((x >= 1) && (x + 2 < filmWidth)) : ((y >= 1) && (y + 2 < filmHeight)))) {
						const u_int pp = p - step;
						const u_int qq = q + step;
						if ((normals[pp].LengthSquared() > 0.f) && (normals[qq].LengthSquared() > 0.f) &&
								(!hasIDs || ((ids[pp] == ids[p]) && (ids[qq] == ids[q])))) {
							const float wideAngle = acosf(Clamp(Dot(normals[pp], normals[qq]), -1.f, 1.f));
							crease = (wideAngle > maxAngle) && (angle > .4f * wideAngle);
						}
					}

					if (crease)
						edge = edgeCentred | edgeInternal;
				}

				edgeSides[2 * p + axis] = pNear ? edge : -edge;
			}
		}
	}

	// Calls f(mid point x, y, direction x, y, crease) for the edges around the
	// pixel (x, y): the pixel centres are at integer coordinates, the direction
	// goes from the near side of the edge to the far side
	auto forEachEdge = [&](const int x, const int y, auto &&f) {
		for (int ey = Max(0, y - fitRadius); ey <= Min(filmHeight - 1, y + fitRadius); ++ey) {
			for (int ex = Max(0, x - fitRadius); ex <= Min(filmWidth - 1, x + fitRadius); ++ex) {
				const u_int e = 2 * (ex + ey * filmWidth);
				if (edgeSides[e])
					f(ex + .5f, static_cast<float>(ey), (edgeSides[e] > 0) ? 1.f : -1.f, 0.f,
							(abs(edgeSides[e]) & edgeShapeMask) == edgeCentred);
				if (edgeSides[e + 1])
					f(static_cast<float>(ex), ey + .5f, 0.f, (edgeSides[e + 1] > 0) ? 1.f : -1.f,
							(abs(edgeSides[e + 1]) & edgeShapeMask) == edgeCentred);
			}
		}
	};

	// The staircase of the edges is anti-aliased: each edge draws a short line
	// fitted to the edges around it, and each pixel takes the fraction of its
	// area that the lines cover, for the external and the internal lines
	vector<float> coverages[2] = { vector<float>(pixelCount, 0.f), vector<float>(pixelCount, 0.f) };

	#pragma omp parallel for schedule(dynamic, 8)
	for (int y = 0; y < filmHeight; ++y) {
		for (int x = 0; x < filmWidth; ++x) {
			const u_int p = x + y * filmWidth;

			for (u_int axis = 0; axis < 2; ++axis) {
				const int side = edgeSides[2 * p + axis];
				if (!side)
					continue;

				const float mx = x + ((axis == 0) ? .5f : 0.f);
				const float my = y + ((axis == 1) ? .5f : 0.f);
				const int edge = abs(side);
				const bool crease = ((edge & edgeShapeMask) == edgeCentred);
				const u_int kind = (edge & edgeInternal) ? 1 : 0;
				const float width = widths[kind];
				const float sx = (axis == 0) ? ((side > 0) ? 1.f : -1.f) : 0.f;
				const float sy = (axis == 1) ? ((side > 0) ? 1.f : -1.f) : 0.f;

				// The normal of the line: the sum of the directions of the
				// edges around it on the same side (the integral of the normal
				// of their staircase). An edge without a near side (a crease,
				// a contact) has its direction turned to the line's.
				float nx = 0.f, ny = 0.f;
				forEachEdge(x, y, [&](const float ex, const float ey, float esx, float esy, const bool eCrease) {
					if (esx * sx + esy * sy < 0.f) {
						if (!crease && !eCrease)
							return;
						esx = -esx;
						esy = -esy;
					}
					const float w = 1.f - sqrtf(Sqr(ex - mx) + Sqr(ey - my)) / (normalRadius + .5f);
					if (w > 0.f) {
						nx += w * esx;
						ny += w * esy;
					}
				});
				const float nLen = sqrtf(nx * nx + ny * ny);
				nx /= nLen;
				ny /= nLen;
				const float tx = -ny;
				const float ty = nx;

				// The position of the line: the mean distance of the edges
				// along it
				float weightSum = 0.f, offsetSum = 0.f;
				forEachEdge(x, y, [&](const float ex, const float ey, const float esx, const float esy, const bool eCrease) {
					if ((esx * nx + esy * ny < -.5f) && !crease && !eCrease)
						return;
					const float d = (ex - mx) * nx + (ey - my) * ny;
					if (fabsf(d) > offsetSpread)
						return;
					const float w = 1.f - fabsf((ex - mx) * tx + (ey - my) * ty) / (tangentRadius + .5f);
					if (w > 0.f) {
						weightSum += w;
						offsetSum += w * d;
					}
				});
				const float offset = offsetSum / weightSum;
				const float lx = mx + offset * nx;
				const float ly = my + offset * ny;

				// The pixels on the two sides of the edge
				const u_int q = (axis == 0) ? (p + 1) : (p + filmWidth);
				const u_int nearPixel = (side > 0) ? p : q;
				const u_int farPixel = (side > 0) ? q : p;

				// The line covers the signed distances from -width (the near
				// side) to 0 (to width when it covers the far side too), from
				// -width / 2 to width / 2 without a near side,
				// segmentHalfLength along it on each side of the edge
				const float lineStart = crease ? (-.5f * width) : -width;
				const float lineEnd = (edge & edgeFarSide) ? width : (lineStart + width);
				const float a = Min(fabsf(nx), fabsf(ny));
				const float b = Max(fabsf(nx), fabsf(ny));
				const float cx = lx + .5f * (lineStart + lineEnd) * nx;
				const float cy = ly + .5f * (lineStart + lineEnd) * ny;
				const float extentX = segmentHalfLength * fabsf(tx) + .5f * (lineEnd - lineStart) * fabsf(nx) + 1.f;
				const float extentY = segmentHalfLength * fabsf(ty) + .5f * (lineEnd - lineStart) * fabsf(ny) + 1.f;
				const int x0 = Max(0, static_cast<int>(floorf(cx - extentX)));
				const int x1 = Min(filmWidth - 1, static_cast<int>(ceilf(cx + extentX)));
				const int y0 = Max(0, static_cast<int>(floorf(cy - extentY)));
				const int y1 = Min(filmHeight - 1, static_cast<int>(ceilf(cy + extentY)));

				for (int py = y0; py <= y1; ++py) {
					for (int px = x0; px <= x1; ++px) {
						const u_int pixel = px + py * filmWidth;
						if (hasIDs && (ids[pixel] != ids[nearPixel]) && (ids[pixel] != ids[farPixel]))
							continue;

						const float dx = px - lx;
						const float dy = py - ly;
						if (fabsf(dx * tx + dy * ty) > segmentHalfLength)
							continue;

						const float d = dx * nx + dy * ny;
						const float coverage = SquareCDF(lineEnd - d, a, b) - SquareCDF(lineStart - d, a, b);
						if (coverage > 0.f)
							AtomicMax(&coverages[kind][pixel], coverage);
					}
				}
			}
		}
	}

	#pragma omp parallel for
	for (
			// Visual C++ 2013 supports only OpenMP 2.5
#if _OPENMP >= 200805
			unsigned
#endif
			int i = 0; i < pixelCount; ++i) {
		// The external lines cover the internal ones
		if (coverages[1][i] > 0.f)
			pixels[i] = Lerp(internal.opacity * coverages[1][i], pixels[i], internal.color);
		if (coverages[0][i] > 0.f)
			pixels[i] = Lerp(external.opacity * coverages[0][i], pixels[i], external.color);
	}
}

// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
