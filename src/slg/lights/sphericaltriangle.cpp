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

// The solid angle sampling of the triangle lights, compiled from the kernel
// source shared with the GPU (single-source, see
// luxrays/devices/cpudevice_oclemul.h)

#include <cmath>

#include "slg/lights/trianglelight.h"

namespace slg { namespace ocl { namespace {
#include "luxrays/devices/cpudevice_oclemul.h"
#include "slg/lights/sphericaltriangle_funcs.cl"

float3 ToFloat3(const luxrays::Point &p) {
	return MAKE_FLOAT3(p.x, p.y, p.z);
}
} } }

using namespace luxrays;

float slg::SphericalTriangle_SolidAngle(const Point &v0, const Point &v1, const Point &v2,
		const Point &p) {
	return slg::ocl::SphericalTriangle_SolidAngle(slg::ocl::ToFloat3(v0), slg::ocl::ToFloat3(v1),
			slg::ocl::ToFloat3(v2), slg::ocl::ToFloat3(p));
}

bool slg::SphericalTriangle_UseSolidAngle(const float solidAngle) {
	return slg::ocl::SphericalTriangle_UseSolidAngle(solidAngle);
}

bool slg::SphericalTriangle_Sample(const Point &v0, const Point &v1, const Point &v2,
		const Point &p, const float u, float *b0, float *b1, float *b2, float *pdfW) {
	float s, t;
	slg::ocl::SphericalTriangle_SplitSample(u, &s, &t);

	return slg::ocl::SphericalTriangle_Sample(slg::ocl::ToFloat3(v0), slg::ocl::ToFloat3(v1),
			slg::ocl::ToFloat3(v2), slg::ocl::ToFloat3(p), s, t, b0, b1, b2, pdfW);
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
