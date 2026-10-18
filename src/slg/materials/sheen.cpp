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

// The microfiber sheen lobe and its LTC table, compiled from the kernel
// sources shared with the GPU (single-source, see
// luxrays/devices/cpudevice_oclemul.h)

#include <cmath>

#include "slg/materials/sheen.h"

namespace slg { namespace ocl { namespace {
#include "luxrays/devices/cpudevice_oclemul.h"
#include "slg/materials/sheen_ltc_tables.cl"
#include "slg/materials/sheen_funcs.cl"
} } }

using namespace luxrays;

void slg::Sheen_Lookup(const float mu, const float roughness,
		float *a, float *b, float *albedo) {
	slg::ocl::SheenLTC_Lookup(mu, roughness, a, b, albedo);
}

float slg::Sheen_LTC(const float a, const float b,
		const Vector &eyeDir, const Vector &lightDir) {
	return slg::ocl::Sheen_LTC(a, b, slg::ocl::make_float3(eyeDir.x, eyeDir.y, eyeDir.z),
			slg::ocl::make_float3(lightDir.x, lightDir.y, lightDir.z));
}

Vector slg::Sheen_SampleLTC(const float a, const float b,
		const Vector &eyeDir, const float u0, const float u1) {
	const auto d = slg::ocl::Sheen_SampleLTC(a, b, slg::ocl::make_float3(eyeDir.x, eyeDir.y, eyeDir.z), u0, u1);
	return Vector(d.x, d.y, d.z);
}
