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

#ifndef _SLG_SHEEN_H
#define	_SLG_SHEEN_H

#include "luxrays/core/geometry/vector.h"

namespace slg {

//------------------------------------------------------------------------------
// The microfiber sheen lobe of Zeltner, Burley & Chiang 2022 (see
// sheen_funcs.cl and sheen_ltc_tables.cl, compiled for the CPU in sheen.cpp):
// f(eye, light) |cos light| = albedo(mu_eye, r) * Sheen_LTC(a, b, eye, light).
// The directions are local, the shading normal is z, both above the surface.
//------------------------------------------------------------------------------

// The LTC parameters and the directional albedo for the view cosine mu and
// the roughness (clamped to [1e-3, 1])
extern void Sheen_Lookup(const float mu, const float roughness,
		float *a, float *b, float *albedo);
// The LTC density of lightDir (also the pdf of Sheen_SampleLTC())
extern float Sheen_LTC(const float a, const float b,
		const luxrays::Vector &eyeDir, const luxrays::Vector &lightDir);
// A light direction distributed as the LTC (it can be below the surface)
extern luxrays::Vector Sheen_SampleLTC(const float a, const float b,
		const luxrays::Vector &eyeDir, const float u0, const float u1);

}

#endif	/* _SLG_SHEEN_H */
