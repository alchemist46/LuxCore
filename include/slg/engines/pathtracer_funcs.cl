#line 2 "pathtracer_funcs.cl"

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

// The path tracer code shared by the CPU engines (compiled through the
// OpenCL emulation, see pathtracer.cpp) and the PathOCL kernels.

//------------------------------------------------------------------------------
// Russian roulette
//------------------------------------------------------------------------------

// The probability of continuing the eye path after a bounce. Both the path
// continuation and the MIS weight of the direct light sampling (the pdf of the
// BSDF technique includes the probability of reaching the light) call this
// function with the same arguments, so the two MIS weights keep summing to one.
//
// bounceWeight is the sample weight of the bounce, f cos / pdf; pathThroughput
// the throughput of the path before the bounce; etaScale the product of the
// eta^2 of the refractions of the path including the bounce, which undoes the
// radiance scaling of the dielectrics so that the test sees energy, not
// radiance (pbrt-v4).
//
// RR_BOUNCE (LuxCore's classic rule): the mean of the bounce weight, clamped
// to [importanceCap, 1]. The accumulated throughput plays no role, so a bright
// path whose last bounce was dim is terminated as often as a dim one.
//
// RR_PATH (pbrt-v4, Mitsuba): the largest component of the throughput after
// the bounce, clamped to [importanceCap, 1]. The surviving paths carry a
// throughput whose largest component is at least one, so the weights stay
// bounded even with a cap of 0.
//
// RR_PATH_SQRT (Cycles): the square root of the same quantity, which
// terminates later on average.
OPENCL_FORCE_INLINE float PathTracer_RussianRouletteProb(const RussianRouletteType type,
		const float importanceCap, const float3 bounceWeight,
		const float3 pathThroughput, const float etaScale) {
	if (type == RR_BOUNCE)
		return clamp((bounceWeight.x + bounceWeight.y + bounceWeight.z) * (1.f / 3.f), importanceCap, 1.f);

	const float3 beta = pathThroughput * bounceWeight * etaScale;
	const float m = fmax(beta.x, fmax(beta.y, beta.z));

	return clamp((type == RR_PATH_SQRT) ? sqrt(m) : m, fmax(importanceCap, 1e-6f), 1.f);
}
