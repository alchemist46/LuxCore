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

// The directional albedo tables of the GGX lobes and their lookups, compiled
// from the kernel source shared with the GPU (single-source, see
// luxrays/devices/cpudevice_oclemul.h)

#include <cmath>

#include "slg/materials/material.h"

namespace slg { namespace ocl { namespace {
#include "luxrays/devices/cpudevice_oclemul.h"
#include "slg/materials/microfacet_energy_tables.cl"

// GGXEnergy_CoatE() and GGXEnergy_CoatEavg() for the three channels of F0: the
// table lookups do not depend on F0, so they are done once instead of once
// per channel (the Spectrum versions are called 3 times per Glossy2,
// GlossyCoating and GlossyTranslucent evaluation). The arithmetic is the one
// of GGXEnergy_CoatE(), GGXEnergy_CoatEavg() and GGXEnergy_CoatFms() in the
// same order, so the results are bit-identical.

float CoatFmsFromEavg(const float Eavg, const float F0) {
	const float Fss = F0 + (1.f - F0) * (1.f / 21.f);

	return Fss * Eavg / fmax(1.f - Fss * (1.f - Eavg), 1e-4f);
}

void CoatE3(const float u, const float v, const float mu, const float F0[3], float result[3]) {
	const float rough = sqrt(sqrt(u * v));
	const float E = GGXEnergy_Lerp2D(GGXEnergy_E_HC, GGXENERGY_N_ROUGH, GGXENERGY_N_MU, rough, mu);
	const float B = GGXEnergy_Lerp2D(GGXEnergy_SchlickB_HC, GGXENERGY_N_ROUGH, GGXENERGY_N_MU, rough, mu);
	const float Eavg = GGXEnergy_Lerp1D(GGXEnergy_Eavg_HC, GGXENERGY_N_ROUGH, rough);

	for (uint i = 0; i < 3; ++i) {
		const float EF = F0[i] * (E - B) + B;
		result[i] = EF * (1.f + CoatFmsFromEavg(Eavg, F0[i]) * (1.f - E) / fmax(E, 1e-4f));
	}
}

void CoatEavg3(const float u, const float v, const float F0[3], float result[3]) {
	const float rough = sqrt(sqrt(u * v));
	const float P = GGXEnergy_Lerp1D(&GGXEnergy_CoatAvg_HC[0 * GGXENERGY_N_ROUGH], GGXENERGY_N_ROUGH, rough);
	const float Q = GGXEnergy_Lerp1D(&GGXEnergy_CoatAvg_HC[1 * GGXENERGY_N_ROUGH], GGXENERGY_N_ROUGH, rough);
	const float R = GGXEnergy_Lerp1D(&GGXEnergy_CoatAvg_HC[2 * GGXENERGY_N_ROUGH], GGXENERGY_N_ROUGH, rough);
	const float S = GGXEnergy_Lerp1D(&GGXEnergy_CoatAvg_HC[3 * GGXENERGY_N_ROUGH], GGXENERGY_N_ROUGH, rough);
	const float Eavg = GGXEnergy_Lerp1D(GGXEnergy_Eavg_HC, GGXENERGY_N_ROUGH, rough);

	for (uint i = 0; i < 3; ++i)
		result[i] = F0[i] * P + Q + CoatFmsFromEavg(Eavg, F0[i]) * (F0[i] * R + S);
}
} } }

float slg::GGXEnergy_E(const float u, const float v, const float mu, const bool separable) {
	return slg::ocl::GGXEnergy_E(u, v, mu, separable);
}

float slg::GGXEnergy_Eavg(const float u, const float v, const bool separable) {
	return slg::ocl::GGXEnergy_Eavg(u, v, separable);
}

float slg::GGXEnergy_GlassE(const float u, const float v, const float mu, const float eta) {
	return slg::ocl::GGXEnergy_GlassE(u, v, mu, eta);
}

float slg::GGXEnergy_ReflectScale(const float u, const float v, const float mu,
		const float Fss, const bool separable) {
	return slg::ocl::GGXEnergy_ReflectScale(slg::ocl::GGXEnergy_E(u, v, mu, separable),
			slg::ocl::GGXEnergy_Eavg(u, v, separable), Fss);
}

luxrays::Spectrum slg::GGXEnergy_ReflectScale(const float u, const float v, const float mu,
		const luxrays::Spectrum &Fss, const bool separable) {
	const float E = slg::ocl::GGXEnergy_E(u, v, mu, separable);
	const float Eavg = slg::ocl::GGXEnergy_Eavg(u, v, separable);

	return luxrays::Spectrum(
			slg::ocl::GGXEnergy_ReflectScale(E, Eavg, Fss.c[0]),
			slg::ocl::GGXEnergy_ReflectScale(E, Eavg, Fss.c[1]),
			slg::ocl::GGXEnergy_ReflectScale(E, Eavg, Fss.c[2]));
}

float slg::SchlickFresnel_Average(const float F0) {
	return F0 + (1.f - F0) * (1.f / 21.f);
}

luxrays::Spectrum slg::SchlickFresnel_Average(const luxrays::Spectrum &F0) {
	return F0 + (luxrays::Spectrum(1.f) - F0) * (1.f / 21.f);
}

luxrays::Spectrum slg::Microfacet_MSScale(
		const float u, const float v, const float mu, const luxrays::Spectrum &Fss) {
	return GGXEnergy_ReflectScale(u, v, mu, Fss, false);
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4

luxrays::Spectrum slg::GGXEnergy_CoatE(const float u, const float v, const float mu,
		const luxrays::Spectrum &F0) {
	luxrays::Spectrum result;
	slg::ocl::CoatE3(u, v, mu, F0.c, result.c);

	return result;
}

luxrays::Spectrum slg::GGXEnergy_CoatEavg(const float u, const float v, const luxrays::Spectrum &F0) {
	luxrays::Spectrum result;
	slg::ocl::CoatEavg3(u, v, F0.c, result.c);

	return result;
}
