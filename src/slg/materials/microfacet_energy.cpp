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

luxrays::Spectrum slg::Microfacet_MSScale(const bool multiscattering,
		const float u, const float v, const float mu, const luxrays::Spectrum &Fss) {
	if (multiscattering)
		return GGXEnergy_ReflectScale(u, v, mu, Fss, false);
	else
		return luxrays::Spectrum(1.f);
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4

luxrays::Spectrum slg::GGXEnergy_CoatE(const float u, const float v, const float mu,
		const luxrays::Spectrum &F0) {
	return luxrays::Spectrum(
			slg::ocl::GGXEnergy_CoatE(u, v, mu, F0.c[0]),
			slg::ocl::GGXEnergy_CoatE(u, v, mu, F0.c[1]),
			slg::ocl::GGXEnergy_CoatE(u, v, mu, F0.c[2]));
}

luxrays::Spectrum slg::GGXEnergy_CoatEavg(const float u, const float v, const luxrays::Spectrum &F0) {
	return luxrays::Spectrum(
			slg::ocl::GGXEnergy_CoatEavg(u, v, F0.c[0]),
			slg::ocl::GGXEnergy_CoatEavg(u, v, F0.c[1]),
			slg::ocl::GGXEnergy_CoatEavg(u, v, F0.c[2]));
}
