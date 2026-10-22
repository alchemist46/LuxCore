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

#include "slg/materials/thinfilmcoating.h"

// The film model of the kernels (materialdefs_funcs_thinfilmcoating.cl),
// compiled for the CPU
namespace slg { namespace ocl { namespace {
#include "luxrays/devices/cpudevice_oclemul.h"

#define BLACK MAKE_FLOAT3(0.f, 0.f, 0.f)
#define WHITE MAKE_FLOAT3(1.f, 1.f, 1.f)
#include "luxrays/core/utils_funcs.cl"
#include "luxrays/core/color/color_funcs.cl"
#include "slg/materials/materialdefs_funcs_thinfilmcoating.cl"
#undef BLACK
#undef WHITE
} } }

using namespace std;
using namespace luxrays;
using namespace slg;

//---------------------------------------------------------------------------------
// Thin film coating (utility functions, to be used in materials where appropriate)
//---------------------------------------------------------------------------------

static slg::ocl::float3 ToFloat3(const Vector &v) {
	return slg::ocl::MAKE_FLOAT3(v.x, v.y, v.z);
}

static slg::ocl::float3 ToFloat3(const Spectrum &c) {
	return slg::ocl::MAKE_FLOAT3(c.c[0], c.c[1], c.c[2]);
}

static Spectrum ToSpectrum(const slg::ocl::float3 &c) {
	return Spectrum(c.x, c.y, c.z);
}

// The kernel functions give Linear Rec.709 on the CPU
// (WorkingColorSpace_FromRec709() is the identity there)
Spectrum slg::CalcFilmColor(const Vector &localFixedDir, const float filmThickness, const float filmIOR,
		const float iorIncident, const float iorSubstrate, const WorkingColorSpace &workingColorSpace) {
	return workingColorSpace.FromRec709(ToSpectrum(slg::ocl::CalcFilmColor(ToFloat3(localFixedDir),
			filmThickness, filmIOR, iorIncident, iorSubstrate)));
}

Spectrum slg::CalcFilmColorConductor(const Vector &localFixedDir, const float filmThickness, const float filmIOR,
		const float iorIncident, const Spectrum &eta, const Spectrum &k,
		const WorkingColorSpace &workingColorSpace) {
	return workingColorSpace.FromRec709(ToSpectrum(slg::ocl::CalcFilmColorConductor(ToFloat3(localFixedDir),
			filmThickness, filmIOR, iorIncident, ToFloat3(eta), ToFloat3(k))));
}

void slg::FilmConductorIOR(const Spectrum &f0, const Spectrum &edgeTint, Spectrum *eta, Spectrum *k) {
	slg::ocl::float3 e, kk;
	slg::ocl::ThinFilm_ConductorIOR(ToFloat3(f0), ToFloat3(edgeTint), &e, &kk);
	*eta = ToSpectrum(e);
	*k = ToSpectrum(kk);
}

// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
