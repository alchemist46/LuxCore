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

#include <stdexcept>

#include "slg/film/film.h"
#include "slg/kernels/kernels.h"
#include "slg/film/imagepipeline/plugins/workingtorec709.h"
#include "luxrays/kernels/kernels.h"

using namespace std;
using namespace luxrays;
using namespace slg;

//------------------------------------------------------------------------------
// Working colour space to Linear Rec.709 plugin
//------------------------------------------------------------------------------

BOOST_CLASS_EXPORT_IMPLEMENT(slg::WorkingToRec709Plugin)

WorkingToRec709Plugin::WorkingToRec709Plugin() {
	applyKernel = nullptr;
}

WorkingToRec709Plugin::WorkingToRec709Plugin(const WorkingColorSpace &workingColorSpace) {
	const float (&toRec709)[3][3] = workingColorSpace.GetToRec709Matrix();
	for (u_int y = 0; y < 3; ++y)
		for (u_int x = 0; x < 3; ++x)
			m.push_back(toRec709[y][x]);

	applyKernel = nullptr;
}

WorkingToRec709Plugin::~WorkingToRec709Plugin() {
}

ImagePipelinePlugin *WorkingToRec709Plugin::Copy() const {
	WorkingToRec709Plugin *plugin = new WorkingToRec709Plugin();
	plugin->m = m;

	return plugin;
}

//------------------------------------------------------------------------------
// CPU version
//------------------------------------------------------------------------------

void WorkingToRec709Plugin::Apply(Film &film, const u_int index) {
	Spectrum *pixels = (Spectrum *)film.channel_IMAGEPIPELINEs[index]->GetPixels();
	const u_int pixelCount = film.GetWidth() * film.GetHeight();

	const bool hasPN = film.HasChannel(Film::RADIANCE_PER_PIXEL_NORMALIZED);
	const bool hasSN = film.HasChannel(Film::RADIANCE_PER_SCREEN_NORMALIZED);

	#pragma omp parallel for
	for (
			// Visual C++ 2013 supports only OpenMP 2.5
#if _OPENMP >= 200805
			unsigned
#endif
			int i = 0; i < pixelCount; ++i) {
		if (film.HasSamples(hasPN, hasSN, i)) {
			const Spectrum c = pixels[i];

			pixels[i].c[0] = m[0] * c.c[0] + m[1] * c.c[1] + m[2] * c.c[2];
			pixels[i].c[1] = m[3] * c.c[0] + m[4] * c.c[1] + m[5] * c.c[2];
			pixels[i].c[2] = m[6] * c.c[0] + m[7] * c.c[1] + m[8] * c.c[2];
		}
	}
}

//------------------------------------------------------------------------------
// HardwareDevice version
//------------------------------------------------------------------------------

void WorkingToRec709Plugin::AddHWChannelsUsed(unordered_set<Film::FilmChannelType, hash<int> > &hwChannelsUsed) const {
	hwChannelsUsed.insert(Film::IMAGEPIPELINE);
}

void WorkingToRec709Plugin::ApplyHW(Film &film, const u_int index) {
	HardwareDevice *hardwareDevice = film.hardwareDevice;

	if (!applyKernel) {
		film.ctx->SetVerbose(true);

		// Compile sources
		const double tStart = WallClockTime();

		vector<string> opts;
		opts.push_back("-D LUXRAYS_OPENCL_KERNEL");
		opts.push_back("-D SLG_OPENCL_KERNEL");

		auto program = hardwareDevice->CompileProgram(
				opts,
				luxrays::ocl::KernelSource_utils_funcs +
				slg::ocl::KernelSource_plugin_workingtorec709_funcs,
				"WorkingToRec709Plugin");

		SLG_LOG("[WorkingToRec709Plugin] Compiling WorkingToRec709Plugin_Apply Kernel");
		applyKernel = hardwareDevice->GetKernel(*program, "WorkingToRec709Plugin_Apply");

		// Set kernel arguments
		u_int argIndex = 0;
		hardwareDevice->SetKernelArg(applyKernel, argIndex++, film.GetWidth());
		hardwareDevice->SetKernelArg(applyKernel, argIndex++, film.GetHeight());
		hardwareDevice->SetKernelArg(applyKernel, argIndex++, film.hw_IMAGEPIPELINE);
		for (u_int i = 0; i < 9; ++i)
			hardwareDevice->SetKernelArg(applyKernel, argIndex++, m[i]);

		const double tEnd = WallClockTime();
		SLG_LOG("[WorkingToRec709Plugin] Kernels compilation time: " << int((tEnd - tStart) * 1000.0) << "ms");

		film.ctx->SetVerbose(false);
	}

	hardwareDevice->EnqueueKernel(applyKernel, HardwareDeviceRange(RoundUp(film.GetWidth() * film.GetHeight(), 256u)),
			HardwareDeviceRange(256));
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
