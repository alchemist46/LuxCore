/***************************************************************************
 * Copyright 1998-2026 by authors (see AUTHORS.txt)
 *
 *   This file is part of LuxCoreRender.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ***************************************************************************/

// Tool used at build time to pre-compile the HIP path kernels. It writes, for
// each kernel variant, the exact HIP source and compiler options used at run
// time, so that the key of the baked kernel matches the one computed at run
// time (see luxrays/utils/hipkernel.h).
//
// The generated .hip files are compiled by the offline clang of the HIP SDK
// or ROCm, with the runtime header of hiprtc (see --hiprtc-probe below), one
// code object per variant and family of architectures (see
// build-system/cmake/Scripts/CompileHIP.cmake), and embedded in LuxCore
// binaries (see src/slg/CMakeLists.txt).
//
// The variants are the ones of PathOCLBaseOCLRenderThread::InitKernels() on a
// HIP device: the render engine (PATHOCL and RTPATHOCL, the ones used by
// BlendLuxCore; TILEPATHOCL is compiled at run time), the private or global
// evaluation stacks and the subsurface random walk. The working colour space
// and the epsilons are kernel constants (PARAM_KERNEL_CONSTANTS), so they
// don't add variants.

#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include <hipew.h>

#include "luxrays/kernels/kernels.h"
#include "luxrays/utils/hipkernel.h"

#include "slg/kernels/kernels.h"
#include "slg/kernels/kernelsources.h"

using namespace std;
using namespace luxrays;
using namespace slg;
using namespace slg::ocl;

// Write a file only if its content changed: the tool runs again whenever it is
// rebuilt (e.g. after any change in luxrays), and a new time stamp would
// compile all the kernels again
static bool WriteFile(const string &fileName, const string &content) {
	{
		ifstream oldFile(fileName, ifstream::in | ifstream::binary);
		if (oldFile) {
			const string oldContent((istreambuf_iterator<char>(oldFile)), istreambuf_iterator<char>());
			if (oldContent == content)
				return true;
		}
	}

	ofstream file(fileName, ofstream::out | ofstream::binary | ofstream::trunc);
	if (!file) {
		cerr << "Unable to write " << fileName << endl;
		return false;
	}
	file << content;
	file.close();

	return !file.fail();
}

// Compile a small program with hiprtc. The offline clang must compile the
// kernels with the header hiprtc includes implicitly (hiprtc_runtime.h, not the
// HIP runtime header of the SDK: they define different types and functions),
// and that header exists only inside the hiprtc library. comgr writes it in
// its temporary directory, which it keeps with AMD_COMGR_SAVE_TEMPS=1: see
// build-system/cmake/Scripts/ExtractHiprtcHeader.cmake.
static int HiprtcProbe() {
	if (hipewInit(HIPEW_INIT_HIPRTC) != HIPEW_SUCCESS) {
		cerr << "The hiprtc library is required to bake the HIP kernels (with the HIP SDK "
				"or ROCm clang) and it was not found" << endl;
		return EXIT_FAILURE;
	}
	cout << "hiprtc library: " << hipewHiprtcLibraryName() << endl;

	const char *source = "extern \"C\" __global__ void Probe(float *a) { a[threadIdx.x] = sqrtf(a[threadIdx.x]); }\n";
	hiprtcProgram prog;
	if (hiprtcCreateProgram(&prog, source, "probe", 0, nullptr, nullptr) != HIPRTC_SUCCESS) {
		cerr << "hiprtcCreateProgram() failed" << endl;
		return EXIT_FAILURE;
	}
	const char *opts[] = { "--offload-arch=gfx1030", "-O3" };
	const hiprtcResult result = hiprtcCompileProgram(prog, 2, opts);
	hiprtcDestroyProgram(&prog);
	if (result != HIPRTC_SUCCESS) {
		cerr << "hiprtcCompileProgram() failed: " << hiprtcGetErrorString(result) << endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}

int main(int argc, char *argv[]) {
	if ((argc == 2) && (string(argv[1]) == "--hiprtc-probe"))
		return HiprtcProbe();

	if (argc != 2) {
		cerr << "Usage: " << argv[0] << " <output dir>" << endl;
		cerr << "       " << argv[0] << " --hiprtc-probe" << endl;
		return EXIT_FAILURE;
	}

	const string outDir = argv[1];

	// The exact source compiled at run time: HIP devices always compile the
	// monolithic source (i.e. with the micro-kernels)
	const string hipSource = GetHIPKernelSource(
			GetPathOCLBaseKernelSources() +
			KernelSource_pathoclbase_kernels_micro);

	const string renderEngineTypes[] = { "PATHOCL", "RTPATHOCL" };

	string manifest;
	for (const string &renderEngineType : renderEngineTypes) {
		for (const bool privateEvalStack : { false, true }) {
			for (const bool subsurface : { false, true }) {
				// The parameters of the program, in the order of
				// PathOCLBaseOCLRenderThread::InitKernels() and
				// HIPDevice::AddKernelOpts(). Note: the epsilon values are not
				// used with the kernel constants.
				vector<string> params;
				GetKernelParamters(params, renderEngineType, 0.f, 0.f, true);
				AddPathKernelVariantParameters(params, privateEvalStack, subsurface);
				const vector<string> deviceParams = GetHIPDeviceKernelParameters();
				params.insert(params.end(), deviceParams.begin(), deviceParams.end());
				const vector<string> compileOpts = GetHIPPathCompileOptions();
				params.insert(params.end(), compileOpts.begin(), compileOpts.end());

				// The file names must be stable (i.e. independent of the key) in
				// order to be known at CMake configuration time
				string stableName = renderEngineType;
				transform(stableName.begin(), stableName.end(), stableName.begin(), ::tolower);
				if (privateEvalStack)
					stableName += "_private";
				if (subsurface)
					stableName += "_subsurface";
				const string baseName = outDir + "/" + stableName;

				if (!WriteFile(baseName + ".hip", hipSource))
					return EXIT_FAILURE;

				// The compiler options: the hashed ones (i.e. without the target
				// architecture), one per line
				vector<string> opts = GetHIPBaseCompilerOptions();
				opts.insert(opts.end(), params.begin(), params.end());
				string optsText;
				for (const string &opt : opts)
					optsText += opt + "\n";
				if (!WriteFile(baseName + ".params", optsText))
					return EXIT_FAILURE;

				manifest += stableName + " " + GetHIPBakedKernelKey(params, hipSource) + "\n";
			}
		}
	}

	if (!WriteFile(outDir + "/hipmanifest.txt", manifest))
		return EXIT_FAILURE;

	return EXIT_SUCCESS;
}
