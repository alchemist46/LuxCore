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

#ifndef _LUXRAYS_HIPCACHE_H
#define	_LUXRAYS_HIPCACHE_H

#include <string>
#include <vector>
#include <tuple>
#include <memory>
#include <filesystem>

#include "luxrays/luxrays.h"
#include "luxrays/utils/hip.h"
#include "luxrays/utils/hipkernel.h"

#if !defined(LUXRAYS_DISABLE_HIP)

#include <mutex>

namespace luxrays {

// Loads programs from the precompiled kernels embedded in the binaries or
// compiles them with hiprtc, and keeps the compiled code objects in a
// persistent cache, one per GPU architecture. A program is looked for, in
// order:
// 1. in the embedded kernels (see embeddedhip.h), when the architecture
//    belongs to a baked family. A code object that fails to load (e.g. an
//    old runtime rejecting the code object v6 or the generic targets) falls
//    through to the next step;
// 2. in the persistent cache;
// 3. it is compiled with hiprtc, if available (it is optional: the HIP SDK on
//    Windows, the ROCm hiprtc library on Linux).
//
// Unlike the CUDA cache, the target architecture is given explicitly: code
// objects are compiled without any HIP device being current, so the parallel
// compilation can run on any thread. Only the module loading, which requires
// the device to be current, is done on the calling thread.
class hipKernelPersistentCache {
public:
	// The largest work group the compiled kernels can be launched with
	static constexpr u_int MaxWorkGroupSize = HIPMaxWorkGroupSize;

	// Where the code object of a program comes from
	enum class CodeOrigin { COMPILED, CACHED, PRECOMPILED };

	hipKernelPersistentCache(const std::string &applicationName, const std::string &gpuArch);
	~hipKernelPersistentCache();

	// Returns the code object of the program, compiling it if it isn't cached
	// (the embedded kernels are not looked for). The extra options are passed
	// to hiprtc but they are not part of the key (i.e. include directories).
	bool CompileCodeObject(const std::vector<std::string> &kernelsParameters,
		const std::string &kernelSource, const std::string &programName,
		std::unique_ptr<char[]> *code, size_t *codeSize, bool *cached, std::string *error,
		const std::vector<std::string> &extraOptions = {});

	// Returns a module loaded on the current device, nullptr on compilation error
	hipModule_t Compile(const std::vector<std::string> &kernelsParameters,
		const std::string &kernelSource, const std::string &programName,
		CodeOrigin *origin, std::string *error,
		const std::vector<std::string> &extraOptions = {});

	// How the code object of a program was obtained, for the log: empty,
	// " (cached)" or " (precompiled kernels, <family>)"
	std::string GetCodeOriginDesc(const CodeOrigin origin) const;

	// Compiles all the programs in parallel and loads them on the current device
	std::vector<hipModule_t> CompileMultiple(
		const std::vector<std::tuple<std::vector<std::string>, std::string, std::string>> &kernels,
		std::vector<CodeOrigin> *origins = nullptr,
		std::vector<std::string> *errors = nullptr
	);

	// The family of architectures of the embedded kernels used for this
	// architecture, empty if they are not used (none for this architecture,
	// none embedded, or disabled)
	const std::string &GetPrecompiledFamily() const { return precompiledFamily; }
	// True if there is an embedded code object for the program
	bool HasPrecompiled(const std::vector<std::string> &kernelsParameters,
		const std::string &kernelSource) const;

	const std::string &GetApplicationName() const { return appName; }
	const std::string &GetGPUArch() const { return arch; }

	static std::filesystem::path GetCacheDir(const std::string &applicationName);

	// Clear all cached kernels for this application
	void ClearCache();

	// Clear all HIP kernel caches (all applications)
	static void ClearAllCaches();

	// The options used for all the programs, in addition to the program ones
	static std::vector<std::string> GetCompilerOptions(const std::string &gpuArch);

	static bool ForcedCompileCodeObject(const std::vector<std::string> &kernelsParameters,
		const std::string &kernelSource, const std::string &programName,
		const std::string &gpuArch,
		std::unique_ptr<char[]> *code, size_t *codeSize, std::string *error,
		const std::vector<std::string> &extraOptions = {});

private:
	// The embedded code object of the program, nullptr if there is none
	const unsigned char *GetPrecompiled(const std::vector<std::string> &kernelsParameters,
		const std::string &kernelSource) const;
	// Loads the embedded code object of the program on the current device.
	// Returns nullptr if there is none or if it fails to load (with a warning).
	hipModule_t LoadPrecompiled(const std::vector<std::string> &kernelsParameters,
		const std::string &kernelSource, std::string *warning) const;

	std::string appName, arch, precompiledFamily;

	std::mutex cacheMutex;
};

}

#endif

#endif	/* _LUXRAYS_HIPCACHE_H */
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
