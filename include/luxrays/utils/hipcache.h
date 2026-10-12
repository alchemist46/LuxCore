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

#if !defined(LUXRAYS_DISABLE_HIP)

#include <mutex>

namespace luxrays {

// Compiles programs with hiprtc and keeps the resulting code objects in a
// persistent cache, one per GPU architecture.
//
// Unlike the CUDA cache, the target architecture is given explicitly: code
// objects are compiled without any HIP device being current, so the parallel
// compilation can run on any thread. Only the module loading, which requires
// the device to be current, is done on the calling thread.
class hipKernelPersistentCache {
public:
	// The largest work group the compiled kernels can be launched with
	static constexpr u_int MaxWorkGroupSize = 256;

	hipKernelPersistentCache(const std::string &applicationName, const std::string &gpuArch);
	~hipKernelPersistentCache();

	// Returns the code object of the program, compiling it if it isn't cached
	bool CompileCodeObject(const std::vector<std::string> &kernelsParameters,
		const std::string &kernelSource, const std::string &programName,
		std::unique_ptr<char[]> *code, size_t *codeSize, bool *cached, std::string *error);

	// Returns a module loaded on the current device, nullptr on compilation error
	hipModule_t Compile(const std::vector<std::string> &kernelsParameters,
		const std::string &kernelSource, const std::string &programName,
		bool *cached, std::string *error);

	// Compiles all the programs in parallel and loads them on the current device
	std::vector<hipModule_t> CompileMultiple(
		const std::vector<std::tuple<std::vector<std::string>, std::string, std::string>> &kernels,
		std::vector<bool> *cached = nullptr,
		std::vector<std::string> *errors = nullptr
	);

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
		std::unique_ptr<char[]> *code, size_t *codeSize, std::string *error);

private:
	std::string appName, arch;

	std::mutex cacheMutex;
};

}

#endif

#endif	/* _LUXRAYS_HIPCACHE_H */
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
