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

#include <memory>
#if !defined(LUXRAYS_DISABLE_HIP)

#include <fstream>
#include <mutex>
#include <vector>
#include <oneapi/tbb.h>

#include "luxrays/luxrays.h"
#include "luxrays/utils/utils.h"
#include "luxrays/utils/config.h"
#include "luxrays/utils/hipcache.h"
#include "luxrays/utils/oclcache.h"

namespace luxrays {

//------------------------------------------------------------------------------
// hipKernelPersistentCache
//------------------------------------------------------------------------------

hipKernelPersistentCache::hipKernelPersistentCache(const std::string &applicationName,
		const std::string &gpuArch) : appName(applicationName), arch(gpuArch) {
	// Create the cache directory
	std::filesystem::create_directories(GetCacheDir(appName));
}

hipKernelPersistentCache::~hipKernelPersistentCache() {
}

std::filesystem::path hipKernelPersistentCache::GetCacheDir(const std::string &applicationName) {
	return luxrays::GetCacheDir() / "hip_kernel_cache" / SanitizeFileName(applicationName);
}

std::vector<std::string> hipKernelPersistentCache::GetCompilerOptions(const std::string &gpuArch) {
	return {
		// Compile for the architecture of the device the program will run on
		"--offload-arch=" + gpuArch,
		"-O3",
		// The OpenCL-style sources produce a lot of harmless warnings once
		// compiled as HIP (unused variables, sign conversions, etc.)
		"-w",
		// By default, kernels are compiled to run work groups of up to 1024
		// threads, which halves the registers available to each thread. The
		// render engines use much smaller work groups and the kernels are
		// register hungry: on gfx1030, this is about 20% faster (the same
		// speed of the AMD OpenCL compiler) and spills less to the stack.
		"--gpu-max-threads-per-block=" + std::to_string(MaxWorkGroupSize)
	};
}

bool hipKernelPersistentCache::ForcedCompileCodeObject(
	const std::vector<std::string> &kernelsParameters, const std::string &kernelSource,
	const std::string &programName, const std::string &gpuArch,
	std::unique_ptr<char[]> *code, size_t *codeSize, std::string *error
) {
	if (error)
		*error = "";

	hiprtcProgram prog;
	CHECK_HIPRTC_ERROR(hiprtcCreateProgram(&prog, kernelSource.c_str(), programName.c_str(), 0, nullptr, nullptr));

	const std::vector<std::string> compilerOpts = GetCompilerOptions(gpuArch);

	std::vector<const char *> hipOpts;
	for (auto const &o : compilerOpts)
		hipOpts.push_back(o.c_str());
	for (auto const &p : kernelsParameters)
		hipOpts.push_back(p.c_str());

	const hiprtcResult compilationResult = hiprtcCompileProgram(prog,
			hipOpts.size(), hipOpts.data());

	size_t logSize;
	CHECK_HIPRTC_ERROR(hiprtcGetProgramLogSize(prog, &logSize));
	if (logSize > 1) {
		auto log = std::make_unique<char[]>(logSize);
		CHECK_HIPRTC_ERROR(hiprtcGetProgramLog(prog, log.get()));
		if (error)
			*error = std::string(log.get());
	}

	if (compilationResult != HIPRTC_SUCCESS) {
		hiprtcDestroyProgram(&prog);
		return false;
	}

	CHECK_HIPRTC_ERROR(hiprtcGetCodeSize(prog, codeSize));
	*code = std::make_unique<char[]>(*codeSize);
	CHECK_HIPRTC_ERROR(hiprtcGetCode(prog, code->get()));

	CHECK_HIPRTC_ERROR(hiprtcDestroyProgram(&prog));

	return true;
}

bool hipKernelPersistentCache::CompileCodeObject(const std::vector<std::string> &kernelsParameters,
		const std::string &kernelSource, const std::string &programName,
		std::unique_ptr<char[]> *code, size_t *codeSize, bool *cached, std::string *error) {
	if (error)
		*error = "";
	*cached = false;

	// Check if the kernel is available in the cache. The compiler options are
	// part of the hash, so a change of them invalidates the cached kernels.
	std::vector<std::string> allOpts = GetCompilerOptions(arch);
	allOpts.insert(allOpts.end(), kernelsParameters.begin(), kernelsParameters.end());
	const std::string kernelName =
			oclKernelPersistentCache::HashString(oclKernelPersistentCache::ToOptsString(allOpts))
			+ "-" +
			oclKernelPersistentCache::HashString(kernelSource) +
			"_" + SanitizeFileName(arch) + ".co";
	const std::filesystem::path dirPath = GetCacheDir(appName);
	const std::filesystem::path filePath = dirPath / kernelName;

	{
		std::lock_guard<std::mutex> lock(cacheMutex);

		if (std::filesystem::exists(filePath)) {
			const size_t fileSize = std::filesystem::file_size(filePath);

			bool valid = false;
			if (fileSize > sizeof(u_int)) {
				*codeSize = fileSize - sizeof(u_int);
				*code = std::make_unique<char[]>(*codeSize);

				std::ifstream file(filePath, std::ifstream::in | std::ifstream::binary);

				// Read the binary hash
				u_int hashBin;
				file.read((char *)&hashBin, sizeof(u_int));
				file.read(code->get(), *codeSize);
				if (file.fail())
					throw std::runtime_error("Unable to read kernel file cache " + filePath.generic_string());
				file.close();

				valid = (hashBin == oclKernelPersistentCache::HashBin(code->get(), *codeSize));
			}

			if (valid) {
				*cached = true;
				return true;
			}

			// Something is wrong in the file, remove it and compile again
			std::filesystem::remove(filePath);
		}
	}

	// It isn't available, compile the source
	if (!ForcedCompileCodeObject(kernelsParameters, kernelSource, programName, arch, code, codeSize, error))
		return false;

	// Add the kernel to the cache
	std::lock_guard<std::mutex> lock(cacheMutex);

	std::filesystem::create_directories(dirPath);

	std::ofstream file(filePath, std::ofstream::out | std::ofstream::binary | std::ofstream::trunc);

	// Write the binary hash
	const u_int hashBin = oclKernelPersistentCache::HashBin(code->get(), *codeSize);
	file.write((char *)&hashBin, sizeof(u_int));
	file.write(code->get(), *codeSize);
	if (file.fail())
		throw std::runtime_error("Unable to write kernel file cache " + filePath.generic_string());
	file.close();

	return true;
}

hipModule_t hipKernelPersistentCache::Compile(const std::vector<std::string> &kernelsParameters,
		const std::string &kernelSource, const std::string &programName,
		bool *cached, std::string *error) {
	std::unique_ptr<char[]> code;
	size_t codeSize;
	if (!CompileCodeObject(kernelsParameters, kernelSource, programName, &code, &codeSize, cached, error))
		return nullptr;

	hipModule_t module;
	CHECK_HIP_ERROR(hipModuleLoadData(&module, code.get()));

	return module;
}

std::vector<hipModule_t> hipKernelPersistentCache::CompileMultiple(
		const std::vector<std::tuple<std::vector<std::string>, std::string, std::string>> &kernels,
		std::vector<bool> *cached, std::vector<std::string> *errors) {
	const size_t count = kernels.size();

	std::vector<std::unique_ptr<char[]>> codes(count);
	std::vector<size_t> codeSizes(count, 0);
	std::vector<char> compiled(count, 0);
	std::vector<char> isCached(count, 0);
	std::vector<std::string> compileErrors(count);

	// Compile the code objects in parallel: no HIP device is required
	tbb::parallel_for(tbb::blocked_range<size_t>(0, count),
		[&](const tbb::blocked_range<size_t> &range) {
			for (size_t i = range.begin(); i < range.end(); ++i) {
				const auto &[params, source, name] = kernels[i];

				bool c = false;
				compiled[i] = CompileCodeObject(params, source, name, &codes[i], &codeSizes[i], &c, &compileErrors[i]);
				isCached[i] = c;
			}
		});

	// Load the modules on the calling thread, where the device is current
	std::vector<hipModule_t> modules(count, nullptr);
	for (size_t i = 0; i < count; ++i) {
		if (compiled[i])
			CHECK_HIP_ERROR(hipModuleLoadData(&modules[i], codes[i].get()));
	}

	if (cached)
		cached->assign(isCached.begin(), isCached.end());
	if (errors)
		*errors = std::move(compileErrors);

	return modules;
}

void hipKernelPersistentCache::ClearCache() {
	const std::filesystem::path cacheDir = GetCacheDir(appName);

	std::lock_guard<std::mutex> lock(cacheMutex);
	if (std::filesystem::exists(cacheDir)) {
		std::filesystem::remove_all(cacheDir);
		std::filesystem::create_directories(cacheDir);
	}
}

void hipKernelPersistentCache::ClearAllCaches() {
	const std::filesystem::path baseCacheDir = luxrays::GetCacheDir() / "hip_kernel_cache";

	static std::mutex globalCacheMutex;
	std::lock_guard<std::mutex> lock(globalCacheMutex);
	if (std::filesystem::exists(baseCacheDir)) {
		std::filesystem::remove_all(baseCacheDir);
		std::filesystem::create_directories(baseCacheDir);
	}
}

}

#endif
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
