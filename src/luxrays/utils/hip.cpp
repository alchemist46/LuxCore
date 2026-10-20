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
#include <sstream>
#include <vector>
#include <oneapi/tbb.h>

#include "luxrays/luxrays.h"
#include "luxrays/utils/utils.h"
#include "luxrays/utils/config.h"
#include "luxrays/utils/embeddedhip.h"
#include "luxrays/utils/hipcache.h"
#include "luxrays/utils/oclcache.h"

namespace luxrays {

//------------------------------------------------------------------------------
// hipKernelPersistentCache
//------------------------------------------------------------------------------

// True if the environment variable is set to anything but "" or "0"
static bool GetEnvFlag(const char *name) {
	const char *value = getenv(name);
	return value && value[0] && (std::string(value) != "0");
}

hipKernelPersistentCache::hipKernelPersistentCache(const std::string &applicationName,
		const std::string &gpuArch) : appName(applicationName), arch(gpuArch) {
	// Create the cache directory
	std::filesystem::create_directories(GetCacheDir(appName));

	// The embedded kernels are not used when the code would differ:
	// LUXRAYS_HIP_COMPILER_OPTIONS adds compiler options. They can be
	// disabled with LUXRAYS_HIP_DISABLE_PRECOMPILED_KERNELS=1, e.g. to compare
	// them with the architecture specific code compiled by hiprtc.
	const char *extraOpts = getenv("LUXRAYS_HIP_COMPILER_OPTIONS");
	if (HasEmbeddedHIPKernels() &&
			!(extraOpts && extraOpts[0]) &&
			!GetEnvFlag("LUXRAYS_HIP_DISABLE_PRECOMPILED_KERNELS"))
		precompiledFamily = GetHIPBakedKernelFamily(arch);
}

hipKernelPersistentCache::~hipKernelPersistentCache() {
}

std::filesystem::path hipKernelPersistentCache::GetCacheDir(const std::string &applicationName) {
	return luxrays::GetCacheDir() / "hip_kernel_cache" / SanitizeFileName(applicationName);
}

std::vector<std::string> hipKernelPersistentCache::GetCompilerOptions(const std::string &gpuArch) {
	std::vector<std::string> opts = {
		// Compile for the architecture of the device the program will run on
		"--offload-arch=" + gpuArch
	};
	// The options shared with the baked kernels (see hipkernel.h)
	const std::vector<std::string> baseOpts = GetHIPBaseCompilerOptions();
	opts.insert(opts.end(), baseOpts.begin(), baseOpts.end());

	// For experiments (e.g. "-mllvm -amdgpu-num-vgpr=192"): extra options,
	// separated by spaces. They are part of the cache key.
	if (const char *extra = getenv("LUXRAYS_HIP_COMPILER_OPTIONS")) {
		std::istringstream ss(extra);
		std::string opt;
		while (ss >> opt)
			opts.push_back(opt);
	}

	return opts;
}

bool hipKernelPersistentCache::ForcedCompileCodeObject(
	const std::vector<std::string> &kernelsParameters, const std::string &kernelSource,
	const std::string &programName, const std::string &gpuArch,
	std::unique_ptr<char[]> *code, size_t *codeSize, std::string *error,
	const std::vector<std::string> &extraOptions
) {
	if (error)
		*error = "";

	if (!isHiprtcAvailable) {
		if (error) {
			*error = "There is no precompiled kernel for this program on " + gpuArch +
					" and it can't be compiled: that requires the HIP run-time compiler (hiprtc), "
#if defined(_WIN32)
					"installed with the AMD HIP SDK (HIP_PATH)";
#else
					"the ROCm hiprtc library (libhiprtc.so)";
#endif
		}
		return false;
	}

	hiprtcProgram prog;
	CHECK_HIPRTC_ERROR(hiprtcCreateProgram(&prog, kernelSource.c_str(), programName.c_str(), 0, nullptr, nullptr));

	const std::vector<std::string> compilerOpts = GetCompilerOptions(gpuArch);

	std::vector<const char *> hipOpts;
	for (auto const &o : compilerOpts)
		hipOpts.push_back(o.c_str());
	for (auto const &p : kernelsParameters)
		hipOpts.push_back(p.c_str());
	for (auto const &o : extraOptions)
		hipOpts.push_back(o.c_str());

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
		std::unique_ptr<char[]> *code, size_t *codeSize, bool *cached, std::string *error,
		const std::vector<std::string> &extraOptions) {
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
	if (!ForcedCompileCodeObject(kernelsParameters, kernelSource, programName, arch, code, codeSize, error, extraOptions))
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

const unsigned char *hipKernelPersistentCache::GetPrecompiled(
		const std::vector<std::string> &kernelsParameters,
		const std::string &kernelSource) const {
	if (precompiledFamily.empty())
		return nullptr;

	return GetEmbeddedHIPKernel(GetHIPBakedKernelKey(kernelsParameters, kernelSource),
			precompiledFamily, nullptr);
}

bool hipKernelPersistentCache::HasPrecompiled(const std::vector<std::string> &kernelsParameters,
		const std::string &kernelSource) const {
	return GetPrecompiled(kernelsParameters, kernelSource) != nullptr;
}

hipModule_t hipKernelPersistentCache::LoadPrecompiled(
		const std::vector<std::string> &kernelsParameters,
		const std::string &kernelSource, std::string *warning) const {
	const unsigned char *code = GetPrecompiled(kernelsParameters, kernelSource);
	if (!code)
		return nullptr;

	hipModule_t module;
	const hipError_t err = hipModuleLoadData(&module, code);
	if (err != hipSuccess) {
		*warning = "Unable to load the precompiled kernels for " + precompiledFamily +
				" (" + hipGetErrorName(err) + "): they are compiled instead\n";
		return nullptr;
	}

	return module;
}

std::string hipKernelPersistentCache::GetCodeOriginDesc(const CodeOrigin origin) const {
	switch (origin) {
		case CodeOrigin::CACHED:
			return " (cached)";
		case CodeOrigin::PRECOMPILED:
			return " (precompiled kernels, " + precompiledFamily + ")";
		default:
			return "";
	}
}

hipModule_t hipKernelPersistentCache::Compile(const std::vector<std::string> &kernelsParameters,
		const std::string &kernelSource, const std::string &programName,
		CodeOrigin *origin, std::string *error,
		const std::vector<std::string> &extraOptions) {
	std::string warning;
	hipModule_t module = LoadPrecompiled(kernelsParameters, kernelSource, &warning);
	if (module) {
		*origin = CodeOrigin::PRECOMPILED;
		if (error)
			*error = "";
		return module;
	}

	std::unique_ptr<char[]> code;
	size_t codeSize;
	bool cached;
	const bool compiled = CompileCodeObject(kernelsParameters, kernelSource, programName,
			&code, &codeSize, &cached, error, extraOptions);
	if (error)
		*error = warning + *error;
	if (!compiled)
		return nullptr;
	*origin = cached ? CodeOrigin::CACHED : CodeOrigin::COMPILED;

	CHECK_HIP_ERROR(hipModuleLoadData(&module, code.get()));

	return module;
}

std::vector<hipModule_t> hipKernelPersistentCache::CompileMultiple(
		const std::vector<std::tuple<std::vector<std::string>, std::string, std::string>> &kernels,
		std::vector<CodeOrigin> *origins, std::vector<std::string> *errors) {
	const size_t count = kernels.size();

	std::vector<hipModule_t> modules(count, nullptr);
	std::vector<std::unique_ptr<char[]>> codes(count);
	std::vector<size_t> codeSizes(count, 0);
	std::vector<char> compiled(count, 0);
	std::vector<CodeOrigin> codeOrigins(count, CodeOrigin::COMPILED);
	std::vector<std::string> warnings(count);
	std::vector<std::string> compileErrors(count);

	// Load the precompiled programs on the calling thread, where the device is
	// current
	for (size_t i = 0; i < count; ++i) {
		const auto &[params, source, name] = kernels[i];

		modules[i] = LoadPrecompiled(params, source, &warnings[i]);
		if (modules[i])
			codeOrigins[i] = CodeOrigin::PRECOMPILED;
	}

	// Compile the other code objects in parallel: no HIP device is required
	tbb::parallel_for(tbb::blocked_range<size_t>(0, count),
		[&](const tbb::blocked_range<size_t> &range) {
			for (size_t i = range.begin(); i < range.end(); ++i) {
				if (modules[i])
					continue;

				const auto &[params, source, name] = kernels[i];

				bool c = false;
				compiled[i] = CompileCodeObject(params, source, name, &codes[i], &codeSizes[i], &c, &compileErrors[i]);
				if (c)
					codeOrigins[i] = CodeOrigin::CACHED;
			}
		});

	// Load the modules on the calling thread, where the device is current
	for (size_t i = 0; i < count; ++i) {
		if (compiled[i])
			CHECK_HIP_ERROR(hipModuleLoadData(&modules[i], codes[i].get()));
		compileErrors[i] = warnings[i] + compileErrors[i];
	}

	if (origins)
		*origins = std::move(codeOrigins);
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
