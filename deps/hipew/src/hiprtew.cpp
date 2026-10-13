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

#include <hiprtew.h>

#include <mutex>
#include <string>
#include <vector>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  define VC_EXTRALEAN
#  include <windows.h>

typedef HMODULE DynamicLibrary;

#  define dynamic_library_open(path)         LoadLibraryA(path)
#  define dynamic_library_close(lib)         FreeLibrary(lib)
#  define dynamic_library_find(lib, symbol)  GetProcAddress(lib, symbol)
#else
#  include <dlfcn.h>

typedef void* DynamicLibrary;

#  define dynamic_library_open(path)         dlopen(path, RTLD_NOW)
#  define dynamic_library_close(lib)         dlclose(lib)
#  define dynamic_library_find(lib, symbol)  dlsym(lib, symbol)
#endif

// All symbols are required: a missing one makes the whole library
// unavailable, so callers never run into a null function pointer
#define HIPRTEW_LIBRARY_FIND(lib, name) \
	name = (t##name *)dynamic_library_find(lib, #name); \
	if (!name) missingSymbol = true;

//------------------------------------------------------------------------------
// Function pointers
//------------------------------------------------------------------------------

thiprtCreateContext *hiprtCreateContext;
thiprtDestroyContext *hiprtDestroyContext;
thiprtCreateGeometry *hiprtCreateGeometry;
thiprtDestroyGeometry *hiprtDestroyGeometry;
thiprtBuildGeometry *hiprtBuildGeometry;
thiprtGetGeometryBuildTemporaryBufferSize *hiprtGetGeometryBuildTemporaryBufferSize;
thiprtCompactGeometry *hiprtCompactGeometry;
thiprtCreateScene *hiprtCreateScene;
thiprtDestroyScene *hiprtDestroyScene;
thiprtBuildScene *hiprtBuildScene;
thiprtGetSceneBuildTemporaryBufferSize *hiprtGetSceneBuildTemporaryBufferSize;
thiprtBuildTraceKernels *hiprtBuildTraceKernels;
thiprtCreateGlobalStackBuffer *hiprtCreateGlobalStackBuffer;
thiprtDestroyGlobalStackBuffer *hiprtDestroyGlobalStackBuffer;
thiprtSetCacheDirPath *hiprtSetCacheDirPath;
thiprtSetLogLevel *hiprtSetLogLevel;

//------------------------------------------------------------------------------
// Library loading
//------------------------------------------------------------------------------

namespace {

#if defined(_WIN32)
const char *hiprtRocmDirs[] = { nullptr };
const char pathSeparator = '\\';
#else
const char *hiprtRocmDirs[] = { "/opt/rocm/lib", nullptr };
const char pathSeparator = '/';
#endif

struct LoadedLibrary {
	~LoadedLibrary() {
		if (lib)
			dynamic_library_close(lib);
	}

	DynamicLibrary lib = nullptr;
	std::string name;
	int result = HIPRTEW_ERROR_OPEN_FAILED;
	std::once_flag once;
};

LoadedLibrary hiprtLib;

// The directory of the binary (i.e. pyluxcore module or executable) this code
// is linked into, so the HIPRT library can be shipped next to it
std::string ThisModuleDir() {
	std::string path;
#if defined(_WIN32)
	HMODULE module = nullptr;
	if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
			GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
			(LPCSTR)&ThisModuleDir, &module)) {
		char buff[MAX_PATH];
		const DWORD size = GetModuleFileNameA(module, buff, MAX_PATH);
		if ((size > 0) && (size < MAX_PATH))
			path = std::string(buff, size);
	}
#else
	Dl_info info;
	if (dladdr((void *)&ThisModuleDir, &info) && info.dli_fname)
		path = info.dli_fname;
#endif

	const size_t pos = path.find_last_of(pathSeparator);
	return (pos == std::string::npos) ? std::string() : path.substr(0, pos);
}

void HiprtInit() {
	std::vector<std::string> paths;
	paths.push_back(HIPRTEW_LIBRARY_NAME);
	const std::string moduleDir = ThisModuleDir();
	if (!moduleDir.empty())
		paths.push_back(moduleDir + pathSeparator + HIPRTEW_LIBRARY_NAME);
	for (int i = 0; hiprtRocmDirs[i]; ++i)
		paths.push_back(std::string(hiprtRocmDirs[i]) + pathSeparator + HIPRTEW_LIBRARY_NAME);

	for (const auto &p : paths) {
		hiprtLib.lib = dynamic_library_open(p.c_str());
		if (hiprtLib.lib) {
			hiprtLib.name = p;
			break;
		}
	}
	if (!hiprtLib.lib)
		return;

	bool missingSymbol = false;
	HIPRTEW_LIBRARY_FIND(hiprtLib.lib, hiprtCreateContext);
	HIPRTEW_LIBRARY_FIND(hiprtLib.lib, hiprtDestroyContext);
	HIPRTEW_LIBRARY_FIND(hiprtLib.lib, hiprtCreateGeometry);
	HIPRTEW_LIBRARY_FIND(hiprtLib.lib, hiprtDestroyGeometry);
	HIPRTEW_LIBRARY_FIND(hiprtLib.lib, hiprtBuildGeometry);
	HIPRTEW_LIBRARY_FIND(hiprtLib.lib, hiprtGetGeometryBuildTemporaryBufferSize);
	HIPRTEW_LIBRARY_FIND(hiprtLib.lib, hiprtCompactGeometry);
	HIPRTEW_LIBRARY_FIND(hiprtLib.lib, hiprtCreateScene);
	HIPRTEW_LIBRARY_FIND(hiprtLib.lib, hiprtDestroyScene);
	HIPRTEW_LIBRARY_FIND(hiprtLib.lib, hiprtBuildScene);
	HIPRTEW_LIBRARY_FIND(hiprtLib.lib, hiprtGetSceneBuildTemporaryBufferSize);
	HIPRTEW_LIBRARY_FIND(hiprtLib.lib, hiprtBuildTraceKernels);
	HIPRTEW_LIBRARY_FIND(hiprtLib.lib, hiprtCreateGlobalStackBuffer);
	HIPRTEW_LIBRARY_FIND(hiprtLib.lib, hiprtDestroyGlobalStackBuffer);
	HIPRTEW_LIBRARY_FIND(hiprtLib.lib, hiprtSetCacheDirPath);
	HIPRTEW_LIBRARY_FIND(hiprtLib.lib, hiprtSetLogLevel);

	hiprtLib.result = missingSymbol ? HIPRTEW_ERROR_MISSING_SYMBOL : HIPRTEW_SUCCESS;
}

}

int hiprtewInit() {
	std::call_once(hiprtLib.once, HiprtInit);

	return hiprtLib.result;
}

const char *hiprtewLibraryName() {
	return hiprtLib.name.c_str();
}

// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
