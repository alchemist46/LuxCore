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

#include <cstdint>
#include <cstdlib>
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
#  include <link.h>
#  include <unistd.h>

typedef void* DynamicLibrary;

// RTLD_DEEPBIND: the library must use its own symbols. A host application
// can have another HIPRT version already loaded (i.e. Blender for Cycles) and
// HIPRT calls its own exported functions: they would be resolved to the
// other version (and crash). The sanitizer runtimes refuse to load a library
// with RTLD_DEEPBIND: it isn't used in sanitized builds.
//
// Only when this code is in a library loaded by a host (pyluxcore): in a
// non-PIE executable (luxcoreui, luxcoreconsole) RTLD_DEEPBIND breaks HIPRT.
// The executable holds copy relocations of libstdc++ data (i.e.
// std::ctype<char>::id) that every other module uses, while a deep-bound
// HIPRT reads the unused originals in libstdc++ and its std::regex crashes
// at the first BVH build. There is no other HIPRT in a LuxCore executable.
#  if defined(RTLD_DEEPBIND) && !defined(__SANITIZE_ADDRESS__)
#    define HIPRTEW_USE_DEEPBIND
#  endif
#  define dynamic_library_open(path)         dlopen(path, HiprtOpenFlags())
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
// Not the HIPRT of the HIP SDK: it misses the headers HIPRT compiles its
// kernels with (see SetHiprtPath())
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

#if !defined(_WIN32)
#if defined(HIPRTEW_USE_DEEPBIND)
// True when this code is linked into the main executable, not into a library
// loaded by a host application
bool InMainExecutable() {
	struct Search {
		uintptr_t address;
		bool found;
	} search = { (uintptr_t)&ThisModuleDir, false };

	dl_iterate_phdr([](struct dl_phdr_info *info, size_t, void *data) -> int {
		Search *s = (Search *)data;
		for (int i = 0; i < info->dlpi_phnum; ++i) {
			const ElfW(Phdr) &phdr = info->dlpi_phdr[i];
			if (phdr.p_type != PT_LOAD)
				continue;

			const uintptr_t start = info->dlpi_addr + phdr.p_vaddr;
			if ((s->address >= start) && (s->address < start + phdr.p_memsz))
				s->found = true;
		}

		// The main executable is always the first object listed
		return 1;
	}, &search);

	return search.found;
}
#endif

int HiprtOpenFlags() {
#if defined(HIPRTEW_USE_DEEPBIND)
	if (!InMainExecutable())
		return RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND;
#endif
	return RTLD_NOW | RTLD_LOCAL;
}
#endif

// HIPRT compiles its kernels at run time with headers of its own (hiprt/*.h,
// hiprt/impl/*.h, also in a "baked" build), which it finds in HIPRT_PATH, or
// in ".." of the current directory without it. When they are shipped next to
// the loaded library (pyluxcore), HIPRT_PATH is set to that folder, unless it
// is already set.
void SetHiprtPath() {
	const char *current = getenv("HIPRT_PATH");
	if (current && current[0])
		return;

	std::string libPath;
#if defined(_WIN32)
	char buff[MAX_PATH];
	const DWORD size = GetModuleFileNameA(hiprtLib.lib, buff, MAX_PATH);
	if ((size > 0) && (size < MAX_PATH))
		libPath = std::string(buff, size);
#else
	Dl_info info;
	if (dladdr((void *)hiprtCreateContext, &info) && info.dli_fname)
		libPath = info.dli_fname;
#endif
	const size_t pos = libPath.find_last_of(pathSeparator);
	if (pos == std::string::npos)
		return;
	const std::string libDir = libPath.substr(0, pos);

	const std::string header = libDir + pathSeparator + "hiprt" + pathSeparator + "hiprt_types.h";
#if defined(_WIN32)
	if (GetFileAttributesA(header.c_str()) == INVALID_FILE_ATTRIBUTES)
		return;
	// HIPRT reads it with _dupenv_s()
	_putenv_s("HIPRT_PATH", libDir.c_str());
#else
	if (access(header.c_str(), R_OK) != 0)
		return;
	setenv("HIPRT_PATH", libDir.c_str(), 0);
#endif
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
#if defined(_WIN32)
		// A full path: its dependencies are next to it
		hiprtLib.lib = (p.find(pathSeparator) == std::string::npos) ?
				dynamic_library_open(p.c_str()) :
				LoadLibraryExA(p.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
#else
		hiprtLib.lib = dynamic_library_open(p.c_str());
#endif
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
	if (hiprtLib.result == HIPRTEW_SUCCESS)
		SetHiprtPath();
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
