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

#include <hipew.h>

#include <mutex>

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

// All symbols of a library are required: a missing one makes the whole
// library unavailable, so callers never run into a null function pointer
#define HIPEW_LIBRARY_FIND(lib, name) \
	name = (t##name *)dynamic_library_find(lib, #name); \
	if (!name) missingSymbol = true;

//------------------------------------------------------------------------------
// Function pointers
//------------------------------------------------------------------------------

thipGetErrorName *hipGetErrorName;
thipGetErrorString *hipGetErrorString;
thipInit *hipInit;
thipDriverGetVersion *hipDriverGetVersion;
thipRuntimeGetVersion *hipRuntimeGetVersion;
thipGetDeviceCount *hipGetDeviceCount;
thipDeviceGet *hipDeviceGet;
thipDeviceGetName *hipDeviceGetName;
thipDeviceGetAttribute *hipDeviceGetAttribute;
thipDeviceTotalMem *hipDeviceTotalMem;
thipGetDevicePropertiesR0600 *hipGetDevicePropertiesR0600;
thipSetDevice *hipSetDevice;
thipGetDevice *hipGetDevice;
thipDeviceSynchronize *hipDeviceSynchronize;
thipMalloc *hipMalloc;
thipMallocManaged *hipMallocManaged;
thipMemAdvise *hipMemAdvise;
thipFree *hipFree;
thipMemcpyHtoD *hipMemcpyHtoD;
thipMemcpyDtoH *hipMemcpyDtoH;
thipMemcpyHtoDAsync *hipMemcpyHtoDAsync;
thipMemcpyDtoHAsync *hipMemcpyDtoHAsync;
thipMemGetAddressRange *hipMemGetAddressRange;
thipStreamSynchronize *hipStreamSynchronize;
thipModuleLoadData *hipModuleLoadData;
thipModuleUnload *hipModuleUnload;
thipModuleGetFunction *hipModuleGetFunction;
thipModuleLaunchKernel *hipModuleLaunchKernel;

thiprtcGetErrorString *hiprtcGetErrorString;
thiprtcVersion *hiprtcVersion;
thiprtcCreateProgram *hiprtcCreateProgram;
thiprtcCompileProgram *hiprtcCompileProgram;
thiprtcGetProgramLogSize *hiprtcGetProgramLogSize;
thiprtcGetProgramLog *hiprtcGetProgramLog;
thiprtcGetCodeSize *hiprtcGetCodeSize;
thiprtcGetCode *hiprtcGetCode;
thiprtcDestroyProgram *hiprtcDestroyProgram;

//------------------------------------------------------------------------------
// Library loading
//------------------------------------------------------------------------------

namespace {

#if defined(_WIN32)
const char *hipLibraryPaths[] = {
	"amdhip64_7.dll", "amdhip64_6.dll", "amdhip64.dll", nullptr
};
const char *hiprtcLibraryPaths[] = {
	"hiprtc0702.dll", "hiprtc0701.dll", "hiprtc0700.dll",
	"hiprtc0605.dll", "hiprtc0604.dll", "hiprtc0603.dll",
	"hiprtc0602.dll", "hiprtc0601.dll", "hiprtc0600.dll",
	"hiprtc.dll", nullptr
};
#else
const char *hipLibraryPaths[] = {
	"libamdhip64.so.7", "libamdhip64.so.6", "libamdhip64.so",
	"/opt/rocm/lib/libamdhip64.so.7", "/opt/rocm/lib/libamdhip64.so.6",
	"/opt/rocm/lib/libamdhip64.so", nullptr
};
const char *hiprtcLibraryPaths[] = {
	"libhiprtc.so.7", "libhiprtc.so.6", "libhiprtc.so",
	"/opt/rocm/lib/libhiprtc.so.7", "/opt/rocm/lib/libhiprtc.so.6",
	"/opt/rocm/lib/libhiprtc.so", nullptr
};
#endif

struct LoadedLibrary {
	~LoadedLibrary() {
		if (lib)
			dynamic_library_close(lib);
	}

	DynamicLibrary lib = nullptr;
	const char *name = "";
	int result = HIPEW_ERROR_OPEN_FAILED;
	std::once_flag once;
};

LoadedLibrary hipLib;
LoadedLibrary hiprtcLib;

DynamicLibrary OpenFirst(const char **paths, const char **name) {
	for (int i = 0; paths[i]; ++i) {
		DynamicLibrary lib = dynamic_library_open(paths[i]);
		if (lib) {
			*name = paths[i];
			return lib;
		}
	}

	return nullptr;
}

void HipInit() {
	hipLib.lib = OpenFirst(hipLibraryPaths, &hipLib.name);
	if (!hipLib.lib)
		return;

	bool missingSymbol = false;
	HIPEW_LIBRARY_FIND(hipLib.lib, hipGetErrorName);
	HIPEW_LIBRARY_FIND(hipLib.lib, hipGetErrorString);
	HIPEW_LIBRARY_FIND(hipLib.lib, hipInit);
	HIPEW_LIBRARY_FIND(hipLib.lib, hipDriverGetVersion);
	HIPEW_LIBRARY_FIND(hipLib.lib, hipRuntimeGetVersion);
	HIPEW_LIBRARY_FIND(hipLib.lib, hipGetDeviceCount);
	HIPEW_LIBRARY_FIND(hipLib.lib, hipDeviceGet);
	HIPEW_LIBRARY_FIND(hipLib.lib, hipDeviceGetName);
	HIPEW_LIBRARY_FIND(hipLib.lib, hipDeviceGetAttribute);
	HIPEW_LIBRARY_FIND(hipLib.lib, hipDeviceTotalMem);
	HIPEW_LIBRARY_FIND(hipLib.lib, hipGetDevicePropertiesR0600);
	HIPEW_LIBRARY_FIND(hipLib.lib, hipSetDevice);
	HIPEW_LIBRARY_FIND(hipLib.lib, hipGetDevice);
	HIPEW_LIBRARY_FIND(hipLib.lib, hipDeviceSynchronize);
	HIPEW_LIBRARY_FIND(hipLib.lib, hipMalloc);
	HIPEW_LIBRARY_FIND(hipLib.lib, hipMallocManaged);
	HIPEW_LIBRARY_FIND(hipLib.lib, hipMemAdvise);
	HIPEW_LIBRARY_FIND(hipLib.lib, hipFree);
	HIPEW_LIBRARY_FIND(hipLib.lib, hipMemcpyHtoD);
	HIPEW_LIBRARY_FIND(hipLib.lib, hipMemcpyDtoH);
	HIPEW_LIBRARY_FIND(hipLib.lib, hipMemcpyHtoDAsync);
	HIPEW_LIBRARY_FIND(hipLib.lib, hipMemcpyDtoHAsync);
	HIPEW_LIBRARY_FIND(hipLib.lib, hipMemGetAddressRange);
	HIPEW_LIBRARY_FIND(hipLib.lib, hipStreamSynchronize);
	HIPEW_LIBRARY_FIND(hipLib.lib, hipModuleLoadData);
	HIPEW_LIBRARY_FIND(hipLib.lib, hipModuleUnload);
	HIPEW_LIBRARY_FIND(hipLib.lib, hipModuleGetFunction);
	HIPEW_LIBRARY_FIND(hipLib.lib, hipModuleLaunchKernel);

	hipLib.result = missingSymbol ? HIPEW_ERROR_MISSING_SYMBOL : HIPEW_SUCCESS;
}

void HiprtcInit() {
	hiprtcLib.lib = OpenFirst(hiprtcLibraryPaths, &hiprtcLib.name);
	if (!hiprtcLib.lib)
		return;

	bool missingSymbol = false;
	HIPEW_LIBRARY_FIND(hiprtcLib.lib, hiprtcGetErrorString);
	HIPEW_LIBRARY_FIND(hiprtcLib.lib, hiprtcVersion);
	HIPEW_LIBRARY_FIND(hiprtcLib.lib, hiprtcCreateProgram);
	HIPEW_LIBRARY_FIND(hiprtcLib.lib, hiprtcCompileProgram);
	HIPEW_LIBRARY_FIND(hiprtcLib.lib, hiprtcGetProgramLogSize);
	HIPEW_LIBRARY_FIND(hiprtcLib.lib, hiprtcGetProgramLog);
	HIPEW_LIBRARY_FIND(hiprtcLib.lib, hiprtcGetCodeSize);
	HIPEW_LIBRARY_FIND(hiprtcLib.lib, hiprtcGetCode);
	HIPEW_LIBRARY_FIND(hiprtcLib.lib, hiprtcDestroyProgram);

	hiprtcLib.result = missingSymbol ? HIPEW_ERROR_MISSING_SYMBOL : HIPEW_SUCCESS;
}

}

int hipewInit(unsigned int flags) {
	if (flags & HIPEW_INIT_HIP) {
		std::call_once(hipLib.once, HipInit);
		if (hipLib.result != HIPEW_SUCCESS)
			return hipLib.result;
	}

	if (flags & HIPEW_INIT_HIPRTC) {
		std::call_once(hiprtcLib.once, HiprtcInit);
		if (hiprtcLib.result != HIPEW_SUCCESS)
			return hiprtcLib.result;
	}

	return HIPEW_SUCCESS;
}

const char *hipewHipLibraryName(void) {
	return hipLib.name;
}

const char *hipewHiprtcLibraryName(void) {
	return hiprtcLib.name;
}

// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
