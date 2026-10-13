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

// Minimal run-time loader for AMD HIPRT, the ray tracing library for AMD GPUs
// (https://github.com/GPUOpen-LibrariesAndSDKs/HIPRT), in the spirit of hipew:
// LuxCore doesn't need the HIPRT SDK at build time and works without HIPRT.
//
// Only the subset of the host API used by LuxCore is declared, with the types,
// enum values and struct layouts of HIPRT 3.1 (MIT license, Copyright (c)
// Advanced Micro Devices, Inc.). hiprtCreateContext() fails with
// hiprtErrorInvalidApiVersion if the loaded library doesn't implement exactly
// HIPRTEW_API_VERSION, so only libraries of that version are looked for.
//
// The device side of HIPRT doesn't need to be declared: the library compiles
// the trace kernels at run time with its own embedded device headers (see
// hiprtBuildTraceKernels()).

#ifndef __HIPRTEW_H__
#define __HIPRTEW_H__

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  define HIPRTAPI __stdcall
#else
#  define HIPRTAPI
#endif

// The HIPRT version implemented by the library looked for
#define HIPRTEW_API_VERSION 3001
#define HIPRTEW_VERSION_STR "03001"

#if defined(_WIN32)
#  define HIPRTEW_LIBRARY_NAME "hiprt" HIPRTEW_VERSION_STR "64.dll"
#else
#  define HIPRTEW_LIBRARY_NAME "libhiprt" HIPRTEW_VERSION_STR "64.so"
#endif

//------------------------------------------------------------------------------
// HIPRT types
//------------------------------------------------------------------------------

struct _hiprtGeometry;
struct _hiprtScene;
struct _hiprtContext;

typedef void *hiprtDevicePtr;
typedef _hiprtGeometry *hiprtGeometry;
typedef _hiprtScene *hiprtScene;
typedef _hiprtContext *hiprtContext;
typedef uint32_t hiprtLogLevel;
typedef uint32_t hiprtBuildFlags;

typedef int hiprtApiDevice; // hipDevice_t
typedef void *hiprtApiCtx; // hipCtx_t
typedef void *hiprtApiStream; // hipStream_t
typedef void *hiprtApiFunction; // hipFunction_t
typedef void *hiprtApiModule; // hipModule_t

enum : uint32_t {
	hiprtInvalidValue = ~0u
};

enum hiprtError {
	hiprtSuccess = 0,
	hiprtErrorNotImplemented = 1,
	hiprtErrorInternal = 2,
	hiprtErrorOutOfHostMemory = 3,
	hiprtErrorOutOfDeviceMemory = 4,
	hiprtErrorInvalidApiVersion = 5,
	hiprtErrorInvalidParameter = 6
};

enum hiprtLogLevelBits {
	hiprtLogLevelNone = 0,
	hiprtLogLevelInfo = 1 << 0,
	hiprtLogLevelWarn = 1 << 1,
	hiprtLogLevelError = 1 << 2
};

enum hiprtBuildOperation {
	hiprtBuildOperationBuild = 1,
	hiprtBuildOperationUpdate = 2
};

enum hiprtBuildFlagBits {
	hiprtBuildFlagBitPreferFastBuild = 0,
	hiprtBuildFlagBitPreferBalancedBuild = 1,
	hiprtBuildFlagBitPreferHighQualityBuild = 2,
	hiprtBuildFlagBitCustomBvhImport = 3,
	hiprtBuildFlagBitDisableSpatialSplits = 1 << 2,
	hiprtBuildFlagBitDisableTrianglePairing = 1 << 3,
	hiprtBuildFlagBitDisableOrientedBoundingBoxes = 1 << 4
};

enum hiprtPrimitiveType {
	hiprtPrimitiveTypeTriangleMesh,
	hiprtPrimitiveTypeAABBList
};

enum hiprtInstanceType {
	hiprtInstanceTypeGeometry,
	hiprtInstanceTypeScene
};

enum hiprtFrameType {
	hiprtFrameTypeSRT,
	hiprtFrameTypeMatrix
};

enum hiprtStackType {
	hiprtStackTypeGlobal,
	hiprtStackTypeDynamic
};

enum hiprtStackEntryType {
	hiprtStackEntryTypeInteger,
	hiprtStackEntryTypeInstance
};

enum hiprtDeviceType {
	hiprtDeviceAMD,
	hiprtDeviceNVIDIA
};

struct hiprtFuncNameSet {
	const char *intersectFuncName = nullptr;
	const char *filterFuncName = nullptr;
};

struct hiprtContextCreationInput {
	hiprtApiCtx ctxt;
	hiprtApiDevice device;
	hiprtDeviceType deviceType;
};

struct hiprtBuildOptions {
	hiprtBuildFlags buildFlags;
	// Batch build max prim count (if 0 then batch build is not used)
	uint32_t batchBuildMaxPrimCount = 0u;
};

struct hiprtTriangleMeshPrimitive {
	hiprtDevicePtr vertices;
	uint32_t vertexCount;
	uint32_t vertexStride;
	hiprtDevicePtr triangleIndices = nullptr;
	uint32_t triangleCount = 0u;
	uint32_t triangleStride = 0u;
	hiprtDevicePtr trianglePairIndices = nullptr;
	uint32_t trianglePairCount = 0u;
};

struct hiprtAABBListPrimitive {
	hiprtDevicePtr aabbs;
	uint32_t aabbCount;
	uint32_t aabbStride;
};

struct hiprtBvhNodeList {
	hiprtDevicePtr internalNodes;
	hiprtDevicePtr leafNodes;
	uint32_t nodeCount;
};

struct alignas(64) hiprtGeometryBuildInput {
	hiprtPrimitiveType type;
	// Geometry type used for custom function table
	uint32_t geomType = hiprtInvalidValue;
	union {
		hiprtTriangleMeshPrimitive triangleMesh;
		hiprtAABBListPrimitive aabbList;
	} primitive{};
	// Custom Bvh nodes (optional)
	hiprtBvhNodeList nodeList;
};

struct alignas(16) hiprtInstance {
	hiprtInstanceType type;
	union {
		hiprtGeometry geometry;
		hiprtScene scene;
	};
};

struct alignas(16) hiprtSceneBuildInput {
	// Array of instanceCount hiprtInstance
	hiprtDevicePtr instances;
	// Array of instanceCount hiprtTransformHeader (optional: one frame per
	// instance if nullptr)
	hiprtDevicePtr instanceTransformHeaders;
	// Array of frameCount frames, ordered according to time
	hiprtDevicePtr instanceFrames;
	// Array of instanceCount masks (optional: full mask if nullptr)
	hiprtDevicePtr instanceMasks;
	// Custom Bvh nodes (optional)
	hiprtBvhNodeList nodeList;
	uint32_t instanceCount;
	uint32_t frameCount;
	hiprtFrameType frameType = hiprtFrameTypeSRT;
};

// The device side of a global stack buffer, passed by value to the kernels
struct hiprtGlobalStackBuffer {
	uint32_t stackSize;
	uint32_t stackCount;
	void *stackData;
};

struct hiprtGlobalStackBufferInput {
	hiprtStackType type = hiprtStackTypeGlobal;
	hiprtStackEntryType entryType = hiprtStackEntryTypeInteger;
	uint32_t stackSize;
	// Total number of threads (for hiprtStackTypeGlobal only)
	uint32_t threadCount;
};

// Object to world transformation
struct alignas(64) hiprtFrameMatrix {
	float matrix[3][4];
	float time;
};

struct alignas(8) hiprtTransformHeader {
	uint32_t frameIndex;
	uint32_t frameCount;
};

// The same checks of the HIPRT headers
static_assert(sizeof(hiprtGeometryBuildInput) == 128, "hiprtGeometryBuildInput size mismatch");
static_assert(sizeof(hiprtInstance) == 16, "hiprtInstance size mismatch");
static_assert(sizeof(hiprtSceneBuildInput) == 80, "hiprtSceneBuildInput size mismatch");
static_assert(sizeof(hiprtFrameMatrix) == 64, "hiprtFrameMatrix size mismatch");
static_assert(sizeof(hiprtTransformHeader) == 8, "hiprtTransformHeader size mismatch");
static_assert(sizeof(hiprtGlobalStackBuffer) == 16, "hiprtGlobalStackBuffer size mismatch");

//------------------------------------------------------------------------------
// HIPRT functions
//------------------------------------------------------------------------------

extern "C" {

typedef hiprtError HIPRTAPI thiprtCreateContext(uint32_t hiprtApiVersion,
	const hiprtContextCreationInput &input, hiprtContext &contextOut);
typedef hiprtError HIPRTAPI thiprtDestroyContext(hiprtContext context);
typedef hiprtError HIPRTAPI thiprtCreateGeometry(hiprtContext context,
	const hiprtGeometryBuildInput &buildInput, const hiprtBuildOptions buildOptions,
	hiprtGeometry &geometryOut);
typedef hiprtError HIPRTAPI thiprtDestroyGeometry(hiprtContext context, hiprtGeometry geometry);
typedef hiprtError HIPRTAPI thiprtBuildGeometry(hiprtContext context,
	hiprtBuildOperation buildOperation, const hiprtGeometryBuildInput &buildInput,
	const hiprtBuildOptions buildOptions, hiprtDevicePtr temporaryBuffer,
	hiprtApiStream stream, hiprtGeometry geometryOut);
typedef hiprtError HIPRTAPI thiprtGetGeometryBuildTemporaryBufferSize(hiprtContext context,
	const hiprtGeometryBuildInput &buildInput, const hiprtBuildOptions buildOptions,
	size_t &sizeOut);
typedef hiprtError HIPRTAPI thiprtCreateScene(hiprtContext context,
	const hiprtSceneBuildInput &buildInput, const hiprtBuildOptions buildOptions,
	hiprtScene &sceneOut);
typedef hiprtError HIPRTAPI thiprtDestroyScene(hiprtContext context, hiprtScene scene);
typedef hiprtError HIPRTAPI thiprtBuildScene(hiprtContext context,
	hiprtBuildOperation buildOperation, const hiprtSceneBuildInput &buildInput,
	const hiprtBuildOptions buildOptions, hiprtDevicePtr temporaryBuffer,
	hiprtApiStream stream, hiprtScene sceneOut);
typedef hiprtError HIPRTAPI thiprtGetSceneBuildTemporaryBufferSize(hiprtContext context,
	const hiprtSceneBuildInput &buildInput, const hiprtBuildOptions buildOptions,
	size_t &sizeOut);
typedef hiprtError HIPRTAPI thiprtBuildTraceKernels(hiprtContext context,
	uint32_t numFunctions, const char **funcNames, const char *src,
	const char *moduleName, uint32_t numHeaders, const char **headers,
	const char **includeNames, uint32_t numOptions, const char **options,
	uint32_t numGeomTypes, uint32_t numRayTypes, hiprtFuncNameSet *funcNameSets,
	hiprtApiFunction *functionsOut, hiprtApiModule *moduleOut, bool cache);
typedef hiprtError HIPRTAPI thiprtCreateGlobalStackBuffer(hiprtContext context,
	const hiprtGlobalStackBufferInput &input, hiprtGlobalStackBuffer &stackBufferOut);
typedef hiprtError HIPRTAPI thiprtDestroyGlobalStackBuffer(hiprtContext context,
	hiprtGlobalStackBuffer stackBuffer);
typedef hiprtError HIPRTAPI thiprtSetCacheDirPath(hiprtContext context, const char *path);
typedef hiprtError HIPRTAPI thiprtSetLogLevel(hiprtContext context, hiprtLogLevel level);

}

extern thiprtCreateContext *hiprtCreateContext;
extern thiprtDestroyContext *hiprtDestroyContext;
extern thiprtCreateGeometry *hiprtCreateGeometry;
extern thiprtDestroyGeometry *hiprtDestroyGeometry;
extern thiprtBuildGeometry *hiprtBuildGeometry;
extern thiprtGetGeometryBuildTemporaryBufferSize *hiprtGetGeometryBuildTemporaryBufferSize;
extern thiprtCreateScene *hiprtCreateScene;
extern thiprtDestroyScene *hiprtDestroyScene;
extern thiprtBuildScene *hiprtBuildScene;
extern thiprtGetSceneBuildTemporaryBufferSize *hiprtGetSceneBuildTemporaryBufferSize;
extern thiprtBuildTraceKernels *hiprtBuildTraceKernels;
extern thiprtCreateGlobalStackBuffer *hiprtCreateGlobalStackBuffer;
extern thiprtDestroyGlobalStackBuffer *hiprtDestroyGlobalStackBuffer;
extern thiprtSetCacheDirPath *hiprtSetCacheDirPath;
extern thiprtSetLogLevel *hiprtSetLogLevel;

//------------------------------------------------------------------------------
// hiprtew
//------------------------------------------------------------------------------

enum {
	HIPRTEW_SUCCESS = 0,
	HIPRTEW_ERROR_OPEN_FAILED = -1,
	HIPRTEW_ERROR_MISSING_SYMBOL = -3
};

// Loads the HIPRT library and resolves its symbols. It is safe to call it more
// than once: the library is loaded only the first time.
//
// The library is looked for in the default search path of the dynamic loader
// (i.e. LD_LIBRARY_PATH or PATH), in the directory of the LuxCore library and
// in the ROCm installation directory.
int hiprtewInit();

// The name of the library actually loaded, empty if none
const char *hiprtewLibraryName();

#endif	/* __HIPRTEW_H__ */

// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
