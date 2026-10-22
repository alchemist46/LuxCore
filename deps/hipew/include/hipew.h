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

// Minimal run-time loader for the AMD HIP runtime (amdhip64) and the HIP
// run-time compiler (hiprtc), in the spirit of cuew and clew: LuxCore doesn't
// need the ROCm SDK at build time and works without ROCm installed.
//
// Only the subset of the API used by LuxCore is declared. Type sizes, enum
// values and the hipDeviceProp_t (R0600) layout match the HIP ABI, which is
// stable since ROCm 6.0.

#ifndef __HIPEW_H__
#define __HIPEW_H__

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#  define HIPAPI __stdcall
#else
#  define HIPAPI
#endif

//------------------------------------------------------------------------------
// HIP runtime types
//------------------------------------------------------------------------------

typedef enum hipError_t {
	hipSuccess = 0,
	hipErrorInvalidValue = 1,
	hipErrorOutOfMemory = 2,
	hipErrorNotInitialized = 3,
	hipErrorNoDevice = 100,
	hipErrorInvalidDevice = 101,
	hipErrorNotReady = 600,
	hipErrorForceSize = 0x7fffffff
} hipError_t;

typedef int hipDevice_t;
typedef struct ihipCtx_t *hipCtx_t;
typedef void *hipDeviceptr_t;
typedef struct ihipStream_t *hipStream_t;
typedef struct ihipEvent_t *hipEvent_t;
typedef struct ihipModule_t *hipModule_t;
typedef struct ihipModuleSymbol_t *hipFunction_t;

typedef enum hipDeviceAttribute_t {
	hipDeviceAttributeClockRate = 5,
	hipDeviceAttributeConcurrentManagedAccess = 9,
	hipDeviceAttributeManagedMemory = 24,
	hipDeviceAttributeMultiprocessorCount = 63,
	hipDeviceAttributeUnifiedAddressing = 85,
	hipDeviceAttributeWarpSize = 87,
	hipDeviceAttributeForceSize = 0x7fffffff
} hipDeviceAttribute_t;

typedef enum hipMemoryAdvise {
	hipMemAdviseSetReadMostly = 1,
	hipMemAdviseForceSize = 0x7fffffff
} hipMemoryAdvise;

#define hipMemAttachGlobal 0x01

typedef enum hipMemcpyKind {
	hipMemcpyHostToHost = 0,
	hipMemcpyHostToDevice = 1,
	hipMemcpyDeviceToHost = 2,
	hipMemcpyDeviceToDevice = 3,
	hipMemcpyDefault = 4
} hipMemcpyKind;

// Arrays and texture objects (driver_types.h and texture_types.h)
typedef struct hipArray *hipArray_t;
typedef struct hipMipmappedArray *hipMipmappedArray_t;
typedef struct __hip_texture *hipTextureObject_t;

#define hipArrayDefault 0x00

typedef enum hipChannelFormatKind {
	hipChannelFormatKindSigned = 0,
	hipChannelFormatKindUnsigned = 1,
	hipChannelFormatKindFloat = 2,
	hipChannelFormatKindNone = 3
} hipChannelFormatKind;

typedef struct hipChannelFormatDesc {
	int x;
	int y;
	int z;
	int w;
	enum hipChannelFormatKind f;
} hipChannelFormatDesc;

typedef enum hipResourceType {
	hipResourceTypeArray = 0x00,
	hipResourceTypeMipmappedArray = 0x01,
	hipResourceTypeLinear = 0x02,
	hipResourceTypePitch2D = 0x03
} hipResourceType;

typedef struct hipResourceDesc {
	enum hipResourceType resType;
	union {
		struct {
			hipArray_t array;
		} array;
		struct {
			hipMipmappedArray_t mipmap;
		} mipmap;
		struct {
			void *devPtr;
			struct hipChannelFormatDesc desc;
			size_t sizeInBytes;
		} linear;
		struct {
			void *devPtr;
			struct hipChannelFormatDesc desc;
			size_t width;
			size_t height;
			size_t pitchInBytes;
		} pitch2D;
	} res;
} hipResourceDesc;

typedef enum hipTextureAddressMode {
	hipAddressModeWrap = 0,
	hipAddressModeClamp = 1,
	hipAddressModeMirror = 2,
	hipAddressModeBorder = 3
} hipTextureAddressMode;

typedef enum hipTextureFilterMode {
	hipFilterModePoint = 0,
	hipFilterModeLinear = 1
} hipTextureFilterMode;

typedef enum hipTextureReadMode {
	hipReadModeElementType = 0,
	hipReadModeNormalizedFloat = 1
} hipTextureReadMode;

typedef struct hipTextureDesc {
	enum hipTextureAddressMode addressMode[3];
	enum hipTextureFilterMode filterMode;
	enum hipTextureReadMode readMode;
	int sRGB;
	float borderColor[4];
	int normalizedCoords;
	unsigned int maxAnisotropy;
	enum hipTextureFilterMode mipmapFilterMode;
	float mipmapLevelBias;
	float minMipmapLevelClamp;
	float maxMipmapLevelClamp;
} hipTextureDesc;

struct hipResourceViewDesc;

// hipDeviceProp_t is a large struct whose layout is versioned through the
// hipGetDevicePropertiesR0600() entry point. Only the fields LuxCore needs are
// read, at their R0600 offsets.
#define HIPEW_DEVICE_PROP_R0600_SIZE 1472
#define HIPEW_DEVICE_PROP_R0600_GCN_ARCH_NAME_OFFSET 1160
#define HIPEW_DEVICE_PROP_GCN_ARCH_NAME_SIZE 256

//------------------------------------------------------------------------------
// hiprtc types
//------------------------------------------------------------------------------

typedef enum hiprtcResult {
	HIPRTC_SUCCESS = 0,
	HIPRTC_ERROR_COMPILATION = 6,
	HIPRTC_ERROR_FORCE_SIZE = 0x7fffffff
} hiprtcResult;

typedef struct _hiprtcProgram *hiprtcProgram;

//------------------------------------------------------------------------------
// HIP runtime functions
//------------------------------------------------------------------------------

typedef const char *HIPAPI thipGetErrorName(hipError_t error);
typedef const char *HIPAPI thipGetErrorString(hipError_t error);
typedef hipError_t HIPAPI thipInit(unsigned int flags);
typedef hipError_t HIPAPI thipDriverGetVersion(int *driverVersion);
typedef hipError_t HIPAPI thipRuntimeGetVersion(int *runtimeVersion);
typedef hipError_t HIPAPI thipGetDeviceCount(int *count);
typedef hipError_t HIPAPI thipDeviceGet(hipDevice_t *device, int ordinal);
typedef hipError_t HIPAPI thipDeviceGetName(char *name, int len, hipDevice_t device);
typedef hipError_t HIPAPI thipDeviceGetAttribute(int *pi, hipDeviceAttribute_t attr, int deviceId);
typedef hipError_t HIPAPI thipDeviceTotalMem(size_t *bytes, hipDevice_t device);
typedef hipError_t HIPAPI thipGetDevicePropertiesR0600(void *prop, int deviceId);
typedef hipError_t HIPAPI thipSetDevice(int deviceId);
typedef hipError_t HIPAPI thipGetDevice(int *deviceId);
typedef hipError_t HIPAPI thipDeviceSynchronize(void);
typedef hipError_t HIPAPI thipDevicePrimaryCtxRetain(hipCtx_t *pctx, hipDevice_t dev);
typedef hipError_t HIPAPI thipDevicePrimaryCtxRelease(hipDevice_t dev);
typedef hipError_t HIPAPI thipMalloc(void **ptr, size_t size);
typedef hipError_t HIPAPI thipMallocManaged(void **devPtr, size_t size, unsigned int flags);
typedef hipError_t HIPAPI thipMemAdvise(const void *devPtr, size_t count, hipMemoryAdvise advice, int device);
typedef hipError_t HIPAPI thipFree(void *ptr);
typedef hipError_t HIPAPI thipMemGetInfo(size_t *free, size_t *total);
typedef hipError_t HIPAPI thipMemcpyHtoD(hipDeviceptr_t dst, const void *src, size_t sizeBytes);
typedef hipError_t HIPAPI thipMemcpyDtoH(void *dst, hipDeviceptr_t src, size_t sizeBytes);
typedef hipError_t HIPAPI thipMemcpyHtoDAsync(hipDeviceptr_t dst, const void *src, size_t sizeBytes, hipStream_t stream);
typedef hipError_t HIPAPI thipMemcpyDtoHAsync(void *dst, hipDeviceptr_t src, size_t sizeBytes, hipStream_t stream);
typedef hipError_t HIPAPI thipHostRegister(void *hostPtr, size_t sizeBytes, unsigned int flags);
typedef hipError_t HIPAPI thipHostUnregister(void *hostPtr);
typedef hipError_t HIPAPI thipMemGetAddressRange(hipDeviceptr_t *pbase, size_t *psize, hipDeviceptr_t dptr);
typedef hipError_t HIPAPI thipStreamSynchronize(hipStream_t stream);
typedef hipError_t HIPAPI thipEventCreate(hipEvent_t *event);
typedef hipError_t HIPAPI thipEventDestroy(hipEvent_t event);
typedef hipError_t HIPAPI thipEventRecord(hipEvent_t event, hipStream_t stream);
typedef hipError_t HIPAPI thipEventElapsedTime(float *ms, hipEvent_t start, hipEvent_t stop);
typedef hipError_t HIPAPI thipModuleLoadData(hipModule_t *module, const void *image);
typedef hipError_t HIPAPI thipModuleUnload(hipModule_t module);
typedef hipError_t HIPAPI thipModuleGetFunction(hipFunction_t *function, hipModule_t module, const char *kname);
typedef hipError_t HIPAPI thipModuleGetGlobal(hipDeviceptr_t *dptr, size_t *bytes, hipModule_t module, const char *name);
typedef enum hipFunction_attribute {
	HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0,
	HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1,
	HIP_FUNC_ATTRIBUTE_CONST_SIZE_BYTES = 2,
	HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = 3,
	HIP_FUNC_ATTRIBUTE_NUM_REGS = 4,
	HIP_FUNC_ATTRIBUTE_PTX_VERSION = 5,
	HIP_FUNC_ATTRIBUTE_BINARY_VERSION = 6,
	HIP_FUNC_ATTRIBUTE_CACHE_MODE_CA = 7,
	HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8,
	HIP_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = 9,
	HIP_FUNC_ATTRIBUTE_MAX
} hipFunction_attribute;
typedef hipError_t HIPAPI thipFuncGetAttribute(int *value, hipFunction_attribute attrib, hipFunction_t hfunc);
typedef hipError_t HIPAPI thipModuleLaunchKernel(hipFunction_t f,
	unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
	unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
	unsigned int sharedMemBytes, hipStream_t stream, void **kernelParams, void **extra);
typedef hipError_t HIPAPI thipMallocArray(hipArray_t *array, const hipChannelFormatDesc *desc,
	size_t width, size_t height, unsigned int flags);
typedef hipError_t HIPAPI thipFreeArray(hipArray_t array);
typedef hipError_t HIPAPI thipMemcpy2DToArray(hipArray_t dst, size_t wOffset, size_t hOffset,
	const void *src, size_t spitch, size_t width, size_t height, hipMemcpyKind kind);
typedef hipError_t HIPAPI thipCreateTextureObject(hipTextureObject_t *pTexObject,
	const hipResourceDesc *pResDesc, const hipTextureDesc *pTexDesc,
	const struct hipResourceViewDesc *pResViewDesc);
typedef hipError_t HIPAPI thipDestroyTextureObject(hipTextureObject_t textureObject);

extern thipGetErrorName *hipGetErrorName;
extern thipGetErrorString *hipGetErrorString;
extern thipInit *hipInit;
extern thipDriverGetVersion *hipDriverGetVersion;
extern thipRuntimeGetVersion *hipRuntimeGetVersion;
extern thipGetDeviceCount *hipGetDeviceCount;
extern thipDeviceGet *hipDeviceGet;
extern thipDeviceGetName *hipDeviceGetName;
extern thipDeviceGetAttribute *hipDeviceGetAttribute;
extern thipDeviceTotalMem *hipDeviceTotalMem;
extern thipGetDevicePropertiesR0600 *hipGetDevicePropertiesR0600;
extern thipSetDevice *hipSetDevice;
extern thipGetDevice *hipGetDevice;
extern thipDeviceSynchronize *hipDeviceSynchronize;
extern thipDevicePrimaryCtxRetain *hipDevicePrimaryCtxRetain;
extern thipDevicePrimaryCtxRelease *hipDevicePrimaryCtxRelease;
extern thipMalloc *hipMalloc;
extern thipMallocManaged *hipMallocManaged;
extern thipMemAdvise *hipMemAdvise;
extern thipFree *hipFree;
extern thipMemGetInfo *hipMemGetInfo;
extern thipMemcpyHtoD *hipMemcpyHtoD;
extern thipMemcpyDtoH *hipMemcpyDtoH;
extern thipMemcpyHtoDAsync *hipMemcpyHtoDAsync;
extern thipMemcpyDtoHAsync *hipMemcpyDtoHAsync;
extern thipHostRegister *hipHostRegister;
extern thipHostUnregister *hipHostUnregister;
extern thipMemGetAddressRange *hipMemGetAddressRange;
extern thipStreamSynchronize *hipStreamSynchronize;
extern thipEventCreate *hipEventCreate;
extern thipEventDestroy *hipEventDestroy;
extern thipEventRecord *hipEventRecord;
extern thipEventElapsedTime *hipEventElapsedTime;
extern thipModuleLoadData *hipModuleLoadData;
extern thipModuleUnload *hipModuleUnload;
extern thipModuleGetFunction *hipModuleGetFunction;
extern thipModuleGetGlobal *hipModuleGetGlobal;
extern thipFuncGetAttribute *hipFuncGetAttribute;
extern thipModuleLaunchKernel *hipModuleLaunchKernel;
extern thipMallocArray *hipMallocArray;
extern thipFreeArray *hipFreeArray;
extern thipMemcpy2DToArray *hipMemcpy2DToArray;
extern thipCreateTextureObject *hipCreateTextureObject;
extern thipDestroyTextureObject *hipDestroyTextureObject;

//------------------------------------------------------------------------------
// hiprtc functions
//------------------------------------------------------------------------------

typedef const char *HIPAPI thiprtcGetErrorString(hiprtcResult result);
typedef hiprtcResult HIPAPI thiprtcVersion(int *major, int *minor);
typedef hiprtcResult HIPAPI thiprtcCreateProgram(hiprtcProgram *prog, const char *src,
	const char *name, int numHeaders, const char **headers, const char **includeNames);
typedef hiprtcResult HIPAPI thiprtcCompileProgram(hiprtcProgram prog, int numOptions, const char * const *options);
typedef hiprtcResult HIPAPI thiprtcGetProgramLogSize(hiprtcProgram prog, size_t *logSizeRet);
typedef hiprtcResult HIPAPI thiprtcGetProgramLog(hiprtcProgram prog, char *log);
typedef hiprtcResult HIPAPI thiprtcGetCodeSize(hiprtcProgram prog, size_t *codeSizeRet);
typedef hiprtcResult HIPAPI thiprtcGetCode(hiprtcProgram prog, char *code);
typedef hiprtcResult HIPAPI thiprtcDestroyProgram(hiprtcProgram *prog);

extern thiprtcGetErrorString *hiprtcGetErrorString;
extern thiprtcVersion *hiprtcVersion;
extern thiprtcCreateProgram *hiprtcCreateProgram;
extern thiprtcCompileProgram *hiprtcCompileProgram;
extern thiprtcGetProgramLogSize *hiprtcGetProgramLogSize;
extern thiprtcGetProgramLog *hiprtcGetProgramLog;
extern thiprtcGetCodeSize *hiprtcGetCodeSize;
extern thiprtcGetCode *hiprtcGetCode;
extern thiprtcDestroyProgram *hiprtcDestroyProgram;

//------------------------------------------------------------------------------
// hipew
//------------------------------------------------------------------------------

enum {
	HIPEW_SUCCESS = 0,
	HIPEW_ERROR_OPEN_FAILED = -1,
	HIPEW_ERROR_ATEXIT_FAILED = -2,
	HIPEW_ERROR_MISSING_SYMBOL = -3
};

enum {
	HIPEW_INIT_HIP = 1,
	HIPEW_INIT_HIPRTC = 2
};

// Loads the requested libraries and resolves their symbols. It is safe to call
// it more than once: libraries are loaded only the first time.
int hipewInit(unsigned int flags);

// The name of the library actually loaded, empty if none
const char *hipewHipLibraryName(void);
const char *hipewHiprtcLibraryName(void);

#ifdef __cplusplus
}
#endif

#endif	/* __HIPEW_H__ */
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
