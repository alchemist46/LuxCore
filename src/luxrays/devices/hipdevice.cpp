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

#include "luxrays/core/hardwaredevice.h"
#include "luxrays/usings.h"
#if !defined(LUXRAYS_DISABLE_HIP)

#include <chrono>
#include <fstream>
#include <cstring>

#include "luxrays/devices/ocldevice.h"
#include "luxrays/devices/hipdevice.h"
#include "luxrays/kernels/kernels.h"
#include "luxrays/utils/hipkernel.h"

using namespace std;
using namespace luxrays;

//------------------------------------------------------------------------------
// HIP Device Description
//------------------------------------------------------------------------------

// If HIPRT can work on a device of the given architecture. A HIPRT library
// with precompiled kernels (LUXRAYS_HIPRT_PRECOMPILED, see
// src/luxrays/CMakeLists.txt) compiles nothing at run time, but its kernels
// exist only for the RDNA2/3/4 architectures, the ones of the baked families.
// Otherwise, HIPRT compiles its kernels at run time: it requires hiprtc, and
// it crashes without it (its loader doesn't check the entry points).
static bool IsHIPRTSupported(const string &arch) {
#if defined(LUXRAYS_HIPRT_PRECOMPILED)
	return !GetHIPBakedKernelFamily(arch).empty();
#else
	return isHiprtcAvailable;
#endif
}

HIPDeviceDescription::HIPDeviceDescription(hipDevice_t dev, const size_t devIndex) :
		DeviceDescription("HIPInitializingDevice", DEVICE_TYPE_HIP_GPU),
		hipDeviceIndex(devIndex), hipDevice(dev), useHIPRT(false) {
	char buff[256];
	CHECK_HIP_ERROR(hipDeviceGetName(buff, 256, hipDevice));
	name = string(buff);

	// The architecture name is only available in hipDeviceProp_t. The buffer
	// is larger than the R0600 struct in case a newer runtime writes more.
	vector<char> prop(HIPEW_DEVICE_PROP_R0600_SIZE + 4096, 0);
	CHECK_HIP_ERROR(hipGetDevicePropertiesR0600(prop.data(), (int)hipDeviceIndex));
	const char *gcnArchName = prop.data() + HIPEW_DEVICE_PROP_R0600_GCN_ARCH_NAME_OFFSET;
	arch = string(gcnArchName, strnlen(gcnArchName, HIPEW_DEVICE_PROP_GCN_ARCH_NAME_SIZE));
	if (arch.compare(0, 3, "gfx") != 0)
		throw runtime_error("Unable to read the architecture of HIP device " + name + ": " + arch);

	useHIPRT = isHiprtAvailable && IsHIPRTSupported(arch);
}

HIPDeviceDescription::~HIPDeviceDescription() {
}

int HIPDeviceDescription::GetComputeUnits() const {
	int v;
	CHECK_HIP_ERROR(hipDeviceGetAttribute(&v, hipDeviceAttributeMultiprocessorCount, (int)hipDeviceIndex));

	return v;
}

int HIPDeviceDescription::GetHIPWarpSize() const {
	int v;
	CHECK_HIP_ERROR(hipDeviceGetAttribute(&v, hipDeviceAttributeWarpSize, (int)hipDeviceIndex));

	return v;
}

u_int HIPDeviceDescription::GetNativeVectorWidthFloat() const {
	return 1;
}

size_t HIPDeviceDescription::GetMaxMemory() const {
	size_t bytes;
	CHECK_HIP_ERROR(hipDeviceTotalMem(&bytes, hipDevice));

	return bytes;
}

size_t HIPDeviceDescription::GetMaxMemoryAllocSize() const {
	return numeric_limits<size_t>::max();
}

bool HIPDeviceDescription::HasOutOfCoreMemorySupport() const {
	int v;
	CHECK_HIP_ERROR(hipDeviceGetAttribute(&v, hipDeviceAttributeManagedMemory, (int)hipDeviceIndex));

	return (v == 1);
}

void HIPDeviceDescription::AddDeviceDescs(std::vector<DeviceDescriptionUPtr> &descriptions) {
	int devCount;
	CHECK_HIP_ERROR(hipGetDeviceCount(&devCount));

	for (int i = 0; i < devCount; i++) {
		hipDevice_t device;
		CHECK_HIP_ERROR(hipDeviceGet(&device, i));

		descriptions.push_back(std::make_unique<HIPDeviceDescription>(device, i));
	}
}

//------------------------------------------------------------------------------
// HIPDevice
//------------------------------------------------------------------------------

// HIP has no context stack like CUDA: the current device is a per-thread
// setting, so Push/PopThreadCurrentDevice() keep their own stack
static thread_local vector<int> hipThreadDeviceStack;

HIPDevice::HIPDevice(
		const Context & context,
		HIPDeviceDescriptionConstRef desc,
		const size_t devIndex) :
		Device(context, devIndex),
		deviceDesc(desc), hiprtCtx(nullptr), hiprtCtxInitialized(false) {
	deviceName = (desc.GetName() + " HIPIntersect").c_str();

	kernelCache = std::make_unique<hipKernelPersistentCache>("LUXRAYS_" LUXRAYS_VERSION, desc.GetHIPArch());

	// Like cuCtxCreate() for CUDA, leave the new device current on this thread
	CHECK_HIP_ERROR(hipSetDevice((int)deviceDesc.GetHIPDeviceIndex()));
}

HIPDevice::~HIPDevice() {
	if (loadedModules.size() > 0) {
		PushThreadCurrentDevice();

		// Free all loaded modules
		for (auto &m : loadedModules) {
			CHECK_HIP_ERROR(hipModuleUnload(m));
		}
		loadedModules.clear();

		PopThreadCurrentDevice();
	}

	if (hiprtCtx) {
		// It unloads the trace kernel modules too
		CHECK_HIPRT_ERROR(hiprtDestroyContext(hiprtCtx));
		CHECK_HIP_ERROR(hipDevicePrimaryCtxRelease(deviceDesc.GetHIPDevice()));
	}
}

hiprtContext HIPDevice::GetHIPRTContext() {
	if (hiprtCtxInitialized)
		return hiprtCtx;
	hiprtCtxInitialized = true;

	if (isHiprtAvailable && deviceDesc.useHIPRT && !IsHIPRTSupported(deviceDesc.GetHIPArch())) {
		// Requested with hip.hiprt.devices.select
#if defined(LUXRAYS_HIPRT_PRECOMPILED)
		const string reason = "its kernels are not precompiled for this architecture";
#else
		const string reason = "it requires hiprtc";
#endif
		LR_LOG(deviceContext, "WARNING HIPRT can't be used on device " << deviceName << " (" <<
				deviceDesc.GetHIPArch() << "): " << reason);
	} else if (isHiprtAvailable && deviceDesc.useHIPRT) {
		// HIPRT works with the driver API: give it the primary context, the
		// one used by the runtime API calls of this device
		hipCtx_t hipCtx;
		CHECK_HIP_ERROR(hipDevicePrimaryCtxRetain(&hipCtx, deviceDesc.GetHIPDevice()));

		hiprtContextCreationInput input;
		input.ctxt = hipCtx;
		input.device = deviceDesc.GetHIPDevice();
		input.deviceType = hiprtDeviceAMD;

		const hiprtError result = hiprtCreateContext(HIPRTEW_API_VERSION, input, hiprtCtx);
		if (result != hiprtSuccess) {
			LR_LOG(deviceContext, "WARNING unable to create HIPRT context for device " << deviceName <<
					" (code: " << result << ")");
			hiprtCtx = nullptr;

			CHECK_HIP_ERROR(hipDevicePrimaryCtxRelease(deviceDesc.GetHIPDevice()));
		} else {
			// HIPRT logs to the standard output otherwise
			CHECK_HIPRT_ERROR(hiprtSetLogLevel(hiprtCtx, hiprtLogLevelError));

			// The kernels compiled by HIPRT (its build kernels, when they are
			// not precompiled; the trace kernel goes through the cache of
			// this device, see HiprtAccel) are cached next to the ones of
			// this device, so they are cleared together
			const std::filesystem::path hiprtCacheDir =
					hipKernelPersistentCache::GetCacheDir(kernelCache->GetApplicationName()) / "hiprt";
			std::filesystem::create_directories(hiprtCacheDir);
			CHECK_HIPRT_ERROR(hiprtSetCacheDirPath(hiprtCtx, hiprtCacheDir.string().c_str()));
		}
	}

	return hiprtCtx;
}

void HIPDevice::PushThreadCurrentDevice() {
	int current;
	CHECK_HIP_ERROR(hipGetDevice(&current));
	hipThreadDeviceStack.push_back(current);

	CHECK_HIP_ERROR(hipSetDevice((int)deviceDesc.GetHIPDeviceIndex()));
}

void HIPDevice::PopThreadCurrentDevice() {
	assert (!hipThreadDeviceStack.empty());

	const int previous = hipThreadDeviceStack.back();
	hipThreadDeviceStack.pop_back();

	CHECK_HIP_ERROR(hipSetDevice(previous));
}

//------------------------------------------------------------------------------
// Kernels handling for hardware (aka GPU) only applications
//------------------------------------------------------------------------------

vector<string> HIPDevice::AddKernelOpts(const vector<string> &programParameters) const {
	vector<string> hipProgramParameters = programParameters;

	const vector<string> deviceParameters = GetHIPDeviceKernelParameters();
	hipProgramParameters.insert(hipProgramParameters.end(),
			deviceParameters.begin(), deviceParameters.end());

	hipProgramParameters.insert(hipProgramParameters.end(),
			additionalCompileOpts.begin(), additionalCompileOpts.end());

	return hipProgramParameters;
}

// Debugging aid: with LUXRAYS_KERNEL_DUMP_DIR set, the source of each program
// is written there (with its compiler options on the first line), so it can be
// compiled offline, e.g. with the ROCm clang and warnings enabled
static void DumpKernelSource(const vector<string> &params, const string &source,
		const string &programName) {
	const char *dir = getenv("LUXRAYS_KERNEL_DUMP_DIR");
	if (!dir)
		return;

	string fileName = programName;
	for (char &c : fileName)
		if (!isalnum(c))
			c = '_';
	ofstream file((string(dir) + "/" + fileName + ".hip").c_str());
	file << "// " << oclKernelPersistentCache::ToOptsString(params) << "\n" << source;
}

// Debugging aid, the reverse of LUXRAYS_KERNEL_DUMP_DIR: with
// LUXRAYS_KERNEL_LOAD_DIR set, a program with a file there (as written by the
// dump, the first line is skipped) is compiled from that file instead, so a
// kernel change can be tried without rebuilding
static void LoadKernelSource(string &source, const string &programName) {
	const char *dir = getenv("LUXRAYS_KERNEL_LOAD_DIR");
	if (!dir)
		return;

	string fileName = programName;
	for (char &c : fileName)
		if (!isalnum(c))
			c = '_';
	ifstream file((string(dir) + "/" + fileName + ".hip").c_str());
	if (!file)
		return;

	string optsLine;
	getline(file, optsLine);
	source.assign(istreambuf_iterator<char>(file), istreambuf_iterator<char>());
}

string HIPDevice::GetKernelSource(const string &kernelSource) const {
	return GetHIPKernelSource(kernelSource);
}

bool HIPDevice::HasPrecompiledPrograms() const {
	return !kernelCache->GetPrecompiledFamily().empty();
}

bool HIPDevice::HasPrecompiledProgram(const std::vector<std::string> &programParameters,
		const std::string &programSource) const {
	return kernelCache->HasPrecompiled(AddKernelOpts(programParameters),
			GetKernelSource(programSource));
}

HardwareDeviceProgramUPtr HIPDevice::CompileProgram(
		const std::vector<std::string> &programParameters,
		const std::string &programSource,
		const std::string &programName
) {
	auto hipProgramParameters = AddKernelOpts(programParameters);

	LR_LOG(deviceContext, "[" << programName << "] Compiler options: " << oclKernelPersistentCache::ToOptsString(hipProgramParameters));
	LR_LOG(deviceContext, "[" << programName << "] Compiling kernels for " << deviceDesc.GetHIPArch());
	LR_LOG(deviceContext, "[" << programName << "] Cache directory: " << kernelCache->GetCacheDir(kernelCache->GetApplicationName()));

	auto startTime = std::chrono::high_resolution_clock::now();

	string hipProgramSource = GetKernelSource(programSource);
	DumpKernelSource(hipProgramParameters, hipProgramSource, programName);
	LoadKernelSource(hipProgramSource, programName);

	hipKernelPersistentCache::CodeOrigin origin;
	string error;
	hipModule_t module = kernelCache->Compile(hipProgramParameters, hipProgramSource, programName, &origin, &error);
	if (!module) {
		LR_LOG(deviceContext, "[" << programName << "] HIP program compilation error: " << endl << error);

		throw runtime_error(programName + " HIP program compilation error");
	} else {
		if (error.length() > 0) {
			LR_LOG(deviceContext, "[" << programName << "] HIP program compilation warnings: " << endl << error);
		}
	}

	auto endTime = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
	LR_LOG(deviceContext, "[" << programName << "] Compilation completed in " << duration.count() << " ms" <<
			kernelCache->GetCodeOriginDesc(origin));

	auto hipDeviceProgram = std::make_unique<HIPDeviceProgram>();
	hipDeviceProgram->Set(module);
	loadedModules.push_back(module);

	return static_cast<HardwareDeviceProgramUPtr>(std::move(hipDeviceProgram));
}

std::vector<HardwareDeviceProgramUPtr> HIPDevice::CompilePrograms(
	const std::vector<ProgramRequest> &requests
) {
	if (requests.empty())
		return std::vector<HardwareDeviceProgramUPtr>();

	auto startTime = std::chrono::high_resolution_clock::now();

	// Apply the same preprocessing of CompileProgram() to all requests
	std::vector<std::tuple<std::vector<std::string>, std::string, std::string>> kernels;
	kernels.reserve(requests.size());
	for (const auto &request : requests) {
		auto hipProgramParameters = AddKernelOpts(request.parameters);

		LR_LOG(deviceContext, "[" << request.name << "] Compiler options: " << oclKernelPersistentCache::ToOptsString(hipProgramParameters));

		kernels.push_back(std::make_tuple(hipProgramParameters,
			GetKernelSource(request.source), request.name));
		DumpKernelSource(hipProgramParameters, std::get<1>(kernels.back()), request.name);
		LoadKernelSource(std::get<1>(kernels.back()), request.name);
	}

	LR_LOG(deviceContext, "[" << requests.size() << " programs] Compiling kernels in parallel for " << deviceDesc.GetHIPArch());

	std::vector<hipKernelPersistentCache::CodeOrigin> origins;
	std::vector<std::string> errors;
	std::vector<hipModule_t> modules = kernelCache->CompileMultiple(kernels, &origins, &errors);

	// Keep track of the loaded modules before checking for errors, so they are
	// unloaded in any case
	for (auto m : modules) {
		if (m)
			loadedModules.push_back(m);
	}

	for (std::size_t i = 0; i < requests.size(); ++i) {
		if (!modules[i]) {
			LR_LOG(deviceContext, "[" << requests[i].name << "] HIP program compilation error: " << endl << errors[i]);

			throw runtime_error(requests[i].name + " HIP program compilation error");
		} else {
			if (errors[i].length() > 0) {
				LR_LOG(deviceContext, "[" << requests[i].name << "] HIP program compilation warnings: " << endl << errors[i]);
			}
		}
	}

	std::vector<HardwareDeviceProgramUPtr> programs;
	programs.reserve(requests.size());
	for (std::size_t i = 0; i < requests.size(); ++i) {
		auto hipDeviceProgram = std::make_unique<HIPDeviceProgram>();
		hipDeviceProgram->Set(modules[i]);

		programs.push_back(static_cast<HardwareDeviceProgramUPtr>(std::move(hipDeviceProgram)));
	}

	auto endTime = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
	for (std::size_t i = 0; i < requests.size(); ++i) {
		LR_LOG(deviceContext, "[" << requests[i].name << "] Compilation completed in " <<
			duration.count() << " ms" << kernelCache->GetCodeOriginDesc(origins[i]));
	}

	return programs;
}

HardwareDeviceKernelUPtr HIPDevice::GetKernel(
		HardwareDeviceProgramRef program,
		const string &kernelName
) {
	auto [kernel, hipDeviceKernel] =
		CreateUniquePtr<HardwareDeviceKernel, HIPDeviceKernel>();

	auto& hipDeviceProgram = dynamic_cast<HIPDeviceProgramRef>(program);

	hipFunction_t function;
	CHECK_HIP_ERROR(
		hipModuleGetFunction(
			&function, hipDeviceProgram.GetModule(), kernelName.c_str())
		);

	int numRegs = 0, scratchBytes = 0;
	if ((hipFuncGetAttribute(&numRegs, HIP_FUNC_ATTRIBUTE_NUM_REGS, function) == hipSuccess) &&
			(hipFuncGetAttribute(&scratchBytes, HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, function) == hipSuccess))
		LR_LOG(deviceContext, "[" << kernelName << "] " << numRegs << " VGPRs, " << scratchBytes << " bytes of scratch");

	hipDeviceKernel.Set(function);

	return std::move(kernel);
}

bool HIPDevice::SetProgramConstant(HardwareDeviceProgramRef program,
		const string &name, const size_t size, const void *data) {
	auto& hipDeviceProgram = dynamic_cast<HIPDeviceProgramRef>(program);

	hipDeviceptr_t ptr;
	size_t bytes;
	if (hipModuleGetGlobal(&ptr, &bytes, hipDeviceProgram.GetModule(), name.c_str()) != hipSuccess)
		return false;
	if (bytes != size)
		throw runtime_error("Wrong size of the program constant " + name + " in HIPDevice::SetProgramConstant(): " +
				ToString(bytes) + " instead of " + ToString(size));

	// The kernels of the program may still be running
	FinishQueue();
	CHECK_HIP_ERROR(hipMemcpyHtoD(ptr, data, size));

	return true;
}

HardwareDeviceKernelUPtr HIPDevice::GetKernel(hipFunction_t function) {
	auto [kernel, hipDeviceKernel] =
		CreateUniquePtr<HardwareDeviceKernel, HIPDeviceKernel>();

	hipDeviceKernel.Set(function);

	return std::move(kernel);
}

u_int HIPDevice::GetKernelWorkGroupSize(HardwareDeviceKernelRPtr kernel) {
	assert (kernel);
	assert (!kernel->IsNull());

	// Same as CUDA: the global sizes used by the render engines are rounded to
	// this value
	return 32;
}

void HIPDevice::SetKernelArg(HardwareDeviceKernelRPtr kernel,
		const u_int index, const size_t size, const void *arg) {
	assert (kernel);
	assert (!kernel->IsNull());

	auto& hipDeviceKernel = dynamic_cast<HIPDeviceKernelRef>(*kernel);

	if (index >= hipDeviceKernel.args.size())
		hipDeviceKernel.args.resize(index + 1, nullptr);

	void *argCpy;
	if (arg) {
		// Copy the argument
		argCpy = new char[size];
		memcpy(argCpy, arg, size);
	} else {
		hipDeviceptr_t p = nullptr;
		argCpy = new char[sizeof(hipDeviceptr_t)];
		memcpy(argCpy, &p, sizeof(hipDeviceptr_t));
	}

	if (hipDeviceKernel.args[index]) {
		delete[] (char *)hipDeviceKernel.args[index];
		hipDeviceKernel.args[index] = nullptr;
	}

	hipDeviceKernel.args[index] = argCpy;
}

void HIPDevice::SetKernelArgBuffer(HardwareDeviceKernelRPtr kernel,
		const u_int index, const HardwareDeviceBuffer *buff) {
	assert (kernel);
	assert (!kernel->IsNull());

	if (buff) {
		const HIPDeviceBuffer *hipDeviceBuff = dynamic_cast<const HIPDeviceBuffer *>(buff);
		assert (hipDeviceBuff);

		SetKernelArg(kernel, index, sizeof(hipDeviceptr_t), &hipDeviceBuff->hipBuff);
	} else
		SetKernelArg(kernel, index, sizeof(hipDeviceptr_t), nullptr);
}

static void ConvertHardwareRange(const u_int globalSize, const u_int workGroupSize,
		u_int &block, u_int &thread) {
	if (workGroupSize == 0) {
		block = globalSize / 32;
		thread = 32;
	} else {
		block = globalSize / workGroupSize;
		thread = workGroupSize;
	}
}

static void ConvertHardwareRange(const HardwareDeviceRange &globalSize,
		const HardwareDeviceRange &workGroupSize,
		u_int &blockX, u_int &blockY, u_int &blockZ,
		u_int &threadX, u_int &threadY, u_int &threadZ) {
	if (globalSize.dimensions == 1) {
		ConvertHardwareRange(globalSize.sizes[0], workGroupSize.sizes[0], blockX, threadX);
		blockY = 1;
		threadY = 1;
		blockZ = 1;
		threadZ = 1;
	} else if (globalSize.dimensions == 2) {
		ConvertHardwareRange(globalSize.sizes[0], workGroupSize.sizes[0], blockX, threadX);
		ConvertHardwareRange(globalSize.sizes[1], workGroupSize.sizes[1], blockY, threadY);
		blockZ = 1;
		threadZ = 1;
	} else {
		ConvertHardwareRange(globalSize.sizes[0], workGroupSize.sizes[0], blockX, threadX);
		ConvertHardwareRange(globalSize.sizes[1], workGroupSize.sizes[1], blockY, threadY);
		ConvertHardwareRange(globalSize.sizes[2], workGroupSize.sizes[2], blockZ, threadZ);
	}
}

void HIPDevice::EnqueueKernel(HardwareDeviceKernelRPtr kernel,
			const HardwareDeviceRange &globalSize,
			const HardwareDeviceRange &workGroupSize) {
	assert (kernel);
	assert (!kernel->IsNull());

	auto& hipDeviceKernel = dynamic_cast<HIPDeviceKernelRef>(*kernel);

	u_int blockX, blockY, blockZ;
	u_int threadX, threadY, threadZ;
	ConvertHardwareRange(globalSize, workGroupSize,
			blockX, blockY, blockZ,
			threadX, threadY, threadZ);

	// The kernels are compiled for a limited work group size
	if (threadX * threadY * threadZ > hipKernelPersistentCache::MaxWorkGroupSize)
		throw runtime_error("HIP work group size " + ToString(threadX * threadY * threadZ) +
				" is larger than the supported maximum of " + ToString(hipKernelPersistentCache::MaxWorkGroupSize) +
				" (check opencl.gpu.workgroup.size)");

	CHECK_HIP_ERROR(hipModuleLaunchKernel(hipDeviceKernel.hipKernel,
			blockX, blockY, blockZ,  // blocks
			threadX, threadY, threadZ,  // threads
			0, nullptr,
			hipDeviceKernel.args.data(),
			nullptr));
}

void HIPDevice::EnqueueReadBuffer(const HardwareDeviceBuffer *buff,
		const bool blocking, const size_t size, void *ptr) {
	assert (buff);
	assert (!buff->IsNull());

	const HIPDeviceBuffer *hipDeviceBuff = dynamic_cast<const HIPDeviceBuffer *>(buff);
	assert (hipDeviceBuff);

	if (blocking) {
		CHECK_HIP_ERROR(hipMemcpyDtoH(ptr, hipDeviceBuff->hipBuff, size));
	} else {
		CHECK_HIP_ERROR(hipMemcpyDtoHAsync(ptr, hipDeviceBuff->hipBuff, size, nullptr));
	}
}

void HIPDevice::EnqueueWriteBuffer(const HardwareDeviceBuffer *buff,
		const bool blocking, const size_t size, const void *ptr) {
	assert (buff);
	assert (!buff->IsNull());

	const HIPDeviceBuffer *hipDeviceBuff = dynamic_cast<const HIPDeviceBuffer *>(buff);
	assert (hipDeviceBuff);

	if (blocking) {
		CHECK_HIP_ERROR(hipMemcpyHtoD(hipDeviceBuff->hipBuff, ptr, size));
	} else {
		CHECK_HIP_ERROR(hipMemcpyHtoDAsync(hipDeviceBuff->hipBuff, ptr, size, nullptr));
	}
}

void HIPDevice::EnqueueWriteBufferRange(const HardwareDeviceBuffer *buff,
		const bool blocking, const size_t offset, const size_t size, const void *ptr) {
	assert (buff);
	assert (!buff->IsNull());

	const HIPDeviceBuffer *hipDeviceBuff = dynamic_cast<const HIPDeviceBuffer *>(buff);
	assert (hipDeviceBuff);

	hipDeviceptr_t dst = (hipDeviceptr_t)((char *)hipDeviceBuff->hipBuff + offset);
	if (blocking) {
		CHECK_HIP_ERROR(hipMemcpyHtoD(dst, ptr, size));
	} else {
		CHECK_HIP_ERROR(hipMemcpyHtoDAsync(dst, ptr, size, nullptr));
	}
}

void HIPDevice::FlushQueue() {
}

void HIPDevice::FinishQueue() {
	CHECK_HIP_ERROR(hipStreamSynchronize(nullptr));
}

void HIPDevice::RegisterHostMemory(void *ptr, const size_t size) {
	// LUXRAYS_DISABLE_PINNED_MEMORY: for the A/B measurements of the transfers
	if (getenv("LUXRAYS_DISABLE_PINNED_MEMORY"))
		return;

	// Not fatal: the transfers just stay staged through pageable memory
	const hipError_t err = hipHostRegister(ptr, size, 0);
	if (err != hipSuccess)
		LR_LOG(deviceContext, "[" << GetName() << "] WARNING: hipHostRegister() failed: " << hipGetErrorString(err));
}

void HIPDevice::UnregisterHostMemory(void *ptr) {
	hipHostUnregister(ptr);
}

//------------------------------------------------------------------------------
// Memory management for hardware (aka GPU) only applications
//------------------------------------------------------------------------------

void HIPDevice::AllocBuffer(HardwareDeviceBuffer **hdBuff, const BufferType type,
		void *src, const size_t size, const string &desc) {
	if (!*hdBuff)
		*hdBuff = new HIPDeviceBuffer();

	HIPDeviceBuffer *hipDeviceBuff = dynamic_cast<HIPDeviceBuffer *>(*hdBuff);
	assert (hipDeviceBuff);

	hipDeviceptr_t *buff = &hipDeviceBuff->hipBuff;

	// Handle the case of an empty buffer
	if (!size) {
		if (*buff) {
			// Free the buffer
			FreeMemory(hipDeviceBuff->hipBuffSize);

			CHECK_HIP_ERROR(hipFree(*buff));
		}

		*buff = nullptr;
		hipDeviceBuff->hipBuffSize = 0;

		return;
	}

	if (*buff) {
		// Check the size of the already allocated buffer
		if (size == hipDeviceBuff->hipBuffSize) {
			// I can reuse the buffer; just update the content
			if (src) {
				CHECK_HIP_ERROR(hipMemcpyHtoD(*buff, src, size));
			}

			return;
		} else {
			// Free the buffer
			FreeMemory(hipDeviceBuff->hipBuffSize);

			CHECK_HIP_ERROR(hipFree(*buff));
			*buff = nullptr;
			hipDeviceBuff->hipBuffSize = 0;
		}
	}

	if (desc != "")
		LR_LOG(deviceContext, "[Device " << GetName() << "] " << desc <<
				" buffer size: " << ToMemString(size) << ((type & BUFFER_TYPE_OUT_OF_CORE) ? " (OUT OF CORE)" : ""));

	// Check if I was asked for out of core support
	const bool outOfCore = (type & BUFFER_TYPE_OUT_OF_CORE) && deviceDesc.HasOutOfCoreMemorySupport();
	if ((type & BUFFER_TYPE_OUT_OF_CORE) && !outOfCore) {
		LR_LOG(deviceContext, "WARNING: HIP device " << deviceDesc.GetName() << " doesn't support out of core memory buffers: " << desc);
	}

	if (outOfCore) {
		CHECK_HIP_ERROR(hipMallocManaged(buff, size, hipMemAttachGlobal));

		if (type & BUFFER_TYPE_READ_ONLY) {
			CHECK_HIP_ERROR(hipMemAdvise(*buff, size, hipMemAdviseSetReadMostly, (int)deviceDesc.GetHIPDeviceIndex()));
		}
	} else {
		CHECK_HIP_ERROR(hipMalloc(buff, size));
	}
	hipDeviceBuff->hipBuffSize = size;

	// The source is copied synchronously: callers can free it as soon as this
	// method returns
	if (src) {
		CHECK_HIP_ERROR(hipMemcpyHtoD(*buff, src, size));
	}

	AllocMemory(size);
}

void HIPDevice::FreeBuffer(HardwareDeviceBuffer **buff) {
	if (*buff) {
		if (!(*buff)->IsNull()) {
			HIPDeviceBuffer *hipDeviceBuff = dynamic_cast<HIPDeviceBuffer *>(*buff);
			assert (hipDeviceBuff);

			FreeMemory(hipDeviceBuff->hipBuffSize);

			CHECK_HIP_ERROR(hipFree(hipDeviceBuff->hipBuff));
		}

		delete *buff;
		*buff = nullptr;
	}
}

#endif
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
