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
#include <cstring>

#include <boost/algorithm/string/replace.hpp>

#include "luxrays/devices/ocldevice.h"
#include "luxrays/devices/hipdevice.h"
#include "luxrays/kernels/kernels.h"

using namespace std;
using namespace luxrays;

//------------------------------------------------------------------------------
// HIP Device Description
//------------------------------------------------------------------------------

HIPDeviceDescription::HIPDeviceDescription(hipDevice_t dev, const size_t devIndex) :
		DeviceDescription("HIPInitializingDevice", DEVICE_TYPE_HIP_GPU),
		hipDeviceIndex(devIndex), hipDevice(dev) {
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
		deviceDesc(desc) {
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

vector<string> HIPDevice::AddKernelOpts(const vector<string> &programParameters) {
	vector<string> hipProgramParameters = programParameters;

	// HIP kernels are compiled from the same CUDA-flavoured sources, so the
	// CUDA code paths are enabled too. LUXRAYS_HIP_DEVICE marks the few places
	// where HIP is different.
	hipProgramParameters.push_back("-D LUXRAYS_CUDA_DEVICE");
	hipProgramParameters.push_back("-D LUXRAYS_HIP_DEVICE");
#if defined (__APPLE__)
	hipProgramParameters.push_back("-D LUXRAYS_OS_APPLE");
#elif defined (WIN32)
	hipProgramParameters.push_back("-D LUXRAYS_OS_WINDOWS");
#elif defined (__linux__)
	hipProgramParameters.push_back("-D LUXRAYS_OS_LINUX");
#endif

	hipProgramParameters.insert(hipProgramParameters.end(),
			additionalCompileOpts.begin(), additionalCompileOpts.end());

	return hipProgramParameters;
}

string HIPDevice::GetKernelSource(const string &kernelSource) {
	// The pragma only covers functions: file scope tables (always declared at
	// the start of a line as "__constant <type> <name>[] = { ... }", while the
	// __constant pointer qualifiers are always indented or follow a "(") must
	// be explicitly moved to the device
	string source = "\n" + kernelSource;
	boost::replace_all(source, "\n__constant ", "\n__constant__ ");

	// The OpenCL sources don't mark device functions: NVRTC handles that with
	// --device-as-default-execution-space, hiprtc with this pragma
	return
		"#pragma clang force_cuda_host_device begin\n" +
		luxrays::ocl::KernelSource_cudadevice_oclemul_types +
		luxrays::ocl::KernelSource_cudadevice_math +
		luxrays::ocl::KernelSource_cudadevice_oclemul_funcs +
		source +
		"\n#pragma clang force_cuda_host_device end\n";
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

	const string hipProgramSource = GetKernelSource(programSource);

	bool cached;
	string error;
	hipModule_t module = kernelCache->Compile(hipProgramParameters, hipProgramSource, programName, &cached, &error);
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
			(cached ? " (cached)" : ""));

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
	}

	LR_LOG(deviceContext, "[" << requests.size() << " programs] Compiling kernels in parallel for " << deviceDesc.GetHIPArch());

	std::vector<bool> cached;
	std::vector<std::string> errors;
	std::vector<hipModule_t> modules = kernelCache->CompileMultiple(kernels, &cached, &errors);

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
			duration.count() << " ms" << (cached[i] ? " (cached)" : ""));
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

void HIPDevice::FlushQueue() {
}

void HIPDevice::FinishQueue() {
	CHECK_HIP_ERROR(hipStreamSynchronize(nullptr));
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
