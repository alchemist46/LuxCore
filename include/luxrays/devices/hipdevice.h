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

#ifndef _LUXRAYS_HIPDEVICE_H
#define	_LUXRAYS_HIPDEVICE_H

#include "luxrays/core/hardwaredevice.h"
#include "luxrays/core/intersectiondevice.h"
#include "luxrays/utils/hip.h"
#include "luxrays/utils/hipcache.h"

#include <mutex>
#include <unordered_map>

#if !defined(LUXRAYS_DISABLE_HIP)

#if defined(LUXRAYS_DISABLE_OPENCL)
#error "HIP support can be enabled only if also OpenCL support is enabled"
#endif

namespace luxrays {

//------------------------------------------------------------------------------
// HIPDeviceDescription
//------------------------------------------------------------------------------

class HIPDeviceDescription : public DeviceDescription {
public:
	HIPDeviceDescription(hipDevice_t hipDevice, const size_t devIndex);
	virtual ~HIPDeviceDescription();

	virtual int GetComputeUnits() const;
	virtual u_int GetNativeVectorWidthFloat() const;
	virtual size_t GetMaxMemory() const;
	virtual size_t GetMaxMemoryAllocSize() const;
	virtual bool HasOutOfCoreMemorySupport() const;

	hipDevice_t GetHIPDevice() const { return hipDevice; }
	size_t GetHIPDeviceIndex() const { return hipDeviceIndex; }
	// The AMD GPU architecture name, i.e. "gfx1030" (it can include target
	// features, i.e. "gfx90a:sramecc+:xnack-")
	const std::string &GetHIPArch() const { return arch; }
	int GetHIPWarpSize() const;
	void SetHIPUseHIPRT(const bool v) { useHIPRT = v; }
	bool GetHIPUseHIPRT() const { return useHIPRT; }

	friend class Context;
	friend class HIPDevice;

protected:
	static void AddDeviceDescs(std::vector<DeviceDescriptionUPtr> &descriptions);

	size_t hipDeviceIndex;
	hipDevice_t hipDevice;
	std::string arch;
	bool useHIPRT;
};

//------------------------------------------------------------------------------
// HIPDeviceKernel
//------------------------------------------------------------------------------

class HIPDeviceKernel : public HardwareDeviceKernel {
public:
	HIPDeviceKernel() : hipKernel(nullptr) { }
	virtual ~HIPDeviceKernel() {
		for (auto p : args)
			delete [] (char *)p;
	}

	bool IsNull() const {
		return (hipKernel == nullptr);
	}

	friend class HIPDevice;

protected:
	void Set(hipFunction_t kernel, const std::string &kernelName) {
		hipKernel = kernel;
		name = kernelName;
	}

	hipFunction_t Get() { return hipKernel; }

	hipFunction_t hipKernel;
	std::string name;
	std::vector<void *> args;
};

//------------------------------------------------------------------------------
// HIPDeviceProgram
//------------------------------------------------------------------------------

class HIPDeviceProgram : public HardwareDeviceProgram {
public:
	HIPDeviceProgram() : hipModule(nullptr) { }
	virtual ~HIPDeviceProgram() {
		// Module is unloaded by the device destructor
	}

	bool IsNull() const {
		return (hipModule == nullptr);
	}

	friend class HIPDevice;

protected:
	void Set(hipModule_t m) {
		hipModule = m;
	}

	hipModule_t GetModule() { return hipModule; }

	hipModule_t hipModule;
};

//------------------------------------------------------------------------------
// HIPDeviceBuffer
//------------------------------------------------------------------------------

class HIPDeviceBuffer : public HardwareDeviceBuffer {
public:
	HIPDeviceBuffer() : hipBuff(nullptr), hipBuffSize(0) { }
	virtual ~HIPDeviceBuffer() {
	}

	bool IsNull() const {
		return (hipBuff == nullptr);
	}

	size_t GetSize() const {
		assert (hipBuff != nullptr);

		return hipBuffSize;
	}

	//--------------------------------------------------------------------------
	// HIPDevice specific methods
	//--------------------------------------------------------------------------

	hipDeviceptr_t &GetHIPDevicePointer() { return hipBuff; }

	friend class HIPDevice;

protected:
	hipDeviceptr_t hipBuff;
	size_t hipBuffSize;
};

//------------------------------------------------------------------------------
// HIPDevice
//------------------------------------------------------------------------------

class HIPDevice : virtual public HardwareDevice {
public:
	HIPDevice(const Context & context,
		HIPDeviceDescriptionConstRef desc, const size_t devIndex);
	virtual ~HIPDevice();

	virtual const DeviceDescription& GetDeviceDesc() const { return deviceDesc; }

	virtual void PushThreadCurrentDevice();
	virtual void PopThreadCurrentDevice();

	//--------------------------------------------------------------------------
	// Kernels handling for hardware (aka GPU) only applications
	//--------------------------------------------------------------------------

	virtual HardwareDeviceProgramUPtr CompileProgram(
		const std::vector<std::string> &programParameters,
		const std::string &programSource,
		const std::string &programName
	) override;

	// Compile a batch of programs in parallel
	virtual std::vector<HardwareDeviceProgramUPtr> CompilePrograms(
		const std::vector<ProgramRequest> &requests
	) override;

	virtual HardwareDeviceKernelUPtr GetKernel(
		HardwareDeviceProgramRef program,
		const std::string &kernelName
	) override;
	virtual u_int GetKernelWorkGroupSize(HardwareDeviceKernelRPtr kernel) override;
	virtual bool HasPrecompiledPrograms() const override;
	virtual bool HasPrecompiledProgram(const std::vector<std::string> &programParameters,
		const std::string &programSource) const override;
	virtual bool HasProgramConstants() const override { return true; }
	virtual bool SetProgramConstant(HardwareDeviceProgramRef program,
			const std::string &name, const size_t size, const void *data) override;
	virtual void SetKernelArg(HardwareDeviceKernelRPtr kernel,
			const u_int index, const size_t size, const void *arg) override;

	virtual void EnqueueKernel(HardwareDeviceKernelRPtr kernel,
			const HardwareDeviceRange &globalSize,
			const HardwareDeviceRange &workGroupSize) override;
	virtual void EnqueueReadBuffer(const HardwareDeviceBuffer *buff,
			const bool blocking, const size_t size, void *ptr) override;
	virtual void EnqueueWriteBuffer(const HardwareDeviceBuffer *buff,
			const bool blocking, const size_t size, const void *ptr) override;
	virtual void EnqueueWriteBufferRange(const HardwareDeviceBuffer *buff,
			const bool blocking, const size_t offset, const size_t size, const void *ptr) override;
	virtual void FlushQueue() override;
	virtual void FinishQueue() override;
	virtual void RegisterHostMemory(void *ptr, const size_t size) override;
	virtual void UnregisterHostMemory(void *ptr) override;
	virtual std::string GetKernelTimingReport() override;

	//--------------------------------------------------------------------------
	// Memory management for hardware (aka GPU) only applications
	//--------------------------------------------------------------------------

	virtual void AllocBuffer(HardwareDeviceBuffer **buff, const BufferType type,
		void *src, const size_t size, const std::string &desc = "");
	virtual void FreeBuffer(HardwareDeviceBuffer **buff);

	//--------------------------------------------------------------------------
	// HIPDevice specific methods
	//--------------------------------------------------------------------------

	const std::unique_ptr<luxrays::hipKernelPersistentCache>&
	GetHIPKernelCache() { return kernelCache; }

	// nullptr if HIPRT isn't available or it isn't enabled for this device. The
	// context is created by the first call: devices used only for the image
	// pipeline don't need it.
	hiprtContext GetHIPRTContext();

	std::vector<std::string> AddKernelOpts(const std::vector<std::string> &programParameters) const;
	std::string GetKernelSource(const std::string &kernelSource) const;

	// Wraps a function of a module not loaded by this device (i.e. a HIPRT
	// trace kernel), so it can be used with SetKernelArg()/EnqueueKernel()
	HardwareDeviceKernelUPtr GetKernel(hipFunction_t function, const std::string &kernelName);

	friend class Context;

protected:
	virtual void SetKernelArgBuffer(
		HardwareDeviceKernelRPtr kernel,
		const u_int index, const HardwareDeviceBuffer *buff
	) override;

	HIPDeviceDescriptionConstRef deviceDesc;
	std::vector<hipModule_t> loadedModules;

	std::unique_ptr<luxrays::hipKernelPersistentCache> kernelCache;

	hiprtContext hiprtCtx;
	bool hiprtCtxInitialized;

	// LUXRAYS_HIP_KERNEL_TIMING: a pair of events around each kernel launch,
	// read back by FinishQueue(), which waits for them anyway
	struct KernelTime {
		u_int launches = 0;
		double ms = 0.0;
	};
	struct KernelLaunchEvents {
		KernelTime *time;
		hipEvent_t start, stop;
	};
	void ReadKernelLaunchEvents();

	bool kernelTiming;
	std::mutex kernelTimingMutex;
	std::unordered_map<std::string, KernelTime> kernelTimes;
	std::vector<KernelLaunchEvents> pendingLaunchEvents;
	std::vector<hipEvent_t> freeEvents;
};

}

#endif

#endif	/* _LUXRAYS_HIPDEVICE_H */
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
