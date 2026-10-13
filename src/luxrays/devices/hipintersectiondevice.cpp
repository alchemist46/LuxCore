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

#if !defined(LUXRAYS_DISABLE_HIP)

#include "luxrays/devices/hipintersectiondevice.h"

using namespace std;

namespace luxrays {

//------------------------------------------------------------------------------
// HIP IntersectionDevice
//------------------------------------------------------------------------------

HIPIntersectionDevice::HIPIntersectionDevice(
	const Context & context,
	HIPDeviceDescriptionConstRef desc,
	const size_t devIndex
) :
	Device(context, devIndex), HIPDevice(context, desc, devIndex),
	HardwareIntersectionDevice(), kernel(nullptr) {
}

HIPIntersectionDevice::~HIPIntersectionDevice() {
}

void HIPIntersectionDevice::SetDataSet(DataSetSPtr newDataSet) {
	IntersectionDevice::SetDataSet(newDataSet);

	if (dataSet) {
		const AcceleratorType accelType = dataSet->GetAcceleratorType();
		if (accelType != ACCEL_AUTO) {
			accel = dataSet->GetAccelerator(accelType);
		} else {
			if (GetHIPRTContext())
				accel = dataSet->GetAccelerator(ACCEL_HIPRT);
			else
				accel = dataSet->GetAccelerator(GetFallbackAcceleratorType());
		}
	}
}

AcceleratorType HIPIntersectionDevice::GetFallbackAcceleratorType() const {
	if (dataSet->RequiresInstanceSupport() || dataSet->RequiresMotionBlurSupport())
		return ACCEL_MBVH;
	else
		return ACCEL_BVH;
}

void HIPIntersectionDevice::Update() {
	kernel->Update(dataSet);
}

void HIPIntersectionDevice::Start() {
	HIPDevice::Start();

	// Compile required kernel
	try {
		kernel = accel->NewHardwareIntersectionKernel(*this);
	} catch (const exception &e) {
		// HIPRT is used by default: if it doesn't work (i.e. it isn't able to
		// compile its kernels), fall back to the LuxCore accelerators instead
		// of failing. An explicitly requested HIPRT accelerator still fails.
		if ((accel->GetType() != ACCEL_HIPRT) || (dataSet->GetAcceleratorType() != ACCEL_AUTO))
			throw;

		accel = dataSet->GetAccelerator(GetFallbackAcceleratorType());
		LR_LOG(deviceContext, "WARNING: unable to use HIPRT on device " << deviceName << ", using the " <<
				Accelerator::AcceleratorType2String(accel->GetType()) << " accelerator: " << e.what());

		kernel = accel->NewHardwareIntersectionKernel(*this);
	}
}

void HIPIntersectionDevice::Stop() {
	kernel.reset();

	HIPDevice::Stop();
}

void HIPIntersectionDevice::EnqueueTraceRayBuffer(HardwareDeviceBuffer *rayBuff,
			HardwareDeviceBuffer *rayHitBuff,
			const unsigned int rayCount) {
	// Enqueue the intersection kernel
	kernel->EnqueueTraceRayBuffer(rayBuff, rayHitBuff, rayCount);
	statsTotalDataParallelRayCount += rayCount;
}

}

#endif
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
