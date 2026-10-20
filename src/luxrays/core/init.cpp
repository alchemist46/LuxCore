/***************************************************************************
 * Copyright 1998-2020 by authors (see AUTHORS.txt)                        *
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

#include <iostream>
#include <filesystem>

#include "luxrays/luxrays.h"
#if !defined(LUXRAYS_DISABLE_OPENCL)
#include "luxrays/utils/ocl.h"
#endif
#if !defined(LUXRAYS_DISABLE_CUDA)
#include "luxrays/utils/cuda.h"
#endif
#if !defined(LUXRAYS_DISABLE_HIP)
#include "luxrays/utils/hip.h"
#endif

using namespace std;
using namespace luxrays;

//------------------------------------------------------------------------------
// Init()
//------------------------------------------------------------------------------

namespace luxrays {

void Init() {

#if !defined(LUXRAYS_DISABLE_OPENCL)
	// Report OpenCL as available only if the OpenCL library can be loaded
	isOpenCLAvilable = (clewInit() == CLEW_SUCCESS);
	if (!isOpenCLAvilable)
		std::cerr << "OpenCL disabled: OpenCL library not found\n";
#endif

#if !defined(LUXRAYS_DISABLE_CUDA)
	// Report CUDA/Optix as available only once fully initialized
	isCudaAvilable = false;
	isOptixAvilable = false;
	if (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != CUEW_SUCCESS) {
		std::cerr << "CUDA disabled: CUDA driver or nvrtc library not found\n";
	} else {
		// Was:
		//CHECK_CUDA_ERROR(cuInit(0));

		const CUresult err = cuInit(0);
		if (err != CUDA_SUCCESS) {
			// Handling multiple cases like:
			//
			// - when CUDA is installed but there are no NVIDIA GPUs available (CUDA_ERROR_NO_DEVICE)
			//
			// - when CUDA is installed but there an external GPU is unplugged (CUDA_ERROR_UNKNOWN)
			//
			// In all above cases, I just disable CUDA (but with this solution, at
			/// the moment I have no way to report the type of error).
			switch(err) {
				case CUDA_ERROR_INVALID_VALUE:
					std::cerr << "Cuda error: invalid value\n"; break;
				case CUDA_ERROR_INVALID_DEVICE:
					std::cerr << "Cuda error: invalid device\n"; break;
				case CUDA_ERROR_SYSTEM_DRIVER_MISMATCH:
					std::cerr << "Cuda error: system driver mismatch\n"; break;
				case CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE:
					std::cerr << "Cuda error: compat not supported on device\n"; break;
				case CUDA_ERROR_OUT_OF_MEMORY:
					std::cerr << "Cuda error: out of memory\n"; break;
				default:
					std::cerr << "Cuda error: undefined\n"; break;
			}


		} else {
			isCudaAvilable = true;

			// Try to initialize Optix too
			if (optixInit() == OPTIX_SUCCESS)
				isOptixAvilable = true;
		}
	}
#endif

#if !defined(LUXRAYS_DISABLE_HIP)
	// Report HIP as available only once fully initialized
	isHipAvailable = false;
	if (hipewInit(HIPEW_INIT_HIP) != HIPEW_SUCCESS) {
		std::cerr << "HIP disabled: HIP runtime library not found\n";
	} else {
		// hipErrorNoDevice is the normal case when ROCm is installed without
		// any supported AMD GPU
		const hipError_t err = hipInit(0);
		if (err == hipSuccess)
			isHipAvailable = true;
		else if (err != hipErrorNoDevice)
			std::cerr << "HIP disabled: hipInit() error " << err << "\n";
	}

	// hiprtc is optional: the precompiled kernels need only the HIP runtime
	// (the AMD driver on Windows). Without it, the programs that aren't
	// precompiled can't be compiled (see hipKernelPersistentCache).
	isHiprtcAvailable = isHipAvailable && (hipewInit(HIPEW_INIT_HIPRTC) == HIPEW_SUCCESS);

	// HIPRT is optional: without it, HIP devices use the BVH/MBVH accelerators.
	// Whether it works without hiprtc depends on the library and on the device
	// (see HIPDeviceDescription).
	isHiprtAvailable = isHipAvailable && (hiprtewInit() == HIPRTEW_SUCCESS);
#endif
}

}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
