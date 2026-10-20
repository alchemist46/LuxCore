#line 2 "kernelconstants_types.cl"

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

// The values the path kernels read from a program scope constant, set by the
// host after the program is loaded (HardwareDevice::SetProgramConstant()), on
// the devices that support it (HIP, with PARAM_KERNEL_CONSTANTS defined). A
// change of these values doesn't change the kernel, so they don't make new
// kernel variants to compile. The other devices get them as -D PARAM_*
// definitions (see WorkingColorSpace::AddKernelParameters() and
// slg::ocl::GetKernelParamters()).
typedef struct {
	// Linear Rec.709 to the working colour space and back (row major), only
	// used when the working colour space is not Rec.709
	float wcsFromRec709[9], wcsToRec709[9];
	// The luminance weights of Spectrum_Y()
	float wcsY[3];
	unsigned int wcsIsRec709;

	// The clamp of MachineEpsilon_E()
	float rayEpsilonMin, rayEpsilonMax;
} KernelConstants;

#if defined(LUXRAYS_OPENCL_KERNEL) && defined(PARAM_KERNEL_CONSTANTS)
__constant KernelConstants kernelConstants;
#endif
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
