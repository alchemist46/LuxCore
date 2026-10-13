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

// The OpenCL C emulation used to compile the kernel sources (the *_funcs.cl
// files) as C++ for the CPU, the way CUDADevice::GetKernelSource() and
// HIPDevice::GetKernelSource() compile them for CUDA and HIP.
//
// A CPU kernel source file includes all its regular headers first (this file
// defines macros like __global and restrict), then this file and the kernel
// sources inside the same namespace, e.g.:
//
//   namespace slg { namespace ocl { namespace {
//   #include "luxrays/devices/cpudevice_oclemul.h"
//   #include "luxrays/core/utils_funcs.cl"
//   ...
//   } } }
//
// The unnamed namespace gives internal linkage to the kernel functions and to
// the __constant tables, so more than one CPU kernel source file can include
// the same kernel sources.

#define LUXRAYS_CPU_DEVICE

#include "luxrays/devices/cpudevice_oclemul_types.h"
#include "luxrays/devices/cudadevice_math.cl"
#include "luxrays/devices/cudadevice_oclemul_funcs.cl"

// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
