#line 2 "cudadevice_oclemul_types.cl"

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

#if defined(LUXRAYS_HIP_DEVICE) && defined(LUXRAYS_HIP_WAVES_PER_EU)
// For experiments (LUXRAYS_HIP_COMPILER_OPTIONS="-DLUXRAYS_HIP_WAVES_PER_EU=N"):
// asks the compiler for at least N waves per SIMD, i.e. a register budget
#define __kernel extern "C" __global__ __attribute__((amdgpu_waves_per_eu(LUXRAYS_HIP_WAVES_PER_EU)))
#else
#define __kernel extern "C" __global__
#endif
#define __global
#define __local __shared__
#define __constant
#define restrict __restrict__

// This is a workaround to long compilation time
#define OPENCL_FORCE_NOT_INLINE __noinline__
#define OPENCL_FORCE_INLINE __forceinline__
#if defined(LUXRAYS_HIP_DEVICE)
// The hot functions with one or two callers are left to the compiler on HIP
// (inlining everything is 4x slower to compile and spills registers)
#define OPENCL_HOT_FUNCTION
#else
#define OPENCL_HOT_FUNCTION __noinline__
#endif

typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;
#if !defined(LUXRAYS_HIP_DEVICE)
// HIP already defines ulong
typedef unsigned long long ulong;
#endif

#define INFINITY __int_as_float(0x7f800000)
#define M_PI_F 3.141592654f
#define M_1_PI_F (1.f / 3.141592654f)
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
