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

// The CPU counterpart of cudadevice_oclemul_types.cl: the OpenCL C types and
// qualifiers of the kernel sources, emulated in C++. It is followed by the
// cudadevice_math.cl and cudadevice_oclemul_funcs.cl files shared with CUDA
// and HIP, see cpudevice_oclemul.h.
//
// It is included inside a namespace, like the *_types.cl files.

#define __kernel
#define __global
#define __local
#define __constant
#define restrict __restrict

#define OPENCL_FORCE_NOT_INLINE inline
#define OPENCL_FORCE_INLINE inline
#define OPENCL_HOT_FUNCTION inline

// Used by cudadevice_math.cl and cudadevice_oclemul_funcs.cl
#define __device__
#define __forceinline__ inline

typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;
typedef unsigned long long ulong;

// INFINITY is defined by <cmath>
#define M_PI_F 3.141592654f
#define M_1_PI_F (1.f / 3.141592654f)

//------------------------------------------------------------------------------
// Vector types (the CUDA ones, without the OpenCL padding of the 3 components
// types, as the kernels are compiled for CUDA and HIP)
//------------------------------------------------------------------------------

struct float2 { float x, y; };
struct float3 { float x, y, z; };
struct float4 { float x, y, z, w; };
struct int2 { int x, y; };
struct int3 { int x, y, z; };
struct int4 { int x, y, z, w; };
struct uint2 { uint x, y; };
struct uint3 { uint x, y, z; };
struct uint4 { uint x, y, z, w; };

inline float2 make_float2(const float x, const float y) { return { x, y }; }
inline float3 make_float3(const float x, const float y, const float z) { return { x, y, z }; }
inline float4 make_float4(const float x, const float y, const float z, const float w) { return { x, y, z, w }; }
inline int2 make_int2(const int x, const int y) { return { x, y }; }
inline int3 make_int3(const int x, const int y, const int z) { return { x, y, z }; }
inline int4 make_int4(const int x, const int y, const int z, const int w) { return { x, y, z, w }; }
inline uint2 make_uint2(const uint x, const uint y) { return { x, y }; }
inline uint3 make_uint3(const uint x, const uint y, const uint z) { return { x, y, z }; }
inline uint4 make_uint4(const uint x, const uint y, const uint z, const uint w) { return { x, y, z, w }; }

//------------------------------------------------------------------------------
// Scalar math functions
//------------------------------------------------------------------------------

// The vector overloads defined in this namespace by cudadevice_math.cl and
// cudadevice_oclemul_funcs.cl would hide the scalar ones
using std::abs;
using std::atan;
using std::atan2;
using std::atanh;
using std::cos;
using std::exp;
using std::fabs;
using std::fmax;
using std::fmin;
using std::isinf;
using std::isnan;
using std::log;
using std::pow;
using std::sin;
using std::sqrt;
using ::floorf;
using ::fmaxf;
using ::fminf;
using ::fmodf;

// The CUDA built-ins used by cudadevice_math.cl
inline int min(const int a, const int b) { return (a < b) ? a : b; }
inline uint min(const uint a, const uint b) { return (a < b) ? a : b; }
inline float min(const float a, const float b) { return fminf(a, b); }
inline int max(const int a, const int b) { return (a > b) ? a : b; }
inline uint max(const uint a, const uint b) { return (a > b) ? a : b; }
inline float max(const float a, const float b) { return fmaxf(a, b); }

inline float rsqrtf(const float x) { return 1.f / std::sqrt(x); }

// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
