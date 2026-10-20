/***************************************************************************
 * Copyright 1998-2026 by authors (see AUTHORS.txt)
 *
 *   This file is part of LuxCoreRender.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ***************************************************************************/

#ifndef _LUXRAYS_HIPKERNEL_H
#define	_LUXRAYS_HIPKERNEL_H

#include <cstdlib>
#include <string>
#include <vector>

#include "luxrays/kernels/kernels.h"
// Note: oclhash.h (and not oclcache.h) so the hip-kernel-baker is
// dependency-free
#include "luxrays/utils/oclhash.h"

namespace luxrays {

// How a HIP program is built from an OpenCL source: the parts shared by
// HIPDevice / hipKernelPersistentCache at run time and by the
// hip-kernel-baker at build time (src/slg/kernels/hipkernelbaker.cpp), so
// that the key of a baked kernel can't drift from the one of the program it
// replaces. Header-only and free of any HIP dependency for the same reason.

// The largest work group the compiled kernels can be launched with
constexpr unsigned int HIPMaxWorkGroupSize = 256;

// The definitions added to the parameters of every program
// (HIPDevice::AddKernelOpts())
inline std::vector<std::string> GetHIPDeviceKernelParameters() {
	return {
		// HIP kernels are compiled from the same CUDA-flavoured sources, so the
		// CUDA code paths are enabled too. LUXRAYS_HIP_DEVICE marks the few
		// places where HIP is different.
		"-D LUXRAYS_CUDA_DEVICE",
		"-D LUXRAYS_HIP_DEVICE"
		// Note: no LUXRAYS_OS_* parameter, as on OpenCL: no kernel uses it, and
		// the compiled code (and its cache key) must be the same on every OS
	};
}

// The compiler options of every program, without the target architecture
// (hipKernelPersistentCache::GetCompilerOptions() adds --offload-arch first
// and LUXRAYS_HIP_COMPILER_OPTIONS last). They are valid for hiprtc and for
// the offline clang alike.
inline std::vector<std::string> GetHIPBaseCompilerOptions() {
	return {
		"-O3",
		// The OpenCL-style sources produce a lot of harmless warnings once
		// compiled as HIP (unused variables, sign conversions, etc.)
		"-w",
		// By default, kernels are compiled to run work groups of up to 1024
		// threads, which halves the registers available to each thread. The
		// render engines use much smaller work groups and the kernels are
		// register hungry: on gfx1030, this is about 20% faster (the same
		// speed of the AMD OpenCL compiler) and spills less to the stack.
		"--gpu-max-threads-per-block=" + std::to_string(HIPMaxWorkGroupSize)
	};
}

// The HIP source of an OpenCL program (HIPDevice::GetKernelSource())
inline std::string GetHIPKernelSource(const std::string &kernelSource) {
	// The pragma only covers functions: file scope tables (always declared at
	// the start of a line as "__constant <type> <name>[] = { ... }", while the
	// __constant pointer qualifiers are always indented or follow a "(") must
	// be explicitly moved to the device
	const std::string from = "\n__constant ";
	const std::string to = "\n__constant__ ";

	const std::string source = "\n" + kernelSource;
	std::string hipSource;
	hipSource.reserve(source.size() + source.size() / 64);
	std::size_t pos = 0;
	for (;;) {
		const std::size_t next = source.find(from, pos);
		if (next == std::string::npos)
			break;
		hipSource.append(source, pos, next - pos);
		hipSource += to;
		pos = next + from.size();
	}
	hipSource.append(source, pos, std::string::npos);

	// The OpenCL sources don't mark device functions: NVRTC handles that with
	// --device-as-default-execution-space, hiprtc with this pragma
	return
		"#pragma clang force_cuda_host_device begin\n" +
		luxrays::ocl::KernelSource_cudadevice_oclemul_types +
		luxrays::ocl::KernelSource_cudadevice_math +
		luxrays::ocl::KernelSource_cudadevice_oclemul_funcs +
		hipSource +
		"\n#pragma clang force_cuda_host_device end\n";
}

// The key of a baked kernel: the hash of hipKernelPersistentCache without
// the --offload-arch option (a baked code object serves a whole family of
// architectures). programParameters are the parameters of the program
// including the device ones (i.e. as returned by HIPDevice::AddKernelOpts())
// and hipSource is the HIP source (GetHIPKernelSource()).
inline std::string GetHIPBakedKernelKey(const std::vector<std::string> &programParameters,
		const std::string &hipSource) {
	std::vector<std::string> opts = GetHIPBaseCompilerOptions();
	opts.insert(opts.end(), programParameters.begin(), programParameters.end());

	return OCLKernelHashString(OCLKernelParamsToString(opts)) + "-" +
			OCLKernelHashString(hipSource);
}

// The families of architectures the kernels are baked for: one generic code
// object (code object v6) per family covers all the RDNA2/3/4 GPUs that
// HIPRT supports
inline std::vector<std::string> GetHIPBakedKernelFamilies() {
	return { "gfx10-3-generic", "gfx11-generic", "gfx12-generic" };
}

// The family of a device architecture (as reported by hipDeviceProp_t's
// gcnArchName, e.g. "gfx1030" or "gfx90a:sramecc+:xnack-"), or an empty
// string if no baked family covers it. The members of each generic target
// are the ones of the LLVM AMDGPU documentation.
inline std::string GetHIPBakedKernelFamily(const std::string &arch) {
	const std::string name = arch.substr(0, arch.find(':'));

	static const char *const gfx103[] = {
		"gfx1030", "gfx1031", "gfx1032", "gfx1033", "gfx1034", "gfx1035", "gfx1036"
	};
	static const char *const gfx11[] = {
		"gfx1100", "gfx1101", "gfx1102", "gfx1103",
		"gfx1150", "gfx1151", "gfx1152", "gfx1153"
	};
	static const char *const gfx12[] = {
		"gfx1200", "gfx1201"
	};

	for (const char *a : gfx103)
		if (name == a)
			return "gfx10-3-generic";
	for (const char *a : gfx11)
		if (name == a)
			return "gfx11-generic";
	for (const char *a : gfx12)
		if (name == a)
			return "gfx12-generic";

	return "";
}

//------------------------------------------------------------------------------
// The HIPRT trace kernel (see HiprtAccel)
//------------------------------------------------------------------------------

// The source of the trace kernel program, as hiprtBuildTraceKernels() builds
// it: HIPRT's device implementation, the dispatch of the custom functions
// (none: the generated switches only have their default case), then the
// kernel. The HIPRT headers are found in an include directory (HIPRT_PATH),
// which isn't part of the key: the HIPRT version is (see
// GetHiprtTraceKernelParameters()).
inline std::string GetHiprtTraceKernelSource() {
	return
		"#include <hiprt/impl/hiprt_device_impl.h>\n"
		"\n"
		"HIPRT_DEVICE bool intersectFunc( uint32_t geomType, uint32_t rayType, const hiprtFuncTableHeader& tableHeader, "
		"const hiprtRay& ray, void* payload, hiprtHit& hit )\n"
		"{\n"
		"\tconst uint32_t index = tableHeader.numGeomTypes * rayType + geomType;\n"
		"\t[[maybe_unused]] const void* data = tableHeader.funcDataSets[index].intersectFuncData;\n"
		"\tswitch ( index ) \n"
		"\t{\n"
		"\t\t default: { return false; }\n"
		"\t}\n"
		"}\n"
		"\n"
		"HIPRT_DEVICE bool filterFunc( uint32_t geomType, uint32_t rayType, const hiprtFuncTableHeader& tableHeader, "
		"const hiprtRay& ray, void* payload, const hiprtHit& hit )\n"
		"{\n"
		"\tconst uint32_t index = tableHeader.numGeomTypes * rayType + geomType;\n"
		"\t[[maybe_unused]] const void* data = tableHeader.funcDataSets[index].filterFuncData;\n"
		"\tswitch ( index ) \n"
		"\t{\n"
		"\t\t default: { return false; }\n"
		"\t}\n"
		"}\n"
		"\n" +
		// The kernel uses only the plain C ray types: the OpenCL emulation of
		// the other HIP kernels isn't required
		luxrays::ocl::KernelSource_luxrays_types +
		luxrays::ocl::KernelSource_point_types +
		luxrays::ocl::KernelSource_vector_types +
		luxrays::ocl::KernelSource_ray_types +
		luxrays::ocl::KernelSource_hiprtaccel;
}

// The work group size and the shared traversal stack size of the trace kernel
// (the latter is the same of the HIPRT default stack). The work group size is
// the one of the device when it is forced, as it is by default: the render
// engines force opencl.gpu.workgroup.size (32 by default) on every GPU, so
// the kernel is baked for that.
constexpr unsigned int HiprtTraceKernelWorkGroupSize = 64;
constexpr unsigned int HiprtTraceKernelBakedWorkGroupSize = 32;
constexpr unsigned int HiprtTraceKernelSharedStackSize = 16;

// The parameters of the trace kernel program. rtip is the one HIPRT builds the
// acceleration structures for (see GetHiprtRtip()), hiprtApiVersion the
// version of the HIPRT library (HIPRTEW_API_VERSION): the device code of
// HIPRT comes from its headers, which are not part of the key.
inline std::vector<std::string> GetHiprtTraceKernelParameters(const unsigned int workGroupSize,
		const bool geometryOnly, const unsigned int rtip, const unsigned int hiprtApiVersion) {
	std::vector<std::string> params = {
		"-D HIPRTACCEL_WORKGROUP_SIZE=" + std::to_string(workGroupSize),
		"-D HIPRTACCEL_SHARED_STACK_SIZE=" + std::to_string(HiprtTraceKernelSharedStackSize)
	};
	// Without instances or motion blur, there is only the merged geometry (or
	// nothing for an empty DataSet)
	if (geometryOnly)
		params.push_back("-D HIPRTACCEL_GEOMETRY_ONLY");
	// HIPRT doesn't use fast math for the trace kernels: it is ~12% faster
	// with motion blur (the frame interpolation) and ~6% with instances, with
	// the same results. Infinite values are still required: the ray maxt can
	// be INFINITY.
	params.push_back("-ffast-math");
	params.push_back("-fno-finite-math-only");
	params.push_back("-D LUXRAYS_HIPRT_API_VERSION=" + std::to_string(hiprtApiVersion));

	// The options added by HIPRT (Compiler::addCommonOpts())
	if (rtip > 0)
		params.push_back("-DHIPRT_RTIP=" + std::to_string(rtip));
	params.push_back("-D__USE_HIP__");
	params.push_back("-std=c++17");

	return params;
}

// The RTIP (ray tracing IP) HIPRT uses for a device (Context::getRtip()): the
// layout of the acceleration structures it builds, which the trace kernel must
// be compiled for. gfx12 gets RTIP 3.1 only from a driver of HIP 6.4
// (Windows) or ROCm 7.0 (Linux), else 2.0. HIPRT also requires hiprtc to
// support RTIP 3.1 when it compiles at run time; a HIPRT with precompiled
// kernels doesn't check it, nor does this function.
inline unsigned int GetHiprtRtip(const std::string &arch, const int driverVersion) {
	// The number after "gfx" (as std::stoi(), i.e. gfx90a is 90)
	unsigned int archNumber = 0;
	if (arch.compare(0, 3, "gfx") == 0)
		archNumber = (unsigned int)strtoul(arch.c_str() + 3, nullptr, 10);

	unsigned int rtip = 0;
	if (archNumber >= 1030)
		rtip = 11;
	if (archNumber >= 1100)
		rtip = 20;
	if (archNumber >= 1200) {
#if defined(_WIN32)
		const bool driverRtip31 = (driverVersion >= 60400000);
#else
		const bool driverRtip31 = (driverVersion >= 70000000);
#endif
		if (driverRtip31)
			rtip = 31;
	}

	return rtip;
}

// The RTIPs the trace kernel is baked for, for each family: the ones
// GetHiprtRtip() can return for its architectures
inline std::vector<unsigned int> GetHiprtBakedRtips(const std::string &family) {
	if (family == "gfx10-3-generic")
		return { 11 };
	if (family == "gfx11-generic")
		return { 20 };
	if (family == "gfx12-generic")
		return { 31, 20 };
	return {};
}

}

#endif	/* _LUXRAYS_HIPKERNEL_H */
