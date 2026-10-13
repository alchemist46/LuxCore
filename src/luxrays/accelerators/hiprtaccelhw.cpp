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

#include <cmath>
#include <limits>
#include <map>

#include "luxrays/core/context.h"
#include "luxrays/core/exttrianglemesh.h"
#include "luxrays/devices/hipintersectiondevice.h"
#include "luxrays/accelerators/hiprtaccel.h"
#include "luxrays/kernels/kernels.h"
#include "luxrays/utils/oclcache.h"

using namespace std;

namespace luxrays {

//------------------------------------------------------------------------------
// HiprtKernel
//------------------------------------------------------------------------------

class HiprtKernel : public HardwareIntersectionKernel {
public:
	HiprtKernel(HardwareIntersectionDevice &dev, const HiprtAccel &hiprtAccel) :
			HardwareIntersectionKernel(dev), hipDevice(nullptr),
			hiprtCtx(nullptr), scene(nullptr),
			tmpBuff(nullptr), rayTimeOffset(0.f), stackBuffer{} {
		hipDevice = dynamic_cast<HIPIntersectionDevice *>(&dev);

		// Safety checks
		if (!hipDevice)
			throw runtime_error("Used a no HIP device in HiprtKernel::HiprtKernel(): " + DeviceDescription::GetDeviceType(dev.GetDeviceDesc().GetType()));
		hiprtCtx = hipDevice->GetHIPRTContext();
		if (!hiprtCtx)
			throw runtime_error("No HIPRT context in HiprtKernel::HiprtKernel()");

		// Building the acceleration structures and loading the kernel require
		// the device to be current
		hipDevice->PushThreadCurrentDevice();
		try {
			// The empty DataSet case is handled by the kernel with a null scene
			if (hiprtAccel.meshes.size() == 0) {
				LR_LOG(device.GetContext(), "HIPRT accelerator is empty");
			} else
				BuildScene(hiprtAccel);

			CompileKernel();
		} catch (...) {
			FreeScene();
			hipDevice->PopThreadCurrentDevice();
			throw;
		}
		hipDevice->PopThreadCurrentDevice();
	}

	virtual ~HiprtKernel() {
		hipDevice->PushThreadCurrentDevice();
		FreeScene();
		hipDevice->PopThreadCurrentDevice();

		// The kernel module is unloaded by hiprtDestroyContext()
	}

	virtual void Update(DataSetConstSPtr newDataSet) { assert(false); }
	virtual void EnqueueTraceRayBuffer(HardwareDeviceBuffer *rayBuff,
			HardwareDeviceBuffer *rayHitBuff, const unsigned int rayCount);

private:
	// Returns the geometry of a mesh, building it only the first time (the
	// same mesh can be used by many instances)
	hiprtGeometry GetGeometry(const TriangleMesh &mesh) {
		auto it = geometries.find(&mesh);
		if (it != geometries.end())
			return it->second;

		// Allocate vertices and triangle vertex indices buffers. They are
		// only used by the build: HIPRT stores the triangles in its nodes.
		HardwareDeviceBuffer *vertsBuff = nullptr;
		hipDevice->AllocBufferRO(&vertsBuff, mesh.GetVertices().data(), sizeof(Point) * mesh.GetTotalVertexCount());
		HardwareDeviceBuffer *trisBuff = nullptr;
		hipDevice->AllocBufferRO(&trisBuff, mesh.GetTriangles().data(), sizeof(Triangle) * mesh.GetTotalTriangleCount());

		hiprtGeometryBuildInput buildInput;
		buildInput.type = hiprtPrimitiveTypeTriangleMesh;
		hiprtTriangleMeshPrimitive &triangleMesh = buildInput.primitive.triangleMesh;
		triangleMesh.vertices = ((HIPDeviceBuffer *)vertsBuff)->GetHIPDevicePointer();
		triangleMesh.vertexCount = mesh.GetTotalVertexCount();
		triangleMesh.vertexStride = sizeof(Point);
		triangleMesh.triangleIndices = ((HIPDeviceBuffer *)trisBuff)->GetHIPDevicePointer();
		triangleMesh.triangleCount = mesh.GetTotalTriangleCount();
		triangleMesh.triangleStride = sizeof(Triangle);

		hiprtBuildOptions buildOptions;
		buildOptions.buildFlags = hiprtBuildFlagBitPreferHighQualityBuild;

		size_t tmpSize;
		CHECK_HIPRT_ERROR(hiprtGetGeometryBuildTemporaryBufferSize(hiprtCtx, buildInput, buildOptions, tmpSize));
		hiprtDevicePtr tmp = GetTemporaryBuffer(tmpSize);

		hiprtGeometry geometry;
		CHECK_HIPRT_ERROR(hiprtCreateGeometry(hiprtCtx, buildInput, buildOptions, geometry));
		geometries[&mesh] = geometry;
		CHECK_HIPRT_ERROR(hiprtBuildGeometry(hiprtCtx, hiprtBuildOperationBuild,
				buildInput, buildOptions, tmp, nullptr, geometry));

		// The build is asynchronous
		hipDevice->FinishQueue();

		hipDevice->FreeBuffer(&vertsBuff);
		hipDevice->FreeBuffer(&trisBuff);

		return geometry;
	}

	// The temporary buffer is shared by all builds
	hiprtDevicePtr GetTemporaryBuffer(const size_t size) {
		if (size == 0)
			return nullptr;

		if (!tmpBuff || (tmpBuff->GetSize() < size)) {
			hipDevice->FreeBuffer(&tmpBuff);
			hipDevice->AllocBufferRW(&tmpBuff, nullptr, size);
		}

		return ((HIPDeviceBuffer *)tmpBuff)->GetHIPDevicePointer();
	}

	static hiprtFrameMatrix ToFrame(const Matrix4x4 &m, const float time) {
		hiprtFrameMatrix frame;
		for (u_int i = 0; i < 3; ++i)
			for (u_int j = 0; j < 4; ++j)
				frame.matrix[i][j] = m.m[i][j];
		frame.time = time;

		return frame;
	}

	void BuildScene(const HiprtAccel &hiprtAccel) {
		const double t0 = WallClockTime();

		LR_LOG(device.GetContext(), "Building HIPRT accelerator");

		// HIPRT uses the first motion frame when the ray time is 0, whatever
		// the frame times are, so all frame times are shifted to be >= 1
		float minMotionTime = numeric_limits<float>::infinity();
		for (auto mesh : hiprtAccel.meshes) {
			if ((mesh->GetType() == TYPE_TRIANGLE_MOTION) || (mesh->GetType() == TYPE_EXT_TRIANGLE_MOTION)) {
				auto& mtm = dynamic_cast<const MotionTriangleMesh &>(*mesh);
				minMotionTime = Min(minMotionTime, mtm.GetMotionSystem().times.front());
			}
		}
		rayTimeOffset = std::isinf(minMotionTime) ? 0.f : (minMotionTime - 1.f);

		// One instance for each mesh, so the instance ID is the mesh index

		vector<hiprtInstance> instances(hiprtAccel.meshes.size());
		vector<hiprtTransformHeader> transformHeaders(hiprtAccel.meshes.size());
		vector<hiprtFrameMatrix> frames;
		frames.reserve(hiprtAccel.meshes.size());

		for (u_int i = 0; i < hiprtAccel.meshes.size(); ++i) {
			auto mesh = hiprtAccel.meshes[i];

			instances[i].type = hiprtInstanceTypeGeometry;
			transformHeaders[i].frameIndex = frames.size();

			switch (mesh->GetType()) {
				case TYPE_TRIANGLE:
				case TYPE_EXT_TRIANGLE: {
					auto& tm = dynamic_cast<const TriangleMesh &>(*mesh);

					// The vertices are already in world space
					instances[i].geometry = GetGeometry(tm);
					frames.push_back(ToFrame(Matrix4x4(), 0.f));
					break;
				}
				case TYPE_TRIANGLE_INSTANCE:
				case TYPE_EXT_TRIANGLE_INSTANCE: {
					auto& itm = dynamic_cast<const InstanceTriangleMesh &>(*mesh);

					instances[i].geometry = GetGeometry(itm.GetTriangleMesh());

					Transform local2World;
					itm.GetLocal2World(0.f, local2World);
					frames.push_back(ToFrame(local2World.m, 0.f));
					break;
				}
				case TYPE_TRIANGLE_MOTION:
				case TYPE_EXT_TRIANGLE_MOTION: {
					auto& mtm = dynamic_cast<const MotionTriangleMesh &>(*mesh);

					instances[i].geometry = GetGeometry(mtm.GetTriangleMesh());

					// One frame for each motion system key: HIPRT interpolates
					// the matrices like Optix does
					const MotionSystem &ms = mtm.GetMotionSystem();
					for (auto t : ms.times)
						frames.push_back(ToFrame(ms.SampleInverse(t), t - rayTimeOffset));
					break;
				}
				default:
					throw runtime_error("Unsupported mesh type in HiprtKernel(): " + ToString(mesh->GetType()));
			}

			transformHeaders[i].frameCount = frames.size() - transformHeaders[i].frameIndex;
		}

		LR_LOG(device.GetContext(), "HIPRT accelerator instances: " << instances.size() <<
				" (geometries: " << geometries.size() << ")");

		// Build the scene

		HardwareDeviceBuffer *instancesBuff = nullptr;
		hipDevice->AllocBufferRO(&instancesBuff, &instances[0], sizeof(hiprtInstance) * instances.size());
		HardwareDeviceBuffer *transformHeadersBuff = nullptr;
		hipDevice->AllocBufferRO(&transformHeadersBuff, &transformHeaders[0], sizeof(hiprtTransformHeader) * transformHeaders.size());
		HardwareDeviceBuffer *framesBuff = nullptr;
		hipDevice->AllocBufferRO(&framesBuff, &frames[0], sizeof(hiprtFrameMatrix) * frames.size());

		hiprtSceneBuildInput buildInput = {};
		buildInput.instances = ((HIPDeviceBuffer *)instancesBuff)->GetHIPDevicePointer();
		buildInput.instanceTransformHeaders = ((HIPDeviceBuffer *)transformHeadersBuff)->GetHIPDevicePointer();
		buildInput.instanceFrames = ((HIPDeviceBuffer *)framesBuff)->GetHIPDevicePointer();
		buildInput.instanceMasks = nullptr;
		buildInput.instanceCount = instances.size();
		buildInput.frameCount = frames.size();
		buildInput.frameType = hiprtFrameTypeMatrix;

		hiprtBuildOptions buildOptions;
		buildOptions.buildFlags = hiprtBuildFlagBitPreferHighQualityBuild;

		size_t tmpSize;
		CHECK_HIPRT_ERROR(hiprtGetSceneBuildTemporaryBufferSize(hiprtCtx, buildInput, buildOptions, tmpSize));
		hiprtDevicePtr tmp = GetTemporaryBuffer(tmpSize);

		CHECK_HIPRT_ERROR(hiprtCreateScene(hiprtCtx, buildInput, buildOptions, scene));
		CHECK_HIPRT_ERROR(hiprtBuildScene(hiprtCtx, hiprtBuildOperationBuild,
				buildInput, buildOptions, tmp, nullptr, scene));

		// The build is asynchronous
		hipDevice->FinishQueue();

		// The scene has its own copy of the instances and frames
		hipDevice->FreeBuffer(&instancesBuff);
		hipDevice->FreeBuffer(&transformHeadersBuff);
		hipDevice->FreeBuffer(&framesBuff);
		hipDevice->FreeBuffer(&tmpBuff);

		// The traversal stack in global memory, used when the one in local
		// memory is full. A dynamic stack is shared by the waves running on the
		// device, so its size doesn't depend on the number of rays.
		hiprtGlobalStackBufferInput stackInput;
		stackInput.type = hiprtStackTypeDynamic;
		stackInput.entryType = hiprtStackEntryTypeInteger;
		stackInput.stackSize = StackSize;
		stackInput.threadCount = 0;
		CHECK_HIPRT_ERROR(hiprtCreateGlobalStackBuffer(hiprtCtx, stackInput, stackBuffer));

		LR_LOG(device.GetContext(), "HIPRT accelerator total build time: " << int((WallClockTime() - t0) * 1000) << "ms");
	}

	void FreeScene() {
		hipDevice->FreeBuffer(&tmpBuff);

		if (stackBuffer.stackData) {
			CHECK_HIPRT_ERROR(hiprtDestroyGlobalStackBuffer(hiprtCtx, stackBuffer));
			stackBuffer = {};
		}

		if (scene) {
			CHECK_HIPRT_ERROR(hiprtDestroyScene(hiprtCtx, scene));
			scene = nullptr;
		}

		for (auto &g : geometries)
			CHECK_HIPRT_ERROR(hiprtDestroyGeometry(hiprtCtx, g.second));
		geometries.clear();
	}

	void CompileKernel() {
		const double t0 = WallClockTime();

		// The kernel uses only the plain C ray types: the OpenCL emulation of
		// the other HIP kernels isn't required
		const string kernelSource =
				luxrays::ocl::KernelSource_luxrays_types +
				luxrays::ocl::KernelSource_point_types +
				luxrays::ocl::KernelSource_vector_types +
				luxrays::ocl::KernelSource_ray_types +
				luxrays::ocl::KernelSource_hiprtaccel;

		// The local memory stack is sized for the work group size
		if (device.GetDeviceDesc().GetForceWorkGroupSize() > 0)
			kernelWorkGroupSize = device.GetDeviceDesc().GetForceWorkGroupSize();
		else
			kernelWorkGroupSize = 64;

		vector<string> opts = hipKernelPersistentCache::GetCompilerOptions(
				dynamic_cast<const HIPDeviceDescription &>(hipDevice->GetDeviceDesc()).GetHIPArch());
		opts.push_back("-D HIPRTACCEL_WORKGROUP_SIZE=" + ToString(kernelWorkGroupSize));
		opts.push_back("-D HIPRTACCEL_SHARED_STACK_SIZE=" + ToString(SharedStackSize));
		vector<const char *> optsPtrs;
		for (auto &o : opts)
			optsPtrs.push_back(o.c_str());
		LR_LOG(device.GetContext(), "[HiprtAccel] Compiler options: " << oclKernelPersistentCache::ToOptsString(opts));

		const char *funcName = "HiprtAccel_Intersect_RayBuffer";
		hiprtApiFunction function;
		// HIPRT keeps its own persistent cache of the compiled kernels
		const hiprtError err = hiprtBuildTraceKernels(hiprtCtx,
				1, &funcName,
				kernelSource.c_str(), "HiprtAccel",
				0, nullptr, nullptr,
				optsPtrs.size(), optsPtrs.data(),
				0, 1, nullptr,
				&function, nullptr, true);
		if (err != hiprtSuccess) {
			// HIPRT prints the compilation errors to the standard error
			LR_LOG(device.GetContext(), "[HiprtAccel] HIPRT program compilation error (code: " << err << ")");

			throw runtime_error("HiprtAccel HIPRT program compilation error");
		}

		kernel = hipDevice->GetKernel((hipFunction_t)function);

		LR_LOG(device.GetContext(), "[HiprtAccel] Compilation completed in " << int((WallClockTime() - t0) * 1000) << "ms");
	}

	// The traversal stack sizes (the same of the HIPRT default stack)
	static constexpr u_int StackSize = 64;
	static constexpr u_int SharedStackSize = 16;

	HIPIntersectionDevice *hipDevice;
	hiprtContext hiprtCtx;

	map<const TriangleMesh *, hiprtGeometry> geometries;
	hiprtScene scene;
	HardwareDeviceBuffer *tmpBuff;
	float rayTimeOffset;
	hiprtGlobalStackBuffer stackBuffer;

	HardwareDeviceKernelUPtr kernel;
	u_int kernelWorkGroupSize;
};

void HiprtKernel::EnqueueTraceRayBuffer(HardwareDeviceBuffer *rayBuff,
			HardwareDeviceBuffer *rayHitBuff, const unsigned int rayCount) {
	device.SetKernelArg(kernel, 0, sizeof(hiprtScene), &scene);
	device.SetKernelArg(kernel, 1, sizeof(hiprtGlobalStackBuffer), &stackBuffer);
	device.SetKernelArg(kernel, 2, rayTimeOffset);
	device.SetKernelArg(kernel, 3, rayBuff);
	device.SetKernelArg(kernel, 4, rayHitBuff);
	device.SetKernelArg(kernel, 5, rayCount);

	const u_int globalRange = RoundUp<u_int>(rayCount, kernelWorkGroupSize);
	device.EnqueueKernel(kernel, HardwareDeviceRange(globalRange),
			HardwareDeviceRange(kernelWorkGroupSize));
}

//------------------------------------------------------------------------------
// HiprtAccel
//------------------------------------------------------------------------------

bool HiprtAccel::HasNativeSupport(const IntersectionDevice &device) const {
	return false;
}

bool HiprtAccel::HasHWSupport(const IntersectionDevice &device) const {
	return device.HasHWSupport();
}

HardwareIntersectionKernelUPtr HiprtAccel::NewHardwareIntersectionKernel(HardwareIntersectionDevice &device) const {
	// Setup the kernel
	auto [kernel, ref] = CreateUniquePtr<HardwareIntersectionKernel, HiprtKernel>(device, *this);
	return std::move(kernel);
}

}

#endif
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
