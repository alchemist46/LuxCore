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
			hiprtCtx(nullptr), mergedGeometry(nullptr), scene(nullptr),
			mergedTrianglesBuff(nullptr), instanceMeshesBuff(nullptr),
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
	// Builds a geometry from triangles in host memory
	hiprtGeometry BuildGeometry(const Point *verts, const u_int vertCount,
			const Triangle *tris, const u_int triCount) {
		// Allocate vertices and triangle vertex indices buffers. They are
		// only used by the build: HIPRT stores the triangles in its nodes.
		HardwareDeviceBuffer *vertsBuff = nullptr;
		hipDevice->AllocBufferRO(&vertsBuff, (void *)verts, sizeof(Point) * vertCount);
		HardwareDeviceBuffer *trisBuff = nullptr;
		hipDevice->AllocBufferRO(&trisBuff, (void *)tris, sizeof(Triangle) * triCount);

		hiprtGeometryBuildInput buildInput;
		buildInput.type = hiprtPrimitiveTypeTriangleMesh;
		hiprtTriangleMeshPrimitive &triangleMesh = buildInput.primitive.triangleMesh;
		triangleMesh.vertices = ((HIPDeviceBuffer *)vertsBuff)->GetHIPDevicePointer();
		triangleMesh.vertexCount = vertCount;
		triangleMesh.vertexStride = sizeof(Point);
		triangleMesh.triangleIndices = ((HIPDeviceBuffer *)trisBuff)->GetHIPDevicePointer();
		triangleMesh.triangleCount = triCount;
		triangleMesh.triangleStride = sizeof(Triangle);

		hiprtBuildOptions buildOptions;
		buildOptions.buildFlags = hiprtBuildFlagBitPreferHighQualityBuild;

		size_t tmpSize;
		CHECK_HIPRT_ERROR(hiprtGetGeometryBuildTemporaryBufferSize(hiprtCtx, buildInput, buildOptions, tmpSize));
		hiprtDevicePtr tmp = GetTemporaryBuffer(tmpSize);

		hiprtGeometry geometry;
		CHECK_HIPRT_ERROR(hiprtCreateGeometry(hiprtCtx, buildInput, buildOptions, geometry));
		geometries.push_back(geometry);
		CHECK_HIPRT_ERROR(hiprtBuildGeometry(hiprtCtx, hiprtBuildOperationBuild,
				buildInput, buildOptions, tmp, nullptr, geometry));

		// The build is asynchronous
		hipDevice->FinishQueue();

		hipDevice->FreeBuffer(&vertsBuff);
		hipDevice->FreeBuffer(&trisBuff);

		return geometry;
	}

	// Returns the geometry of an instanced mesh, building it only the first
	// time (the same mesh can be used by many instances)
	hiprtGeometry GetGeometry(const TriangleMesh &mesh) {
		auto it = instancedGeometries.find(&mesh);
		if (it != instancedGeometries.end())
			return it->second;

		hiprtGeometry geometry = BuildGeometry(mesh.GetVertices().data(), mesh.GetTotalVertexCount(),
				mesh.GetTriangles().data(), mesh.GetTotalTriangleCount());
		instancedGeometries[&mesh] = geometry;

		return geometry;
	}

	// Merges all the meshes without a transformation (their vertices are
	// already in world space) in a single geometry: a ray has to traverse a
	// single BVH instead of the one of each mesh with an overlapping bounding
	// box
	hiprtGeometry BuildMergedGeometry(const HiprtAccel &hiprtAccel) {
		u_longlong vertCount = 0;
		u_longlong triCount = 0;
		for (auto mesh : hiprtAccel.meshes) {
			if ((mesh->GetType() == TYPE_TRIANGLE) || (mesh->GetType() == TYPE_EXT_TRIANGLE)) {
				vertCount += mesh->GetTotalVertexCount();
				triCount += mesh->GetTotalTriangleCount();
			}
		}

		if (triCount == 0)
			return nullptr;
		if ((vertCount > numeric_limits<u_int>::max()) || (triCount > numeric_limits<u_int>::max()))
			throw runtime_error("Too many triangles for the HIPRT accelerator: " + ToString(triCount));

		vector<Point> verts;
		verts.reserve(vertCount);
		vector<Triangle> tris;
		tris.reserve(triCount);
		// The mesh index and triangle index of each merged triangle
		vector<u_int> triMeshes;
		triMeshes.reserve(2 * triCount);

		for (u_int i = 0; i < hiprtAccel.meshes.size(); ++i) {
			auto mesh = hiprtAccel.meshes[i];
			if ((mesh->GetType() != TYPE_TRIANGLE) && (mesh->GetType() != TYPE_EXT_TRIANGLE))
				continue;

			auto& tm = dynamic_cast<const TriangleMesh &>(*mesh);

			const u_int vertOffset = verts.size();
			verts.insert(verts.end(), tm.GetVertices().begin(), tm.GetVertices().end());

			const Triangle *meshTris = tm.GetTriangles().data();
			for (u_int j = 0; j < tm.GetTotalTriangleCount(); ++j) {
				tris.push_back(Triangle(meshTris[j].v[0] + vertOffset,
						meshTris[j].v[1] + vertOffset, meshTris[j].v[2] + vertOffset));
				triMeshes.push_back(i);
				triMeshes.push_back(j);
			}
		}

		hiprtGeometry geometry = BuildGeometry(verts.data(), verts.size(), tris.data(), tris.size());

		hipDevice->AllocBufferRO(&mergedTrianglesBuff, triMeshes.data(), sizeof(u_int) * triMeshes.size(),
				"HIPRT merged triangles");

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

		mergedGeometry = BuildMergedGeometry(hiprtAccel);

		// HIPRT uses the first motion frame when the ray time is 0, whatever
		// the frame times are, so all frame times are shifted to be >= 1
		float minMotionTime = numeric_limits<float>::infinity();
		bool hasTransformedMeshes = false;
		for (auto mesh : hiprtAccel.meshes) {
			if ((mesh->GetType() == TYPE_TRIANGLE_MOTION) || (mesh->GetType() == TYPE_EXT_TRIANGLE_MOTION)) {
				auto& mtm = dynamic_cast<const MotionTriangleMesh &>(*mesh);
				minMotionTime = Min(minMotionTime, mtm.GetMotionSystem().times.front());
			}

			if ((mesh->GetType() != TYPE_TRIANGLE) && (mesh->GetType() != TYPE_EXT_TRIANGLE))
				hasTransformedMeshes = true;
		}
		rayTimeOffset = std::isinf(minMotionTime) ? 0.f : (minMotionTime - 1.f);

		if (hasTransformedMeshes) {
			// An instance for the merged geometry (if any) and one for each
			// instanced or motion blurred mesh

			vector<hiprtInstance> instances;
			vector<u_int> instanceMeshes;
			vector<hiprtTransformHeader> transformHeaders;
			vector<hiprtFrameMatrix> frames;

			if (mergedGeometry) {
				hiprtInstance instance;
				instance.type = hiprtInstanceTypeGeometry;
				instance.geometry = mergedGeometry;
				instances.push_back(instance);
				instanceMeshes.push_back(NULL_INDEX);
				transformHeaders.push_back({ (u_int)frames.size(), 1 });
				frames.push_back(ToFrame(Matrix4x4(), 0.f));
			}

			for (u_int i = 0; i < hiprtAccel.meshes.size(); ++i) {
				auto mesh = hiprtAccel.meshes[i];

				hiprtInstance instance;
				instance.type = hiprtInstanceTypeGeometry;
				hiprtTransformHeader transformHeader;
				transformHeader.frameIndex = frames.size();

				switch (mesh->GetType()) {
					case TYPE_TRIANGLE:
					case TYPE_EXT_TRIANGLE:
						// Already in the merged geometry
						continue;
					case TYPE_TRIANGLE_INSTANCE:
					case TYPE_EXT_TRIANGLE_INSTANCE: {
						auto& itm = dynamic_cast<const InstanceTriangleMesh &>(*mesh);

						instance.geometry = GetGeometry(itm.GetTriangleMesh());

						Transform local2World;
						itm.GetLocal2World(0.f, local2World);
						frames.push_back(ToFrame(local2World.m, 0.f));
						break;
					}
					case TYPE_TRIANGLE_MOTION:
					case TYPE_EXT_TRIANGLE_MOTION: {
						auto& mtm = dynamic_cast<const MotionTriangleMesh &>(*mesh);

						instance.geometry = GetGeometry(mtm.GetTriangleMesh());

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

				transformHeader.frameCount = frames.size() - transformHeader.frameIndex;

				instances.push_back(instance);
				instanceMeshes.push_back(i);
				transformHeaders.push_back(transformHeader);
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

			hipDevice->AllocBufferRO(&instanceMeshesBuff, &instanceMeshes[0], sizeof(u_int) * instanceMeshes.size());
		} else
			LR_LOG(device.GetContext(), "HIPRT accelerator is a single geometry");

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

		for (auto g : geometries)
			CHECK_HIPRT_ERROR(hiprtDestroyGeometry(hiprtCtx, g));
		geometries.clear();
		instancedGeometries.clear();
		mergedGeometry = nullptr;

		hipDevice->FreeBuffer(&mergedTrianglesBuff);
		hipDevice->FreeBuffer(&instanceMeshesBuff);
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
		// Without instances or motion blur, there is only the merged geometry
		// (or nothing for an empty DataSet)
		if (!scene)
			opts.push_back("-D HIPRTACCEL_GEOMETRY_ONLY");
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

	// All the geometries built
	vector<hiprtGeometry> geometries;
	map<const TriangleMesh *, hiprtGeometry> instancedGeometries;
	hiprtGeometry mergedGeometry;
	hiprtScene scene;
	HardwareDeviceBuffer *mergedTrianglesBuff;
	HardwareDeviceBuffer *instanceMeshesBuff;
	HardwareDeviceBuffer *tmpBuff;
	float rayTimeOffset;
	hiprtGlobalStackBuffer stackBuffer;

	HardwareDeviceKernelUPtr kernel;
	u_int kernelWorkGroupSize;
};

void HiprtKernel::EnqueueTraceRayBuffer(HardwareDeviceBuffer *rayBuff,
			HardwareDeviceBuffer *rayHitBuff, const unsigned int rayCount) {
	if (scene)
		device.SetKernelArg(kernel, 0, sizeof(hiprtScene), &scene);
	else
		device.SetKernelArg(kernel, 0, sizeof(hiprtGeometry), &mergedGeometry);
	device.SetKernelArg(kernel, 1, sizeof(hiprtGlobalStackBuffer), &stackBuffer);
	device.SetKernelArg(kernel, 2, rayTimeOffset);
	device.SetKernelArg(kernel, 3, instanceMeshesBuff);
	device.SetKernelArg(kernel, 4, mergedTrianglesBuff);
	device.SetKernelArg(kernel, 5, rayBuff);
	device.SetKernelArg(kernel, 6, rayHitBuff);
	device.SetKernelArg(kernel, 7, rayCount);

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
