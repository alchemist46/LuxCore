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
#include <algorithm>
#include <map>

#include "luxrays/core/context.h"
#include "luxrays/core/exttrianglemesh.h"
#include "luxrays/devices/hipintersectiondevice.h"
#include "luxrays/accelerators/hiprtaccel.h"
#include "luxrays/kernels/kernels.h"
#include "luxrays/utils/hipkernel.h"
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
			hiprtCtx(nullptr), buildFlags(hiprtBuildFlagBitPreferHighQualityBuild),
			mergedGeometry(nullptr), scene(nullptr),
			mergedTrianglesBuff(nullptr), instanceMeshesBuff(nullptr),
			tmpBuff(nullptr), rayTimeOffset(0.f), stackBuffer{},
			kernelModule(nullptr) {
		hipDevice = dynamic_cast<HIPIntersectionDevice *>(&dev);

		// Safety checks
		if (!hipDevice)
			throw runtime_error("Used a no HIP device in HiprtKernel::HiprtKernel(): " + DeviceDescription::GetDeviceType(dev.GetDeviceDesc().GetType()));
		hiprtCtx = hipDevice->GetHIPRTContext();
		if (!hiprtCtx)
			throw runtime_error("No HIPRT context in HiprtKernel::HiprtKernel()");

		// The BVH quality: a high quality BVH is faster to trace and slower to
		// build
		const string builderType = device.GetContext().GetConfig().Get(
				Property("accelerator.hiprt.builder.type")("HIGH_QUALITY")).Get<string>();
		if (builderType == "HIGH_QUALITY")
			buildFlags = hiprtBuildFlagBitPreferHighQualityBuild;
		else if (builderType == "BALANCED")
			buildFlags = hiprtBuildFlagBitPreferBalancedBuild;
		else if (builderType == "FAST")
			buildFlags = hiprtBuildFlagBitPreferFastBuild;
		else
			throw runtime_error("Unknown HIPRT builder type in HiprtKernel::HiprtKernel(): " + builderType);

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
			FreeKernel();
			FreeScene();
			hipDevice->PopThreadCurrentDevice();
			throw;
		}
		hipDevice->PopThreadCurrentDevice();
	}

	virtual ~HiprtKernel() {
		hipDevice->PushThreadCurrentDevice();
		FreeKernel();
		FreeScene();
		hipDevice->PopThreadCurrentDevice();
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
		buildOptions.buildFlags = buildFlags;
		size_t tmpSize;
		CHECK_HIPRT_ERROR(hiprtGetGeometryBuildTemporaryBufferSize(hiprtCtx, buildInput, buildOptions, tmpSize));
		if ((buildFlags == hiprtBuildFlagBitPreferHighQualityBuild) && !FitsInFreeMemory(tmpSize)) {
			LR_LOG(device.GetContext(), "Not enough free memory for a high quality HIPRT build of " << triCount <<
					" triangles (" << ToMemString(tmpSize) << " of temporary memory): using a balanced build");

			buildOptions.buildFlags = hiprtBuildFlagBitPreferBalancedBuild;
			CHECK_HIPRT_ERROR(hiprtGetGeometryBuildTemporaryBufferSize(hiprtCtx, buildInput, buildOptions, tmpSize));
		}
		hiprtDevicePtr tmp = GetTemporaryBuffer(tmpSize);

		hiprtGeometry geometry;
		CHECK_HIPRT_ERROR(hiprtCreateGeometry(hiprtCtx, buildInput, buildOptions, geometry));
		CHECK_HIPRT_ERROR(hiprtBuildGeometry(hiprtCtx, hiprtBuildOperationBuild,
				buildInput, buildOptions, tmp, nullptr, geometry));

		// The build is asynchronous
		hipDevice->FinishQueue();

		hipDevice->FreeBuffer(&vertsBuff);
		hipDevice->FreeBuffer(&trisBuff);

		// The memory allocated for the BVH is sized for the worst case (i.e. of
		// the spatial splits): copy it in a buffer of the right size (the input
		// geometry is destroyed). It halves the memory used by the high quality
		// BVH, with the same performance.
		hiprtGeometry compactGeometry;
		CHECK_HIPRT_ERROR(hiprtCompactGeometry(hiprtCtx, nullptr, geometry, compactGeometry));
		geometry = compactGeometry;
		geometries.push_back(geometry);

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

	static bool IsInstance(const Mesh *mesh) {
		return (mesh->GetType() == TYPE_TRIANGLE_INSTANCE) || (mesh->GetType() == TYPE_EXT_TRIANGLE_INSTANCE);
	}

	// Selects the meshes merged in a single geometry: the ones without a
	// transformation and the instances that can be copied in world space
	// without using too much memory. Traversing a single BVH is faster than
	// traversing the BVH of each instance with an overlapping bounding box
	// (~20% faster on a scene with a few instances).
	void SelectMergedMeshes(const HiprtAccel &hiprtAccel) {
		const u_int meshCount = hiprtAccel.meshes.size();
		mergedMeshes.assign(meshCount, false);

		// The instances of each instanced mesh
		map<const TriangleMesh *, vector<u_int>> meshInstances;
		for (u_int i = 0; i < meshCount; ++i) {
			auto mesh = hiprtAccel.meshes[i];

			if ((mesh->GetType() == TYPE_TRIANGLE) || (mesh->GetType() == TYPE_EXT_TRIANGLE))
				mergedMeshes[i] = true;
			else if (IsInstance(mesh))
				meshInstances[&dynamic_cast<const InstanceTriangleMesh &>(*mesh).GetTriangleMesh()].push_back(i);
		}

		// A mesh used by a single instance is copied for free (it doesn't need
		// its own geometry anymore). The others are copied, starting from the
		// ones requiring less memory, as long as the added triangles are
		// within a budget.
		vector<pair<u_longlong, const vector<u_int> *>> sharedMeshes;
		for (auto &mi : meshInstances) {
			const u_longlong addedTriangles = (mi.second.size() - 1) * (u_longlong)mi.first->GetTotalTriangleCount();
			sharedMeshes.push_back(make_pair(addedTriangles, &mi.second));
		}
		sort(sharedMeshes.begin(), sharedMeshes.end(),
				[](const auto &a, const auto &b) { return a.first < b.first; });

		u_longlong totalAddedTriangles = 0;
		for (auto &sm : sharedMeshes) {
			if (totalAddedTriangles + sm.first > MergedInstancesTriangleBudget)
				break;

			totalAddedTriangles += sm.first;
			for (auto i : *sm.second)
				mergedMeshes[i] = true;
		}
	}

	// Merges the meshes selected by SelectMergedMeshes() in a single geometry
	// (in world space): a ray has to traverse a single BVH instead of the one
	// of each mesh with an overlapping bounding box
	hiprtGeometry BuildMergedGeometry(const HiprtAccel &hiprtAccel) {
		SelectMergedMeshes(hiprtAccel);

		u_longlong vertCount = 0;
		u_longlong triCount = 0;
		for (u_int i = 0; i < hiprtAccel.meshes.size(); ++i) {
			auto mesh = hiprtAccel.meshes[i];
			if (mergedMeshes[i]) {
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
			if (!mergedMeshes[i])
				continue;

			const u_int vertOffset = verts.size();
			const TriangleMesh *tm;
			if (IsInstance(mesh)) {
				// Copy the vertices in world space
				auto& itm = dynamic_cast<const InstanceTriangleMesh &>(*mesh);
				tm = &itm.GetTriangleMesh();

				Transform local2World;
				itm.GetLocal2World(0.f, local2World);
				for (auto &v : tm->GetVertices())
					verts.push_back(local2World * v);
			} else {
				tm = &dynamic_cast<const TriangleMesh &>(*mesh);
				verts.insert(verts.end(), tm->GetVertices().begin(), tm->GetVertices().end());
			}

			const Triangle *meshTris = tm->GetTriangles().data();
			for (u_int j = 0; j < tm->GetTotalTriangleCount(); ++j) {
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

	// The high quality build (with spatial splits) traces rays faster than
	// the balanced one (up to 4% less time per sample on the PATHOCL test
	// scenes, 2026-10) but takes 4 times longer (~100 ms for 1.25M triangles)
	// and its temporary memory is ~17 times larger (~1.5KB per triangle
	// instead of ~90 bytes): i.e. 11GB for 8M triangles.
	// It is used only if the temporary memory (and the BVH, ~25% of it before
	// the compaction) fits in the free memory, with a margin for the other
	// applications and the render buffers allocated later.
	//
	// Note: the radix sort of the balanced build is compiled by Orochi, which
	// caches it in "./cache" whatever hiprtSetCacheDirPath() is (HIPRT 3.1).
	bool FitsInFreeMemory(const size_t tmpSize) const {
		size_t freeMem, totalMem;
		CHECK_HIP_ERROR(hipMemGetInfo(&freeMem, &totalMem));

		// The temporary buffer is already allocated if it is large enough
		const size_t tmpBuffSize = tmpBuff ? tmpBuff->GetSize() : 0;
		const size_t required = ((tmpSize > tmpBuffSize) ? (tmpSize - tmpBuffSize) : 0) + tmpSize / 4;

		return required <= freeMem / 4 * 3;
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
		for (u_int i = 0; i < hiprtAccel.meshes.size(); ++i) {
			auto mesh = hiprtAccel.meshes[i];
			if ((mesh->GetType() == TYPE_TRIANGLE_MOTION) || (mesh->GetType() == TYPE_EXT_TRIANGLE_MOTION)) {
				auto& mtm = dynamic_cast<const MotionTriangleMesh &>(*mesh);
				minMotionTime = Min(minMotionTime, mtm.GetMotionSystem().times.front());
			}

			if (!mergedMeshes[i])
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

				// Already in the merged geometry
				if (mergedMeshes[i])
					continue;

				switch (mesh->GetType()) {
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
			buildOptions.buildFlags = buildFlags;
			size_t tmpSize;
			CHECK_HIPRT_ERROR(hiprtGetSceneBuildTemporaryBufferSize(hiprtCtx, buildInput, buildOptions, tmpSize));
			if ((buildFlags == hiprtBuildFlagBitPreferHighQualityBuild) && !FitsInFreeMemory(tmpSize)) {
				LR_LOG(device.GetContext(), "Not enough free memory for a high quality HIPRT build of " << instances.size() <<
						" instances (" << ToMemString(tmpSize) << " of temporary memory): using a balanced build");

				buildOptions.buildFlags = hiprtBuildFlagBitPreferBalancedBuild;
				CHECK_HIPRT_ERROR(hiprtGetSceneBuildTemporaryBufferSize(hiprtCtx, buildInput, buildOptions, tmpSize));
			}
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

	// The trace kernel program is built as hiprtBuildTraceKernels() builds it
	// (see GetHiprtTraceKernelSource()), but through the kernel cache of the
	// device: it is precompiled like the other kernels, and hiprtc is required
	// only when it isn't
	void CompileKernel() {
		const double t0 = WallClockTime();

		// The local memory stack is sized for the work group size
		if (device.GetDeviceDesc().GetForceWorkGroupSize() > 0)
			kernelWorkGroupSize = device.GetDeviceDesc().GetForceWorkGroupSize();
		else
			kernelWorkGroupSize = HiprtTraceKernelWorkGroupSize;

		// The kernel must be compiled for the RTIP of the acceleration
		// structures HIPRT builds
		const string &arch = dynamic_cast<const HIPDeviceDescription &>(hipDevice->GetDeviceDesc()).GetHIPArch();
		int driverVersion;
		CHECK_HIP_ERROR(hipDriverGetVersion(&driverVersion));
		const u_int rtip = GetHiprtRtip(arch, driverVersion);

		// Without instances or motion blur, there is only the merged geometry
		// (or nothing for an empty DataSet)
		const vector<string> params = GetHiprtTraceKernelParameters(kernelWorkGroupSize,
				!scene, rtip, HIPRTEW_API_VERSION);
		const string source = GetHiprtTraceKernelSource();
		LR_LOG(device.GetContext(), "[HiprtAccel] Compiler options: " << oclKernelPersistentCache::ToOptsString(params));

		// hiprtc reads the HIPRT headers from HIPRT_PATH (see SetHiprtPath()
		// in hiprtew)
		vector<string> extraOpts;
		const char *hiprtPath = getenv("HIPRT_PATH");
		if (hiprtPath && hiprtPath[0])
			extraOpts.push_back(string("-I") + hiprtPath);

		hipKernelPersistentCache &kernelCache = *hipDevice->GetHIPKernelCache();
		hipKernelPersistentCache::CodeOrigin origin;
		string error;
		kernelModule = kernelCache.Compile(params, source, "HiprtAccel", &origin, &error, extraOpts);
		if (!kernelModule) {
			LR_LOG(device.GetContext(), "[HiprtAccel] HIP program compilation error: " << endl << error);

			throw runtime_error("HiprtAccel HIP program compilation error");
		} else if (error.length() > 0)
			LR_LOG(device.GetContext(), "[HiprtAccel] HIP program compilation warnings: " << endl << error);

		hipFunction_t function;
		CHECK_HIP_ERROR(hipModuleGetFunction(&function, kernelModule, "HiprtAccel_Intersect_RayBuffer"));
		kernel = hipDevice->GetKernel(function, "HiprtAccel_Intersect_RayBuffer");

		LR_LOG(device.GetContext(), "[HiprtAccel] Compilation completed in " <<
				int((WallClockTime() - t0) * 1000) << "ms" << kernelCache.GetCodeOriginDesc(origin));
	}

	void FreeKernel() {
		kernel.reset();

		if (kernelModule) {
			CHECK_HIP_ERROR(hipModuleUnload(kernelModule));
			kernelModule = nullptr;
		}
	}

	// The traversal stack sizes (the same of the HIPRT default stack)
	static constexpr u_int StackSize = 64;
	static constexpr u_int SharedStackSize = HiprtTraceKernelSharedStackSize;
	// The triangles that can be added to the merged geometry by copying the
	// instances of a mesh (~100MB)
	static constexpr u_longlong MergedInstancesTriangleBudget = 1000000;

	HIPIntersectionDevice *hipDevice;
	hiprtContext hiprtCtx;
	// accelerator.hiprt.builder.type
	hiprtBuildFlags buildFlags;

	// All the geometries built
	vector<hiprtGeometry> geometries;
	map<const TriangleMesh *, hiprtGeometry> instancedGeometries;
	// The meshes of the DataSet merged in a single geometry
	vector<bool> mergedMeshes;
	hiprtGeometry mergedGeometry;
	hiprtScene scene;
	HardwareDeviceBuffer *mergedTrianglesBuff;
	HardwareDeviceBuffer *instanceMeshesBuff;
	HardwareDeviceBuffer *tmpBuff;
	float rayTimeOffset;
	hiprtGlobalStackBuffer stackBuffer;

	hipModule_t kernelModule;
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
