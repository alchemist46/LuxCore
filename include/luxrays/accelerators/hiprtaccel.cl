#line 2 "hiprtaccel.cl"

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

// This kernel is compiled by HIPRT (with hiprtBuildTraceKernels()), not as an
// OpenCL source: the HIPRT device headers are already included and only the
// plain C ray types of LuxCore are prepended.

#include <hiprt/hiprt_device.h>

// The traversal stack: the first HIPRTACCEL_SHARED_STACK_SIZE entries of each
// thread are in local (LDS) memory, the others in a global memory buffer
// shared by the waves running on the device. The kernel must be launched with
// work groups of HIPRTACCEL_WORKGROUP_SIZE threads.
typedef hiprtDynamicStack HiprtAccelStack;

// The scene is a single level of instances, one for each mesh of the DataSet,
// so the instance ID is the mesh index.
//
// rayTimeOffset is subtracted from the ray time: HIPRT treats a time of 0 as
// "use the first motion frame", so the frame times are shifted to be >= 1
extern "C" __global__ void HiprtAccel_Intersect_RayBuffer(
		hiprtScene scene,
		hiprtGlobalStackBuffer globalStackBuffer,
		const float rayTimeOffset,
		const Ray *rays,
		RayHit *rayHits,
		const unsigned int rayCount) {
	// Select the ray to check
	const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
	if (gid >= rayCount)
		return;

	const Ray *ray = &rays[gid];

	// Check if I have to trace the ray
	const unsigned int flags = ray->flags;
	if (flags & RAY_FLAGS_MASKED)
		return;

	RayHit *rayHit = &rayHits[gid];

	// The empty DataSet case
	if (!scene) {
		rayHit->t = ray->maxt;
		rayHit->meshIndex = NULL_INDEX;
		rayHit->triangleIndex = NULL_INDEX;
		return;
	}

	hiprtRay hipRay;
	hipRay.origin = make_float3(ray->o.x, ray->o.y, ray->o.z);
	hipRay.direction = make_float3(ray->d.x, ray->d.y, ray->d.z);
	hipRay.minT = ray->mint;
	// HIPRT doesn't support an infinite maxT (the default of the rays not
	// pointing to a light source): rays miss what they should hit. The HIPRT
	// default is FLT_MAX.
	hipRay.maxT = fminf(ray->maxt, 3.402823466e+38f);

	__shared__ unsigned int sharedStackCache[HIPRTACCEL_SHARED_STACK_SIZE * HIPRTACCEL_WORKGROUP_SIZE];
	hiprtSharedStackBuffer sharedStackBuffer{ HIPRTACCEL_SHARED_STACK_SIZE, sharedStackCache };
	HiprtAccelStack stack(globalStackBuffer, sharedStackBuffer);
	// A single level of instancing
	hiprtEmptyInstanceStack instanceStack;

	hiprtSceneTraversalClosestCustomStack<HiprtAccelStack, hiprtEmptyInstanceStack> traversal(
			scene, hipRay, stack, instanceStack, hiprtFullRayMask,
			hiprtTraversalHintDefault, nullptr, nullptr, 0, ray->time - rayTimeOffset);
	const hiprtHit hit = traversal.getNextHit();

	// Write result
	if (hit.hasHit()) {
		rayHit->t = hit.t;
		// HIPRT barycentrics are the weights of the 2nd and 3rd vertex, like
		// LuxCore ones
		rayHit->b1 = hit.uv.x;
		rayHit->b2 = hit.uv.y;
		rayHit->meshIndex = hit.instanceID;
		rayHit->triangleIndex = hit.primID;
	} else {
		rayHit->t = ray->maxt;
		rayHit->meshIndex = NULL_INDEX;
		rayHit->triangleIndex = NULL_INDEX;
	}
}

// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
