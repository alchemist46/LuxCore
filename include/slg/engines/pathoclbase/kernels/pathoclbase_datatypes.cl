#line 2 "pathoclstatebase_datatypes.cl"

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

//------------------------------------------------------------------------------
// Some OpenCL specific definition
//------------------------------------------------------------------------------

#if defined(SLG_OPENCL_KERNEL)

#ifndef TRUE
#define TRUE 1
#endif

#ifndef FALSE
#define FALSE 0
#endif

#endif

//------------------------------------------------------------------------------
// GPUTask data types
//------------------------------------------------------------------------------

typedef enum {
	// Micro-kernel states
	//
	// One kernel, AdvancePaths_MK_RT, runs all the steps: between two kernel
	// runs, a task waits for the trace of its vertex ray and/or of its shadow
	// ray (see AdvancePaths_MK_RT for their slots) in one of the MK_RT_* states
	// (MK_GENERATE_CAMERA_RAY after Init() and MK_DONE are the exceptions).
	// 3 and 4 were MK_DL_ILLUMINATE and MK_DL_SAMPLE_BSDF.

	// The vertex ray
	MK_RT_NEXT_VERTEX = 0,
	// The vertex ray and the shadow ray of the previous vertex: the next
	// vertex ray was sampled without waiting for the shadow ray (see
	// AdvancePaths_MK_RT)
	MK_RT_NEXT_VERTEX_DL = 1,
	// The shadow ray of the last vertex of the path: the sample is splat
	// after it
	MK_RT_DL_SPLAT = 2,
	// The shadow ray, then the next vertex ray is sampled (shadow catchers,
	// and the scenes where the shadow rays can cross a surface)
	MK_RT_DL = 5,
	MK_GENERATE_NEXT_VERTEX_RAY = 6,
	MK_SPLAT_SAMPLE = 7,
	MK_NEXT_SAMPLE = 8,
	MK_GENERATE_CAMERA_RAY = 9,
	MK_DONE = 10
} PathState;

typedef struct {
	union {
		struct {
			// Must be a power of 2
			unsigned int previewResolutionReduction, previewResolutionReductionStep;
			unsigned int resolutionReduction;
		} rtpathocl;
	} renderEngine;

	Scene scene;
	Sampler sampler;
	PathTracer pathTracer;
	Filter pixelFilter;
	Film film;
} GPUTaskConfiguration;

typedef struct {
	unsigned int lightIndex;
	float pickPdf;

	float directPdfW;

	// Radiance to add to the result if light source is visible
	// Note: it doesn't include the pathThroughput
	Spectrum lightRadiance;
	// This is used only if Film channel IRRADIANCE is enabled and
	// only for the first path vertex
	Spectrum lightIrradiance;

	unsigned int lightID;
} DirectLightIlluminateInfo;

// The state used to keep track of the rendered path
typedef struct {
	Spectrum throughput;
	BSDF bsdf; // Variable size structure

	Seed seedPassThroughEvent;
	
	int albedoToDo, photonGICacheEnabledOnLastHit,
			photonGICausticCacheUsed, photonGIShowIndirectPathMixUsed,
			// The shadow transparency lag used by Scene_Intersect()
			throughShadowTransparency;
} GPUTaskState;

typedef enum {
	ILLUMINATED, SHADOWED, NOT_VISIBLE
} DirectLightResult;

typedef struct {
	// Used to store some intermediate result
	DirectLightIlluminateInfo illumInfo;

	Seed seedPassThroughEvent;

	DirectLightResult directLightResult;

	// The shadow transparency flag used by Scene_Intersect()
	int throughShadowTransparency;
} GPUTaskDirectLight;

// The path values at the vertex of a shadow ray traced with the next vertex
// ray (MK_RT_NEXT_VERTEX_DL): the next vertex ray was sampled before the
// shadow ray was traced, and it changed them. In their own buffer, so the
// other paths don't load a larger GPUTaskDirectLight.
typedef struct {
	Spectrum throughput;
	Spectrum irradiancePathThroughput;
	int firstPathVertex;
} GPUTaskDirectLightVertex;

typedef struct {
	// The task seed
	Seed seed;
} GPUTask;

// Space for temporary storage, used only within one run of AdvancePaths_MK_RT.
// On CUDA and HIP it is a private variable of the kernel (scratch memory, laid
// out lane by lane, so a wave's accesses to a member are coalesced), except
// for the HitPoint of Scene_Intersect(), which has its own global buffer (see
// AdvancePaths_MK_RT). The OpenCL device can't pass private pointers where the
// functions expect __global ones, so it keeps a global buffer of them.
typedef struct {
	BSDF tmpBsdf; // Variable size structure

	// This is used by TriangleLight_Illuminate() to temporary store the
	// point on the light sources.
	// Also used by Scene_Intersect() for evaluating volume textures.
	HitPoint tmpHitPoint;

	// This is used by DirectLight_BSDFSampling()
	PathDepthInfo tmpPathDepthInfo;
} GPUTaskTmp;

typedef struct {
	unsigned int sampleCount;
} GPUTaskStats;
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
