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
	// 4 was MK_DL_SAMPLE_BSDF.

	// The vertex ray
	MK_RT_NEXT_VERTEX = 0,
	// The vertex ray and the shadow ray of the previous vertex: the next
	// vertex ray was sampled without waiting for the shadow ray (see
	// AdvancePaths_MK_RT)
	MK_RT_NEXT_VERTEX_DL = 1,
	// The shadow ray of the last vertex of the path: the sample is splat
	// after it
	MK_RT_DL_SPLAT = 2,
	// A step of the random walk of a subsurface material (GPUTaskSSSWalk):
	// the ray goes from a point in the object to the next scattering point,
	// or to the exit point, the next path vertex
	MK_RT_SSS_WALK = 3,
	// The shadow ray of the light sample of a volume segment of the vertex
	// ray (GPUTaskVolumeSegment) when it is traced alone: after a vertex
	// without light sampling, or where it needs more traces than the
	// vertex's shadow ray it was traced with. Then
	// GPUTaskVolumeSegment::nextState.
	MK_RT_SEGMENT_DL = 4,
	// The shadow ray, then the next vertex ray is sampled (shadow catchers,
	// and the scenes where the shadow rays can cross a surface)
	MK_RT_DL = 5,
	MK_GENERATE_NEXT_VERTEX_RAY = 6,
	MK_SPLAT_SAMPLE = 7,
	MK_NEXT_SAMPLE = 8,
	MK_GENERATE_CAMERA_RAY = 9,
	MK_DONE = 10
} PathState;

// The size in bytes of the random walk of a task (GPUTaskSSSWalk, see
// subsurface_funcs.cl, which checks it)
#define GPUTASK_SSSWALK_SIZE 256

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
	// The factor of lightRadiance if the shadow ray goes through a shadow
	// transparency: it takes the MIS weight out where the path doesn't go
	// through it (see DirectLight_BSDFSampling())
	float shadowTransparencyScale;

	unsigned int lightID;
} DirectLightIlluminateInfo;

// The state used to keep track of the rendered path. The BSDF of the last path
// vertex has its own buffer (tasksBSDF), so the accesses to these members
// don't each touch a cache line per lane: on CUDA and HIP the vertex is shaded
// with a private copy of it, and only what the next iteration reads is stored
// there (see AdvancePaths_MK_RT)
typedef struct {
	Spectrum throughput;

	// The pass-through events of the vertex ray: the sampler's value for the
	// first one (negative once used), then the sequence of seedPassThroughEvent
	// (as Scene::Intersect())
	Seed seedPassThroughEvent;
	float firstPassThroughEvent;
	
	int albedoToDo, photonGICacheEnabledOnLastHit,
			// See PhotonGICache_NextCausticChain(): the chain, and if the
			// caustic cache was looked up with a result at this vertex
			photonGICausticChain, photonGICausticLookup,
			photonGIShowIndirectPathMixUsed,
			// The shadow transparency lag used by Scene_Intersect()
			throughShadowTransparency;
	// At a vertex that uses the PhotonGI indirect cache (see
	// AdvancePaths_PhotonGIIndirect()), the index of its radiance group in
	// pgicRadiancePhotonsValues times 2, plus 1 if the path ends after the
	// light sampling of the vertex; NULL_INDEX elsewhere
	unsigned int photonGIIndirect;
} GPUTaskState;

typedef enum {
	ILLUMINATED, SHADOWED, NOT_VISIBLE
} DirectLightResult;

typedef struct {
	// Used to store some intermediate result
	DirectLightIlluminateInfo illumInfo;

	// As GPUTaskState, for the shadow ray
	Seed seedPassThroughEvent;
	float firstPassThroughEvent;

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

// The direct light along the first segment of the vertex ray in a homogeneous
// volume that can scatter (path.volume.equiangular.enable, see
// EyeVolumeSegmentLight in pathtracer.cpp): the light sample of a point drawn
// by equiangular sampling, set up when the vertex ray has crossed the
// segment (AdvancePaths_VolumeSegmentLight()) and traced after the light
// sampling of the vertex (MK_RT_SEGMENT_DL). One element when the scene
// doesn't use it.
typedef struct {
	// The light sample: its shadow ray, its light and the radiance it adds
	// (the shadow ray's transmittance aside) for the throughput below
	DirectLightIlluminateInfo illumInfo;
	Ray shadowRay;
	Spectrum throughput;
	// The volumes along the segment, for the shadow ray
	PathVolumeInfo volInfo;
	// The point of the sample and its landing normal (the light tree learning)
	Point p;
	Normal landingN;
	// As GPUTaskDirectLight, for the shadow ray
	Seed seedPassThroughEvent;
	float firstPassThroughEvent;
	int throughShadowTransparency;
	// The MIS weight of the light sample of the vertex the distance sampling
	// scattered at in the segment (1 elsewhere)
	float scatterWeight;
	// 0 the vertex ray has crossed no such segment yet, 1 it has, 2 and the
	// shadow ray is to trace, 3 the shadow ray is traced in the second slot
	// of the ray buffer, with the shadow ray of the vertex (MK_RT_DL)
	int state;
	// The state of the path after the shadow ray, when it is traced alone
	// (MK_RT_SEGMENT_DL)
	PathState nextState;
} GPUTaskVolumeSegment;

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
