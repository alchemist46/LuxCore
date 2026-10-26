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

#ifndef _SLG_COMPILEDSESSION_H
#define	_SLG_COMPILEDSESSION_H

#if !defined(LUXRAYS_DISABLE_OPENCL)

#include <map>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>

#include "slg/slg.h"
#include "slg/editaction.h"

#include "slg/core/indexbvh.h"
#include "slg/film/film.h"
#include "slg/scene/scene.h"
#include "slg/scene/sceneobject.h"
#include "slg/lights/strategies/lighttree.h"
#include "slg/lights/visibility/envlightvisibilitycache.h"
#include "slg/engines/pathtracer.h"
#include "slg/cameras/camera.h"

namespace slg {

// An allocator whose resize() leaves the new elements of a plain struct
// uninitialized, for a vector the caller fills (and clears) in parallel
template <typename T> class DefaultInitAllocator : public std::allocator<T> {
public:
	template <typename U> struct rebind { using other = DefaultInitAllocator<U>; };

	DefaultInitAllocator() noexcept { }
	template <typename U> DefaultInitAllocator(const DefaultInitAllocator<U> &) noexcept { }

	template <typename U> void construct(U *p) noexcept(std::is_nothrow_default_constructible<U>::value) {
		::new (static_cast<void *>(p)) U;
	}
	template <typename U, typename... Args> void construct(U *p, Args &&... args) {
		::new (static_cast<void *>(p)) U(std::forward<Args>(args)...);
	}
};

class CompiledScene {
public:
	CompiledScene(SceneConstRef scn, const PathTracer *pt);
	~CompiledScene();
	
	void SetMaxMemPageSize(const size_t maxSize);
	void EnableCode(const std::string &tags);

	void Compile();
	void Recompile(const EditActionList &editActions);
	void RecompilePhotonGI() { CompilePhotonGI(); }

	static void CompileFilm(const Film &film, slg::ocl::Film &oclFilm);

	static std::tuple<std::vector<float>, size_t>
	CompileDistribution1D(luxrays::Distribution1DConstRef dist);


	static std::tuple<std::vector<float>, size_t>
	CompileDistribution2D(luxrays::Distribution2DConstRef dist);

	static std::string ToOCLString(const slg::ocl::Spectrum &v);

	// Compiled Camera
	slg::ocl::Camera camera;
	std::vector<float> cameraBokehDistribution;
	u_int cameraBokehDistributionSize;

	// Compiled Scene Meshes
	std::vector<luxrays::Point> verts;
	std::vector<luxrays::Normal> normals;
	std::vector<luxrays::Normal> triNormals;
	std::vector<luxrays::UV> uvs;
	std::vector<luxrays::Spectrum> cols;
	std::vector<float> alphas;
	std::vector<float> vertexAOVs;
	std::vector<float> triAOVs;
	std::vector<luxrays::Triangle> tris;
	std::vector<luxrays::ocl::InterpolatedTransform> interpolatedTransforms;
	std::vector<luxrays::ocl::ExtMesh> meshDescs;
	luxrays::BSphere worldBSphere;

	// Compiled Scene Objects
	std::vector<slg::ocl::SceneObject> sceneObjs;

	// Compiled Lights (each one cleared and filled by CompileLights())
	std::vector<slg::ocl::LightSource, DefaultInitAllocator<slg::ocl::LightSource>> lightDefs;
	// Additional light related information
	std::vector<u_int> envLightIndices;
	std::vector<u_int> lightIndexOffsetByMeshIndex, lightIndexByTriIndex;
	// Env. light Distribution2Ds
	std::vector<float> envLightDistributions;
	// Compiled light sampling strategy
	std::vector<float> lightsDistribution;
	u_int lightsDistributionSize;
	std::vector<float> infiniteLightSourcesDistribution;
	u_int infiniteLightSourcesDistributionSize;
	// Light tree related data (see lighttree_funcs.cl); lightTreeLightIndex
	// is empty when the direct light sampling doesn't use the tree
	std::vector<slg::ocl::LightTreeNode> lightTreeNodes;
	std::vector<slg::ocl::LightTreeDistantLight> lightTreeDistantLights;
	std::vector<u_int> lightTreeLightIndex;
	// The learned slots of the tree (lightTreeLearnCellCount is 0 when off);
	// each device thread learns from its own samples, with these parameters
	slg::ocl::LightTreeLearn lightTreeLearn;
	u_int lightTreeLearnCellCount;
	float lightTreeLearnMaxWeight, lightTreeLearnSamples;
	// EnvLightVisibilityCache related data
	std::vector<slg::ocl::ELVCacheEntry> elvcAllEntries;
	std::vector<float> elvcDistributions;
	std::vector<u_int> elvcTileDistributionOffsets; 
	std::vector<luxrays::ocl::IndexBVHArrayNode> elvcBVHArrayNode;
	float elvcRadius2, elvcNormalCosAngle;
	u_int elvcTilesXCount, elvcTilesYCount;
	
	// Compiled Materials (and Volumes)
	std::vector<slg::ocl::Material> mats;
	std::vector<slg::ocl::MaterialEvalOp> matEvalOps;
	// Expressed in float
	u_int maxMaterialEvalStackSize;
	// The bump cache at the end of each material evaluation stack (see
	// Material_BumpSubMaterial()), in float: 3 per slot
	u_int materialBumpCacheSize;
	u_int defaultWorldVolumeIndex;
	// True if a shadow ray can be let through by a material or a volume, so
	// Scene_Intersect() needs the closest hit and its BSDF
	bool shadowRaysNeedBSDF;
	// True if a material is a random-walk subsurface one (PathOCL then needs
	// the buffer of the walks)
	bool hasSubsurface;

	// Compiled Textures
	std::vector<slg::ocl::Texture> texs;
	std::vector<slg::ocl::TextureEvalOp> texEvalOps;
	// Expressed in float
	u_int maxTextureEvalStackSize;

	// Compiled ImageMaps
	std::vector<slg::ocl::ImageMap> imageMapDescs;
	std::vector<std::vector<float> > imageMapMemBlocks;
	
	// Compiled PhotonGI cache

	// PhotonGI indirect cache
	std::vector<slg::ocl::RadiancePhoton> pgicRadiancePhotons;
	u_int pgicLightGroupCounts;
	std::vector<slg::ocl::Spectrum> pgicRadiancePhotonsValues;
	std::vector<luxrays::ocl::IndexBVHArrayNode> pgicRadiancePhotonsBVHArrayNode;
	// PhotonGI caustic cache
	std::vector<slg::ocl::Photon> pgicCausticPhotons;
	std::vector<luxrays::ocl::IndexBVHArrayNode> pgicCausticPhotonsBVHArrayNode;

	// All global settings
	slg::ocl::PathTracer compiledPathTracer;

	// Elements compiled during the last call to Compile()/Recompile()
	bool wasCameraCompiled, wasSceneObjectsCompiled, wasGeometryCompiled, 
		wasMaterialsCompiled, wasLightsCompiled, wasImageMapsCompiled,
		wasPhotonGICompiled;

private:
	// The place of an image map's pixels in the pages: the pages are allocated
	// and filled once every image map has its place (CompileImageMaps())
	void AddToImageMapMem(slg::ocl::ImageMap &im, const size_t memSize, std::vector<size_t> &pageSizes);
	u_int CompileImageMap(ImageMapConstRef im, std::vector<size_t> &pageSizes);

	void CompileCamera();
	void CompileSceneObjects();
	void CompileGeometry();
	u_int CompileMaterialConditionalOps(const u_int matIndex,
		const std::vector<slg::ocl::MaterialEvalOp> &evalOpsA, const u_int evalOpStackSizeA,
		const std::vector<slg::ocl::MaterialEvalOp> &evalOpsB, const u_int evalOpStackSizeB,
		std::vector<slg::ocl::MaterialEvalOp> &evalOps) const;
	u_int CompileMaterialConditionalOps(const u_int matIndex,
			const u_int matAIndex, const slg::ocl::MaterialEvalOpType opTypeA, const u_int bumpSlotBaseA,
			const u_int matBIndex, const slg::ocl::MaterialEvalOpType opTypeB, const u_int bumpSlotBaseB,
			std::vector<slg::ocl::MaterialEvalOp> &evalOps) const;
	bool MaterialHasBump(const u_int matIndex) const;
	bool MaterialHasPassThroughTransparency(const u_int matIndex) const;
	u_int MaterialBumpSlotCount(const u_int matIndex) const;
	u_int CompileMaterialOps(const u_int matIndex, const slg::ocl::MaterialEvalOpType opType,
			const u_int bumpSlotBase, std::vector<slg::ocl::MaterialEvalOp> &evalOps) const;
	void CompileMaterialOps();
	void CompileMaterials();
	void CompileTextureMapping2D(
		slg::ocl::TextureMapping2D *mapping,
		TextureMapping2DConstRef m
	);
	void CompileTextureMapping3D(
		slg::ocl::TextureMapping3D *mapping,
		TextureMapping3DConstRef m
	);
public:
	// The stack an evaluation needs: the peak depth, and the floats it leaves
	// on the stack (its result). The ops of a sub-node leave only its result,
	// so a node needs max(what its previous sub-nodes left + the peak of the
	// next one), not the sum of the sub-nodes. "+= n" pushes n floats that
	// stay until the op of the node, "+= sub-node" evaluates a sub-node; sum
	// is the bound used before, for the log.
	struct EvalStackNeed {
		u_int peak = 0, result = 0, sum = 0;

		EvalStackNeed &operator+=(const EvalStackNeed &subNode) {
			peak = luxrays::Max(peak, result + subNode.peak);
			result += subNode.result;
			sum += subNode.sum;
			return *this;
		}
		EvalStackNeed &operator+=(const u_int pushed) {
			result += pushed;
			peak = luxrays::Max(peak, result);
			sum += pushed;
			return *this;
		}
		// The op of the node pops everything and pushes its result
		EvalStackNeed &Result(const u_int resultSize) {
			result = resultSize;
			peak = luxrays::Max(peak, result);
			return *this;
		}
	};

private:
	EvalStackNeed CompileTextureOpsGenericBumpMap(const u_int texIndex);
	EvalStackNeed CompileTextureOps(const u_int texIndex, const slg::ocl::TextureEvalOpType opType);
	EvalStackNeed CompileTextureOpsNode(const u_int texIndex, const slg::ocl::TextureEvalOpType opType);
	EvalStackNeed CompileTextureBumpOps(const u_int texIndex);

	// The sharing of the evaluations repeated inside one bump evaluation
	// (see CompileTextureOps()): an evaluation is a texture, an eval type
	// and the hit point it sees, a number that changes when an op moves the
	// hit point
	struct TextureEvalShare {
		bool enabled = false;
		// Inside a texture that moves the hit point with its own ops
		// (Triplanar, Distort, Bombing): no sharing
		u_int opaqueDepth = 0;
		u_int point = 0, pointCount = 0;
		std::map<std::tuple<u_int, u_int, u_int>, u_int> ids;
		// Per shared evaluation: result size, used by a later one, and the
		// op count it stands for
		std::vector<u_int> sizes, unsharedOps;
		std::vector<bool> used;
		// The ops the shared evaluations save, minus the copies
		int extraOps = 0;

		u_int NewPoint() { return point = ++pointCount; }
	} texEvalShare;
	// For the log: the bump ops with and without the sharing
	struct {
		size_t sharedOps = 0, unsharedOps = 0;
		u_int maxSlotsSize = 0;
		// Not shared: the slots would not fit the private stack
		u_int unsharedTextures = 0;
	} texEvalShareStats;
	void CompileTextureOps();
	void CompileTextures();
	void CompileImageMaps();
	void CompileLights();

	void CompileELVC(EnvLightVisibilityCacheRPtr visibilityMapCache);
	void CompileLightStrategy();

	void CompilePhotonGI();
	void CompilePathTracer();

	SceneConstRef scene;
	const PathTracer *pathTracer;

	size_t maxMemPageSize;
	std::unordered_set<std::string> enabledCode;
};

}

#endif

#endif	/* _SLG_COMPILEDSESSION_H */
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
