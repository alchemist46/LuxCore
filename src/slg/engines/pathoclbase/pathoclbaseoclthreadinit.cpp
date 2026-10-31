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

#if !defined(LUXRAYS_DISABLE_OPENCL)

#include <cstddef>

#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string/replace.hpp>

#include "luxcore/cfg.h"
#include "luxrays/core/geometry/transform.h"
#include "luxrays/utils/ocl.h"
#include "luxrays/devices/ocldevice.h"
#include "luxrays/devices/hipdevice.h"
#include "luxrays/kernels/kernels.h"

#include "slg/slg.h"
#include "slg/kernels/kernels.h"
#include "slg/renderconfig.h"
#include "slg/engines/pathoclbase/pathoclbase.h"
#include "slg/samplers/sobol.h"
#include "slg/utils/pathinfo.h"

using namespace std;
using namespace luxrays;
using namespace slg;

//------------------------------------------------------------------------------
// PathOCLBaseOCLRenderThread initialization methods
//------------------------------------------------------------------------------

void PathOCLBaseOCLRenderThread::InitFilm() {
	if (threadFilms.size() == 0)
		IncThreadFilms();

	u_int threadFilmWidth, threadFilmHeight, threadFilmSubRegion[4];
	GetThreadFilmSize(&threadFilmWidth, &threadFilmHeight, threadFilmSubRegion);

	for(ThreadFilmRPtr threadFilm: threadFilms)
		threadFilm->Init(renderEngine->GetFilm(), threadFilmWidth, threadFilmHeight,
			threadFilmSubRegion);
}

void PathOCLBaseOCLRenderThread::InitCamera() {
	CompiledScene *cscene = renderEngine->compiledScene;

	intersectionDevice.AllocBufferRO(&cameraBuff, &cscene->camera,
			sizeof(slg::ocl::Camera), "Camera");
	if (not cscene->cameraBokehDistribution.empty())
		intersectionDevice.AllocBufferRO(&cameraBokehDistributionBuff, cscene->cameraBokehDistribution.data(),
				cscene->cameraBokehDistributionSize, "CameraBokehDistribution");
	else
		intersectionDevice.FreeBuffer(&cameraBokehDistributionBuff);
}

void PathOCLBaseOCLRenderThread::InitGeometry() {
	CompiledScene *cscene = renderEngine->compiledScene;

	const BufferType memTypeFlags = renderEngine->ctx->GetUseOutOfCoreBuffers() ?
		((BufferType)(BUFFER_TYPE_READ_ONLY | BUFFER_TYPE_OUT_OF_CORE)) :
		BUFFER_TYPE_READ_ONLY;

	if (cscene->normals.size() > 0)
		intersectionDevice.AllocBuffer(&normalsBuff,
				memTypeFlags,
				&cscene->normals[0],
				sizeof(Normal) * cscene->normals.size(), "Normals");
	else
		intersectionDevice.FreeBuffer(&normalsBuff);

	if (cscene->uvs.size() > 0)
		intersectionDevice.AllocBuffer(&uvsBuff,
				memTypeFlags,
				&cscene->uvs[0],
				sizeof(UV) * cscene->uvs.size(), "UVs");
	else
		intersectionDevice.FreeBuffer(&uvsBuff);

	if (cscene->cols.size() > 0)
		intersectionDevice.AllocBuffer(&colsBuff,
				memTypeFlags,
				&cscene->cols[0],
				sizeof(Spectrum) * cscene->cols.size(), "Colors");
	else
		intersectionDevice.FreeBuffer(&colsBuff);

	if (cscene->alphas.size() > 0)
		intersectionDevice.AllocBuffer(&alphasBuff,
				memTypeFlags,
				&cscene->alphas[0],
				sizeof(float) * cscene->alphas.size(), "Alphas");
	else
		intersectionDevice.FreeBuffer(&alphasBuff);

	if (cscene->vertexAOVs.size() > 0)
		intersectionDevice.AllocBuffer(&vertexAOVBuff,
				memTypeFlags,
				&cscene->vertexAOVs[0],
				sizeof(float) * cscene->vertexAOVs.size(), "Vertex AOVs");
	else
		intersectionDevice.FreeBuffer(&vertexAOVBuff);

	if (cscene->triAOVs.size() > 0)
		intersectionDevice.AllocBuffer(&triAOVBuff,
				memTypeFlags,
				&cscene->triAOVs[0],
				sizeof(float) * cscene->triAOVs.size(), "Triangle AOVs");
	else
		intersectionDevice.FreeBuffer(&triAOVBuff);

	intersectionDevice.AllocBuffer(&triNormalsBuff,
			memTypeFlags,
			&cscene->triNormals[0],
			sizeof(Normal) * cscene->triNormals.size(), "Triangle normals");

	intersectionDevice.AllocBuffer(&vertsBuff,
			memTypeFlags,
			&cscene->verts[0],
			sizeof(Point) * cscene->verts.size(), "Vertices");

	intersectionDevice.AllocBuffer(&trianglesBuff,
			memTypeFlags,
			&cscene->tris[0],
			sizeof(Triangle) * cscene->tris.size(), "Triangles");

	if (cscene->interpolatedTransforms.size() > 0) {
		intersectionDevice.AllocBuffer(&interpolatedTransformsBuff,
				memTypeFlags,
				&cscene->interpolatedTransforms[0],
				sizeof(luxrays::ocl::InterpolatedTransform) * cscene->interpolatedTransforms.size(), "Interpolated transformations");
	} else
		intersectionDevice.FreeBuffer(&interpolatedTransformsBuff);

	intersectionDevice.AllocBufferRO(&meshDescsBuff, &cscene->meshDescs[0],
			sizeof(slg::ocl::ExtMesh) * cscene->meshDescs.size(), "Mesh description");
}

void PathOCLBaseOCLRenderThread::InitMaterials() {
	const size_t materialsCount = renderEngine->compiledScene->mats.size();
	intersectionDevice.AllocBufferRO(&materialsBuff, &renderEngine->compiledScene->mats[0],
			sizeof(slg::ocl::Material) * materialsCount, "Materials");

	intersectionDevice.AllocBufferRO(&materialEvalOpsBuff, &renderEngine->compiledScene->matEvalOps[0],
			sizeof(slg::ocl::MaterialEvalOp) * renderEngine->compiledScene->matEvalOps.size(), "Material evaluation ops");

	const u_int taskCount = renderEngine->taskCount;
	// Each task's block ends with the bump cache (see
	// Material_BumpSubMaterial()), also with the private stacks
	intersectionDevice.AllocBufferRW(&materialEvalStackBuff,
			nullptr, sizeof(float) * (renderEngine->compiledScene->maxMaterialEvalStackSize +
			renderEngine->compiledScene->materialBumpCacheSize) *
			taskCount, "Material evaluation stacks");

	// The random walks of the subsurface materials (one element when there is
	// none, only to keep the same kernel arguments)
	intersectionDevice.AllocBufferRW(&tasksSSSWalkBuff, nullptr,
			GPUTASK_SSSWALK_SIZE * (renderEngine->compiledScene->hasSubsurface ? taskCount : 1),
			"GPUTaskSSSWalk");
}

void PathOCLBaseOCLRenderThread::InitSceneObjects() {
	const BufferType memTypeFlags = renderEngine->ctx->GetUseOutOfCoreBuffers() ?
		((BufferType)(BUFFER_TYPE_READ_ONLY | BUFFER_TYPE_OUT_OF_CORE)) :
		BUFFER_TYPE_READ_ONLY;

	const u_int sceneObjsCount = renderEngine->compiledScene->sceneObjs.size();
	intersectionDevice.AllocBuffer(&scnObjsBuff, memTypeFlags,
			&renderEngine->compiledScene->sceneObjs[0],
			sizeof(slg::ocl::SceneObject) * sceneObjsCount, "Scene objects");
}

void PathOCLBaseOCLRenderThread::InitTextures() {
	const size_t texturesCount = renderEngine->compiledScene->texs.size();
	intersectionDevice.AllocBufferRO(&texturesBuff, &renderEngine->compiledScene->texs[0],
			sizeof(slg::ocl::Texture) * texturesCount, "Textures");

	intersectionDevice.AllocBufferRO(&textureEvalOpsBuff, &renderEngine->compiledScene->texEvalOps[0],
			sizeof(slg::ocl::TextureEvalOp) * renderEngine->compiledScene->texEvalOps.size(), "Texture evaluation ops");

	const u_int taskCount = renderEngine->taskCount;
	intersectionDevice.AllocBufferRW(&textureEvalStackBuff, 
			nullptr, sizeof(float) * renderEngine->compiledScene->maxTextureEvalStackSize *
			taskCount, "Texture evaluation stacks");
}

void PathOCLBaseOCLRenderThread::InitLights() {
	CompiledScene *cscene = renderEngine->compiledScene;

	intersectionDevice.AllocBufferRO(&lightsBuff, &cscene->lightDefs[0],
		sizeof(slg::ocl::LightSource) * cscene->lightDefs.size(), "Lights");
	if (cscene->envLightIndices.size() > 0) {
		intersectionDevice.AllocBufferRO(&envLightIndicesBuff, &cscene->envLightIndices[0],
				sizeof(u_int) * cscene->envLightIndices.size(), "Env. light indices");
	} else
		intersectionDevice.FreeBuffer(&envLightIndicesBuff);

	if (cscene->lightIndexOffsetByMeshIndex.size() > 0) {
		intersectionDevice.AllocBufferRO(&lightIndexOffsetByMeshIndexBuff, &cscene->lightIndexOffsetByMeshIndex[0],
			sizeof(u_int) * cscene->lightIndexOffsetByMeshIndex.size(), "Light offsets (Part I)");
	} else {
		intersectionDevice.FreeBuffer(&lightIndexOffsetByMeshIndexBuff);
	}
	if (cscene->lightIndexByTriIndex.size() > 0) {
		intersectionDevice.AllocBufferRO(&lightIndexByTriIndexBuff, &cscene->lightIndexByTriIndex[0],
			sizeof(u_int) * cscene->lightIndexByTriIndex.size(), "Light offsets (Part II)");
	} else {
		intersectionDevice.FreeBuffer(&lightIndexByTriIndexBuff);
	}


	if (cscene->envLightDistributions.size() > 0) {
		intersectionDevice.AllocBufferRO(&envLightDistributionsBuff, &cscene->envLightDistributions[0],
			sizeof(float) * cscene->envLightDistributions.size(), "Env. light distributions");
	} else
		intersectionDevice.FreeBuffer(&envLightDistributionsBuff);

	if (cscene->lightsDistributionSize > 0) {
		intersectionDevice.AllocBufferRO(
			&lightsDistributionBuff,
			cscene->lightsDistribution.data(),
			cscene->lightsDistributionSize,
			"LightsDistribution"
		);
	} else {
		intersectionDevice.FreeBuffer(&lightsDistributionBuff);
	}

	if (cscene->infiniteLightSourcesDistributionSize > 0) {
		intersectionDevice.AllocBufferRO(&infiniteLightSourcesDistributionBuff, cscene->infiniteLightSourcesDistribution.data(),
			cscene->infiniteLightSourcesDistributionSize, "InfiniteLightSourcesDistribution");
	} else {
		intersectionDevice.FreeBuffer(&infiniteLightSourcesDistributionBuff);
	}

	if (cscene->lightTreeLightIndex.size() > 0) {
		if (cscene->lightTreeNodes.size() > 0) {
			intersectionDevice.AllocBufferRO(&lightTreeNodesBuff, &cscene->lightTreeNodes[0],
				cscene->lightTreeNodes.size() * sizeof(slg::ocl::LightTreeNode), "Light tree nodes");
		} else
			intersectionDevice.FreeBuffer(&lightTreeNodesBuff);
		if (cscene->lightTreeDistantLights.size() > 0) {
			intersectionDevice.AllocBufferRO(&lightTreeDistantLightsBuff, &cscene->lightTreeDistantLights[0],
				cscene->lightTreeDistantLights.size() * sizeof(slg::ocl::LightTreeDistantLight), "Light tree distant lights");
		} else
			intersectionDevice.FreeBuffer(&lightTreeDistantLightsBuff);
		intersectionDevice.AllocBufferRO(&lightTreeLightIndexBuff, &cscene->lightTreeLightIndex[0],
			cscene->lightTreeLightIndex.size() * sizeof(u_int), "Light tree light index");
	} else {
		intersectionDevice.FreeBuffer(&lightTreeNodesBuff);
		intersectionDevice.FreeBuffer(&lightTreeDistantLightsBuff);
		intersectionDevice.FreeBuffer(&lightTreeLightIndexBuff);
	}

	// The learned slots of the tree: the cells start with no learned weight
	// and the sums at zero (each render starts learning afresh). Not with the
	// Metropolis sampler: each epoch of the learning changes the light
	// sampling, so the samples the chains stand on no longer have the
	// luminance the next steps are compared with, and the many short chains
	// of the GPU start again from a sample that is not drawn in proportion to
	// the new luminance (the simple scene, 150x100: -2.2% at 64 spp with the
	// learning, -0.5% without). The tree is used without learning.
	if ((cscene->lightTreeLightIndex.size() > 0) && (cscene->lightTreeLearnCellCount > 0) &&
			(renderEngine->oclSampler->type != slg::ocl::METROPOLIS)) {
		const u_int cellCount = cscene->lightTreeLearnCellCount;
		const u_int slotCount = cscene->lightTreeLearn.slotCount;
		lightTreeLearnCells.assign((size_t)cellCount * (slotCount + 1), 0.f);
		lightTreeLearnSums.assign((size_t)cellCount * slotCount, 0.f);
		lightTreeLearnBoundsSums.assign((size_t)cellCount * slotCount, 0.f);
		lightTreeLearnCounts.assign(cellCount, 0.f);
		lightTreeLearnNextSamples = 0.0;

		intersectionDevice.AllocBufferRO(&lightTreeLearnBuff, &cscene->lightTreeLearn,
			sizeof(slg::ocl::LightTreeLearn), "Light tree learning parameters");
		intersectionDevice.AllocBufferRW(&lightTreeLearnCellsBuff, lightTreeLearnCells.data(),
			lightTreeLearnCells.size() * sizeof(float), "Light tree learned cells");
		intersectionDevice.AllocBufferRW(&lightTreeLearnSumsBuff, lightTreeLearnSums.data(),
			lightTreeLearnSums.size() * sizeof(float), "Light tree learning sums");
		intersectionDevice.AllocBufferRW(&lightTreeLearnBoundsSumsBuff, lightTreeLearnBoundsSums.data(),
			lightTreeLearnBoundsSums.size() * sizeof(float), "Light tree learning bounds sums");
		intersectionDevice.AllocBufferRW(&lightTreeLearnCountsBuff, lightTreeLearnCounts.data(),
			lightTreeLearnCounts.size() * sizeof(float), "Light tree learning counts");
	} else {
		intersectionDevice.FreeBuffer(&lightTreeLearnBuff);
		intersectionDevice.FreeBuffer(&lightTreeLearnCellsBuff);
		intersectionDevice.FreeBuffer(&lightTreeLearnSumsBuff);
		intersectionDevice.FreeBuffer(&lightTreeLearnBoundsSumsBuff);
		intersectionDevice.FreeBuffer(&lightTreeLearnCountsBuff);
	}

	if (cscene->elvcAllEntries.size() > 0) {
		intersectionDevice.AllocBufferRO(&elvcAllEntriesBuff, &cscene->elvcAllEntries[0],
			cscene->elvcAllEntries.size() * sizeof(slg::ocl::ELVCacheEntry), "ELVC all entries");
		intersectionDevice.AllocBufferRO(&elvcDistributionsBuff, &cscene->elvcDistributions[0],
			cscene->elvcDistributions.size() * sizeof(float), "ELVC distributions table");
		if (cscene->elvcTileDistributionOffsets.size() > 0) {
			intersectionDevice.AllocBufferRO(&elvcTileDistributionOffsetsBuff, &cscene->elvcTileDistributionOffsets[0],
					cscene->elvcTileDistributionOffsets.size() * sizeof(u_int), "ELVC tile distribution offsets table");
		} else
			intersectionDevice.FreeBuffer(&elvcTileDistributionOffsetsBuff);
		intersectionDevice.AllocBufferRO(&elvcBVHNodesBuff, &cscene->elvcBVHArrayNode[0],
			cscene->elvcBVHArrayNode.size() * sizeof(luxrays::ocl::IndexBVHArrayNode), "ELVC BVH nodes");
	} else {
		intersectionDevice.FreeBuffer(&elvcAllEntriesBuff);
		intersectionDevice.FreeBuffer(&elvcDistributionsBuff);
		intersectionDevice.FreeBuffer(&elvcTileDistributionOffsetsBuff);
		intersectionDevice.FreeBuffer(&elvcBVHNodesBuff);
	}
}

void PathOCLBaseOCLRenderThread::InitPhotonGI() {
	CompiledScene *cscene = renderEngine->compiledScene;

	const BufferType memTypeFlags = renderEngine->ctx->GetUseOutOfCoreBuffers() ?
		((BufferType)(BUFFER_TYPE_READ_ONLY | BUFFER_TYPE_OUT_OF_CORE)) :
		BUFFER_TYPE_READ_ONLY;

	if (cscene->pgicRadiancePhotons.size() > 0) {
		intersectionDevice.AllocBuffer(&pgicRadiancePhotonsBuff, memTypeFlags, &cscene->pgicRadiancePhotons[0],
			cscene->pgicRadiancePhotons.size() * sizeof(slg::ocl::RadiancePhoton), "PhotonGI indirect cache all entries");
		intersectionDevice.AllocBuffer(&pgicRadiancePhotonsValuesBuff, memTypeFlags, &cscene->pgicRadiancePhotonsValues[0],
			cscene->pgicRadiancePhotonsValues.size() * sizeof(slg::ocl::Spectrum), "PhotonGI indirect cache all entry values");
		intersectionDevice.AllocBuffer(&pgicRadiancePhotonsBVHNodesBuff, memTypeFlags, &cscene->pgicRadiancePhotonsBVHArrayNode[0],
			cscene->pgicRadiancePhotonsBVHArrayNode.size() * sizeof(luxrays::ocl::IndexBVHArrayNode), "PhotonGI indirect cache BVH nodes");
	} else {
		intersectionDevice.FreeBuffer(&pgicRadiancePhotonsBuff);
		intersectionDevice.FreeBuffer(&pgicRadiancePhotonsValuesBuff);
		intersectionDevice.FreeBuffer(&pgicRadiancePhotonsBVHNodesBuff);
	}

	if (cscene->pgicCausticPhotons.size() > 0) {
		intersectionDevice.AllocBuffer(&pgicCausticPhotonsBuff, memTypeFlags, &cscene->pgicCausticPhotons[0],
			cscene->pgicCausticPhotons.size() * sizeof(slg::ocl::Photon), "PhotonGI caustic cache all entries");
		intersectionDevice.AllocBuffer(&pgicCausticPhotonsBVHNodesBuff, memTypeFlags, &cscene->pgicCausticPhotonsBVHArrayNode[0],
			cscene->pgicCausticPhotonsBVHArrayNode.size() * sizeof(luxrays::ocl::IndexBVHArrayNode), "PhotonGI caustic cache BVH nodes");
	} else {
		intersectionDevice.FreeBuffer(&pgicCausticPhotonsBuff);
		intersectionDevice.FreeBuffer(&pgicCausticPhotonsBVHNodesBuff);
	}
}

// On a HIP device, the lookups of the image maps use the texture hardware: two
// texture objects per image map (filtered and single texels), in the device's
// copy of the image map descriptions. The density grids index their pixels
// directly and keep them, and so do the image maps with a white wrap: the
// border colour of a texture object reads black on AMD (the WrapWhite unit
// tests). LUXRAYS_HIP_TEXTURES=0 reads every image map without the texture
// hardware.
void PathOCLBaseOCLRenderThread::CreateImageMapTextures(vector<slg::ocl::ImageMap> &imageMapDescs) {
#if !defined(LUXRAYS_DISABLE_HIP)
	HIPDevice *hipDevice = dynamic_cast<HIPDevice *>(&intersectionDevice);
	if (!hipDevice)
		return;

	const char *enable = getenv("LUXRAYS_HIP_TEXTURES");
	if (enable && (string(enable) == "0"))
		return;

	CompiledScene *cscene = renderEngine->compiledScene;

	const double startTime = WallClockTime();
	size_t textureBytes = 0;

	vector<bool> densityGrid(imageMapDescs.size(), false);
	for (auto &tex : cscene->texs) {
		if (tex.type == slg::ocl::DENSITYGRID_TEX)
			densityGrid[tex.densityGrid.imageMapIndex] = true;
	}

	for (u_int i = 0; i < imageMapDescs.size(); ++i) {
		slg::ocl::ImageMap &im = imageMapDescs[i];
		if (densityGrid[i] || (im.wrapType == slg::ocl::WRAP_WHITE))
			continue;

		u_int channelSize;
		switch (im.storageType) {
			case slg::ocl::BYTE:
				channelSize = 1;
				break;
			case slg::ocl::HALF:
				channelSize = 2;
				break;
			case slg::ocl::FLOAT:
			default:
				channelSize = 4;
				break;
		}

		HIPDevice::TextureAddressMode addressMode;
		switch (im.wrapType) {
			case slg::ocl::WRAP_REPEAT:
				addressMode = HIPDevice::TEXTURE_WRAP;
				break;
			case slg::ocl::WRAP_CLAMP:
				addressMode = HIPDevice::TEXTURE_CLAMP;
				break;
			case slg::ocl::WRAP_BLACK:
			default:
				addressMode = HIPDevice::TEXTURE_BORDER;
				break;
		}

		im.textureObject = hipDevice->CreateTexture2D(
				&cscene->imageMapMemBlocks[im.pageIndex][im.pixelsIndex],
				im.width, im.height, im.channelCount, channelSize,
				im.filterType == slg::ocl::FILTER_LINEAR, addressMode, 0.f,
				"ImageMap", &im.texelTextureObject);
		if (im.textureObject) {
			imageMapTextures.push_back(im.textureObject);
			textureBytes += (size_t)im.width * im.height * im.channelCount * channelSize;
		}
	}

	if (imageMapTextures.size() > 0)
		SLG_LOG("[PathOCLBaseRenderThread::" << threadIndex << "] Image maps read with the texture hardware: " <<
				imageMapTextures.size() << "/" << imageMapDescs.size() << " (" << textureBytes / (1024 * 1024) <<
				"MB uploaded in " << int((WallClockTime() - startTime) * 1000.0) << "ms)");
#endif
}

void PathOCLBaseOCLRenderThread::FreeImageMapTextures() {
#if !defined(LUXRAYS_DISABLE_HIP)
	if (imageMapTextures.size() == 0)
		return;

	HIPDevice *hipDevice = dynamic_cast<HIPDevice *>(&intersectionDevice);
	for (auto t : imageMapTextures)
		hipDevice->FreeTexture2D(t);
	imageMapTextures.clear();
#endif
}

void PathOCLBaseOCLRenderThread::InitImageMaps() {
	CompiledScene *cscene = renderEngine->compiledScene;

	FreeImageMapTextures();

	if (cscene->imageMapDescs.size() > 0) {
		vector<slg::ocl::ImageMap> imageMapDescs = cscene->imageMapDescs;
		CreateImageMapTextures(imageMapDescs);

		intersectionDevice.AllocBufferRO(&imageMapDescsBuff,
				&imageMapDescs[0],
				sizeof(slg::ocl::ImageMap) * imageMapDescs.size(), "ImageMap descriptions");

		// The pixels are not needed when every image map has its textures
		const bool needPixels = (imageMapTextures.size() < imageMapDescs.size());
		const size_t pageCount = needPixels ? cscene->imageMapMemBlocks.size() : 0;

		// Free unused pages
		for (u_int i = pageCount; i < imageMapsBuff.size(); ++i)
			intersectionDevice.FreeBuffer(&imageMapsBuff[i]);
		imageMapsBuff.resize(pageCount, NULL);

		const BufferType memTypeFlags = renderEngine->ctx->GetUseOutOfCoreBuffers() ?
			((BufferType)(BUFFER_TYPE_READ_ONLY | BUFFER_TYPE_OUT_OF_CORE)) :
			BUFFER_TYPE_READ_ONLY;

		for (u_int i = 0; i < imageMapsBuff.size(); ++i) {
			intersectionDevice.AllocBuffer(&(imageMapsBuff[i]),
					memTypeFlags,
					&(cscene->imageMapMemBlocks[i][0]),
					sizeof(float) * cscene->imageMapMemBlocks[i].size(), "ImageMaps");
		}
	} else {
		intersectionDevice.FreeBuffer(&imageMapDescsBuff);
		for (u_int i = 0; i < imageMapsBuff.size(); ++i)
			intersectionDevice.FreeBuffer(&imageMapsBuff[i]);
		imageMapsBuff.resize(0);
	}
}

void PathOCLBaseOCLRenderThread::UpdateGPUTaskConfigurationBuffer() {
	// The scene configuration (default volume, shadow ray any-hit) depends on
	// the camera volume and on the materials: re-upload it after their edit
	intersectionDevice.AllocBufferRO(&taskConfigBuff, &renderEngine->taskConfig,
			sizeof(slg::ocl::pathoclbase::GPUTaskConfiguration), "GPUTaskConfiguration");
}

void PathOCLBaseOCLRenderThread::InitGPUTaskBuffer() {
	const u_int taskCount = renderEngine->taskCount;

	//--------------------------------------------------------------------------
	// Allocate tasksConfigBuff
	//--------------------------------------------------------------------------

	intersectionDevice.AllocBufferRO(&taskConfigBuff, &renderEngine->taskConfig, sizeof(slg::ocl::pathoclbase::GPUTaskConfiguration), "GPUTaskConfiguration");

	//--------------------------------------------------------------------------
	// Allocate tasksBuff
	//--------------------------------------------------------------------------

	intersectionDevice.AllocBufferRW(&tasksBuff, nullptr, sizeof(slg::ocl::pathoclbase::GPUTask) * taskCount, "GPUTask");

	//--------------------------------------------------------------------------
	// Allocate tasksTmpBuff and tasksIntersectHitPointBuff: the temporary
	// storage of AdvancePaths_MK_RT is a private variable of the kernel on CUDA
	// and HIP, but for the HitPoint of Scene_Intersect(). The unused buffer
	// has one element (only to keep the same kernel arguments).
	//--------------------------------------------------------------------------

	const bool privateTaskTmp = intersectionDevice.GetDeviceDesc().GetType() & (DEVICE_TYPE_CUDA_ALL | DEVICE_TYPE_HIP_ALL);
	intersectionDevice.AllocBufferRW(&tasksTmpBuff, nullptr, sizeof(slg::ocl::pathoclbase::GPUTaskTmp) * (privateTaskTmp ? 1 : taskCount), "GPUTaskTmp");
	intersectionDevice.AllocBufferRW(&tasksIntersectHitPointBuff, nullptr, sizeof(slg::ocl::HitPoint) * (privateTaskTmp ? taskCount : 1), "GPUTask intersection HitPoint");

	//--------------------------------------------------------------------------
	// Allocate tasksDirectLightBuff
	//--------------------------------------------------------------------------

	intersectionDevice.AllocBufferRW(&tasksDirectLightBuff, nullptr, sizeof(slg::ocl::pathoclbase::GPUTaskDirectLight) * taskCount, "GPUTaskDirectLight");
	intersectionDevice.AllocBufferRW(&tasksDirectLightVertexBuff, nullptr, sizeof(slg::ocl::pathoclbase::GPUTaskDirectLightVertex) * taskCount, "GPUTaskDirectLightVertex");

	//--------------------------------------------------------------------------
	// Allocate tasksStateBuff
	//--------------------------------------------------------------------------

	intersectionDevice.AllocBufferRW(&tasksStateBuff, nullptr, sizeof(slg::ocl::pathoclbase::GPUTaskState) * taskCount, "GPUTaskState");

	//--------------------------------------------------------------------------
	// Allocate tasksBSDFBuff: the BSDF of the last path vertex, in its own
	// buffer to keep GPUTaskState small
	//--------------------------------------------------------------------------

	intersectionDevice.AllocBufferRW(&tasksBSDFBuff, nullptr, sizeof(slg::ocl::BSDF) * taskCount, "GPUTask BSDF");

	//--------------------------------------------------------------------------
	// Allocate pathStatesBuff: the state of each path is kept in its own
	// compact array, so the kernels can check it without touching the large
	// GPUTaskState
	//--------------------------------------------------------------------------

	intersectionDevice.AllocBufferRW(&pathStatesBuff, nullptr, sizeof(slg::ocl::pathoclbase::PathState) * taskCount, "PathState");
}

void PathOCLBaseOCLRenderThread::InitSamplerSharedDataBuffer() {
	const u_int *subRegion = renderEngine->GetFilm().GetSubRegion();
	const u_int filmRegionPixelCount = (subRegion[1] - subRegion[0] + 1) * (subRegion[3] - subRegion[2] + 1);

	size_t size = 0;
	if (renderEngine->oclSampler->type == slg::ocl::METROPOLIS) {
		size += sizeof(slg::ocl::MetropolisSamplerSharedData);

		// Plus the luminance of the large steps of each task and the task
		// whose first sample starts its chain
		size += (sizeof(slg::ocl::MetropolisTaskLuminance) + sizeof(u_int)) * renderEngine->taskCount;
	} else if (renderEngine->oclSampler->type == slg::ocl::SOBOL) {
		size += sizeof(slg::ocl::SobolSamplerSharedData);

		// Plus the a pass field for each pixel
		size += sizeof(u_int) * filmRegionPixelCount;

		// Plus the Sobol directions array
		size += sizeof(u_int) * renderEngine->pathTracer.eyeSampleSize * SOBOL_BITS;

		// Plus the list of the pixels to converge of the per-pixel noise
		// threshold halt test
		if (renderEngine->oclSampler->sobol.halfReplicas)
			size += sizeof(u_int) * filmRegionPixelCount;
	} else if (renderEngine->oclSampler->type == slg::ocl::TILEPATHSAMPLER) {
		size += sizeof(slg::ocl::TilePathSamplerSharedData);

		switch (renderEngine->GetType()) {
			case TILEPATHOCL:
				size += sizeof(u_int) * renderEngine->pathTracer.eyeSampleSize * SOBOL_BITS;
				break;
			case RTPATHOCL:
				break;
			default:
				throw runtime_error("Unknown render engine in PathOCLBaseRenderThread::InitSamplerSharedDataBuffer(): " +
						ToString(renderEngine->GetType()));
		}
	} else
		throw runtime_error("Unknown sampler.type in PathOCLBaseRenderThread::InitSamplerSharedDataBuffer(): " +
				ToString(renderEngine->oclSampler->type));

	if (size == 0)
		intersectionDevice.FreeBuffer(&samplerSharedDataBuff);
	else
		intersectionDevice.AllocBufferRW(&samplerSharedDataBuff, nullptr, size, "SamplerSharedData");

	// Initialize the sampler shared data
	if (renderEngine->oclSampler->type == slg::ocl::SOBOL) {
		auto _buffer = std::make_unique<char[]>(size);
		auto buffer = _buffer.get();

		// Initialize SobolSamplerSharedData fields
		slg::ocl::SobolSamplerSharedData *sssd = (slg::ocl::SobolSamplerSharedData *)buffer;

		// Each device must have its own seed: each one has its own pass
		// counters so, with the same seed, all of them would render the
		// same samples
		sssd->seedBase = renderEngine->seedBase + threadIndex;
		sssd->bucketIndex = 0;
		sssd->filmRegionPixelCount = filmRegionPixelCount;
		sssd->todoPixelCount = 0;
		sssd->todoBucketCount = 0;
		sssd->todoPixelsOffset = filmRegionPixelCount + renderEngine->pathTracer.eyeSampleSize * SOBOL_BITS;
		sssd->drain = 0;
		uploadedTodoPixels.reset();

		// Initialize all pass values. The pass buffer is attached at the
		// end of slg::ocl::SobolSamplerSharedData
		u_int *passBuffer = (u_int *)(buffer + sizeof(slg::ocl::SobolSamplerSharedData));
		fill(passBuffer, passBuffer + filmRegionPixelCount, SOBOL_STARTOFFSET);

		// Initialize the Sobol directions array values. The pass buffer is attached at the
		// end of slg::ocl::SobolSamplerSharedData + all pass values

		u_int *sobolDirections = (u_int *)(buffer + sizeof(slg::ocl::SobolSamplerSharedData) + sizeof(u_int) * filmRegionPixelCount);
		SobolSequence::GenerateDirectionVectors(sobolDirections, renderEngine->pathTracer.eyeSampleSize);

		// Write the data
		intersectionDevice.EnqueueWriteBuffer(samplerSharedDataBuff, CL_TRUE, size, buffer);

	} else if (renderEngine->oclSampler->type == slg::ocl::METROPOLIS) {
		// The per task data is written by the kernels and by
		// PathOCLOpenCLRenderThread::MetropolisBootstrap()
		slg::ocl::MetropolisSamplerSharedData msd;
		msd.drain = 0;
		msd.invMeanI = 1.f;
		msd.taskCount = renderEngine->taskCount;

		intersectionDevice.EnqueueWriteBufferRange(samplerSharedDataBuff, CL_TRUE,
				0, sizeof(msd), &msd);
	} else if (renderEngine->oclSampler->type == slg::ocl::TILEPATHSAMPLER) {
		// TilePathSamplerSharedData is updated in PathOCLBaseOCLRenderThread::UpdateSamplerData()
		
		switch (renderEngine->GetType()) {
			case TILEPATHOCL: {
				auto _buffer = std::make_unique<char[]>(size);
				auto buffer = _buffer.get();

				// Initialize the Sobol directions array values
				u_int *sobolDirections = (u_int *)(buffer + sizeof(slg::ocl::TilePathSamplerSharedData));
				SobolSequence::GenerateDirectionVectors(sobolDirections, renderEngine->pathTracer.eyeSampleSize);

				intersectionDevice.EnqueueWriteBuffer(samplerSharedDataBuff, CL_TRUE, size, &buffer[0]);
				break;
			}
			case RTPATHOCL:
				break;
			default:
				throw runtime_error("Unknown render engine in PathOCLBaseRenderThread::InitSamplerSharedDataBuffer(): " +
						ToString(renderEngine->GetType()));
		}
	}
}

void PathOCLBaseOCLRenderThread::UpdateSamplerTodoPixels() {
	// The list of the pixels to converge of the per-pixel noise threshold
	// halt test, read by SobolSampler_InitNewSample()
	if (!samplerSharedDataBuff || (renderEngine->oclSampler->type != slg::ocl::SOBOL) ||
			!renderEngine->oclSampler->sobol.halfReplicas)
		return;

	std::shared_ptr<const std::vector<u_int>> todoPixels = renderEngine->GetFilm().GetHalfBufferTodoPixels();
	if (todoPixels == uploadedTodoPixels)
		return;
	uploadedTodoPixels = todoPixels;

	const u_int *subRegion = renderEngine->GetFilm().GetSubRegion();
	const u_int filmRegionPixelCount = (subRegion[1] - subRegion[0] + 1) * (subRegion[3] - subRegion[2] + 1);

	u_int counts[2] = { 0, 0 };
	if (todoPixels && !todoPixels->empty() && (todoPixels->size() <= filmRegionPixelCount)) {
		const u_int bucketSize = renderEngine->oclSampler->sobol.bucketSize;
		const u_int listBucketCount = (todoPixels->size() + bucketSize - 1) / bucketSize;
		// Each pixel of the list gets at most about 64 of the samples of a
		// round of all the tasks; the rest sweep the whole film
		const u_int minBucketCount = (renderEngine->taskCount + 64 * bucketSize - 1) / (64 * bucketSize);

		counts[0] = todoPixels->size();
		counts[1] = Max(listBucketCount, minBucketCount);

		const size_t listOffset = sizeof(slg::ocl::SobolSamplerSharedData) +
				sizeof(u_int) * (filmRegionPixelCount + renderEngine->pathTracer.eyeSampleSize * SOBOL_BITS);
		intersectionDevice.EnqueueWriteBufferRange(samplerSharedDataBuff, CL_TRUE,
				listOffset, sizeof(u_int) * todoPixels->size(), todoPixels->data());
	}

	intersectionDevice.EnqueueWriteBufferRange(samplerSharedDataBuff, CL_TRUE,
			offsetof(slg::ocl::SobolSamplerSharedData, todoPixelCount), sizeof(counts), counts);
}

void PathOCLBaseOCLRenderThread::InitSamplesBuffer() {
	const u_int taskCount = renderEngine->taskCount;

	//--------------------------------------------------------------------------
	// Sample size
	//--------------------------------------------------------------------------

	size_t sampleSize = 0;

	// Add Sample memory size
	switch (renderEngine->oclSampler->type) {
		case  slg::ocl::METROPOLIS: {
			sampleSize += sizeof(slg::ocl::MetropolisSample);
			break;
		}
		case slg::ocl::SOBOL: {
			sampleSize += sizeof(slg::ocl::SobolSample);
			break;
		}
		case slg::ocl::TILEPATHSAMPLER: {
			sampleSize += sizeof(slg::ocl::TilePathSample);
			break;
		}
		default:
			throw runtime_error("Unknown sampler.type in PathOCLBaseRenderThread::InitSamplesBuffer(): " +
					ToString(renderEngine->oclSampler->type));
	}

	SLG_LOG("[PathOCLBaseRenderThread::" << threadIndex << "] Size of a Sample: " << sampleSize << "bytes");
	intersectionDevice.AllocBufferRW(&samplesBuff, nullptr, sampleSize * taskCount, "Sample");
}

void PathOCLBaseOCLRenderThread::InitSampleResultsBuffer() {
	const u_int taskCount = renderEngine->taskCount;

	const size_t sampleResultSize = sizeof(slg::ocl::SampleResult);

	SLG_LOG("[PathOCLBaseRenderThread::" << threadIndex << "] Size of a SampleResult: " << sampleResultSize << "bytes");
	intersectionDevice.AllocBufferRW(&sampleResultsBuff, nullptr, sampleResultSize * taskCount, "SampleResult");
}

void PathOCLBaseOCLRenderThread::InitSampleDataBuffer() {
	const u_int taskCount = renderEngine->taskCount;

	size_t uDataSize;
	if (renderEngine->oclSampler->type == slg::ocl::SOBOL) {
		// To store IDX_SCREEN_X and IDX_SCREEN_Y
		uDataSize = 2 * sizeof(float);
	} else if (renderEngine->oclSampler->type == slg::ocl::METROPOLIS) {
		// Metropolis needs 2 sets of samples, the current and the proposed mutation
		uDataSize = 2 * sizeof(float) * renderEngine->pathTracer.eyeSampleSize;
	} else if (renderEngine->oclSampler->type == slg::ocl::TILEPATHSAMPLER) {
		// To store IDX_SCREEN_X and IDX_SCREEN_Y
		uDataSize = 2 * sizeof(float);
	} else
		throw runtime_error("Unknown sampler.type in PathOCLBaseRenderThread::InitSampleDataBuffer(): " + ToString(renderEngine->oclSampler->type));

	SLG_LOG("[PathOCLBaseRenderThread::" << threadIndex << "] Size of a SampleData: " << uDataSize << "bytes");

	intersectionDevice.AllocBufferRW(&sampleDataBuff, nullptr, uDataSize * taskCount, "SampleData");
}

void PathOCLBaseOCLRenderThread::InitRender() {
	//--------------------------------------------------------------------------
	// Film definition
	//--------------------------------------------------------------------------

	InitFilm();

	//--------------------------------------------------------------------------
	// Camera definition
	//--------------------------------------------------------------------------

	InitCamera();

	//--------------------------------------------------------------------------
	// Scene geometry
	//--------------------------------------------------------------------------

	InitGeometry();

	//--------------------------------------------------------------------------
	// Image maps
	//--------------------------------------------------------------------------

	InitImageMaps();

	//--------------------------------------------------------------------------
	// Texture definitions
	//--------------------------------------------------------------------------

	InitTextures();

	//--------------------------------------------------------------------------
	// Material definitions
	//--------------------------------------------------------------------------

	InitMaterials();

	//--------------------------------------------------------------------------
	// Mesh <=> Material links
	//--------------------------------------------------------------------------

	InitSceneObjects();

	//--------------------------------------------------------------------------
	// Light definitions
	//--------------------------------------------------------------------------

	InitLights();

	//--------------------------------------------------------------------------
	// Light definitions
	//--------------------------------------------------------------------------

	InitPhotonGI();

	//--------------------------------------------------------------------------
	// GPUTaskStats
	//--------------------------------------------------------------------------

	const u_int taskCount = renderEngine->taskCount;

	// In case renderEngine->taskCount has changed
	gpuTaskStats = std::move(
		std::make_unique<slg::ocl::pathoclbase::GPUTaskStats[]>(taskCount)
	);
	for (u_int i = 0; i < taskCount; ++i)
		gpuTaskStats[i].sampleCount = 0;

	//--------------------------------------------------------------------------
	// Allocate Ray/RayHit buffers
	//--------------------------------------------------------------------------

	// The vertex rays of the tasks, then their shadow rays
	intersectionDevice.AllocBufferRW(&raysBuff, nullptr, sizeof(Ray) * 2 * taskCount, "Ray");
	intersectionDevice.AllocBufferRW(&hitsBuff, nullptr, sizeof(RayHit) * 2 * taskCount, "RayHit");

	//--------------------------------------------------------------------------
	// Allocate GPU task buffers
	//--------------------------------------------------------------------------

	InitGPUTaskBuffer();

	//--------------------------------------------------------------------------
	// Allocate GPU task statistic buffers
	//--------------------------------------------------------------------------

	intersectionDevice.AllocBufferRW(&taskStatsBuff, nullptr, sizeof(slg::ocl::pathoclbase::GPUTaskStats) * taskCount, "GPUTask Stats");

	//--------------------------------------------------------------------------
	// Allocate sampler shared data buffer
	//--------------------------------------------------------------------------

	InitSamplerSharedDataBuffer();

	//--------------------------------------------------------------------------
	// Allocate sample buffers
	//--------------------------------------------------------------------------

	InitSamplesBuffer();

	//--------------------------------------------------------------------------
	// Allocate sample data buffers
	//--------------------------------------------------------------------------

	InitSampleDataBuffer();

	//--------------------------------------------------------------------------
	// Allocate sample result buffers
	//--------------------------------------------------------------------------

	InitSampleResultsBuffer();

	//--------------------------------------------------------------------------
	// Allocate volume info buffers if required
	//--------------------------------------------------------------------------

	intersectionDevice.AllocBufferRW(&eyePathInfosBuff, nullptr, sizeof(slg::ocl::EyePathInfo) * taskCount, "PathInfo");

	//--------------------------------------------------------------------------
	// Allocate volume info buffers if required
	//--------------------------------------------------------------------------

	intersectionDevice.AllocBufferRW(&directLightVolInfosBuff, nullptr, sizeof(slg::ocl::PathVolumeInfo) * taskCount, "DirectLightVolumeInfo");

	//--------------------------------------------------------------------------
	// Allocate GPU pixel filter distribution
	//--------------------------------------------------------------------------

	intersectionDevice.AllocBufferRO(&pixelFilterBuff, renderEngine->pixelFilterDistribution.data(),
			renderEngine->pixelFilterDistributionSize, "Pixel Filter Distribution");

	//--------------------------------------------------------------------------
	// Compile kernels
	//--------------------------------------------------------------------------

	InitKernels();

	//--------------------------------------------------------------------------
	// Initialize
	//--------------------------------------------------------------------------

	// Set kernel arguments
	SetKernelArgs();

	// Clear all thread films
	for(ThreadFilmRPtr threadFilm: threadFilms) {
		intersectionDevice.PushThreadCurrentDevice();
		threadFilm->ClearFilm(intersectionDevice, filmClearKernel, filmClearWorkGroupSize);
		intersectionDevice.PopThreadCurrentDevice();
	}

	intersectionDevice.FinishQueue();

	// Reset statistics in order to be more accurate
	intersectionDevice.ResetPerformaceStats();
}

#endif
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
