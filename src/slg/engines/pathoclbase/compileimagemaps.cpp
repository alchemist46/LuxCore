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

#include <iosfwd>
#include <limits>

#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>

#include "oneapi/tbb.h"

#include "slg/engines/pathoclbase/compiledscene.h"
#include "slg/kernels/kernels.h"

#include "slg/imagemap/imagemap.h"
#include "slg/textures/imagemaptex.h"

using namespace std;
using namespace luxrays;
using namespace slg;

void CompiledScene::AddToImageMapMem(slg::ocl::ImageMap &im, const size_t dataSize, vector<size_t> &pageSizes) {
	const size_t memSize = RoundUp(dataSize, sizeof(float));

	if (memSize > maxMemPageSize)
		throw runtime_error("An image data block is too big to fit in a single block of memory");

	// pageSizes are in floats
	bool found = false;
	u_int page;
	for (u_int j = 0; j < pageSizes.size(); ++j) {
		// Check if it fits in this page
		if (memSize + pageSizes[j] * sizeof(float) <= maxMemPageSize) {
			found = true;
			page = j;
			break;
		}
	}

	if (!found) {
		// Check if I can add a new page
		if (pageSizes.size() > 8)
			throw runtime_error("More than 8 blocks of memory are required for image maps");

		// Add a new page
		pageSizes.push_back(0);
		page = pageSizes.size() - 1;
	}

	const size_t start = pageSizes[page];
	pageSizes[page] += memSize / sizeof(float);

	im.pageIndex = page;
	im.pixelsIndex = start;
	// Set by the HIP render threads (see PathOCLBaseOCLRenderThread::InitImageMaps())
	im.textureObject = 0;
	im.texelTextureObject = 0;
}

u_int CompiledScene::CompileImageMap(ImageMapConstRef im, vector<size_t> &pageSizes) {
	const u_int imgMapIndex = imageMapDescs.size();

	imageMapDescs.resize(imageMapDescs.size() + 1);
	slg::ocl::ImageMap *imd = &imageMapDescs[imgMapIndex];

	imd->channelCount = im.GetChannelCount();
	imd->width = im.GetWidth();
	imd->height = im.GetHeight();

	switch (im.GetStorage().GetWrapType()) {
		case ImageMapStorage::REPEAT:
			imd->wrapType = slg::ocl::WRAP_REPEAT;
			break;
		case ImageMapStorage::BLACK:
			imd->wrapType = slg::ocl::WRAP_BLACK;
			break;
		case ImageMapStorage::WHITE:
			imd->wrapType = slg::ocl::WRAP_WHITE;
			break;
		case ImageMapStorage::CLAMP:
			imd->wrapType = slg::ocl::WRAP_CLAMP;
			break;
		default:
			throw runtime_error("Unknown wrap type in CompiledScene::CompileImageMap(): " +
					ToString(im.GetStorage().GetWrapType()));
	}

	switch (im.GetStorage().GetFilterType()) {
		case ImageMapStorage::NEAREST:
			imd->filterType = slg::ocl::FILTER_NEAREST;
			break;
		case ImageMapStorage::LINEAR:
			imd->filterType = slg::ocl::FILTER_LINEAR;
			break;
		default:
			throw runtime_error("Unknown filter type in CompiledScene::CompileImageMap(): " +
					ToString(im.GetStorage().GetFilterType()));
	}

	switch (im.GetStorage().GetStorageType()) {
		case ImageMapStorage::BYTE:
			imd->storageType = slg::ocl::BYTE;
			break;
		case ImageMapStorage::HALF:
			imd->storageType = slg::ocl::HALF;
			break;
		case ImageMapStorage::FLOAT:
			imd->storageType = slg::ocl::FLOAT;
			break;
		default:
			throw runtime_error("Unknown storage type in CompiledScene::CompileImageMap(): " +
					ToString(im.GetStorage().GetStorageType()));
	}

	AddToImageMapMem(*imd, im.GetStorage().GetMemorySize(), pageSizes);

	return imgMapIndex;
}

void CompiledScene::CompileImageMaps() {
	SLG_LOG("Compile ImageMaps");
	wasImageMapsCompiled = true;

	imageMapDescs.resize(0);
	imageMapMemBlocks.resize(0);

	//--------------------------------------------------------------------------
	// Translate image maps
	//--------------------------------------------------------------------------

	const double tStart = WallClockTime();

	vector<std::reference_wrapper<const ImageMap> > ims;
	scene.GetImageMaps().GetImageMaps(ims);

	vector<size_t> pageSizes;
	for (u_int i = 0; i < ims.size(); ++i)
		CompileImageMap(ims[i], pageSizes);

	// Each page is allocated once (the rounding of an image map's size to a
	// float stays zero) and the pixels are copied in parallel: hundreds of MB
	// in a scene with large textures
	imageMapMemBlocks.resize(pageSizes.size());
	for (u_int i = 0; i < pageSizes.size(); ++i)
		imageMapMemBlocks[i].resize(pageSizes[i]);
	tbb::parallel_for(tbb::blocked_range<size_t>(0, ims.size(), 1),
		[&](const tbb::blocked_range<size_t> &r) {
			for (size_t i = r.begin(); i < r.end(); ++i) {
				const ImageMapStorage &storage = ims[i].get().GetStorage();
				const slg::ocl::ImageMap &imd = imageMapDescs[i];
				memcpy(&imageMapMemBlocks[imd.pageIndex][imd.pixelsIndex],
						storage.GetPixelsData(), storage.GetMemorySize());
			}
		});

	SLG_LOG("Image maps page(s) count: " << imageMapMemBlocks.size());
	for (u_int i = 0; i < imageMapMemBlocks.size(); ++i)
		SLG_LOG(" RGB channel page " << i << " size: " << imageMapMemBlocks[i].size() * sizeof(float) / 1024 << "Kbytes");

	const double tEnd = WallClockTime();
	SLG_LOG("Image maps compilation time: " << int((tEnd - tStart) * 1000.0) << "ms");
}

#endif
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
