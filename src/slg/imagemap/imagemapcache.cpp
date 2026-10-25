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

#include <boost/lexical_cast.hpp>
#include <atomic>
#include <set>
#include <thread>

#include <boost/format.hpp>

#include "luxrays/utils/thread.h"
#include "slg/core/sdl.h"
#include "slg/imagemap/imagemapcache.h"
#include "slg/textures/imagemaptex.h"
#include "slg/usings.h"

using namespace std;
using namespace luxrays;
using namespace slg;

//------------------------------------------------------------------------------
// ImageMapCache
//------------------------------------------------------------------------------

ImageMapCache::ImageMapCache() {
	resizePolicy = new ImageMapResizeNonePolicy();
}

ImageMapCache::~ImageMapCache() {
	//for(ImageMapPtr m: maps) {
		//// I avoid to free the static global ImageMapTexture::randomImageMap
		//if (m != ImageMapTexture::randomImageMap)
			//delete m;
	//}

	delete resizePolicy;
}

void ImageMapCache::SetImageResizePolicy(ImageMapResizePolicy *policy) {
	delete resizePolicy;
	resizePolicy = policy;
}

string ImageMapCache::GetCacheKey(const string &fileName, const ImageMapConfig &imgCfg) const {
	string key = fileName + "_#_";

	switch (imgCfg.colorSpaceCfg.colorSpaceType) {
		case ColorSpaceConfig::NOP_COLORSPACE:
			key += "CS_NOP_#_";
			break;
		case ColorSpaceConfig::LUXCORE_COLORSPACE:
			key += "CS_LUXCORE_#_" + ToString(imgCfg.colorSpaceCfg.luxcore.gamma) + "_#_";
			break;
		case ColorSpaceConfig::OPENCOLORIO_COLORSPACE:
			key += "CS_OPENCOLORIO_#_" +
					ToString(imgCfg.colorSpaceCfg.ocio.configName) + "_#_" +
					ToString(imgCfg.colorSpaceCfg.ocio.colorSpaceName) + "_#_";
			break;
		default:
			throw runtime_error("Unknown color space type in ImageMapCache::GetCacheKey(): " + 
					ToString(imgCfg.colorSpaceCfg.colorSpaceType));
	}
	
	key += ToString(imgCfg.GetStorageType()) + "_#_" +
			ToString(imgCfg.GetWrapType()) + "_#_" +
			ToString(imgCfg.GetSelectionType());

	return key;
}

string ImageMapCache::GetCacheKey(const string &fileName) const {
	return fileName;
}

ImageMapRef ImageMapCache::GetImageMap(
	const string &fileName, const ImageMapConfig &cfg, const bool applyResizePolicy
) {
	ImageMapConfig imgCfg = cfg;
	imgCfg.workingColorSpace = workingColorSpace;

	// Compose the cache key
	string key = GetCacheKey(fileName);

	// Check if the image map has been already defined
	auto it = mapByKey.find(key);

	if (it != mapByKey.end()) {
		//SDL_LOG("Cached defined image map: " << fileName);
		auto im = (it->second);
		return im.get();
	}

	// Check if it is a reference to a file
	key = GetCacheKey(fileName, imgCfg);
	it = mapByKey.find(key);

	if (it != mapByKey.end()) {
		//SDL_LOG("Cached file image map: " << fileName);
		auto im = (it->second);
		return im.get();
	}

	// I haven't yet loaded the file

	ImageMapUPtr im;
	auto preloaded = applyResizePolicy ? preloadedImageMaps.end() : preloadedImageMaps.find(key);
	if (preloaded != preloadedImageMaps.end()) {
		// Decoded by PreloadImageMaps()
		im = std::move(preloaded->second);
		preloadedImageMaps.erase(preloaded);

		resizePolicyToApply.push_back(false);
	} else if (applyResizePolicy) {
		// Scale the image if required
		bool toApply;
		im = resizePolicy->ApplyResizePolicy(fileName, imgCfg, toApply);

		resizePolicyToApply.push_back(toApply);
	} else {
		im = std::make_unique<ImageMap>(fileName, imgCfg);

		resizePolicyToApply.push_back(false);
	}

	auto& imRef = AppendImageMap(ImageMapSPtr(std::move(im)), fileName, key);

	return std::ref(imRef);
}

void ImageMapCache::PreloadImageMaps(const vector<pair<string, ImageMapConfig>> &requests,
		const vector<pair<string, ImageMapConfig>> &laterRequests) {
	preloadedImageMaps.clear();

	struct Job {
		string fileName;
		ImageMapConfig cfg;
		string key;
		bool later;
		ImageMapUPtr im;
	};
	vector<Job> jobs;
	set<string> jobKeys;
	auto addJobs = [&](const vector<pair<string, ImageMapConfig>> &reqs, const bool later) {
		for (const auto &request : reqs) {
			ImageMapConfig imgCfg = request.second;
			imgCfg.workingColorSpace = workingColorSpace;

			// The checks of GetImageMap(): a defined image map or a file loaded
			// with the same configuration
			if (mapByKey.find(GetCacheKey(request.first)) != mapByKey.end())
				continue;
			const string key = GetCacheKey(request.first, imgCfg);
			if ((mapByKey.find(key) != mapByKey.end()) || !jobKeys.insert(key).second)
				continue;

			jobs.push_back(Job{request.first, imgCfg, key, later, nullptr});
		}
	};
	// The other policies scale the image or write a TX file: they load the
	// image maps of the textures one at a time. The later requests are read
	// without the resize policy
	if (dynamic_cast<const ImageMapResizeNonePolicy *>(resizePolicy))
		addJobs(requests, false);
	addJobs(laterRequests, true);
	if (jobs.size() < 2)
		return;

	const double startTime = WallClockTime();

	atomic<size_t> nextJob(0);
	auto loader = [&jobs, &nextJob]() {
		for (;;) {
			const size_t i = nextJob++;
			if (i >= jobs.size())
				return;

			try {
				jobs[i].im = std::make_unique<ImageMap>(jobs[i].fileName, jobs[i].cfg);
			} catch (...) {
				// GetImageMap() loads the file again and reports the error
			}
		}
	};

	const size_t threadCount = Min<size_t>(jobs.size(), Max<size_t>(1, GetHardwareThreadCount()));
	vector<thread> threads;
	for (size_t i = 1; i < threadCount; ++i)
		threads.emplace_back(loader);
	loader();
	for (auto &t : threads)
		t.join();

	// In the order of the requests; the later ones wait for GetImageMap()
	for (auto &job : jobs) {
		if (!job.im)
			continue;

		if (job.later)
			preloadedImageMaps[job.key] = std::move(job.im);
		else {
			resizePolicyToApply.push_back(false);
			AppendImageMap(ImageMapSPtr(std::move(job.im)), job.fileName, job.key);
		}
	}

	SDL_LOG("Preloaded " << jobs.size() << " image maps with " << threadCount <<
			" threads in " << int((WallClockTime() - startTime) * 1000.0) << "ms");
}

ImageMapRef ImageMapCache::DefineImageMap(ImageMapUPtr&& im) {
	// Move im into a shared_ptr
	auto ims = ImageMapSPtr(std::move(im));

	return DefineImageMap(ims);
}

ImageMapRef ImageMapCache::DefineImageMap(ImageMapSPtr im) {
	const string name = im->GetName();

	SDL_LOG("Define ImageMap: " << name);

	// Compose the cache key
	const string key = GetCacheKey(name);

	auto it = mapByKey.find(key);

	if (it == mapByKey.end()) {
		// New value: append to container
		auto& imRef = AppendImageMap(im, name, key);
		return std::ref(imRef);
	}

	// Existing value: overwrite the existing image definition
	const u_int index = GetImageMapIndex(it->second.get());
	std::swap(maps[index], im);

	resizePolicyToApply[index] = false;

	// I have to modify mapByName for last or it iterator would be modified
	// otherwise (it->second would point to the new ImageMap and not to the old one)
	mapByKey.erase(key);
	mapByKey.insert(make_pair(key, std::ref(*maps[index])));

	return std::ref(*maps[index]);
}

void ImageMapCache::DeleteImageMap(ImageMapConstRef im) {
	for (auto it = mapByKey.begin(); it != mapByKey.end(); ++it) {
		if (&it->second.get() == &im) {
			mapByKey.erase(it);

			for (u_int i = 0; i < maps.size(); ++i) {
				if (&*maps[i] == &im) {
					mapNames.erase(mapNames.begin() + i);
					maps.erase(maps.begin() + i);
					resizePolicyToApply.erase(resizePolicyToApply.begin() + i);
					break;
				}
			}

			return;
		}
	}
}

string ImageMapCache::GetSequenceFileName(ImageMapConstRef im) const {
	return ("imagemap-" + ((boost::format("%05d") % GetImageMapIndex(im)).str()) +
			"." + im.GetFileExtension());
}

u_int ImageMapCache::GetImageMapIndex(ImageMapConstRef im) const {
	for (u_int i = 0; i < maps.size(); ++i) {
		if (&*maps[i] == &im)
			return i;
	}

	throw runtime_error("Unknown image map: " + ToString(&im));
}
u_int ImageMapCache::GetImageMapIndex(ImageMapConstPtr p) const {
	return GetImageMapIndex(*p);
}

void ImageMapCache::GetImageMaps(std::vector<std::reference_wrapper<const ImageMap>> &ims) const {
	ims.reserve(maps.size());

	for(auto& im: maps)
		ims.push_back(std::ref(*im));
}

void ImageMapCache::Preprocess(SceneConstRef scene, const bool useRTMode) {
	resizePolicy->Preprocess(*this, scene, useRTMode);
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
