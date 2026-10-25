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

#ifndef _SLG_IMAGEMAPCACHE_H
#define	_SLG_IMAGEMAPCACHE_H

#include <string>
#include <vector>
#include <ranges>

#include "luxrays/devices/ocldevice.h"
#include "slg/usings.h"
#include "slg/imagemap/imagemap.h"
#include "slg/imagemap/resizepolicies/resizepolicies.h"
#include "slg/core/sdl.h"

namespace slg {

//------------------------------------------------------------------------------
// ImageMapCache
//------------------------------------------------------------------------------

class ImageMapCache {
public:
	ImageMapCache();
	~ImageMapCache();

	void SetImageResizePolicy(ImageMapResizePolicy *policy);
	const ImageMapResizePolicy *GetImageResizePolicy() const { return resizePolicy; }

	// The image maps loaded by GetImageMap() are converted to this space. The
	// ones already in the cache (only when a scene is loaded, their pixels are
	// already converted) get it as the space of their luminance.
	void SetWorkingColorSpace(const WorkingColorSpace &wcs) {
		workingColorSpace = wcs;
		// Grey maps don't depend on it (and the random image map is shared
		// by all the scenes)
		for (auto &im : maps) {
			if (im->GetChannelCount() >= 3)
				im->SetWorkingColorSpace(wcs);
		}
	}

	// Prefered insertion will require a unique_ptr owned object
	// However, we also have to handle the case of ImageMapTexture::randomImageMap,
	// which is shared between ImageMapTexture and ImageMapCache
	ImageMapRef DefineImageMap(ImageMapUPtr&& im);
	ImageMapRef DefineImageMap(ImageMapSPtr im);

	ImageMapRef GetImageMap(const std::string &fileName, const ImageMapConfig &imgCfg,
			const bool applyResizePolicy);

	// Loads in parallel the image files that GetImageMap(file, config, true) is
	// going to be asked for (each file is decoded by one thread, the decoding is
	// most of the time of loading a scene with many textures). The image maps
	// are added to the cache in the order of the requests; GetImageMap() then
	// finds them. A file that can't be loaded is left to GetImageMap(), which
	// fails as it does without the preload. Only with the resize policy that
	// loads the file as it is.
	void PreloadImageMaps(const std::vector<std::pair<std::string, ImageMapConfig>> &requests);

	void DeleteImageMap(ImageMapConstRef im);

	std::string GetSequenceFileName(ImageMapConstRef im) const;
	u_int GetImageMapIndex(ImageMapConstRef im) const;
	u_int GetImageMapIndex(ImageMapConstPtr p) const;

	void GetImageMaps(std::vector<std::reference_wrapper<const ImageMap>> &ims) const;

	auto ViewImageMaps() const {
		// Create and return a view of references to the values
		return maps | std::views::transform([](const auto& ptr) -> const ImageMap&
				{ return *ptr; });
	}
	auto GetImageMaps() const {
		auto view = ViewImageMaps();
		return std::vector<std::reference_wrapper<const ImageMap>>(view.begin(), view.end());
	}


	u_int GetSize()const { return static_cast<u_int>(mapByKey.size()); }
	bool IsImageMapDefined(const std::string &name) const { return mapByKey.find(name) != mapByKey.end(); }


	friend class Scene;
	friend class ImageMapResizePolicy;
	friend class ImageMapResizeMinMemPolicy;
	friend class ImageMapResizeMipMapMemPolicy;
	friend class boost::serialization::access;

private:
	// Used for the support of resize policies
	void Preprocess(SceneConstRef scene, const bool useRTMode);

	// Add a new image to the container, at the last position, without checking
	// existing images. Warning: this is just a helper; for secure inserting,
	// use DefineImageMap
	ImageMapRef AppendImageMap(
		ImageMapSPtr im, const std::string& name, const std::string& key
	) {
		// Add the new image
		maps.push_back(im);
		auto& imRef = *maps.back();

		// Update index
		mapByKey.insert(make_pair(key, std::ref(imRef)));
		mapNames.push_back(name);

		return std::ref(imRef);
	}

	std::string GetCacheKey(const std::string &fileName,
				const ImageMapConfig &imgCfg) const;
	std::string GetCacheKey(const std::string &fileName) const;

	template<class Archive> void save(Archive &ar, const unsigned int version) const;
	template<class Archive>	void load(Archive &ar, const unsigned int version);
	BOOST_SERIALIZATION_SPLIT_MEMBER()

	// Here is the main owner of ImageMap objects
	// However, this is a shared_ptr container, as we have to handle the case
	// of ImageMapTexture::randomImageMap, which belongs also to ImageMapTexture
	// as a class static singleton
	std::vector<ImageMapSPtr> maps;

	std::unordered_map<std::string, std::reference_wrapper<ImageMap> > mapByKey;
	// Used to preserve insertion order and to retrieve insertion index
	std::vector<std::string> mapNames;

	ImageMapResizePolicy *resizePolicy;
	WorkingColorSpace workingColorSpace;
	std::vector<bool> resizePolicyToApply;
};


}

BOOST_CLASS_VERSION(slg::ImageMapCache, 2)

BOOST_CLASS_EXPORT_KEY(slg::ImageMapCache)

#endif	/* _SLG_IMAGEMAPCACHE_H */
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
