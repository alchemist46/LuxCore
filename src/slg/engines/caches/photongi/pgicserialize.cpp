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

#include <cstring>
#include <set>
#include <sstream>

#include "luxrays/utils/safesave.h"

#include "slg/engines/caches/photongi/photongicache.h"
#include "slg/scene/scene.h"

using namespace std;
using namespace luxrays;
using namespace slg;

//------------------------------------------------------------------------------
// PhotonGICache signature
//------------------------------------------------------------------------------

// FNV-1a, 8 bytes at a time
static void HashBytes(u_longlong &hash, const void *data, const size_t size) {
	const u_longlong prime = 1099511628211ull;

	const u_char *bytes = (const u_char *)data;
	size_t i = 0;
	for (; i + 8 <= size; i += 8) {
		u_longlong word;
		memcpy(&word, bytes + i, 8);
		hash = (hash ^ word) * prime;
	}
	for (; i < size; ++i)
		hash = (hash ^ bytes[i]) * prime;
}

static void HashString(u_longlong &hash, const string &s) {
	HashBytes(hash, s.data(), s.size());
	// A separator, so "ab" + "c" and "a" + "bc" differ
	hash = (hash ^ 0xffull) * 1099511628211ull;
}

static bool IsUUIDAt(const string &s, const size_t pos) {
	// "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx", as NamedObject::GetUniqueName()
	// writes it
	if (pos + 36 > s.size())
		return false;
	for (size_t i = 0; i < 36; ++i) {
		const char c = s[pos + i];
		if ((i == 8) || (i == 13) || (i == 18) || (i == 23)) {
			if (c != '-')
				return false;
		} else if (!(((c >= '0') && (c <= '9')) || ((c >= 'a') && (c <= 'f'))))
			return false;
	}

	return true;
}

// The implicit textures of a material (a metal2 preset, a constant value, etc.)
// are named with a random UUID (NamedObject::GetUniqueName()), different in
// every process: the UUIDs are replaced by a fixed token
static string StripUUIDs(const string &s) {
	string result;
	result.reserve(s.size());
	for (size_t i = 0; i < s.size();) {
		if (IsUUIDAt(s, i)) {
			result += "<uuid>";
			i += 36;
		} else
			result += s[i++];
	}

	return result;
}

// The parameters that change what the cache holds, as the configuration asks
// them (before Preprocess() fills in the radius etc.), and a hash of the scene
// without the camera: the scene properties, the meshes and the image maps. The
// camera is left out so a fly-through animation can reuse the cache.
string PhotonGICache::ComputeSignature() const {
	stringstream ss;
	ss.precision(9);
	ss << "sampler " << SamplerType2String(params.samplerType) <<
			" photon " << params.photon.maxTracedCount << " " << params.photon.maxPathDepth <<
			" " << params.photon.timeStart << " " << params.photon.timeEnd <<
			" visibility " << params.visibility.lookUpRadius << " " << params.visibility.lookUpNormalAngle <<
			" " << params.visibility.targetHitRate << " " << params.visibility.maxSampleCount <<
			" glossiness " << params.glossinessUsageThreshold;
	if (params.indirect.enabled)
		ss << " indirect " << params.indirect.maxSize << " " << params.indirect.haltThreshold <<
				" " << params.indirect.lookUpRadius << " " << params.indirect.lookUpNormalAngle <<
				" " << params.indirect.usageThresholdScale << " " << params.indirect.filterRadiusScale;
	if (params.caustic.enabled)
		ss << " caustic " << params.caustic.maxSize << " " << params.caustic.updateSpp <<
				" " << params.caustic.lookUpRadius << " " << params.caustic.lookUpNormalAngle <<
				" " << params.caustic.radiusReduction << " " << params.caustic.minLookUpRadius;

	u_longlong sceneHash = 14695981039346656037ull;
	try {
		const PropertiesUPtr sceneProps = scene->ToProperties(false);
		for (auto const &name : sceneProps->GetAllNames()) {
			if (name.starts_with("scene.camera."))
				continue;
			// The material, object and volume ids (of the ID AOVs) are random
			// when the scene doesn't set them (the light group ids,
			// "emission.id", are kept)
			if ((name.starts_with("scene.materials.") || name.starts_with("scene.objects.") ||
					name.starts_with("scene.volumes.")) &&
					name.ends_with(".id") && !name.ends_with(".emission.id"))
				continue;
			HashString(sceneHash, StripUUIDs(sceneProps->Get(name).ToString()));
		}
	} catch (const exception &e) {
		SLG_LOG("PhotonGI cache can not compute the scene signature: " << e.what());
		// An empty signature never matches a file
		return "";
	}

	// The meshes (the instances share theirs)
	set<const void *> meshesDone;
	const SceneObjectDefinitions &objDefs = scene->GetObjects();
	for (u_int i = 0; i < objDefs.GetSize(); ++i) {
		ExtMeshConstRef mesh = objDefs.GetSceneObject(i).GetExtMesh();

		const auto vertices = mesh.GetVertices();
		if (!meshesDone.insert(vertices.data()).second)
			continue;
		HashBytes(sceneHash, vertices.data(), vertices.size_bytes());
		const auto triangles = mesh.GetTriangles();
		HashBytes(sceneHash, triangles.data(), triangles.size_bytes());
	}

	// The image maps
	for (const ImageMap &imageMap : scene->GetImageMaps().GetImageMaps()) {
		const ImageMapStorage &storage = imageMap.GetStorage();
		HashBytes(sceneHash, storage.GetPixelsData(), storage.GetMemorySize());
	}

	ss << " scene " << std::hex << sceneHash;

	return ss.str();
}

//------------------------------------------------------------------------------
// PhotonGICache persistent file
//------------------------------------------------------------------------------

bool PhotonGICache::LoadPersistentCache(const std::string &fileName) {
	SLG_LOG("Loading persistent PhotonGI cache: " + fileName);

	try {
		SerializationInputFile sif(fileName);

		PhotonGICacheParams fileParams;
		sif.GetArchive() >> fileParams;

		if (!sif.IsGood())
			throw runtime_error("error while reading the parameters");

		// The files of older versions have no signature
		if (fileParams.signature.empty() || (fileParams.signature != params.signature)) {
			SLG_LOG("The persistent PhotonGI cache was computed for another scene or other parameters: computing it again");
			return false;
		}

		sif.GetArchive() >> visibilityParticles;
		sif.GetArchive() >> visibilityParticlesKdTree;

		sif.GetArchive() >> radiancePhotons;
		sif.GetArchive() >> radiancePhotonsBVH;
		sif.GetArchive() >> indirectPhotonTracedCount;

		sif.GetArchive() >> causticPhotons;
		sif.GetArchive() >> causticPhotonsBVH;
		sif.GetArchive() >> causticPhotonTracedCount;
		sif.GetArchive() >> causticPhotonPass;

		if (!sif.IsGood())
			throw runtime_error("error while reading the cache");

		// The file holds the parameters computed by the pre-processing (the
		// radius etc.); the debug mode and the persistent file settings are
		// the configuration's
		fileParams.debugType = params.debugType;
		fileParams.persistent = params.persistent;
		params = fileParams;
	} catch (const exception &e) {
		SLG_LOG("Error while loading PhotonGI persistent cache " << fileName << " (" << e.what() << "): computing it again");

		visibilityParticles.clear();
		delete visibilityParticlesKdTree;
		visibilityParticlesKdTree = nullptr;
		radiancePhotons.clear();
		delete radiancePhotonsBVH;
		radiancePhotonsBVH = nullptr;
		indirectPhotonTracedCount = 0;
		causticPhotons.clear();
		delete causticPhotonsBVH;
		causticPhotonsBVH = nullptr;
		causticPhotonTracedCount = 0;
		causticPhotonPass = 0;

		return false;
	}

	return true;
}

void PhotonGICache::SavePersistentCache(const std::string &fileName) {
	SLG_LOG("Saving persistent PhotonGI cache: " + fileName);

	SafeSave safeSave(fileName);
	{
		SerializationOutputFile sof(params.persistent.safeSave ? safeSave.GetSaveFileName() : fileName);

		sof.GetArchive() << params;

		sof.GetArchive() << visibilityParticles;
		sof.GetArchive() << visibilityParticlesKdTree;

		sof.GetArchive() << radiancePhotons;
		sof.GetArchive() << radiancePhotonsBVH;
		sof.GetArchive() << indirectPhotonTracedCount;

		sof.GetArchive() << causticPhotons;
		sof.GetArchive() << causticPhotonsBVH;
		sof.GetArchive() << causticPhotonTracedCount;
		sof.GetArchive() << causticPhotonPass;

		if (!sof.IsGood())
			throw runtime_error("Error while saving PhotonGI persistent cache: " + fileName);

		sof.Flush();

		SLG_LOG("PhotonGI persistent cache saved: " << (sof.GetPosition() / 1024) << " Kbytes");
	}
	// Now sof is closed and I can call safeSave.Process()
	
	if (params.persistent.safeSave)
		safeSave.Process();
}

//------------------------------------------------------------------------------
// PhotonGICache serialization
//------------------------------------------------------------------------------

BOOST_CLASS_EXPORT_IMPLEMENT(slg::GenericPhoton)
BOOST_CLASS_EXPORT_IMPLEMENT(slg::PGICVisibilityParticle)
BOOST_CLASS_EXPORT_IMPLEMENT(slg::Photon)
BOOST_CLASS_EXPORT_IMPLEMENT(slg::RadiancePhoton)
BOOST_CLASS_EXPORT_IMPLEMENT(slg::PhotonGICacheParams)
BOOST_CLASS_EXPORT_IMPLEMENT(slg::PhotonGICache)

template<class Archive>
void PhotonGICache::serialize(Archive &ar, const u_int version) {
	ar & params;
	ar & threadCount;
	ar & lastUpdateSpp;
	ar & updateSeedBase;

	ar & visibilityParticles;
	ar & visibilityParticlesKdTree;

	ar & radiancePhotons;
	ar & radiancePhotonsBVH;
	ar & indirectPhotonTracedCount;

	ar & causticPhotons;
	ar & causticPhotonsBVH;
	ar & causticPhotonTracedCount;
	ar & causticPhotonPass;

	if constexpr (Archive::is_loading::value) {
		// The render engine calls InitUpdate() again with its own thread
		// count: the resume may run another number of threads
		InitUpdate(threadCount);
		updateGeneration = 0;
	}
}

namespace slg {
// Explicit instantiations for portable archives
template void PhotonGICache::serialize(LuxOutputArchive &ar, const u_int version);
template void PhotonGICache::serialize(LuxInputArchive &ar, const u_int version);
template void PhotonGICache::serialize(LuxOutputArchiveText &ar, const u_int version);
template void PhotonGICache::serialize(LuxInputArchiveText &ar, const u_int version);
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
