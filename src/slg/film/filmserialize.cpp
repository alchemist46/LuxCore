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

#include <algorithm>
#include <memory>

#include <boost/lexical_cast.hpp>
#include <boost/serialization/unordered_set.hpp>
#include <boost/serialization/shared_ptr.hpp>

#include "slg/film/film.h"
#include "slg/film/imagepipeline/imagepipeline.h"

using namespace std;
using namespace luxrays;
using namespace slg;

//------------------------------------------------------------------------------
// Film serialization
//------------------------------------------------------------------------------

BOOST_CLASS_EXPORT_IMPLEMENT(slg::Film)

void Film::SaveSerialized(const string &fileName, FilmRef  film) {
	SerializationOutputFile sof(fileName);

	// Serialized as a pointer: it is the format used by LoadSerialized() (and
	// by the files of the previous versions)
	const Film *filmPtr = &film;
	sof.GetArchive() << filmPtr;

	if (!sof.IsGood())
		throw runtime_error("Error while saving serialized film: " + fileName);

	sof.Flush();
	SLG_LOG("Film saved: " << (sof.GetPosition() / 1024) << " Kbytes");
}

FilmUPtr Film::LoadSerialized(const string &fileName) {
	SerializationInputFile sif(fileName);

	Film *film = nullptr;
	sif.GetArchive() >> film;
	FilmUPtr result(film);

	if (!sif.IsGood())
		throw runtime_error("Error while loading serialized film: " + fileName);

	return result;
}

template<class Archive> void Film::load(Archive &ar, const u_int version) {
	ar & channel_RADIANCE_PER_PIXEL_NORMALIZEDs;
	ar & channel_RADIANCE_PER_SCREEN_NORMALIZEDs;
	ar & channel_ALPHA;
	ar & channel_IMAGEPIPELINEs;
	ar & channel_DEPTH;
	ar & channel_POSITION;
	ar & channel_GEOMETRY_NORMAL;
	ar & channel_SHADING_NORMAL;
	ar & channel_AVG_SHADING_NORMAL;
	ar & channel_MATERIAL_ID;
	ar & channel_DIRECT_DIFFUSE;
	ar & channel_DIRECT_DIFFUSE_REFLECT;
	ar & channel_DIRECT_DIFFUSE_TRANSMIT;
	ar & channel_DIRECT_GLOSSY;
	ar & channel_DIRECT_GLOSSY_REFLECT;
	ar & channel_DIRECT_GLOSSY_TRANSMIT;
	ar & channel_EMISSION;
	ar & channel_INDIRECT_DIFFUSE;
	ar & channel_INDIRECT_DIFFUSE_REFLECT;
	ar & channel_INDIRECT_DIFFUSE_TRANSMIT;
	ar & channel_INDIRECT_GLOSSY;
	ar & channel_INDIRECT_GLOSSY_REFLECT;
	ar & channel_INDIRECT_GLOSSY_TRANSMIT;
	ar & channel_INDIRECT_SPECULAR;
	ar & channel_INDIRECT_SPECULAR_REFLECT;
	ar & channel_INDIRECT_SPECULAR_TRANSMIT;
	ar & channel_MATERIAL_ID_MASKs;
	ar & channel_DIRECT_SHADOW_MASK;
	ar & channel_INDIRECT_SHADOW_MASK;
	ar & channel_UV;
	ar & channel_RAYCOUNT;
	ar & channel_BY_MATERIAL_IDs;
	ar & channel_IRRADIANCE;
	ar & channel_OBJECT_ID;
	ar & channel_OBJECT_ID_MASKs;
	ar & channel_BY_OBJECT_IDs;
	ar & channel_SAMPLECOUNT;
	ar & channel_CONVERGENCE;
	ar & channel_MATERIAL_ID_COLOR;
	ar & channel_ALBEDO;
	ar & channel_NOISE;
	ar & channel_USER_IMPORTANCE;
	if (version >= 29) {
		ar & channel_RADIANCE_PER_PIXEL_NORMALIZED_HALF;
		ar & channel_RADIANCE_PER_SCREEN_NORMALIZED_HALF;
	}

	ar & channels;
	ar & width;
	ar & height;
	ar & subRegion[0];
	ar & subRegion[1];
	ar & subRegion[2];
	ar & subRegion[3];
	ar & pixelCount;
	ar & radianceGroupCount;
	ar & maskMaterialIDs;
	ar & byMaterialIDs;

	ar & statsStartSampleTime;
	ar & statsConvergence;
	ar & statsStartSampleTime;
	ar & samplesCounts;

	ar & imagePipelines;

	ar & convTest;
	ar & noiseEstimation;
	ar & haltTime;
	ar & haltSPP;
	ar & haltSPP_PixelNormalized;
	ar & haltSPP_ScreenNormalized;
	ar & haltNoiseThreshold;
	ar & haltNoiseThresholdWarmUp;
	ar & haltNoiseThresholdTestStep;
	ar & haltNoiseThresholdUseFilter;
	ar & haltNoiseThresholdStopRendering;
	if (version >= 28) {
		ar & haltNoiseThresholdTolerance;
		ar & haltNoiseThresholdPasses;
	} else {
		haltNoiseThresholdTolerance = 0.f;
		haltNoiseThresholdPasses = 0;
	}
	if (version >= 29)
		ar & haltNoiseThresholdHalfBuffer;
	else
		haltNoiseThresholdHalfBuffer = false;

	ar & noiseEstimationWarmUp;
	ar & noiseEstimationTestStep;
	ar & noiseEstimationFilterScale;

	ar & filmOutputs;

	ar & initialized;

	if (version >= 30) {
		int workingColorSpaceType;
		ar & workingColorSpaceType;
		workingColorSpace = WorkingColorSpace((WorkingColorSpace::WorkingColorSpaceType)workingColorSpaceType);
	} else
		workingColorSpace = WorkingColorSpace();

	if (version >= 31) {
		ar & maskObjectIDs;
		ar & byObjectIDs;
	} else {
		// Older files have no object ID lists: rebuild them from the outputs,
		// in the order AddChannel() registered them
		maskObjectIDs.clear();
		byObjectIDs.clear();
		for (u_int i = 0; i < filmOutputs.GetCount(); ++i) {
			const FilmOutputs::FilmOutputType type = filmOutputs.GetType(i);
			if ((type != FilmOutputs::OBJECT_ID_MASK) && (type != FilmOutputs::BY_OBJECT_ID))
				continue;

			const u_int id = filmOutputs.GetProperties(i)->Get(Property("id")(255)).Get<u_int>();
			vector<u_int> &ids = (type == FilmOutputs::OBJECT_ID_MASK) ? maskObjectIDs : byObjectIDs;
			if (count(ids.begin(), ids.end(), id) == 0)
				ids.push_back(id);
		}

		// Never more IDs than buffers (Output() indexes the buffers with them)
		if (maskObjectIDs.size() > channel_OBJECT_ID_MASKs.size())
			maskObjectIDs.resize(channel_OBJECT_ID_MASKs.size());
		if (byObjectIDs.size() > channel_BY_OBJECT_IDs.size())
			byObjectIDs.resize(channel_BY_OBJECT_IDs.size());
	}

	SetUpHW();
}

template<class Archive> void Film::save(Archive &ar, const u_int version) const {
	// I can not really serialize the film while a pipeline is running
	if (isAsyncImagePipelineRunning)
		throw runtime_error("It is not possible to serialize a Film while an AsyncExecuteImagePipeline() is still running");
	
	ar & channel_RADIANCE_PER_PIXEL_NORMALIZEDs;
	ar & channel_RADIANCE_PER_SCREEN_NORMALIZEDs;
	ar & channel_ALPHA;
	ar & channel_IMAGEPIPELINEs;
	ar & channel_DEPTH;
	ar & channel_POSITION;
	ar & channel_GEOMETRY_NORMAL;
	ar & channel_SHADING_NORMAL;
	ar & channel_AVG_SHADING_NORMAL;
	ar & channel_MATERIAL_ID;
	ar & channel_DIRECT_DIFFUSE;
	ar & channel_DIRECT_DIFFUSE_REFLECT;
	ar & channel_DIRECT_DIFFUSE_TRANSMIT;
	ar & channel_DIRECT_GLOSSY;
	ar & channel_DIRECT_GLOSSY_REFLECT;
	ar & channel_DIRECT_GLOSSY_TRANSMIT;
	ar & channel_EMISSION;
	ar & channel_INDIRECT_DIFFUSE;
	ar & channel_INDIRECT_DIFFUSE_REFLECT;
	ar & channel_INDIRECT_DIFFUSE_TRANSMIT;
	ar & channel_INDIRECT_GLOSSY;
	ar & channel_INDIRECT_GLOSSY_REFLECT;
	ar & channel_INDIRECT_GLOSSY_TRANSMIT;
	ar & channel_INDIRECT_SPECULAR;
	ar & channel_INDIRECT_SPECULAR_REFLECT;
	ar & channel_INDIRECT_SPECULAR_TRANSMIT;
	ar & channel_MATERIAL_ID_MASKs;
	ar & channel_DIRECT_SHADOW_MASK;
	ar & channel_INDIRECT_SHADOW_MASK;
	ar & channel_UV;
	ar & channel_RAYCOUNT;
	ar & channel_BY_MATERIAL_IDs;
	ar & channel_IRRADIANCE;
	ar & channel_OBJECT_ID;
	ar & channel_OBJECT_ID_MASKs;
	ar & channel_BY_OBJECT_IDs;
	ar & channel_SAMPLECOUNT;
	ar & channel_CONVERGENCE;
	ar & channel_MATERIAL_ID_COLOR;
	ar & channel_ALBEDO;
	ar & channel_NOISE;
	ar & channel_USER_IMPORTANCE;
	ar & channel_RADIANCE_PER_PIXEL_NORMALIZED_HALF;
	ar & channel_RADIANCE_PER_SCREEN_NORMALIZED_HALF;

	ar & channels;
	ar & width;
	ar & height;
	ar & subRegion[0];
	ar & subRegion[1];
	ar & subRegion[2];
	ar & subRegion[3];
	ar & pixelCount;
	ar & radianceGroupCount;
	ar & maskMaterialIDs;
	ar & byMaterialIDs;

	ar & statsStartSampleTime;
	ar & statsConvergence;
	ar & statsStartSampleTime;
	ar & samplesCounts;

	ar & imagePipelines;

	ar & convTest;
	ar & noiseEstimation;
	ar & haltTime;
	ar & haltSPP;
	ar & haltSPP_PixelNormalized;
	ar & haltSPP_ScreenNormalized;
	ar & haltNoiseThreshold;
	ar & haltNoiseThresholdWarmUp;
	ar & haltNoiseThresholdTestStep;
	ar & haltNoiseThresholdUseFilter;
	ar & haltNoiseThresholdStopRendering;
	ar & haltNoiseThresholdTolerance;
	ar & haltNoiseThresholdPasses;
	ar & haltNoiseThresholdHalfBuffer;

	ar & noiseEstimationWarmUp;
	ar & noiseEstimationTestStep;
	ar & noiseEstimationFilterScale;

	ar & filmOutputs;

	ar & initialized;

	const int workingColorSpaceType = workingColorSpace.GetType();
	ar & workingColorSpaceType;

	ar & maskObjectIDs;
	ar & byObjectIDs;
}

namespace slg {
// Explicit instantiations for portable archives
template void Film::save(LuxOutputArchive &ar, const u_int version) const;
template void Film::load(LuxInputArchive &ar, const u_int version);
template void Film::save(LuxOutputArchiveText &ar, const u_int version) const;
template void Film::load(LuxInputArchiveText &ar, const u_int version);
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
