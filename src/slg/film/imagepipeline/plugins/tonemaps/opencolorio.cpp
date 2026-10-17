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

#include <mutex>

#include <OpenColorIO/OpenColorIO.h>
namespace OCIO = OCIO_NAMESPACE;

#include "luxrays/utils/serializationutils.h"
#include "slg/kernels/kernels.h"
#include "slg/film/film.h"
#include "slg/film/imagepipeline/plugins/tonemaps/opencolorio.h"
#include "slg/utils/filenameresolver.h"

using namespace std;
using namespace luxrays;
using namespace slg;

//------------------------------------------------------------------------------
// OpenColorIO tone mapping
//------------------------------------------------------------------------------

BOOST_CLASS_EXPORT_IMPLEMENT(slg::OpenColorIOToneMap)

struct OpenColorIOToneMap::ProcessorCache {
	// A nested class member: it can read the plugin's private settings
	static OCIO::ConstCPUProcessorRcPtr Create(const OpenColorIOToneMap &tm);

	std::mutex mutex;
	OCIO::ConstCPUProcessorRcPtr cpu;
};

OpenColorIOToneMap::OpenColorIOToneMap() {
}

OpenColorIOToneMap::~OpenColorIOToneMap() {
}

ToneMap *OpenColorIOToneMap::Copy() const {
	OpenColorIOToneMap *ociotm = new OpenColorIOToneMap();

	ociotm->conversionType = conversionType;

	ociotm->configFileName = configFileName;

	// COLORSPACE_CONVERSION
	ociotm->inputColorSpace = inputColorSpace;
	ociotm->outputColorSpace = outputColorSpace;

	// LUT_CONVERSION
	ociotm->lutFileName = lutFileName;

	// DISPLAY_CONVERSION
	ociotm->displayConversion = displayConversion;

	// LOOK_CONVERSION
	ociotm->lookInputColorSpace = lookInputColorSpace;
	ociotm->lookName = lookName;

	return ociotm;
}

OpenColorIOToneMap *OpenColorIOToneMap::CreateColorSpaceConversion(const string &configFileName,
		const string &inputColorSpace, const string &outputColorSpace) {
	OpenColorIOToneMap *ociotm = new OpenColorIOToneMap();

	ociotm->conversionType = COLORSPACE_CONVERSION;

	ociotm->configFileName = configFileName;
	ociotm->inputColorSpace = inputColorSpace;
	ociotm->outputColorSpace = outputColorSpace;

	return ociotm;
}

OpenColorIOToneMap *OpenColorIOToneMap::CreateLUTConversion(const string &lutFileName) {
	OpenColorIOToneMap *ociotm = new OpenColorIOToneMap();

	ociotm->conversionType = LUT_CONVERSION;

	ociotm->lutFileName = lutFileName;

	return ociotm;
}

OpenColorIOToneMap *OpenColorIOToneMap::CreateDisplayConversion(const string &configFileName,
		const string &inputColorSpace, const string &displayName,
		const string &viewName, const string &lookName) {
	OpenColorIOToneMap *ociotm = new OpenColorIOToneMap();

	ociotm->conversionType = DISPLAY_CONVERSION;

	ociotm->configFileName = configFileName;
	ociotm->displayConversion.inputColorSpace = inputColorSpace;
	ociotm->displayConversion.displayName = displayName;
	ociotm->displayConversion.viewName = viewName;
	ociotm->displayConversion.lookName = lookName;

	return ociotm;
}

OpenColorIOToneMap *OpenColorIOToneMap::CreateLookConversion(const string &configFileName,
		const string &lookInputColorSpace, const string &lookName) {
	OpenColorIOToneMap *ociotm = new OpenColorIOToneMap();

	ociotm->conversionType = LOOK_CONVERSION;

	ociotm->configFileName = configFileName;
	ociotm->lookInputColorSpace = lookInputColorSpace;
	ociotm->lookName = lookName;

	return ociotm;
}

//------------------------------------------------------------------------------
// CPU version
//------------------------------------------------------------------------------

// The current config ($OCIO) when the name is empty, the file found by the
// file name resolver otherwise
static OCIO::ConstConfigRcPtr LoadConfig(const string &configFileName) {
	return (configFileName == "") ?
		OCIO::GetCurrentConfig() :
		OCIO::Config::CreateFromFile(SLG_FileNameResolver.ResolveFile(configFileName).c_str());
}

static OCIO::ConstCPUProcessorRcPtr CreateProcessor(const OCIO::ConstConfigRcPtr &config,
		const OCIO::ConstTransformRcPtr &transform) {
	return config->getProcessor(transform)->getDefaultCPUProcessor();
}

OCIO::ConstCPUProcessorRcPtr OpenColorIOToneMap::ProcessorCache::Create(const OpenColorIOToneMap &tm) {
	switch (tm.conversionType) {
		case COLORSPACE_CONVERSION: {
			OCIO::ConstConfigRcPtr config = LoadConfig(tm.configFileName);

			return config->getProcessor(tm.inputColorSpace.c_str(), tm.outputColorSpace.c_str())->
					getDefaultCPUProcessor();
		}
		case LUT_CONVERSION: {
			OCIO::ConstConfigRcPtr config = OCIO::Config::CreateRaw();

			OCIO::FileTransformRcPtr transform = OCIO::FileTransform::Create();
			transform->setSrc(SLG_FileNameResolver.ResolveFile(tm.lutFileName).c_str());
			transform->setInterpolation(OCIO::INTERP_BEST);

			return CreateProcessor(config, transform);
		}
		case DISPLAY_CONVERSION: {
			OCIO::ConstConfigRcPtr config = LoadConfig(tm.configFileName);

			OCIO::GroupTransformRcPtr group = OCIO::GroupTransform::Create();

			const char *currentInputColorSpace = tm.displayConversion.inputColorSpace.c_str();
			if (tm.displayConversion.lookName != "") {
				const char *lookOutputColorSpace = OCIO::LookTransform::GetLooksResultColorSpace(config,
					config->getCurrentContext(), tm.displayConversion.lookName.c_str());

				if (lookOutputColorSpace && lookOutputColorSpace[0] != 0) {
					OCIO::LookTransformRcPtr transform = OCIO::LookTransform::Create();
					transform->setSrc(currentInputColorSpace);
					transform->setDst(lookOutputColorSpace);
					transform->setLooks(tm.displayConversion.lookName.c_str());
					group->appendTransform(transform);

					currentInputColorSpace = lookOutputColorSpace;
				}
			}

			// An empty display or view means the default one of the config
			const string displayName = (tm.displayConversion.displayName == "") ?
				string(config->getDefaultDisplay()) : tm.displayConversion.displayName;
			const string viewName = (tm.displayConversion.viewName == "") ?
				string(config->getDefaultView(displayName.c_str())) : tm.displayConversion.viewName;

			OCIO::DisplayViewTransformRcPtr transform = OCIO::DisplayViewTransform::Create();
			transform->setSrc(currentInputColorSpace);
			transform->setDisplay(displayName.c_str());
			transform->setView(viewName.c_str());
			transform->setLooksBypass(tm.displayConversion.lookName != "");
			group->appendTransform(transform);

			return CreateProcessor(config, group);
		}
		case LOOK_CONVERSION: {
			OCIO::ConstConfigRcPtr config = LoadConfig(tm.configFileName);

			const char *lookOutputColorSpace = OCIO::LookTransform::GetLooksResultColorSpace(config,
					config->getCurrentContext(), tm.lookName.c_str());
			if (!lookOutputColorSpace || (lookOutputColorSpace[0] == 0))
				throw runtime_error("Unknown look destination color space in OpenColorIOToneMap::Apply()");

			OCIO::LookTransformRcPtr transform = OCIO::LookTransform::Create();
			transform->setSrc(tm.lookInputColorSpace.c_str());
			transform->setDst(lookOutputColorSpace);
			transform->setLooks(tm.lookName.c_str());

			return CreateProcessor(config, transform);
		}
		default:
			throw runtime_error("Unknown mode in OpenColorIOToneMap::Apply(): " + ToString(tm.conversionType));
	}
}

void OpenColorIOToneMap::Apply(Film &film, const u_int index) {
	Spectrum *pixels = (Spectrum *)film.channel_IMAGEPIPELINEs[index]->GetPixels();

	try {
		if (!processorCache)
			processorCache = make_unique<ProcessorCache>();

		OCIO::ConstCPUProcessorRcPtr cpu;
		{
			std::lock_guard<std::mutex> lock(processorCache->mutex);
			if (!processorCache->cpu)
				processorCache->cpu = ProcessorCache::Create(*this);
			cpu = processorCache->cpu;
		}

		// Apply the color transform with OpenColorIO
		OCIO::PackedImageDesc img(pixels, film.GetWidth(), film.GetHeight(), 3);
		cpu->apply(img);
	} catch (OCIO::Exception &exception) {
		throw runtime_error("OpenColorIO Error in OpenColorIOToneMap::Apply(): " + string(exception.what()));
	}
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
