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
	
#include "luxrays/utils/strutils.h"
#include "slg/scene/colorspaceconverters.h"
#include "slg/utils/filenameresolver.h"

using namespace std;
using namespace luxrays;
using namespace slg;

//------------------------------------------------------------------------------
// ColorSpaceConverters
//------------------------------------------------------------------------------

ColorSpaceConverters::ColorSpaceConverters() {
}

ColorSpaceConverters::~ColorSpaceConverters() {
}

void ColorSpaceConverters::ConvertFrom(const ColorSpaceConfig &cfg, float &v) {
	switch (cfg.colorSpaceType) {
		case ColorSpaceConfig::NOP_COLORSPACE:
			// Nothing to do
			break;
		case ColorSpaceConfig::LUXCORE_COLORSPACE:
			ConvertFromLuxCore(cfg.luxcore.gamma, v);
			break;
		case ColorSpaceConfig::OPENCOLORIO_COLORSPACE:
			ConvertFromOpenColorIO(cfg.ocio.configName, cfg.ocio.colorSpaceName, v);
			break;
		default:
			throw runtime_error("Unknown color space type in ColorSpaceConverters::ConvertFrom(): " + ToString(cfg.colorSpaceType));
	}
}

void ColorSpaceConverters::ConvertFrom(const ColorSpaceConfig &cfg, luxrays::Spectrum &c) {
	switch (cfg.colorSpaceType) {
		case ColorSpaceConfig::NOP_COLORSPACE:
			// Nothing to do
			break;
		case ColorSpaceConfig::LUXCORE_COLORSPACE:
			ConvertFromLuxCore(cfg.luxcore.gamma, c);
			break;
		case ColorSpaceConfig::OPENCOLORIO_COLORSPACE:
			ConvertFromOpenColorIO(cfg.ocio.configName, cfg.ocio.colorSpaceName, c);
			break;
		default:
			throw runtime_error("Unknown color space type in ColorSpaceConverters::ConvertFrom(): " + ToString(cfg.colorSpaceType));
	}
}

//------------------------------------------------------------------------------
// LuxCore color space
//------------------------------------------------------------------------------

void ColorSpaceConverters::ConvertFromLuxCore(const float gamma, float &v) {
	v = powf(v, gamma);
}

void ColorSpaceConverters::ConvertFromLuxCore(const float gamma, Spectrum &c) {
	c.c[0] = powf(c.c[0], gamma);
	c.c[1] = powf(c.c[1], gamma);
	c.c[2] = powf(c.c[2], gamma);

	c = workingColorSpace.FromRec709(c);
}

//------------------------------------------------------------------------------
// OpenColorIO color space
//------------------------------------------------------------------------------

void ColorSpaceConverters::ConvertFromOpenColorIO(const string &configFileName,
		const string &inputColorSpace, float &v) {
	Spectrum c(v);

	ConvertFromOpenColorIO(configFileName, inputColorSpace, c);

	v = c.Y();
}

static string GetOCIOProcessorCacheKey(const string &configFileName,
		const string &inputColorSpace, const WorkingColorSpace &wcs) {
	return configFileName + "_#_" + inputColorSpace + "_#_" +
			WorkingColorSpace::WorkingColorSpaceType2String(wcs.GetType());
}

void ColorSpaceConverters::ConvertFromOpenColorIO(const string &configFileName,
		const string &inputColorSpace, Spectrum &c) {
	try {
		OCIO::ConstCPUProcessorRcPtr cpu;

		// Look for the processor inside the cache
		const string key = GetOCIOProcessorCacheKey(configFileName, inputColorSpace, workingColorSpace);
		auto it = ocioProcessorCache.find(key);
		if (it == ocioProcessorCache.end()) {
			// Not found, insert a new one into the cache
			OCIO::ConstConfigRcPtr config = (configFileName == "") ?
				OCIO::GetCurrentConfig() :
				OCIO::Config::CreateFromFile(SLG_FileNameResolver.ResolveFile(configFileName).c_str());

			// The target is the working space, or the scene linear role taken
			// as Rec.709 when the config has no such colour space
			string outputColorSpace = OCIO::ROLE_SCENE_LINEAR;
			OCIO::ConstProcessorRcPtr processor;
			if (!workingColorSpace.IsRec709()) {
				const string workingColorSpaceName = workingColorSpace.FindOpenColorIOColorSpace(configFileName);
				if (workingColorSpaceName != "")
					outputColorSpace = workingColorSpaceName;
				else {
					OCIO::ConstColorSpaceRcPtr cs = config->getColorSpace(inputColorSpace.c_str());
					if (!(cs && cs->isData())) {
						// Append Rec.709 to the working space to the transform
						const float (&m)[3][3] = workingColorSpace.GetFromRec709Matrix();
						const double m44[16] = {
							m[0][0], m[0][1], m[0][2], 0.0,
							m[1][0], m[1][1], m[1][2], 0.0,
							m[2][0], m[2][1], m[2][2], 0.0,
							0.0, 0.0, 0.0, 1.0
						};
						OCIO::MatrixTransformRcPtr matrix = OCIO::MatrixTransform::Create();
						matrix->setMatrix(m44);

						OCIO::GroupTransformRcPtr group = OCIO::GroupTransform::Create();
						OCIO::ColorSpaceTransformRcPtr toSceneLinear = OCIO::ColorSpaceTransform::Create();
						toSceneLinear->setSrc(inputColorSpace.c_str());
						toSceneLinear->setDst(OCIO::ROLE_SCENE_LINEAR);
						group->appendTransform(toSceneLinear);
						group->appendTransform(matrix);

						processor = config->getProcessor(group);
					}
				}
			}

			if (!processor)
				processor = config->getProcessor(inputColorSpace.c_str(), outputColorSpace.c_str());

			cpu = processor->getDefaultCPUProcessor();

			ocioProcessorCache[key] = cpu;
		} else
			cpu = it->second;

		/// Apply the color transform with OpenColorIO
		OCIO::PackedImageDesc img(c.c, 1, 1, 3);
		cpu->apply(img);
	} catch (OCIO::Exception &exception) {
		throw runtime_error("OpenColorIO Error in ColorSpaceConverters::ConvertFromOpenColorIO(): " + string(exception.what()));
	}
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
