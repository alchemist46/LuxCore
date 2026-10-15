/***************************************************************************
 * Copyright 1998-2026 by authors (see AUTHORS.txt)                        *
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

#ifndef _SLG_WORKINGTOREC709_PLUGIN_H
#define	_SLG_WORKINGTOREC709_PLUGIN_H

#include <vector>
#include <memory>
#include <typeinfo> 

#include "luxrays/core/color/color.h"
#include "luxrays/core/hardwaredevice.h"
#include "luxrays/utils/serializationutils.h"
#include "slg/core/workingcolorspace.h"
#include "slg/film/imagepipeline/imagepipeline.h"

namespace slg {

class Film;

//------------------------------------------------------------------------------
// Working colour space to Linear Rec.709 plugin
//
// Added by the film in front of the plugins that produce display values
// (GAMMA_CORRECTION, CAMERA_RESPONSE_FUNC) when the working colour space is
// not Rec.709: their curves are for Rec.709 primaries.
//------------------------------------------------------------------------------

class WorkingToRec709Plugin : public ImagePipelinePlugin {
public:
	WorkingToRec709Plugin(const WorkingColorSpace &workingColorSpace);
	virtual ~WorkingToRec709Plugin();

	virtual ImagePipelinePlugin *Copy() const;

	virtual void Apply(Film &film, const u_int index);

	virtual bool CanUseHW() const { return true; }
	virtual void AddHWChannelsUsed(Film::FilmChannels &hwChannelsUsed) const;
	virtual void ApplyHW(Film &film, const u_int index);

	friend class boost::serialization::access;

private:
	// Used by serialization
	WorkingToRec709Plugin();

	template<class Archive> void serialize(Archive &ar, const u_int version) {
		ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(ImagePipelinePlugin);
		ar & m;
	}

	// Row-major
	std::vector<float> m;

	luxrays::HardwareDeviceKernelUPtr applyKernel;
};

}

BOOST_CLASS_VERSION(slg::WorkingToRec709Plugin, 1)

BOOST_CLASS_EXPORT_KEY(slg::WorkingToRec709Plugin)

#endif	/*  _SLG_WORKINGTOREC709_PLUGIN_H */
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
