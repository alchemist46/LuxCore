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

#ifndef _SLG_COLORABERRATION_PLUGIN_H
#define	_SLG_COLORABERRATION_PLUGIN_H

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
// Color aberration plugin
//------------------------------------------------------------------------------

class ColorAberrationPlugin : public ImagePipelinePlugin {
public:
	typedef enum {
		// A wavelength dependent magnification about the center, from the
		// dispersion of a glass (the default of the scene files)
		SPECTRAL,
		// The code of LuxCore 2.x: each pixel blended with a shifted copy
		LEGACY
	} Mode;

	// colorSpace is the one of the image in front of the plugin: the film's
	// working space, or Rec.709 after a display curve
	ColorAberrationPlugin(const float amountX = .005f, const float amountY = .005f,
			const Mode mode = LEGACY, const u_int samples = 16,
			const float centerX = .5f, const float centerY = .5f,
			const WorkingColorSpace &colorSpace = WorkingColorSpace());
	virtual ~ColorAberrationPlugin();

	virtual ImagePipelinePlugin *Copy() const;

	virtual void Apply(Film &film, const u_int index);

	virtual bool CanUseHW() const { return true; }
	virtual void AddHWChannelsUsed(Film::FilmChannels &hwChannelsUsed) const;
	virtual void ApplyHW(Film &film, const u_int index);

	static Mode String2Mode(const std::string &type);
	static std::string Mode2String(const Mode type);

	// samples wavelengths at the centers of strata of 380-780 nm, 4 floats
	// each: the wavelength (nm) and the weights of the 3 channels of its
	// colour in colorSpace. The weights of a channel add up to 1, so a flat
	// image stays unchanged (also used by GLARE)
	static std::vector<float> WavelengthWeights(const u_int samples,
			const WorkingColorSpace &colorSpace);

	// The SPECTRAL mode uses only amountX
	float amountX, amountY;
	Mode mode;
	u_int samples;
	float centerX, centerY;

	friend class boost::serialization::access;

private:
	template<class Archive> void serialize(Archive &ar, const u_int version) {
		ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(ImagePipelinePlugin);
		ar & amountX;
		ar & amountY;
		if (version >= 3) {
			ar & mode;
			ar & samples;
			ar & centerX;
			ar & centerY;
			ar & spectralTable;
		} else {
			// Saved before the SPECTRAL mode
			mode = LEGACY;
			samples = 16;
			centerX = .5f;
			centerY = .5f;
		}
	}

	void InitSpectralTable(const WorkingColorSpace &colorSpace);
	void ApplyLegacy(Film &film, const u_int index);
	void ApplySpectral(Film &film, const u_int index);

	// For each wavelength of the SPECTRAL mode: the magnification minus 1 and
	// the weights of the 3 channels
	std::vector<float> spectralTable;

	luxrays::Spectrum *tmpBuffer;
	size_t tmpBufferSize;

	// Used inside the object destructor to free buffers
	luxrays::HardwareDevice *hardwareDevice;
	luxrays::HardwareDeviceBuffer *hwTmpBuffer;
	luxrays::HardwareDeviceBuffer *hwSpectralTable;

	luxrays::HardwareDeviceKernelUPtr applyKernel;
	luxrays::HardwareDeviceKernelUPtr copyKernel;
};

}

BOOST_CLASS_VERSION(slg::ColorAberrationPlugin, 3)

BOOST_CLASS_EXPORT_KEY(slg::ColorAberrationPlugin)

#endif	/*  _SLG_COLORABERRATION_PLUGIN_H */
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
