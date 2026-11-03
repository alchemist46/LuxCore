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

#ifndef _SLG_GLARE_PLUGIN_H
#define	_SLG_GLARE_PLUGIN_H

#include <vector>
#include <complex>

#include "luxrays/luxrays.h"
#include "luxrays/utils/serializationutils.h"
#include "slg/core/workingcolorspace.h"
#include "slg/film/imagepipeline/imagepipeline.h"

namespace slg {

class Film;

//------------------------------------------------------------------------------
// Glare plugin
//
// The point spread function of a lens, convolved with the linear image:
// (1 - scatter) times the diffraction of the aperture plus scatter times a
// heavy tailed veiling glare, normalized to 1, so the plugin moves light
// without adding or removing any (up to what leaves the image).
//
// The diffraction is built in the frequency domain of the image: its optical
// transfer function at the spatial frequency nu is the autocorrelation of the
// aperture at the shift lambda * f * nu, divided by the aperture area. The
// autocorrelation is tabulated once with a FFT of the rasterized aperture, and
// summed over the wavelengths with the colours of COLOR_ABERRATION's SPECTRAL
// mode. At the physical scale the Airy core is smaller than a pixel: what
// shows is the tail, with the 2N spikes of N (odd) straight blades.
//------------------------------------------------------------------------------

class GlarePlugin : public ImagePipelinePlugin {
public:
	// colorSpace is the one of the image in front of the plugin: the film's
	// working space, or Rec.709 after a display curve
	GlarePlugin(const float fstop, const float sensorWidth,
			const u_int blades, const float rotation, const float roundness,
			const float obstruction, const float scatter,
			const float scatterRadius, const float scatterFalloff,
			const float scale, const float radius, const u_int samples,
			const WorkingColorSpace &colorSpace = WorkingColorSpace());
	virtual ~GlarePlugin();

	virtual ImagePipelinePlugin *Copy() const;

	virtual void Apply(Film &film, const u_int index);

	// The f-number and the sensor width (mm): the pixel pitch is
	// sensorWidth / film width
	float fstop, sensorWidth;
	// The aperture: the blade count (0 for a circle), the rotation of the
	// polygon (degrees), its blend toward the circle (0-1) and the radius of a
	// central obstruction (a fraction of the aperture radius)
	u_int blades;
	float rotation, roundness, obstruction;
	// The energy fraction of the veiling glare, the radius of its core (a
	// fraction of the image diagonal) and the exponent of its tail
	float scatter, scatterRadius, scatterFalloff;
	// 1 is the physical size of the diffraction, larger values widen it
	float scale;
	// The radius of the point spread function, a fraction of the image diagonal
	float radius;
	// The wavelength count
	u_int samples;

	friend class boost::serialization::access;

private:
	// Used by serialization
	GlarePlugin();

	template<class Archive> void serialize(Archive &ar, const u_int version) {
		ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(ImagePipelinePlugin);
		ar & fstop;
		ar & sensorWidth;
		ar & blades;
		ar & rotation;
		ar & roundness;
		ar & obstruction;
		ar & scatter;
		ar & scatterRadius;
		ar & scatterFalloff;
		ar & scale;
		ar & radius;
		ar & samples;
		ar & wavelengthTable;
	}

	void InitWavelengthTable(const WorkingColorSpace &colorSpace);
	bool InsideAperture(const float x, const float y) const;
	void InitApertureOTF();
	float ApertureOTF(const float sx, const float sy) const;
	void InitOTF(const u_int width, const u_int height);

	// For each wavelength: the wavelength (nm) and the weights of the 3 channels
	std::vector<float> wavelengthTable;

	// The autocorrelation of the aperture divided by its area, on a
	// apertureOTFSize^2 grid with the origin at (0, 0) (wrapped), the
	// aperture diameter being APERTURE_DIAMETER samples
	std::vector<float> apertureOTF;

	// The image size the transfer functions are for, and the padded size of
	// the FFTs
	u_int otfWidth, otfHeight;
	u_int paddedWidth, paddedHeight;
	// The transfer function of each channel (real, the point spread function
	// being symmetric), times the 1 / (paddedWidth * paddedHeight) of the
	// inverse FFT, on the paddedHeight x (paddedWidth / 2 + 1) half spectrum
	std::vector<float> otf[3];

	std::vector<float> realBuffer;
	std::vector<std::complex<float> > complexBuffer;
};

}

BOOST_CLASS_VERSION(slg::GlarePlugin, 1)

BOOST_CLASS_EXPORT_KEY(slg::GlarePlugin)

#endif	/*  _SLG_GLARE_PLUGIN_H */
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
