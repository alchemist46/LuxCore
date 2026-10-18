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

#ifndef _SLG_OPENPBRMAT_H
#define	_SLG_OPENPBRMAT_H

#include <array>

#include "slg/materials/material.h"

namespace slg {

//------------------------------------------------------------------------------
// OpenPBR Surface material (specification v1.1)
//
// Every input is a texture named as the specification's parameter with a dot
// for the first underscore (base_color is base.color, specular_ior is
// specular.ior). An input that is not set (nullptr) has the default of the
// specification. The BSDF is the kernel code shared with the GPU, see
// materialdefs_funcs_openpbr_bsdf.cl.
//------------------------------------------------------------------------------

class OpenPBRMaterial : public Material {
public:
	typedef std::array<TextureConstPtr, ocl::OPENPBR_INPUT_COUNT> Inputs;

	OpenPBRMaterial(TextureConstPtr frontTransp, TextureConstPtr backTransp,
			TextureConstPtr emitted, TextureConstPtr bump,
			const Inputs &inputs, const bool thinWalled);

	virtual MaterialType GetType() const { return OPENPBR; }
	virtual BSDFEvent GetEventTypes() const { return DIFFUSE | GLOSSY | REFLECT; };

	virtual luxrays::Spectrum Albedo(const HitPoint &hitPoint) const;

	virtual luxrays::Spectrum Evaluate(const HitPoint &hitPoint,
		const luxrays::Vector &localLightDir, const luxrays::Vector &localEyeDir, BSDFEvent *event,
		float *directPdfW = NULL, float *reversePdfW = NULL) const;
	virtual luxrays::Spectrum Sample(const HitPoint &hitPoint,
		const luxrays::Vector &localFixedDir, luxrays::Vector *localSampledDir,
		const float u0, const float u1, const float passThroughEvent,
		float *pdfW, BSDFEvent *event) const;
	virtual void Pdf(const HitPoint &hitPoint,
		const luxrays::Vector &localLightDir, const luxrays::Vector &localEyeDir,
		float *directPdfW, float *reversePdfW) const;

	virtual void AddReferencedTextures(std::unordered_set<const Texture *> &referencedTexs) const;
	virtual void UpdateTextureReferences(TextureConstRef oldTex, TextureRef newTex);

	virtual luxrays::PropertiesUPtr ToProperties(const ImageMapCache &imgMapCache, const bool useRealFileName) const;

	// nullptr when the input is not set
	TextureConstPtr GetInput(const ocl::OpenPBRInput input) const { return inputs[input]; }
	bool IsThinWalled() const { return thinWalled; }

	// The property name of an input ("base.color") and its default
	static const char *GetInputName(const ocl::OpenPBRInput input);
	static luxrays::Spectrum GetInputDefault(const ocl::OpenPBRInput input);

	float GetFloat(const ocl::OpenPBRInput input, const HitPoint &hitPoint) const;
	luxrays::Spectrum GetSpectrum(const ocl::OpenPBRInput input, const HitPoint &hitPoint) const;

private:
	void UpdateGlossiness();

	Inputs inputs;
	bool thinWalled;
};

}

#endif	/* _SLG_OPENPBRMAT_H */
