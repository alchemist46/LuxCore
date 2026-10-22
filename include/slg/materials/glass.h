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

#ifndef _SLG_GLASSMAT_H
#define	_SLG_GLASSMAT_H

#include "slg/materials/material.h"

namespace slg {

//------------------------------------------------------------------------------
// Glass material
//------------------------------------------------------------------------------

class GlassMaterial : public Material {
public:
	using TexRef = TextureConstPtr;
	GlassMaterial(TexRef frontTransp, TexRef backTransp,
			TexRef emitted, TexRef bump,
			TexRef refl, TexRef trans,
			TexRef exteriorIorFact, TexRef interiorIorFact,
			TexRef B, 
			TexRef filmThickness, TexRef filmIor);

	virtual MaterialType GetType() const { return GLASS; }
	virtual BSDFEvent GetEventTypes() const { return SPECULAR | REFLECT | TRANSMIT; };

	virtual bool IsDelta() const { return true; }
	virtual bool IsRegularizedDelta() const { return false; }
	virtual BSDFEvent GetRegularizedEventTypes() const { return GLOSSY | REFLECT | TRANSMIT; };
	virtual const Material *GetRefractiveMaterial() const { return this; }
	virtual bool GetRefractionIors(const HitPoint &hitPoint, float *nc, float *nt) const;

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

	virtual void AddReferencedTextures(std::unordered_set<const Texture *>  &referencedTexsreferencedTexs) const;
	virtual void UpdateTextureReferences(TextureConstRef oldTex, TextureRef newTex);

	virtual luxrays::PropertiesUPtr ToProperties(const ImageMapCache &imgMapCache, const bool useRealFileName) const;

	TexRef GetKr() const { return Kr; }
	TexRef GetKt() const { return Kt; }
	TexRef GetExteriorIOR() const { return exteriorIor; }
	TexRef GetInteriorIOR() const { return interiorIor; }
	TexRef GetCauchyB() const { return cauchyB; }
	TexRef GetFilmThickness() const { return filmThickness; }
	TexRef GetFilmIOR() const { return filmIor; }

	// The colour of the dispersion of the wavelength (nm) in the working colour
	// space: its average over the visible range (380-780 nm) is white
	static luxrays::Spectrum WaveLengthColor(const float waveLength, const WorkingColorSpace &workingColorSpace);

	static luxrays::Spectrum EvalSpecularReflection(const HitPoint &hitPoint,
			const luxrays::Vector &localFixedDir,
			const luxrays::Spectrum &kr, const float nc, const float nt,
			luxrays::Vector *localSampledDir,
			const float localFilmThickness, const float localFilmIor,
			const WorkingColorSpace &workingColorSpace);
	static luxrays::Spectrum EvalSpecularTransmission(const HitPoint &hitPoint,
			const luxrays::Vector &localFixedDir, const luxrays::Spectrum &kt,
			const float nc, const float nt,
			luxrays::Vector *localSampledDir,
			const float localFilmThickness, const float localFilmIor,
			const WorkingColorSpace &workingColorSpace);

private:

	TexRef Kr;
	TexRef Kt;
	TexRef exteriorIor;
	TexRef interiorIor;
	TexRef cauchyB;
	TexRef filmThickness;
	TexRef filmIor;
};

}

#endif	/* _SLG_GLASSMAT_H */
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
