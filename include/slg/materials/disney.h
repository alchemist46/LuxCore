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

#ifndef _SLG_DISNEYMAT_H
#define	_SLG_DISNEYMAT_H

#include "slg/materials/material.h"

namespace slg {

//------------------------------------------------------------------------------
//							Disney BRDF
// Based on "Physically Based Shading at Disney" presentet SIGGRAPH 2012 
//------------------------------------------------------------------------------

class DisneyMaterial : public Material {
public:
	DisneyMaterial(
		TextureConstPtr frontTransp,
		TextureConstPtr backTransp,
		TextureConstPtr emitted,
		TextureConstPtr bump,
		TextureConstPtr baseColor,
		TextureConstPtr subsurface,
		TextureConstPtr roughness,
		TextureConstPtr metallic,
		TextureConstPtr specular,
		TextureConstPtr specularTint,
		TextureConstPtr clearcoat,
		TextureConstPtr clearcoatGloss,
		TextureConstPtr anisotropic,
		TextureConstPtr sheen,
		TextureConstPtr sheenTint,
		TextureConstPtr filmAmount, 
		TextureConstPtr filmThickness, 
		TextureConstPtr filmIor
	);

	virtual MaterialType GetType() const { return DISNEY; }
	virtual BSDFEvent GetEventTypes() const { return GLOSSY | REFLECT; };

	virtual luxrays::Spectrum Albedo(
		const HitPoint &hitPoint
	) const;

	virtual luxrays::Spectrum Evaluate(
		const HitPoint &hitPoint,
		const luxrays::Vector &localLightDir, 
		const luxrays::Vector &localEyeDir, 
		BSDFEvent *event,
		float *directPdfW = NULL, 
		float *reversePdfW = NULL
	) const;

	virtual luxrays::Spectrum Sample(
		const HitPoint &hitPoint,
		const luxrays::Vector &localFixedDir, 
		luxrays::Vector *localSampledDir,
		const float u0, 
		const float u1, 
		const float passThroughEvent,
		float *pdfW,
		BSDFEvent *event
	) const;

	virtual void Pdf(
		const HitPoint &hitPoint,
		const luxrays::Vector &localLightDir, 
		const luxrays::Vector &localEyeDir,
		float *directPdfW, 
		float *reversePdfW
	) const;

	luxrays::PropertiesUPtr ToProperties(
		const ImageMapCache &imgMapCache, 
		const bool useRealFileName
	) const;

	void UpdateTextureReferences(
		TextureConstRef  oldTex, 
		TextureRef  newTex
	);

	void AddReferencedTextures(std::unordered_set<const Texture *>& referencedTexs) const;

	TextureConstPtr GetBaseColor() const { return BaseColor; };
	TextureConstPtr GetSubsurface() const { return Subsurface; };
	TextureConstPtr GetRoughness() const { return Roughness; };
	TextureConstPtr GetMetallic() const { return Metallic; };
	TextureConstPtr GetSpecular() const { return Specular; };
	TextureConstPtr GetSpecularTint() const { return SpecularTint; };
	TextureConstPtr GetClearcoat() const { return Clearcoat; };
	TextureConstPtr GetClearcoatGloss() const { return ClearcoatGloss; };
	TextureConstPtr GetAnisotropic() const { return Anisotropic; };
	TextureConstPtr GetSheen() const { return Sheen; };
	TextureConstPtr GetSheenTint() const { return SheenTint; };
	TextureConstPtr GetFilmAmount() const { return filmAmount; }
	TextureConstPtr GetFilmThickness() const { return filmThickness; }
	TextureConstPtr GetFilmIOR() const { return filmIor; }

	// The parameters at a hit point (defined in disney.cpp)
	struct Params;

private:
	TextureConstPtr BaseColor;
	TextureConstPtr Subsurface;
	TextureConstPtr Roughness;
	TextureConstPtr Metallic;
	TextureConstPtr Specular;
	TextureConstPtr SpecularTint;
	TextureConstPtr Clearcoat;
	TextureConstPtr ClearcoatGloss;
	TextureConstPtr Anisotropic;
	TextureConstPtr Sheen;
	TextureConstPtr SheenTint;
	TextureConstPtr filmAmount;
	TextureConstPtr filmThickness;
	TextureConstPtr filmIor;

	void UpdateGlossiness();

	// The lobe stack (see disney.cpp)
	void GetParams(const HitPoint &hitPoint, Params &p) const;
	luxrays::Spectrum EvaluateLayers(const Params &p, const bool fromLight,
			const luxrays::Vector &localLightDir, const luxrays::Vector &localEyeDir) const;
	float PdfLobes(const Params &p, const bool fromLight,
			const luxrays::Vector &localFixedDir, const luxrays::Vector &localSampledDir) const;

	luxrays::Spectrum CalculateTint(const luxrays::Spectrum &color) const;
};

}

#endif	/* _SLG_DISNEYMAT_H */// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
