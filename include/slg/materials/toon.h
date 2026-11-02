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

#ifndef _SLG_TOONMAT_H
#define	_SLG_TOONMAT_H

#include <vector>

#include "slg/materials/material.h"

namespace slg {

//------------------------------------------------------------------------------
// Toon material
//
// Non-physical by design, with the parameters of Redshift's Toon material.
// The direct light of each light sample is a ramp of colours on the cosine of
// the light direction (the direction of the centre of the light with
// lightCenter), plus a toon highlight, a ramp of colours on the intensity of
// the GGX highlight of the roughness, and a rim, each with a weight, the rim
// with an optional ramp of colours (see materialdefs_funcs_toon_ramp.cl): the
// toon part. The indirect light is the
// one of a matte material of colour kd, times indirectScale: the Lambert lobe
// of Sample() has it, so only the light sampling sees the steps. The path
// tracers take the whole direct light from the light sampling
// (BSDF::IsDirectLightSamplingOnly()), so the BSDF sampling doesn't blur the
// steps through the MIS.
//------------------------------------------------------------------------------

// A ramp of the toon material: its keys (positions in ascending order,
// colours and interpolations, one for all or one per key). A factor ramp
// (shadowFill) is the legacy base ramp: grey colours, 0 below the first key,
// and the shadow colour fills what it leaves. No keys: no ramp.
struct ToonRamp {
	std::vector<float> positions;
	std::vector<luxrays::Spectrum> colors;
	std::vector<slg::ocl::ToonRampInterpolation> interpolations;
	float smooth = 0.f;
	bool shadowFill = false;
};

// The parameters of the toon material besides its textures
struct ToonOptions {
	// The base ramp on the cosine of the light direction, the specular one on
	// the intensity of the glossy highlight, the rim one across the rim
	ToonRamp ramp, specularRamp, rimRamp;
	float specularSmooth = .05f, specularStretchU = 0.f, specularStretchV = 0.f;
	float rimSize = .2f, rimSmooth = .05f;
	float baseWeight = 1.f, specularWeight = 1.f, rimWeight = 1.f;
	float indirectScale = 1.f;
	bool lightCenter = true;
};

class ToonMaterial : public Material {
public:
	ToonMaterial(TextureConstPtr frontTransp, TextureConstPtr backTransp,
			TextureConstPtr emitted, TextureConstPtr bump,
			TextureConstPtr kd, TextureConstPtr shadowColor,
			TextureConstPtr ks, TextureConstPtr rim, TextureConstPtr roughness,
			const ToonOptions &options);

	virtual MaterialType GetType() const { return TOON; }
	virtual BSDFEvent GetEventTypes() const { return DIFFUSE | REFLECT; };

	virtual luxrays::Spectrum Albedo(const HitPoint &hitPoint) const;
	virtual luxrays::Spectrum EvaluateDiffuse(const HitPoint &hitPoint,
		const luxrays::Vector &localLightDir, const luxrays::Vector &localEyeDir,
		float *pdfW) const;
	virtual bool HasDiffuseLobe() const { return true; }
	virtual bool HasOnlyDiffuseLobe() const { return true; }
	virtual luxrays::Spectrum EvaluateTotal(const HitPoint &hitPoint) const;

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

	// The direct light of an eye path: the toon part of the ramp direction
	// localRampDir, localLightDir or the direction of the centre of the light
	// of angular radius lightRadius (the size of the highlight, see
	// ToonMaterial_GlossyIntensity(); Evaluate() takes localLightDir and 0)
	luxrays::Spectrum EvaluateToon(const HitPoint &hitPoint,
		const luxrays::Vector &localLightDir, const luxrays::Vector &localRampDir,
		const luxrays::Vector &localEyeDir, const float lightRadius) const;

	virtual void AddReferencedTextures(std::unordered_set<const Texture *>  &referencedTexs) const;
	virtual void UpdateTextureReferences(TextureConstRef oldTex, TextureRef newTex);

	virtual luxrays::PropertiesUPtr ToProperties(const ImageMapCache &imgMapCache, const bool useRealFileName) const;

	TextureConstPtr GetKd() const { return Kd; }
	TextureConstPtr GetShadowColor() const { return shadowColor; }
	TextureConstPtr GetKs() const { return Ks; }
	TextureConstPtr GetRim() const { return rim; }
	TextureConstPtr GetRoughness() const { return roughness; }
	bool UsesLightCenter() const { return params.lightCenter; }
	// The ramps and the scalars, the texture indices left to the caller
	const slg::ocl::ToonParam &GetParams() const { return params; }

private:
	TextureConstPtr Kd, shadowColor, Ks, rim, roughness;
	slg::ocl::ToonParam params;
	// The SDL form of the base ramp (ToProperties())
	bool factorRamp;
};

}

#endif	/* _SLG_TOONMAT_H */
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
