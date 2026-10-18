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

#include "slg/materials/roughmattetranslucent.h"

using namespace std;
using namespace luxrays;
using namespace slg;

//------------------------------------------------------------------------------
// RoughMatteTranslucent material
//------------------------------------------------------------------------------

RoughMatteTranslucentMaterial::RoughMatteTranslucentMaterial(
		TextureConstPtr frontTransp, TextureConstPtr backTransp,
		TextureConstPtr emitted, TextureConstPtr bump,
		TextureConstPtr refl, TextureConstPtr trans, TextureConstPtr s) :
			Material(frontTransp, backTransp, emitted, bump), Kr(refl), Kt(trans), sigma(s) {
}

Spectrum RoughMatteTranslucentMaterial::Albedo(const HitPoint &hitPoint) const {
	const Spectrum r = Kr->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
	const Spectrum t = Kt->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f) * 
		// Energy conservation
		(Spectrum(1.f) - r);

	return r + t;
}

// The reflection lobe r and the transmission lobe t = kt (1 - kr) have the
// shape of EON with albedo 1 (Portsmouth et al. 2025, see EON_Evaluate()),
// so the directional albedo of the material is r + t at every roughness; the
// transmission lobe is EON evaluated with the light direction mirrored
// through the surface (diffuse transmission keeps the azimuth). Each lobe is
// sampled with the cosine distribution of its hemisphere, chosen in
// proportion to its weight.

namespace {

// The probability of sampling the reflection lobe, 0 or less if there is no lobe
float RoughMatteTranslucent_ReflectProbability(const Spectrum &r, const Spectrum &t) {
	const float rf = r.Filter();
	const float tf = t.Filter();
	return ((rf + tf) > 0.f) ? rf / (rf + tf) : -1.f;
}

}

Spectrum RoughMatteTranslucentMaterial::Evaluate(const HitPoint &hitPoint,
	const Vector &localLightDir, const Vector &localEyeDir, BSDFEvent *event,
	float *directPdfW, float *reversePdfW) const {
	if (directPdfW)
		*directPdfW = 0.f;
	if (reversePdfW)
		*reversePdfW = 0.f;

	const Spectrum r = Kr->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
	const Spectrum t = Kt->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f) * 
		// Energy conservation
		(Spectrum(1.f) - r);

	const float pReflect = RoughMatteTranslucent_ReflectProbability(r, t);
	if (pReflect < 0.f)
		return Spectrum();

	const bool reflected = (CosTheta(localLightDir) * CosTheta(localEyeDir) > 0.f);
	const float weight = reflected ? pReflect : (1.f - pReflect);

	if (directPdfW)
		*directPdfW = fabsf((hitPoint.fromLight ? CosTheta(localEyeDir) : CosTheta(localLightDir)) * (weight * INV_PI));

	if (reversePdfW)
		*reversePdfW = fabsf((hitPoint.fromLight ? CosTheta(localLightDir) : CosTheta(localEyeDir)) * (weight * INV_PI));

	const float roughness = Clamp(sigma->GetFloatValue(hitPoint), 0.f, 1.f);
	if (reflected) {
		*event = DIFFUSE | REFLECT;
		return r * EON_Evaluate(Spectrum(1.f), roughness, localLightDir, localEyeDir) * fabsf(localLightDir.z);
	} else {
		*event = DIFFUSE | TRANSMIT;
		const Vector mirroredLightDir(localLightDir.x, localLightDir.y, -localLightDir.z);
		return t * EON_Evaluate(Spectrum(1.f), roughness, mirroredLightDir, localEyeDir) * fabsf(localLightDir.z);
	}
}

Spectrum RoughMatteTranslucentMaterial::Sample(const HitPoint &hitPoint,
	const Vector &localFixedDir, Vector *localSampledDir,
	const float u0, const float u1, const float passThroughEvent,
	float *pdfW, BSDFEvent *event) const {
	if (fabsf(localFixedDir.z) < DEFAULT_COS_EPSILON_STATIC)
		return Spectrum();

	const Spectrum r = Kr->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
	const Spectrum t = Kt->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f) * 
		// Energy conservation
		(Spectrum(1.f) - r);
	const float pReflect = RoughMatteTranslucent_ReflectProbability(r, t);
	if (pReflect < 0.f)
		return Spectrum();

	float cosPdf;
	*localSampledDir = CosineSampleHemisphere(u0, u1, &cosPdf);
	if (fabsf(localSampledDir->z) < DEFAULT_COS_EPSILON_STATIC)
		return Spectrum();

	// Only the z flips: a diffuse transmission keeps the azimuth
	if (passThroughEvent < pReflect)
		localSampledDir->z *= Sgn(localFixedDir.z);
	else
		localSampledDir->z *= -Sgn(localFixedDir.z);

	// The value and the pdf of the mixture from the evaluation
	const Vector &localLightDir = hitPoint.fromLight ? localFixedDir : *localSampledDir;
	const Vector &localEyeDir = hitPoint.fromLight ? *localSampledDir : localFixedDir;
	const Spectrum f = Evaluate(hitPoint, localLightDir, localEyeDir, event, pdfW, nullptr);
	if (f.Black() || !(*pdfW > 0.f))
		return Spectrum();

	return f / *pdfW;
}

void RoughMatteTranslucentMaterial::Pdf(const HitPoint &hitPoint,
		const Vector &localLightDir, const Vector &localEyeDir,
		float *directPdfW, float *reversePdfW) const {
	BSDFEvent event;
	Evaluate(hitPoint, localLightDir, localEyeDir, &event, directPdfW, reversePdfW);
}

void RoughMatteTranslucentMaterial::AddReferencedTextures(std::unordered_set<const Texture *>  &referencedTexs) const {
	Material::AddReferencedTextures(referencedTexs);

	Kr->AddReferencedTextures(referencedTexs);
	Kt->AddReferencedTextures(referencedTexs);
	sigma->AddReferencedTextures(referencedTexs);
}

void RoughMatteTranslucentMaterial::UpdateTextureReferences(TextureConstRef oldTex, TextureRef newTex) {
	Material::UpdateTextureReferences(oldTex, newTex);

	if (Kr == &oldTex)
		Kr = &newTex;
	if (Kt == &oldTex)
		Kt = &newTex;
	if (sigma == &oldTex)
		sigma = &newTex;
}

PropertiesUPtr RoughMatteTranslucentMaterial::ToProperties(const ImageMapCache &imgMapCache, const bool useRealFileName) const  {
	auto props = std::make_unique<Properties>();

	const string name = GetName();
	props->Set(Property("scene.materials." + name + ".type")("roughmattetranslucent"));
	props->Set(Property("scene.materials." + name + ".kr")(Kr->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".kt")(Kt->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".sigma")(sigma->GetSDLValue()));
	props->Set(Material::ToProperties(imgMapCache, useRealFileName));

	return props;
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
