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

#include "slg/materials/velvet.h"
#include "slg/materials/sheen.h"

using namespace std;
using namespace luxrays;
using namespace slg;

//------------------------------------------------------------------------------
// Velvet material
//
// Since 2026-10-18 the microfiber sheen of Zeltner, Burley & Chiang 2022 (the
// model of Cycles' Sheen BSDF and of the OpenPBR fuzz lobe, see sheen.h): a
// layer of fibers with the color kd and the roughness r. Its directional
// albedo is the one of the fiber layer alone (0.3-0.75 for a white kd), as
// for Cycles' node. The old model (LuxRender's thin-layer polynomial with
// the p1, p2, p3 and thickness parameters) was not normalized.
//
// Eye paths sample the LTC exactly. Light paths have the light direction
// fixed, where the LTC (oriented by the eye direction) can't be sampled, and
// sample the cosine distribution: Evaluate() returns the pdf of the sampler
// of each path type.
//------------------------------------------------------------------------------

VelvetMaterial::VelvetMaterial(TextureConstPtr frontTransp, TextureConstPtr backTransp,
		TextureConstPtr emitted, TextureConstPtr bump,
		TextureConstPtr kd, TextureConstPtr roughness) :
			Material(frontTransp, backTransp, emitted, bump), Kd(kd), Roughness(roughness) {
	glossiness = 1.f;
}

Spectrum VelvetMaterial::Albedo(const HitPoint &hitPoint) const {
	return Kd->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
}

Spectrum VelvetMaterial::Evaluate(const HitPoint &hitPoint,
	const Vector &localLightDir, const Vector &localEyeDir, BSDFEvent *event,
	float *directPdfW, float *reversePdfW) const {
	if (directPdfW)
		*directPdfW = 0.f;
	if (reversePdfW)
		*reversePdfW = 0.f;
	*event = DIFFUSE | REFLECT;

	if (localLightDir.z * localEyeDir.z <= 0.f)
		return Spectrum();

	// Both faces: the back face is the mirror image of the front one
	const float flip = (localEyeDir.z < 0.f) ? -1.f : 1.f;
	const Vector eyeDir(localEyeDir.x, localEyeDir.y, flip * localEyeDir.z);
	const Vector lightDir(localLightDir.x, localLightDir.y, flip * localLightDir.z);

	float a, b, albedo;
	Sheen_Lookup(eyeDir.z, Roughness->GetFloatValue(hitPoint), &a, &b, &albedo);
	// An invalid LTC (grazing angles at small roughness): no lobe, as Cycles
	if ((fabsf(a) < 1e-5f) || (albedo < 1e-5f))
		return Spectrum();

	const float ltc = Sheen_LTC(a, b, eyeDir, lightDir);
	if (!(ltc > 0.f))
		return Spectrum();

	// The pdf of sampling the light direction from the eye one (the LTC, eye
	// paths) and the eye direction from the light one (cosine, light paths)
	const float pdfLightGivenEye = ltc;
	const float pdfEyeGivenLight = eyeDir.z * INV_PI;
	if (directPdfW)
		*directPdfW = hitPoint.fromLight ? pdfEyeGivenLight : pdfLightGivenEye;
	if (reversePdfW)
		*reversePdfW = hitPoint.fromLight ? pdfLightGivenEye : pdfEyeGivenLight;

	return Kd->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f) * (albedo * ltc);
}

Spectrum VelvetMaterial::Sample(const HitPoint &hitPoint,
	const Vector &localFixedDir, Vector *localSampledDir,
	const float u0, const float u1, const float passThroughEvent,
	float *pdfW, BSDFEvent *event) const {
	if (fabsf(localFixedDir.z) < DEFAULT_COS_EPSILON_STATIC)
		return Spectrum();

	const float flip = (localFixedDir.z < 0.f) ? -1.f : 1.f;
	if (!hitPoint.fromLight) {
		// Eye path: the LTC oriented by the eye direction
		const Vector eyeDir(localFixedDir.x, localFixedDir.y, flip * localFixedDir.z);

		float a, b, albedo;
		Sheen_Lookup(eyeDir.z, Roughness->GetFloatValue(hitPoint), &a, &b, &albedo);
		if ((fabsf(a) < 1e-5f) || (albedo < 1e-5f))
			return Spectrum();

		const Vector lightDir = Sheen_SampleLTC(a, b, eyeDir, u0, u1);
		if (lightDir.z < DEFAULT_COS_EPSILON_STATIC)
			return Spectrum();
		*localSampledDir = Vector(lightDir.x, lightDir.y, flip * lightDir.z);
	} else {
		// Light path: the cosine distribution on the side of the light
		float pdf;
		*localSampledDir = CosineSampleHemisphere(u0, u1, &pdf);
		localSampledDir->z *= flip;
		if (fabsf(localSampledDir->z) < DEFAULT_COS_EPSILON_STATIC)
			return Spectrum();
	}

	// The value and the pdf from the evaluation
	const Vector &localLightDir = hitPoint.fromLight ? localFixedDir : *localSampledDir;
	const Vector &localEyeDir = hitPoint.fromLight ? *localSampledDir : localFixedDir;
	const Spectrum f = Evaluate(hitPoint, localLightDir, localEyeDir, event, pdfW, nullptr);
	if (f.Black() || !(*pdfW > 0.f))
		return Spectrum();

	return f / *pdfW;
}

void VelvetMaterial::Pdf(const HitPoint &hitPoint,
		const Vector &localLightDir, const Vector &localEyeDir,
		float *directPdfW, float *reversePdfW) const {
	BSDFEvent event;
	Evaluate(hitPoint, localLightDir, localEyeDir, &event, directPdfW, reversePdfW);
}

void VelvetMaterial::AddReferencedTextures(std::unordered_set<const Texture *>  &referencedTexs) const {
	Material::AddReferencedTextures(referencedTexs);

	Kd->AddReferencedTextures(referencedTexs);
	Roughness->AddReferencedTextures(referencedTexs);
}

void VelvetMaterial::UpdateTextureReferences(TextureConstRef oldTex, TextureRef newTex) {
	Material::UpdateTextureReferences(oldTex, newTex);

	if (Kd == &oldTex)
		Kd = &newTex;
	if (Roughness == &oldTex)
		Roughness = &newTex;
}

PropertiesUPtr VelvetMaterial::ToProperties(const ImageMapCache &imgMapCache, const bool useRealFileName) const  {
	auto props = std::make_unique<Properties>();

	const std::string name = GetName();
	props->Set(Property("scene.materials." + name + ".type")("velvet"));
	props->Set(Property("scene.materials." + name + ".kd")(Kd->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".roughness")(Roughness->GetSDLValue()));
	props->Set(Material::ToProperties(imgMapCache, useRealFileName));

	return props;
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
