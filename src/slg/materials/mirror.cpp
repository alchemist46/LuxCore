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

#include "slg/materials/mirror.h"

using namespace std;
using namespace luxrays;
using namespace slg;

//------------------------------------------------------------------------------
// Mirror material
//------------------------------------------------------------------------------

MirrorMaterial::MirrorMaterial(TextureConstPtr frontTransp, TextureConstPtr backTransp,
		TextureConstPtr emitted, TextureConstPtr bump,
		TextureConstPtr refl) : Material(frontTransp, backTransp, emitted, bump), Kr(refl) {
}

// Path regularization (HitPoint::regularization > 0): the mirror is a GGX
// reflection with the alpha floor and Kr as its Fresnel term

Spectrum MirrorMaterial::Evaluate(const HitPoint &hitPoint,
	const Vector &localLightDir, const Vector &localEyeDir, BSDFEvent *event,
	float *directPdfW, float *reversePdfW) const {
	if (hitPoint.regularization > 0.f) {
		if (localLightDir.z * localEyeDir.z <= 0.f)
			return Spectrum();

		const float alpha = Microfacet_Regularize(0.f, hitPoint.regularization);
		const Vector wh(Normalize(localLightDir + localEyeDir));

		const Vector &localFixedDir = hitPoint.fromLight ? localLightDir : localEyeDir;
		const Vector &localSampledDir = hitPoint.fromLight ? localEyeDir : localLightDir;
		if (directPdfW)
			*directPdfW = Microfacet_PdfReflect(alpha, alpha, localFixedDir, localSampledDir);
		if (reversePdfW)
			*reversePdfW = Microfacet_PdfReflect(alpha, alpha, localSampledDir, localFixedDir);

		*event = GLOSSY | REFLECT;
		return (Microfacet_D(alpha, alpha, wh) * Microfacet_G(alpha, alpha, localLightDir, localEyeDir) /
				(4.f * fabsf(localEyeDir.z))) * Kr->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
	}

	return Spectrum();
}

Spectrum MirrorMaterial::Sample(const HitPoint &hitPoint,
	const Vector &localFixedDir, Vector *localSampledDir,
	const float u0, const float u1, const float passThroughEvent,
	float *pdfW, BSDFEvent *event) const {
	if (hitPoint.regularization > 0.f) {
		if (fabsf(localFixedDir.z) < DEFAULT_COS_EPSILON_STATIC)
			return Spectrum();

		const float alpha = Microfacet_Regularize(0.f, hitPoint.regularization);
		*pdfW = Microfacet_SampleReflect(alpha, alpha, localFixedDir, u0, u1, localSampledDir);

		const float coso = fabsf(localFixedDir.z);
		const float cosi = fabsf(localSampledDir->z);
		if ((cosi < DEFAULT_COS_EPSILON_STATIC) || (localFixedDir.z * localSampledDir->z < 0.f) || (*pdfW <= 0.f))
			return Spectrum();

		const Vector wh(Normalize(localFixedDir + *localSampledDir));
		const float G = Microfacet_G(alpha, alpha, localFixedDir, *localSampledDir);
		const float d = Microfacet_D(alpha, alpha, wh);

		// f * |cos(sampled)| / pdf
		const float factor = d * G / (4.f * *pdfW * (hitPoint.fromLight ? cosi : coso));

		*event = GLOSSY | REFLECT;
		return factor * Kr->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
	}

	*event = SPECULAR | REFLECT;

	*localSampledDir = Vector(-localFixedDir.x, -localFixedDir.y, localFixedDir.z);
	*pdfW = 1.f;

	return Kr->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
}

void MirrorMaterial::Pdf(const HitPoint &hitPoint,
		const Vector &localLightDir, const Vector &localEyeDir,
		float *directPdfW, float *reversePdfW) const {
	if ((hitPoint.regularization > 0.f) && (localLightDir.z * localEyeDir.z > 0.f)) {
		const float alpha = Microfacet_Regularize(0.f, hitPoint.regularization);
		const Vector &localFixedDir = hitPoint.fromLight ? localLightDir : localEyeDir;
		const Vector &localSampledDir = hitPoint.fromLight ? localEyeDir : localLightDir;
		if (directPdfW)
			*directPdfW = Microfacet_PdfReflect(alpha, alpha, localFixedDir, localSampledDir);
		if (reversePdfW)
			*reversePdfW = Microfacet_PdfReflect(alpha, alpha, localSampledDir, localFixedDir);
		return;
	}

	if (directPdfW)
		*directPdfW = 0.f;
	if (reversePdfW)
		*reversePdfW = 0.f;
}

void MirrorMaterial::AddReferencedTextures(std::unordered_set<const Texture *>  &referencedTexs) const {
	Material::AddReferencedTextures(referencedTexs);

	Kr->AddReferencedTextures(referencedTexs);
}

void MirrorMaterial::UpdateTextureReferences(TextureConstRef oldTex, TextureRef newTex) {
	Material::UpdateTextureReferences(oldTex, newTex);

	if (Kr == &oldTex)
		Kr = &newTex;
}

PropertiesUPtr MirrorMaterial::ToProperties(const ImageMapCache &imgMapCache, const bool useRealFileName) const  {
	auto props = std::make_unique<Properties>();

	const string name = GetName();
	props->Set(Property("scene.materials." + name + ".type")("mirror"));
	props->Set(Property("scene.materials." + name + ".kr")(Kr->GetSDLValue()));
	props->Set(Material::ToProperties(imgMapCache, useRealFileName));

	return props;
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
