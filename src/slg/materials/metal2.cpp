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

#include "slg/materials/metal2.h"

using namespace std;
using namespace luxrays;
using namespace slg;

//------------------------------------------------------------------------------
// Metal2 material
//
// LuxRender Metal2 material porting.
//------------------------------------------------------------------------------

Metal2Material::Metal2Material(
	TextureConstPtr frontTransp,
	TextureConstPtr backTransp,
	TextureConstPtr emitted,
	TextureConstPtr bump,
	TextureConstPtr nn,
	TextureConstPtr kk,
	TextureConstPtr u,
	TextureConstPtr v
) :
	Material(frontTransp, backTransp, emitted, bump),
	fresnelTex(nullptr),
	n(nn),
	k(kk),
	nu(u),
	nv(v)
{
	glossiness = ComputeGlossiness(nu, nv);
}

Metal2Material::Metal2Material(
	TextureConstPtr frontTransp,
	TextureConstPtr backTransp,
	TextureConstPtr emitted,
	TextureConstPtr bump,
	FresnelTextureConstPtr ft,
	TextureConstPtr u,
	TextureConstPtr v)
	:
	Material(frontTransp, backTransp, emitted, bump),
	fresnelTex(ft),
	n(nullptr),
	k(nullptr),
	nu(u),
	nv(v)
{
	glossiness = ComputeGlossiness(nu, nv);
}

Spectrum Metal2Material::EvaluateFresnel(const HitPoint &hitPoint, const float cosi) const {
	Spectrum F;
	if (fresnelTex)
		F = fresnelTex->Evaluate(hitPoint, cosi);
	else {
		// For compatibility with the past
		const Spectrum etaVal = n->GetSpectrumValue(hitPoint).Clamp(.001f);
		const Spectrum kVal = k->GetSpectrumValue(hitPoint).Clamp(.001f);
		F = FresnelTexture::GeneralEvaluate(etaVal, kVal, cosi);
	}
	F.Clamp(0.f, 1.f);

	return F;
}

// The multiple-scattering scale of the lobe seen from localEyeDir: the average
// Fresnel reflectance 2 * integral F(mu) mu dmu by a 4-point Gauss-Legendre
// quadrature on [0, 1]
Spectrum Metal2Material::MSScale(const HitPoint &hitPoint, const float u, const float v,
		const Vector &localEyeDir) const {
	if (!IsMultiscattering())
		return Spectrum(1.f);

	static const float nodes[4] = { .069431844f, .330009478f, .669990522f, .930568156f };
	static const float weights[4] = { .173927423f, .326072577f, .326072577f, .173927423f };
	Spectrum Fss;
	for (u_int i = 0; i < 4; ++i)
		Fss += (2.f * weights[i] * nodes[i]) * EvaluateFresnel(hitPoint, nodes[i]);

	return GGXEnergy_ReflectScale(u, v, fabsf(localEyeDir.z), Fss.Clamp(0.f, 1.f), false);
}

Spectrum Metal2Material::Albedo(const HitPoint &hitPoint) const {
	Spectrum F;
	if (fresnelTex)
		F = fresnelTex->Evaluate(hitPoint, 1.f);
	else {
		// For compatibility with the past
		const Spectrum etaVal = n->GetSpectrumValue(hitPoint).Clamp(.001f);
		const Spectrum kVal = k->GetSpectrumValue(hitPoint).Clamp(.001f);
		F = FresnelTexture::GeneralEvaluate(etaVal, kVal, 1.f);
	}
	F.Clamp(0.f, 1.f);
	
	return F;
}
	
Spectrum Metal2Material::Evaluate(const HitPoint &hitPoint,
	const Vector &localLightDir, const Vector &localEyeDir, BSDFEvent *event,
	float *directPdfW, float *reversePdfW) const {
	const float u = Microfacet_Regularize(Clamp(nu->GetFloatValue(hitPoint), 1e-9f, 1.f), hitPoint.regularization);
	const float v = Microfacet_Regularize(Clamp(nv->GetFloatValue(hitPoint), 1e-9f, 1.f), hitPoint.regularization);

	const Vector wh(Normalize(localLightDir + localEyeDir));
	const float cosWH = Dot(localLightDir, wh);

	const Vector &localFixedDir = hitPoint.fromLight ? localLightDir : localEyeDir;
	const Vector &localSampledDir = hitPoint.fromLight ? localEyeDir : localLightDir;
	if (directPdfW)
		*directPdfW = Microfacet_PdfReflect(u, v, localFixedDir, localSampledDir);

	if (reversePdfW)
		*reversePdfW = Microfacet_PdfReflect(u, v, localSampledDir, localFixedDir);

	Spectrum F;
	if (fresnelTex)
		F = fresnelTex->Evaluate(hitPoint, cosWH);
	else {
		// For compatibility with the past
		const Spectrum etaVal = n->GetSpectrumValue(hitPoint).Clamp(.001f);
		const Spectrum kVal = k->GetSpectrumValue(hitPoint).Clamp(.001f);
		F = FresnelTexture::GeneralEvaluate(etaVal, kVal, cosWH);
	}
	F.Clamp(0.f, 1.f);

	const float G = Microfacet_G(u, v, localLightDir, localEyeDir);

	*event = GLOSSY | REFLECT;
	return (Microfacet_D(u, v, wh) * G / (4.f * fabsf(localEyeDir.z))) * F *
			MSScale(hitPoint, u, v, localEyeDir);
}

Spectrum Metal2Material::Sample(const HitPoint &hitPoint,
	const Vector &localFixedDir, Vector *localSampledDir,
	const float u0, const float u1, const float passThroughEvent,
	float *pdfW, BSDFEvent *event) const {
	if (fabsf(localFixedDir.z) < DEFAULT_COS_EPSILON_STATIC)
		return Spectrum();

	const float u = Microfacet_Regularize(Clamp(nu->GetFloatValue(hitPoint), 1e-9f, 1.f), hitPoint.regularization);
	const float v = Microfacet_Regularize(Clamp(nv->GetFloatValue(hitPoint), 1e-9f, 1.f), hitPoint.regularization);

	*pdfW = Microfacet_SampleReflect(u, v, localFixedDir, u0, u1, localSampledDir);

	const float coso = fabsf(localFixedDir.z);
	const float cosi = fabsf(localSampledDir->z);
	if ((cosi < DEFAULT_COS_EPSILON_STATIC) || (localFixedDir.z * localSampledDir->z < 0.f))
		return Spectrum();

	if (*pdfW <= 0.f)
		return Spectrum();

	const Vector wh(Normalize(localFixedDir + *localSampledDir));
	const float cosWH = AbsDot(localFixedDir, wh);

	const float G = Microfacet_G(u, v, localFixedDir, *localSampledDir);
	const float d = Microfacet_D(u, v, wh);

	Spectrum F;
	if (fresnelTex)
		F = fresnelTex->Evaluate(hitPoint, cosWH);
	else {
		// For compatibility with the past
		const Spectrum etaVal = n->GetSpectrumValue(hitPoint).Clamp(.001f);
		const Spectrum kVal = k->GetSpectrumValue(hitPoint).Clamp(.001f);
		F = FresnelTexture::GeneralEvaluate(etaVal, kVal, cosWH);
	}
	F.Clamp(0.f, 1.f);

	// f * |cos(sampled)| / pdf
	float factor = d * G / (4.f * *pdfW);
	if (!hitPoint.fromLight)
		factor /= coso;
	else
		factor /= cosi;

	*event = GLOSSY | REFLECT;

	return factor * F * MSScale(hitPoint, u, v, hitPoint.fromLight ? *localSampledDir : localFixedDir);
}

void Metal2Material::Pdf(const HitPoint &hitPoint,
		const Vector &localLightDir, const Vector &localEyeDir,
		float *directPdfW, float *reversePdfW) const {
	const float u = Microfacet_Regularize(Clamp(nu->GetFloatValue(hitPoint), 1e-9f, 1.f), hitPoint.regularization);
	const float v = Microfacet_Regularize(Clamp(nv->GetFloatValue(hitPoint), 1e-9f, 1.f), hitPoint.regularization);

	const Vector &localFixedDir = hitPoint.fromLight ? localLightDir : localEyeDir;
	const Vector &localSampledDir = hitPoint.fromLight ? localEyeDir : localLightDir;

	if (directPdfW)
		*directPdfW = Microfacet_PdfReflect(u, v, localFixedDir, localSampledDir);

	if (reversePdfW)
		*reversePdfW = Microfacet_PdfReflect(u, v, localSampledDir, localFixedDir);
}

void Metal2Material::AddReferencedTextures(std::unordered_set<const Texture *>  &referencedTexs) const {
	Material::AddReferencedTextures(referencedTexs);

	if (fresnelTex)
		fresnelTex->AddReferencedTextures(referencedTexs);
	if (n)
		n->AddReferencedTextures(referencedTexs);
	if (k)
		k->AddReferencedTextures(referencedTexs);

	nu->AddReferencedTextures(referencedTexs);
	nv->AddReferencedTextures(referencedTexs);
}

void Metal2Material::UpdateTextureReferences(TextureConstRef oldTex, TextureRef newTex) {
	Material::UpdateTextureReferences(oldTex, newTex);

	bool updateGlossiness = false;
	if (fresnelTex == static_cast<const FresnelTexture *>(std::addressof(oldTex)))
		fresnelTex = static_cast<const FresnelTexture *>(&newTex);
	if (n == &oldTex)
		n = &newTex;
	if (k == &oldTex)
		k = &newTex;
	if (nu == &oldTex) {
		nu = &newTex;
		updateGlossiness = true;
	}
	if (nv == &oldTex) {
		nv = &newTex;
		updateGlossiness = true;
	}
	
	if (updateGlossiness)
		glossiness = ComputeGlossiness(nu, nv);
}

PropertiesUPtr Metal2Material::ToProperties(const ImageMapCache &imgMapCache, const bool useRealFileName) const  {
	auto props = std::make_unique<Properties>();

	const string name = GetName();
	props->Set(Property("scene.materials." + name + ".type")("metal2"));
	if (fresnelTex)
		props->Set(Property("scene.materials." + name + ".fresnel")(fresnelTex->GetSDLValue()));
	if (n)
		props->Set(Property("scene.materials." + name + ".n")(n->GetSDLValue()));
	if (k)
		props->Set(Property("scene.materials." + name + ".k")(k->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".uroughness")(nu->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".vroughness")(nv->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".multiscattering")(IsMultiscattering()));
	props->Set(Material::ToProperties(imgMapCache, useRealFileName));

	return props;
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
