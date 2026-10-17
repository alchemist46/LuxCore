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

#include "slg/textures/fresnel/fresneltexture.h"
#include "slg/materials/roughglass.h"
#include "slg/materials/thinfilmcoating.h"
#include "slg/usings.h"

using namespace std;
using namespace luxrays;
using namespace slg;

//------------------------------------------------------------------------------
// RoughGlass material
//
// LuxRender RoughGlass material porting.
//------------------------------------------------------------------------------

RoughGlassMaterial::RoughGlassMaterial(TextureConstPtr frontTransp, TextureConstPtr backTransp,
		TextureConstPtr emitted, TextureConstPtr bump,
		TextureConstPtr refl, TextureConstPtr trans,
		TextureConstPtr exteriorIorFact, TextureConstPtr interiorIorFact,
		TextureConstPtr u, TextureConstPtr v,
		TextureConstPtr filmThickness, TextureConstPtr filmIor) :
			Material(frontTransp, backTransp, emitted, bump), Kr(refl), Kt(trans),
			exteriorIor(exteriorIorFact), interiorIor(interiorIorFact), nu(u), nv(v),
			filmThickness(filmThickness), filmIor(filmIor) {
	glossiness = ComputeGlossiness(nu, nv);
}

// The multiple-scattering scale of the whole interface seen from localEyeDir:
// 1 / E(mu, alpha, eta) with the albedo of the rough dielectric interface (the
// other side over the eye side as relative IOR), as in Cycles
static float RoughGlass_MSScale(const bool multiscattering,
		const float u, const float v, const Vector &localEyeDir, const float ntc) {
	if (!multiscattering)
		return 1.f;

	const float eta = (localEyeDir.z > 0.f) ? ntc : 1.f / ntc;

	return 1.f / Max(GGXEnergy_GlassE(u, v, fabsf(localEyeDir.z), eta), 1e-3f);
}

Spectrum RoughGlassMaterial::EvaluateRough(const HitPoint &hitPoint,
	const Vector &localLightDir, const Vector &localEyeDir, BSDFEvent *event,
	float *directPdfW, float *reversePdfW,
	const Spectrum &kt, const Spectrum &kr, const float nc, const float nt,
	const float u, const float v, const bool multiscattering,
	const float localFilmThickness, const float localFilmIor,
	const WorkingColorSpace &workingColorSpace) {
	const bool isKtBlack = kt.Black();
	const bool isKrBlack = kr.Black();
	if (isKtBlack && isKrBlack)
		return Spectrum();

	const float ntc = nt / nc;

	const float threshold = isKrBlack ? 1.f : (isKtBlack ? 0.f : .5f);
	const float msScale = RoughGlass_MSScale(multiscattering, u, v, localEyeDir, ntc);
	if (localLightDir.z * localEyeDir.z < 0.f) {
		// Transmit

		const bool entering = (CosTheta(localLightDir) > 0.f);
		const float eta = entering ? (nc / nt) : ntc;

		Vector wh = eta * localLightDir + localEyeDir;
		if (wh.z < 0.f)
			wh = -wh;

		const float lengthSquared = wh.LengthSquared();
		if (!(lengthSquared > 0.f))
			return Spectrum();
		wh /= sqrtf(lengthSquared);
		const float cosThetaI = fabsf(CosTheta(localEyeDir));
		const float cosThetaIH = AbsDot(localEyeDir, wh);
		const float cosThetaOH = Dot(localLightDir, wh);

		const float D = Microfacet_D(u, v, wh);
		const float G = Microfacet_G(u, v, localLightDir, localEyeDir);
		const float F = FresnelTexture::CauchyEvaluate(ntc, cosThetaOH);

		const Vector &localFixedDir = hitPoint.fromLight ? localLightDir : localEyeDir;
		const Vector &localSampledDir = hitPoint.fromLight ? localEyeDir : localLightDir;
		if (directPdfW)
			*directPdfW = threshold * Microfacet_PdfH(u, v, localFixedDir, wh) * (hitPoint.fromLight ? fabsf(cosThetaIH) : (fabsf(cosThetaOH) * eta * eta)) / lengthSquared;

		if (reversePdfW)
			*reversePdfW = threshold * Microfacet_PdfH(u, v, localSampledDir, wh) * (hitPoint.fromLight ? (fabsf(cosThetaOH) * eta * eta) : fabsf(cosThetaIH)) / lengthSquared;

		const Spectrum result = (fabsf(cosThetaOH) * cosThetaIH * D *
			G / (cosThetaI * lengthSquared)) *
			kt * (1.f - F) * msScale;

		*event = GLOSSY | TRANSMIT;

		return result;
	} else {
		// Reflect
		const float cosThetaO = fabsf(CosTheta(localLightDir));
		const float cosThetaI = fabsf(CosTheta(localEyeDir));
		if (cosThetaO == 0.f || cosThetaI == 0.f)
			return Spectrum();
		Vector wh = localLightDir + localEyeDir;
		if (wh == Vector(0.f))
			return Spectrum();
		wh = Normalize(wh);
		if (wh.z < 0.f)
			wh = -wh;

		float cosThetaH = Dot(localEyeDir, wh);
		const float D = Microfacet_D(u, v, wh);
		const float G = Microfacet_G(u, v, localLightDir, localEyeDir);
		const float F = FresnelTexture::CauchyEvaluate(ntc, cosThetaH);

		const Vector &localFixedDir = hitPoint.fromLight ? localLightDir : localEyeDir;
		const Vector &localSampledDir = hitPoint.fromLight ? localEyeDir : localLightDir;
		if (directPdfW)
			*directPdfW = (1.f - threshold) * Microfacet_PdfH(u, v, localFixedDir, wh) / (4.f * AbsDot(localLightDir, wh));

		if (reversePdfW)
			*reversePdfW = (1.f - threshold) * Microfacet_PdfH(u, v, localSampledDir, wh) / (4.f * AbsDot(localLightDir, wh));

		const Spectrum result = (D * G / (4.f * cosThetaI)) * kr * F * msScale;

		*event = GLOSSY | REFLECT;
		
		if (localFilmThickness > 0.f) {
			return result * CalcFilmColor(localEyeDir, localFilmThickness, localFilmIor, workingColorSpace);
		}
		return result;
	}
}

bool RoughGlassMaterial::GetRefractionIors(const HitPoint &hitPoint, float *nc, float *nt) const {
	*nc = ExtractExteriorIors(hitPoint, exteriorIor);
	*nt = ExtractInteriorIors(hitPoint, interiorIor);

	return true;
}

Spectrum RoughGlassMaterial::SampleRough(const HitPoint &hitPoint,
		const Vector &localFixedDir, Vector *localSampledDir,
		const float u0, const float u1, const float passThroughEvent,
		float *pdfW, BSDFEvent *event,
		const Spectrum &kt, const Spectrum &kr, const float nc, const float nt,
		const float u, const float v, const bool multiscattering,
		const float localFilmThickness, const float localFilmIor,
		const WorkingColorSpace &workingColorSpace) {
	if (fabsf(localFixedDir.z) < DEFAULT_COS_EPSILON_STATIC)
		return Spectrum();

	const bool isKtBlack = kt.Black();
	const bool isKrBlack = kr.Black();
	if (isKtBlack && isKrBlack)
		return Spectrum();

	Vector wh;
	float d, specPdf;
	Microfacet_SampleH(u, v, localFixedDir, u0, u1, &wh, &d, &specPdf);
	if (wh.z < 0.f)
		wh = -wh;
	const float cosThetaOH = Dot(localFixedDir, wh);

	const float ntc = nt / nc;

	const float coso = fabsf(localFixedDir.z);

	// Decide to transmit or reflect
	float threshold;
	if (!isKrBlack) {
		if (!isKtBlack)
			threshold = .5f;
		else
			threshold = 0.f;
	} else {
		if (!isKtBlack)
			threshold = 1.f;
		else
			return Spectrum();
	}

	Spectrum result;
	if (passThroughEvent < threshold) {
		// Transmit

		const bool entering = (CosTheta(localFixedDir) > 0.f);
		const float eta = entering ? (nc / nt) : ntc;
		const float eta2 = eta * eta;
		const float sinThetaIH2 = eta2 * Max(0.f, 1.f - cosThetaOH * cosThetaOH);
		if (sinThetaIH2 >= 1.f)
			return Spectrum();
		float cosThetaIH = sqrtf(1.f - sinThetaIH2);
		if (entering)
			cosThetaIH = -cosThetaIH;
		const float length = eta * cosThetaOH + cosThetaIH;
		*localSampledDir = length * wh - eta * localFixedDir;

		const float lengthSquared = length * length;
		*pdfW = specPdf * fabsf(cosThetaIH) / lengthSquared;
		if (*pdfW <= 0.f)
			return Spectrum();

		const float cosi = fabsf(localSampledDir->z);

		const float G = Microfacet_G(u, v, localFixedDir, *localSampledDir);
		float factor = (d / specPdf) * G * fabsf(cosThetaOH) / threshold;

		if (!hitPoint.fromLight) {
			const float F = FresnelTexture::CauchyEvaluate(ntc, cosThetaIH);
			result = (factor / coso) * kt * (1.f - F);
		} else {
			const float F = FresnelTexture::CauchyEvaluate(ntc, cosThetaOH);
			result = (factor / cosi) * kt * (1.f - F);
		}

		*pdfW *= threshold;
		*event = GLOSSY | TRANSMIT;
	} else {
		// Reflect
		*pdfW = specPdf / (4.f * fabsf(cosThetaOH));
		if (*pdfW <= 0.f)
			return Spectrum();

		*localSampledDir = 2.f * cosThetaOH * wh - localFixedDir;

		const float cosi = fabsf(localSampledDir->z);
		if ((cosi < DEFAULT_COS_EPSILON_STATIC) || (localFixedDir.z * localSampledDir->z < 0.f))
			return Spectrum();

		const float G = Microfacet_G(u, v, localFixedDir, *localSampledDir);
		float factor = (d / specPdf) * G * fabsf(cosThetaOH) / (1.f - threshold);

		const float F = FresnelTexture::CauchyEvaluate(ntc, cosThetaOH);
		factor /= (!hitPoint.fromLight) ? coso : cosi;
		result = factor * F * kr;
		
		if (localFilmThickness > 0.f) {
			result *= CalcFilmColor(localFixedDir, localFilmThickness, localFilmIor, workingColorSpace);
		}

		*pdfW *= (1.f - threshold);
		*event = GLOSSY | REFLECT;
	}

	return result * RoughGlass_MSScale(multiscattering, u, v,
			hitPoint.fromLight ? *localSampledDir : localFixedDir, ntc);
}

void RoughGlassMaterial::PdfRough(const HitPoint &hitPoint,
		const Vector &localLightDir, const Vector &localEyeDir,
		float *directPdfW, float *reversePdfW,
		const Spectrum &kt, const Spectrum &kr, const float nc, const float nt,
		const float u, const float v) {
	if (directPdfW)
		*directPdfW = 0.f;
	if (reversePdfW)
		*reversePdfW = 0.f;

	const bool isKtBlack = kt.Black();
	const bool isKrBlack = kr.Black();
	if (isKtBlack && isKrBlack)
		return;

	const float ntc = nt / nc;

	const float threshold = isKrBlack ? 1.f : (isKtBlack ? 0.f : .5f);
	if (localLightDir.z * localEyeDir.z < 0.f) {
		// Transmit

		const bool entering = (CosTheta(localLightDir) > 0.f);
		const float eta = entering ? (nc / nt) : ntc;

		Vector wh = eta * localLightDir + localEyeDir;
		if (wh.z < 0.f)
			wh = -wh;

		const float lengthSquared = wh.LengthSquared();
		if (!(lengthSquared > 0.f))
			return;

		wh /= sqrtf(lengthSquared);
		const float cosThetaIH = AbsDot(localEyeDir, wh);
		const float cosThetaOH = AbsDot(localLightDir, wh);

		if (directPdfW)
			*directPdfW = threshold * Microfacet_PdfH(u, v, localLightDir, wh) * cosThetaIH / lengthSquared;

		if (reversePdfW)
			*reversePdfW = threshold * Microfacet_PdfH(u, v, localEyeDir, wh) * cosThetaOH * eta * eta / lengthSquared;
	} else {
		// Reflect
		const float cosThetaO = fabsf(CosTheta(localLightDir));
		const float cosThetaI = fabsf(CosTheta(localEyeDir));
		if (cosThetaO == 0.f || cosThetaI == 0.f)
			return;

		Vector wh = localLightDir + localEyeDir;
		if (wh == Vector(0.f))
			return;
		wh = Normalize(wh);
		if (wh.z < 0.f)
			wh = -wh;

		if (directPdfW)
			*directPdfW = (1.f - threshold) * Microfacet_PdfH(u, v, localLightDir, wh) / (4.f * AbsDot(localLightDir, wh));

		if (reversePdfW)
			*reversePdfW = (1.f - threshold) * Microfacet_PdfH(u, v, localEyeDir, wh) / (4.f * AbsDot(localLightDir, wh));
	}
}

//------------------------------------------------------------------------------
// The material parameters at the hit point
//------------------------------------------------------------------------------

Spectrum RoughGlassMaterial::Evaluate(const HitPoint &hitPoint,
	const Vector &localLightDir, const Vector &localEyeDir, BSDFEvent *event,
	float *directPdfW, float *reversePdfW) const {
	const Spectrum kt = Kt->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
	const Spectrum kr = Kr->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
	const float nc = ExtractExteriorIors(hitPoint, exteriorIor);
	const float nt = ExtractInteriorIors(hitPoint, interiorIor);
	const float u = Microfacet_Regularize(Clamp(nu->GetFloatValue(hitPoint), 1e-9f, 1.f), hitPoint.regularization);
	const float v = Microfacet_Regularize(Clamp(nv->GetFloatValue(hitPoint), 1e-9f, 1.f), hitPoint.regularization);
	const float localFilmThickness = filmThickness ? filmThickness->GetFloatValue(hitPoint) : 0.f;
	const float localFilmIor = ((localFilmThickness > 0.f) && filmIor) ? filmIor->GetFloatValue(hitPoint) : 1.f;

	return EvaluateRough(hitPoint, localLightDir, localEyeDir, event, directPdfW, reversePdfW,
			kt, kr, nc, nt, u, v, IsMultiscattering(), localFilmThickness, localFilmIor, workingColorSpace);
}

Spectrum RoughGlassMaterial::Sample(const HitPoint &hitPoint,
		const Vector &localFixedDir, Vector *localSampledDir,
		const float u0, const float u1, const float passThroughEvent,
		float *pdfW, BSDFEvent *event) const {
	const Spectrum kt = Kt->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
	const Spectrum kr = Kr->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
	const float nc = ExtractExteriorIors(hitPoint, exteriorIor);
	const float nt = ExtractInteriorIors(hitPoint, interiorIor);
	const float u = Microfacet_Regularize(Clamp(nu->GetFloatValue(hitPoint), 1e-9f, 1.f), hitPoint.regularization);
	const float v = Microfacet_Regularize(Clamp(nv->GetFloatValue(hitPoint), 1e-9f, 1.f), hitPoint.regularization);
	const float localFilmThickness = filmThickness ? filmThickness->GetFloatValue(hitPoint) : 0.f;
	const float localFilmIor = ((localFilmThickness > 0.f) && filmIor) ? filmIor->GetFloatValue(hitPoint) : 1.f;

	return SampleRough(hitPoint, localFixedDir, localSampledDir, u0, u1, passThroughEvent, pdfW, event,
			kt, kr, nc, nt, u, v, IsMultiscattering(), localFilmThickness, localFilmIor, workingColorSpace);
}

void RoughGlassMaterial::Pdf(const HitPoint &hitPoint,
		const Vector &localLightDir, const Vector &localEyeDir,
		float *directPdfW, float *reversePdfW) const {
	const Spectrum kt = Kt->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
	const Spectrum kr = Kr->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
	const float nc = ExtractExteriorIors(hitPoint, exteriorIor);
	const float nt = ExtractInteriorIors(hitPoint, interiorIor);
	const float u = Microfacet_Regularize(Clamp(nu->GetFloatValue(hitPoint), 1e-9f, 1.f), hitPoint.regularization);
	const float v = Microfacet_Regularize(Clamp(nv->GetFloatValue(hitPoint), 1e-9f, 1.f), hitPoint.regularization);

	PdfRough(hitPoint, localLightDir, localEyeDir, directPdfW, reversePdfW, kt, kr, nc, nt, u, v);
}

void RoughGlassMaterial::AddReferencedTextures(std::unordered_set<const Texture *>  &referencedTexs) const {
	Material::AddReferencedTextures(referencedTexs);

	Kr->AddReferencedTextures(referencedTexs);
	Kt->AddReferencedTextures(referencedTexs);
	if (exteriorIor)
		exteriorIor->AddReferencedTextures(referencedTexs);
	if (interiorIor)
		interiorIor->AddReferencedTextures(referencedTexs);
	nu->AddReferencedTextures(referencedTexs);
	nv->AddReferencedTextures(referencedTexs);
	if (filmThickness)
		filmThickness->AddReferencedTextures(referencedTexs);
	if (filmIor)
		filmIor->AddReferencedTextures(referencedTexs);
}

void RoughGlassMaterial::UpdateTextureReferences(TextureConstRef oldTex, TextureRef newTex) {
	Material::UpdateTextureReferences(oldTex, newTex);

	bool updateGlossiness = false;
	if (Kr == &oldTex)
		Kr = &newTex;
	if (Kt == &oldTex)
		Kt = &newTex;
	if (exteriorIor == &oldTex)
		exteriorIor = &newTex;
	if (interiorIor == &oldTex)
		interiorIor = &newTex;
	if (nu == &oldTex) {
		nu = &newTex;
		updateGlossiness = true;
	}
	if (nv == &oldTex) {
		nv = &newTex;
		updateGlossiness = true;
	}
	if (filmThickness == &oldTex)
		filmThickness = &newTex;
	if (filmIor == &oldTex)
		filmIor = &newTex;

	if (updateGlossiness)
		glossiness = ComputeGlossiness(nu, nv);
}

PropertiesUPtr RoughGlassMaterial::ToProperties(const ImageMapCache &imgMapCache, const bool useRealFileName) const  {
	auto props = std::make_unique<Properties>();

	const string name = GetName();
	props->Set(Property("scene.materials." + name + ".type")("roughglass"));
	props->Set(Property("scene.materials." + name + ".kr")(Kr->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".kt")(Kt->GetSDLValue()));
	if (exteriorIor)
		props->Set(Property("scene.materials." + name + ".exteriorior")(exteriorIor->GetSDLValue()));
	if (interiorIor)
		props->Set(Property("scene.materials." + name + ".interiorior")(interiorIor->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".uroughness")(nu->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".vroughness")(nv->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".multiscattering")(IsMultiscattering()));
	if (filmThickness)
		props->Set(Property("scene.materials." + name + ".filmthickness")(filmThickness->GetSDLValue()));
	if (filmIor)
		props->Set(Property("scene.materials." + name + ".filmior")(filmIor->GetSDLValue()));
	props->Set(Material::ToProperties(imgMapCache, useRealFileName));

	return props;
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
