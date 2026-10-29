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
// other side over the eye side as relative IOR), as in Cycles. Rough glass
// always has it; the regularized glass (glass.cpp) turns it off.
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
	if (directPdfW)
		*directPdfW = 0.f;
	if (reversePdfW)
		*reversePdfW = 0.f;

	const bool isKtBlack = kt.Black();
	const bool isKrBlack = kr.Black();
	if (isKtBlack && isKrBlack)
		return Spectrum();

	const float ntc = nt / nc;

	const float threshold = isKrBlack ? 1.f : (isKtBlack ? 0.f : .5f);
	const float msScale = RoughGlass_MSScale(multiscattering, u, v, localEyeDir, ntc);
	if (localLightDir.z * localEyeDir.z < 0.f) {
		// Transmit

		// eta = eta_light / eta_eye (the IOR of the light side over the one
		// of the eye side)
		const bool entering = (CosTheta(localLightDir) > 0.f);
		const float eta = entering ? (nc / nt) : ntc;

		Vector wh = eta * localLightDir + localEyeDir;
		if (wh.z < 0.f)
			wh = -wh;

		const float lengthSquared = wh.LengthSquared();
		if (!(lengthSquared > 0.f))
			return Spectrum();
		wh /= sqrtf(lengthSquared);

		// Discard the back-facing microfacets (as pbrt-v4's DielectricBxDF):
		// the pair can not refract through wh, and Sample() never generates
		// it. Without this, light sampling added energy that BSDF sampling
		// could not produce (white furnace above 1 at high roughness).
		if ((Dot(localEyeDir, wh) * localEyeDir.z <= 0.f) ||
				(Dot(localLightDir, wh) * localLightDir.z <= 0.f))
			return Spectrum();
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

		// The radiance form f_rad |cos light|, on light paths too: the path
		// integral has the radiance BTDF of the physical pair (light to eye)
		// at every vertex, whichever side sampled it. Only a delta refraction
		// differs between the two (the Jacobian of its delta, see
		// GlassMaterial), not a rough one (as pbrt's MicrofacetTransmission,
		// equal in its Radiance and Importance modes)
		Spectrum result = (fabsf(cosThetaOH) * cosThetaIH * D *
			G / (cosThetaI * lengthSquared)) *
			kt * (1.f - F) * msScale;
		if (localFilmThickness > 0.f) {
			// The film of the microfacet, seen from the light side: a lossless
			// film transmits what it does not reflect (the same from both sides)
			const Spectrum filmColor = CalcFilmColor(Vector(0.f, 0.f, cosThetaOH), localFilmThickness, localFilmIor,
					entering ? nc : nt, entering ? nt : nc, workingColorSpace);
			result *= FilmTransmissionScale(F, filmColor);
		}

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
			// The film of the microfacet, where F is evaluated too
			const bool outside = (CosTheta(localEyeDir) > 0.f);
			return result * CalcFilmColor(Vector(0.f, 0.f, cosThetaH), localFilmThickness, localFilmIor,
					outside ? nc : nt, outside ? nt : nc, workingColorSpace);
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

	// A visible microfacet normal (only the direction is used here, the pdf
	// comes from EvaluateRough() below)
	Vector wh;
	float d, specPdf;
	Microfacet_SampleH(u, v, localFixedDir, u0, u1, &wh, &d, &specPdf);
	if (wh.z < 0.f)
		wh = -wh;
	const float cosThetaOH = Dot(localFixedDir, wh);

	const float ntc = nt / nc;

	// Decide to transmit or reflect (the same probabilities as EvaluateRough())
	const float threshold = isKrBlack ? 1.f : (isKtBlack ? 0.f : .5f);

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

		// A refraction through a steep microfacet can point back into the
		// side it came from: that is not a transmission (and Evaluate()
		// would see a reflection), so it is rejected (as pbrt-v4 does)
		if (localFixedDir.z * localSampledDir->z >= 0.f)
			return Spectrum();
	} else {
		// Reflect
		*localSampledDir = 2.f * cosThetaOH * wh - localFixedDir;

		if (localFixedDir.z * localSampledDir->z <= 0.f)
			return Spectrum();
	}
	if (fabsf(localSampledDir->z) < DEFAULT_COS_EPSILON_STATIC)
		return Spectrum();

	// The value and the pdf from the evaluation: the pdf of the lobe choice
	// times the VNDF and the Jacobian of the reflection or the refraction is
	// what EvaluateRough() returns as directPdfW, and the value is then
	// consistent on eye and light paths by construction
	const Vector &localLightDir = hitPoint.fromLight ? localFixedDir : *localSampledDir;
	const Vector &localEyeDir = hitPoint.fromLight ? *localSampledDir : localFixedDir;
	const Spectrum f = EvaluateRough(hitPoint, localLightDir, localEyeDir, event, pdfW, nullptr,
			kt, kr, nc, nt, u, v, multiscattering, localFilmThickness, localFilmIor, workingColorSpace);
	if (f.Black() || !(*pdfW > 0.f))
		return Spectrum();

	return f / *pdfW;
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

	// The same pdfs as EvaluateRough(): the direct one is the pdf of
	// generating the sampled direction from the fixed one (the fixed direction
	// is the light one on light paths)
	const Vector &localFixedDir = hitPoint.fromLight ? localLightDir : localEyeDir;
	const Vector &localSampledDir = hitPoint.fromLight ? localEyeDir : localLightDir;

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

		// Back-facing microfacets (see EvaluateRough())
		if ((Dot(localEyeDir, wh) * localEyeDir.z <= 0.f) ||
				(Dot(localLightDir, wh) * localLightDir.z <= 0.f))
			return;

		const float cosThetaIH = AbsDot(localEyeDir, wh);
		const float cosThetaOH = AbsDot(localLightDir, wh);

		if (directPdfW)
			*directPdfW = threshold * Microfacet_PdfH(u, v, localFixedDir, wh) *
					(hitPoint.fromLight ? cosThetaIH : (cosThetaOH * eta * eta)) / lengthSquared;

		if (reversePdfW)
			*reversePdfW = threshold * Microfacet_PdfH(u, v, localSampledDir, wh) *
					(hitPoint.fromLight ? (cosThetaOH * eta * eta) : cosThetaIH) / lengthSquared;
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
			*directPdfW = (1.f - threshold) * Microfacet_PdfH(u, v, localFixedDir, wh) / (4.f * AbsDot(localLightDir, wh));

		if (reversePdfW)
			*reversePdfW = (1.f - threshold) * Microfacet_PdfH(u, v, localSampledDir, wh) / (4.f * AbsDot(localLightDir, wh));
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
	const float u = Microfacet_Regularize(Clamp(nu->GetFloatValue(hitPoint), MICROFACET_MIN_ROUGHNESS, 1.f), hitPoint.regularization);
	const float v = Microfacet_Regularize(Clamp(nv->GetFloatValue(hitPoint), MICROFACET_MIN_ROUGHNESS, 1.f), hitPoint.regularization);
	const float localFilmThickness = filmThickness ? filmThickness->GetFloatValue(hitPoint) : 0.f;
	const float localFilmIor = ((localFilmThickness > 0.f) && filmIor) ? filmIor->GetFloatValue(hitPoint) : 1.f;

	return EvaluateRough(hitPoint, localLightDir, localEyeDir, event, directPdfW, reversePdfW,
			kt, kr, nc, nt, u, v, true, localFilmThickness, localFilmIor, workingColorSpace);
}

Spectrum RoughGlassMaterial::Sample(const HitPoint &hitPoint,
		const Vector &localFixedDir, Vector *localSampledDir,
		const float u0, const float u1, const float passThroughEvent,
		float *pdfW, BSDFEvent *event) const {
	const Spectrum kt = Kt->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
	const Spectrum kr = Kr->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
	const float nc = ExtractExteriorIors(hitPoint, exteriorIor);
	const float nt = ExtractInteriorIors(hitPoint, interiorIor);
	const float u = Microfacet_Regularize(Clamp(nu->GetFloatValue(hitPoint), MICROFACET_MIN_ROUGHNESS, 1.f), hitPoint.regularization);
	const float v = Microfacet_Regularize(Clamp(nv->GetFloatValue(hitPoint), MICROFACET_MIN_ROUGHNESS, 1.f), hitPoint.regularization);
	const float localFilmThickness = filmThickness ? filmThickness->GetFloatValue(hitPoint) : 0.f;
	const float localFilmIor = ((localFilmThickness > 0.f) && filmIor) ? filmIor->GetFloatValue(hitPoint) : 1.f;

	return SampleRough(hitPoint, localFixedDir, localSampledDir, u0, u1, passThroughEvent, pdfW, event,
			kt, kr, nc, nt, u, v, true, localFilmThickness, localFilmIor, workingColorSpace);
}

void RoughGlassMaterial::Pdf(const HitPoint &hitPoint,
		const Vector &localLightDir, const Vector &localEyeDir,
		float *directPdfW, float *reversePdfW) const {
	const Spectrum kt = Kt->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
	const Spectrum kr = Kr->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
	const float nc = ExtractExteriorIors(hitPoint, exteriorIor);
	const float nt = ExtractInteriorIors(hitPoint, interiorIor);
	const float u = Microfacet_Regularize(Clamp(nu->GetFloatValue(hitPoint), MICROFACET_MIN_ROUGHNESS, 1.f), hitPoint.regularization);
	const float v = Microfacet_Regularize(Clamp(nv->GetFloatValue(hitPoint), MICROFACET_MIN_ROUGHNESS, 1.f), hitPoint.regularization);

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
	if (filmThickness)
		props->Set(Property("scene.materials." + name + ".filmthickness")(filmThickness->GetSDLValue()));
	if (filmIor)
		props->Set(Property("scene.materials." + name + ".filmior")(filmIor->GetSDLValue()));
	props->Set(Material::ToProperties(imgMapCache, useRealFileName));

	return props;
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
