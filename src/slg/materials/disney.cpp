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

#include "slg/materials/disney.h"
#include "slg/materials/thinfilmcoating.h"
#include "slg/materials/sheen.h"
#include "slg/textures/fresnel/fresneltexture.h"

using namespace std;
using namespace luxrays;
using namespace slg;

DisneyMaterial::DisneyMaterial(
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
) : Material(frontTransp, backTransp, emitted, bump), 
	BaseColor(baseColor), 
	Subsurface(subsurface),
	Roughness(roughness),
	Metallic(metallic),
	Specular(specular),
	SpecularTint(specularTint),
	Clearcoat(clearcoat),
	ClearcoatGloss(clearcoatGloss),
	Anisotropic(anisotropic),
	Sheen(sheen),
	SheenTint(sheenTint),
	filmAmount(filmAmount),
	filmThickness(filmThickness),
	filmIor(filmIor) {
	UpdateGlossiness();
}

void DisneyMaterial::UpdateGlossiness() {
	// Disney glossiness requires a very special care because he material can be
	// matte, glossy or specular and everything in between

	// I first decide if it is going to look like a glossy/specular material
	const float metallicFitler = Metallic->Filter();
	const float specularFitler = Specular->Filter();
	if ((metallicFitler >= .5f) || (specularFitler >= .5f)) {
		// I use the sqrtf() because the difference between Disney microfacet model
		// and Glossy2/Metal/etc. models
		const float g = ComputeGlossiness(Roughness);
		
		if (g > 0.f)
			glossiness = sqrtf(ComputeGlossiness(Roughness));
		else
			glossiness = 0.f;
	} else
		glossiness = 1.f;
}

Spectrum DisneyMaterial::Albedo(const HitPoint &hitPoint) const {
	return BaseColor->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
}

//------------------------------------------------------------------------------
// Since 2026-10-18 the Disney parameters drive an energy-conserving stack of
// lobes, as Cycles' Principled BSDF v2 does, from the top:
//
//  clearcoat  w_cc: GGX with the Schlick Fresnel of 0.04 and Disney's gloss to
//             alpha mapping, energy compensated; the layers beneath are scaled
//             by 1 - w_cc Ecc(mu_eye) (exact over any base, see glossycoating.cpp)
//  metallic   m:  the specular lobe with the F0 of the base color
//  dielectric 1 - m:
//    sheen    s:  the microfiber sheen (Zeltner 2022, roughness 0.5, color
//                 from the sheen tint), the layers beneath scaled by
//                 1 - s albedo(mu_eye) as Cycles does
//    specular     the specular lobe with F0 = specular 0.08 tint
//    diffuse      EON (roughness r), blended with Disney's flattened
//                 subsurface lobe, scaled by the reciprocal albedo layering of
//                 the specular lobe above (see glossy2.cpp)
//
// The specular lobe is GGX with the height-correlated Smith masking and the
// visible-normal sampling of the Microfacet_* layer, energy compensated
// (alpha = roughness^2, the anisotropic split of Disney). A white material has
// the albedo 1 for every combination of metallic, specular, sheen and
// clearcoat. The
// old model added the clearcoat and the sheen on top without taking their
// energy away, used Burley's diffuse (no energy conservation) and a GTR1
// clearcoat; the sheen above 1 is clamped now.
//------------------------------------------------------------------------------

namespace {

const float DISNEY_SHEEN_ROUGHNESS = .5f;
const float DISNEY_CLEARCOAT_F0 = .04f;

}

struct DisneyMaterial::Params {
	Spectrum color;
	float subsurface, roughness, metallic;
	// The specular alphas and the dielectric specular F0
	float ax, ay;
	Spectrum F0d;
	float clearcoat, alphaCC;
	float sheen;
	Spectrum sheenColor;
	float filmAmount, filmThickness, filmIor;
	// The luminance weights of the working colour space (the lobe
	// probabilities don't depend on the working space)
	Spectrum lumWeights;
};

void DisneyMaterial::GetParams(const HitPoint &hitPoint, Params &p) const {
	p.color = BaseColor->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
	p.subsurface = Clamp(Subsurface->GetFloatValue(hitPoint), 0.f, 1.f);
	p.roughness = Disney_RegularizeRoughness(Clamp(Roughness->GetFloatValue(hitPoint), 0.f, 1.f), hitPoint.regularization);
	p.metallic = Clamp(Metallic->GetFloatValue(hitPoint), 0.f, 1.f);

	const float anisotropic = Clamp(Anisotropic->GetFloatValue(hitPoint), 0.f, 1.f);
	const float aspect = sqrtf(1.f - .9f * anisotropic);
	p.ax = Max(.001f, Sqr(p.roughness) / aspect);
	p.ay = Max(.001f, Sqr(p.roughness) * aspect);

	const Spectrum tint = CalculateTint(p.color);
	const float specular = Clamp(Specular->GetFloatValue(hitPoint), 0.f, 1.f);
	const float specularTint = Clamp(SpecularTint->GetFloatValue(hitPoint), 0.f, 1.f);
	p.F0d = (specular * .08f * Lerp(specularTint, Spectrum(1.f), tint)).Clamp(0.f, 1.f);

	p.clearcoat = Clamp(Clearcoat->GetFloatValue(hitPoint), 0.f, 1.f);
	const float clearcoatGloss = Disney_RegularizeClearcoatGloss(Clamp(ClearcoatGloss->GetFloatValue(hitPoint), 0.f, 1.f),
			hitPoint.regularization);
	p.alphaCC = Lerp(clearcoatGloss, .1f, .001f);

	p.sheen = Clamp(Sheen->GetFloatValue(hitPoint), 0.f, 1.f);
	const float sheenTint = Clamp(SheenTint->GetFloatValue(hitPoint), 0.f, 1.f);
	// Clamped in Rec.709, so that the colour doesn't depend on the working space
	p.sheenColor = workingColorSpace.FromRec709(
			workingColorSpace.ToRec709(Lerp(sheenTint, Spectrum(1.f), tint)).Clamp(0.f, 1.f));
	p.lumWeights = workingColorSpace.GetLuminanceWeights();

	p.filmAmount = filmAmount ? Clamp(filmAmount->GetFloatValue(hitPoint), 0.f, 1.f) : 1.f;
	p.filmThickness = filmThickness ? filmThickness->GetFloatValue(hitPoint) : 0.f;
	p.filmIor = (p.filmThickness > 0.f && filmIor) ? filmIor->GetFloatValue(hitPoint) : 1.f;
}

namespace {

// The probabilities of the lobes from the direction with the cosine mu
void Disney_LobeWeights(const DisneyMaterial::Params &p, const float mu,
		float *wCC, float *wSpec, float *wSheen, float *wDiff, float *sheenA, float *sheenB) {
	const float ccE = p.clearcoat * GGXEnergy_CoatE(p.alphaCC, p.alphaCC, mu, Spectrum(DISNEY_CLEARCOAT_F0)).c[0];
	const float rest = 1.f - ccE;

	const Spectrum EspecM = GGXEnergy_CoatE(p.ax, p.ay, mu, p.color);
	const Spectrum EspecD = GGXEnergy_CoatE(p.ax, p.ay, mu, p.F0d);

	float albedoSheen = 0.f;
	*sheenA = 0.f;
	*sheenB = 0.f;
	if (p.sheen > 0.f) {
		Sheen_Lookup(mu, DISNEY_SHEEN_ROUGHNESS, sheenA, sheenB, &albedoSheen);
		if ((fabsf(*sheenA) < 1e-5f) || (albedoSheen < 1e-5f))
			albedoSheen = 0.f;
	}
	const float sheenW = p.sheen * albedoSheen;

	// By luminance: the same choices in every working colour space (the
	// diffuse weight is (1 - Y(Espec)) Y(color), not Y((1 - Espec) color))
	const Spectrum &w = p.lumWeights;
	auto Y = [&w](const Spectrum &c) { return Max(0.f, w.c[0] * c.c[0] + w.c[1] * c.c[1] + w.c[2] * c.c[2]); };
	*wCC = ccE;
	*wSpec = rest * (p.metallic * Y(EspecM) + (1.f - p.metallic) * (1.f - sheenW) * Y(EspecD));
	*wSheen = rest * (1.f - p.metallic) * sheenW * Y(p.sheenColor);
	*wDiff = rest * (1.f - p.metallic) * (1.f - sheenW) * Max(0.f, 1.f - Y(EspecD)) * Y(p.color);

	const float sum = *wCC + *wSpec + *wSheen + *wDiff;
	if (sum > 0.f) {
		const float invSum = 1.f / sum;
		*wCC *= invSum;
		*wSpec *= invSum;
		*wSheen *= invSum;
		*wDiff *= invSum;
	}
}

// Disney's flattened subsurface lobe (Hanrahan-Krueger approximation)
Spectrum Disney_Subsurface(const Spectrum &color, const float roughness,
		const float NdotL, const float NdotV, const float LdotH) {
	const float FL = powf(1.f - NdotL, 5.f);
	const float FV = powf(1.f - NdotV, 5.f);
	const float Fss90 = LdotH * LdotH * roughness;
	const float Fss = Lerp(FL, 1.f, Fss90) * Lerp(FV, 1.f, Fss90);
	const float ss = 1.25f * (Fss * (1.f / (NdotL + NdotV) - .5f) + .5f);

	return INV_PI * ss * color;
}

}

Spectrum DisneyMaterial::EvaluateLayers(const Params &p, const bool fromLight,
		const Vector &localLightDir, const Vector &localEyeDir) const {
	if (localLightDir.z * localEyeDir.z <= 0.f)
		return Spectrum();

	// Both faces: the back face is the mirror image of the front one
	const float flip = (localEyeDir.z < 0.f) ? -1.f : 1.f;
	const Vector lightDir(localLightDir.x, localLightDir.y, flip * localLightDir.z);
	const Vector eyeDir(localEyeDir.x, localEyeDir.y, flip * localEyeDir.z);
	const float muL = lightDir.z;
	const float muE = eyeDir.z;
	if ((muL < DEFAULT_COS_EPSILON_STATIC) || (muE < DEFAULT_COS_EPSILON_STATIC))
		return Spectrum();

	const Vector &fixedDir = fromLight ? lightDir : eyeDir;
	const Vector &sampledDir = fromLight ? eyeDir : lightDir;

	// The specular lobes (f |cos light|, energy compensated), with the thin film
	Spectrum specM = SchlickBSDF_CoatingF(fromLight, p.color, p.ax, p.ay, fixedDir, sampledDir);
	Spectrum specD = SchlickBSDF_CoatingF(fromLight, p.F0d, p.ax, p.ay, fixedDir, sampledDir);
	if (p.filmThickness > 0.f) {
		// The film over the dielectric (the IOR of its mean F0) and over the
		// metal (the complex IOR of its F0, white at the edge as Schlick's)
		const float sqrtF0d = sqrtf(Clamp((p.F0d.c[0] + p.F0d.c[1] + p.F0d.c[2]) / 3.f, 0.f, .99f));
		const float iorD = (1.f + sqrtF0d) / (1.f - sqrtF0d);
		Spectrum etaM, kM;
		FilmConductorIOR(workingColorSpace.ToRec709(p.color), Spectrum(1.f), &etaM, &kM);
		const Spectrum specMFilm = specM * CalcFilmColorConductor(eyeDir, p.filmThickness, p.filmIor,
				1.f, etaM, kM, workingColorSpace);
		const Spectrum specDFilm = specD * CalcFilmColor(eyeDir, p.filmThickness, p.filmIor,
				1.f, iorD, workingColorSpace);
		specM = Lerp(p.filmAmount, specM, specMFilm);
		specD = Lerp(p.filmAmount, specD, specDFilm);
	}

	// The diffuse lobe under the dielectric specular
	Spectrum diffuse = EON_Evaluate(p.color, p.roughness, lightDir, eyeDir) * muL;
	if (p.subsurface > 0.f) {
		const Vector H = Normalize(lightDir + eyeDir);
		const Spectrum flattened = Disney_Subsurface(p.color, p.roughness, muL, muE, Dot(lightDir, H)) * muL;
		diffuse = Lerp(p.subsurface, diffuse, flattened);
	}
	const Spectrum EdLight = GGXEnergy_CoatE(p.ax, p.ay, muL, p.F0d);
	const Spectrum EdEye = GGXEnergy_CoatE(p.ax, p.ay, muE, p.F0d);
	const Spectrum EdAvg = GGXEnergy_CoatEavg(p.ax, p.ay, p.F0d);
	Spectrum layerD;
	for (u_int c = 0; c < 3; ++c)
		layerD.c[c] = (1.f - EdLight.c[c]) * (1.f - EdEye.c[c]) / Max(1.f - EdAvg.c[c], 1e-4f);

	// The sheen above the dielectric part
	Spectrum sheenF;
	float sheenScale = 1.f;
	if (p.sheen > 0.f) {
		float a, b, albedo;
		Sheen_Lookup(muE, DISNEY_SHEEN_ROUGHNESS, &a, &b, &albedo);
		if ((fabsf(a) >= 1e-5f) && (albedo >= 1e-5f)) {
			sheenF = p.sheen * albedo * Sheen_LTC(a, b, eyeDir, lightDir) * p.sheenColor;
			sheenScale = 1.f - p.sheen * albedo;
		}
	}

	const Spectrum dielectric = sheenF + sheenScale * (specD + layerD * diffuse);
	const Spectrum base = p.metallic * specM + (1.f - p.metallic) * dielectric;

	// The clearcoat above everything
	if (p.clearcoat > 0.f) {
		const Spectrum F0cc(DISNEY_CLEARCOAT_F0);
		const Spectrum ccF = p.clearcoat * SchlickBSDF_CoatingF(fromLight, F0cc, p.alphaCC, p.alphaCC, fixedDir, sampledDir);
		// Albedo scaling by the eye direction, exact for any base (see
		// glossycoating.cpp): the base is not Lambertian here
		const float EccEye = p.clearcoat * GGXEnergy_CoatE(p.alphaCC, p.alphaCC, muE, F0cc).c[0];

		return ccF + (1.f - EccEye) * base;
	}

	return base;
}

float DisneyMaterial::PdfLobes(const Params &p, const bool fromLight,
		const Vector &localFixedDir, const Vector &localSampledDir) const {
	if (localFixedDir.z * localSampledDir.z <= 0.f)
		return 0.f;

	const float flip = (localFixedDir.z < 0.f) ? -1.f : 1.f;
	const Vector fixedDir(localFixedDir.x, localFixedDir.y, flip * localFixedDir.z);
	const Vector sampledDir(localSampledDir.x, localSampledDir.y, flip * localSampledDir.z);

	float wCC, wSpec, wSheen, wDiff, sheenA, sheenB;
	Disney_LobeWeights(p, fixedDir.z, &wCC, &wSpec, &wSheen, &wDiff, &sheenA, &sheenB);

	float pdf = wDiff * sampledDir.z * INV_PI;
	if (wSpec > 0.f)
		pdf += wSpec * Microfacet_PdfReflect(p.ax, p.ay, fixedDir, sampledDir);
	if (wCC > 0.f)
		pdf += wCC * Microfacet_PdfReflect(p.alphaCC, p.alphaCC, fixedDir, sampledDir);
	if (wSheen > 0.f) {
		// Eye paths sample the LTC of the eye direction, light paths the cosine
		pdf += wSheen * (fromLight ? (sampledDir.z * INV_PI) : Sheen_LTC(sheenA, sheenB, fixedDir, sampledDir));
	}

	return pdf;
}

Spectrum DisneyMaterial::Evaluate(const HitPoint &hitPoint,
		const Vector &localLightDir, const Vector &localEyeDir, BSDFEvent *event,
		float *directPdfW, float *reversePdfW) const {
	Params p;
	GetParams(hitPoint, p);

	const Vector &localFixedDir = hitPoint.fromLight ? localLightDir : localEyeDir;
	const Vector &localSampledDir = hitPoint.fromLight ? localEyeDir : localLightDir;
	// The reverse pdf is the one of the other path type generating the fixed
	// direction (the sheen sampling depends on the path type)
	if (directPdfW)
		*directPdfW = PdfLobes(p, hitPoint.fromLight, localFixedDir, localSampledDir);
	if (reversePdfW)
		*reversePdfW = PdfLobes(p, !hitPoint.fromLight, localSampledDir, localFixedDir);

	*event = GLOSSY | REFLECT;

	return EvaluateLayers(p, hitPoint.fromLight, localLightDir, localEyeDir);
}

Spectrum DisneyMaterial::Sample(const HitPoint &hitPoint,
		const Vector &localFixedDir, Vector *localSampledDir,
		const float u0, const float u1, const float passThroughEvent,
		float *pdfW, BSDFEvent *event) const {
	if (fabsf(localFixedDir.z) < DEFAULT_COS_EPSILON_STATIC)
		return Spectrum();

	Params p;
	GetParams(hitPoint, p);

	const float flip = (localFixedDir.z < 0.f) ? -1.f : 1.f;
	const Vector fixedDir(localFixedDir.x, localFixedDir.y, flip * localFixedDir.z);

	float wCC, wSpec, wSheen, wDiff, sheenA, sheenB;
	Disney_LobeWeights(p, fixedDir.z, &wCC, &wSpec, &wSheen, &wDiff, &sheenA, &sheenB);
	if (wCC + wSpec + wSheen + wDiff <= 0.f)
		return Spectrum();

	Vector sampledDir;
	if (passThroughEvent < wCC) {
		if (Microfacet_SampleReflect(p.alphaCC, p.alphaCC, fixedDir, u0, u1, &sampledDir) <= 0.f)
			return Spectrum();
	} else if (passThroughEvent < wCC + wSpec) {
		if (Microfacet_SampleReflect(p.ax, p.ay, fixedDir, u0, u1, &sampledDir) <= 0.f)
			return Spectrum();
	} else if ((passThroughEvent < wCC + wSpec + wSheen) && !hitPoint.fromLight)
		sampledDir = Sheen_SampleLTC(sheenA, sheenB, fixedDir, u0, u1);
	else {
		// The diffuse lobe, and the sheen on light paths
		float pdf;
		sampledDir = CosineSampleHemisphere(u0, u1, &pdf);
	}
	if (sampledDir.z < DEFAULT_COS_EPSILON_STATIC)
		return Spectrum();
	*localSampledDir = Vector(sampledDir.x, sampledDir.y, flip * sampledDir.z);

	// The value and the pdf of the mixture
	*pdfW = PdfLobes(p, hitPoint.fromLight, localFixedDir, *localSampledDir);
	if (!(*pdfW > 0.f))
		return Spectrum();

	const Vector &localLightDir = hitPoint.fromLight ? localFixedDir : *localSampledDir;
	const Vector &localEyeDir = hitPoint.fromLight ? *localSampledDir : localFixedDir;
	*event = GLOSSY | REFLECT;

	return EvaluateLayers(p, hitPoint.fromLight, localLightDir, localEyeDir) / *pdfW;
}

void DisneyMaterial::Pdf(const HitPoint &hitPoint,
		const Vector &localLightDir, const Vector &localEyeDir,
		float *directPdfW, float *reversePdfW) const {
	Params p;
	GetParams(hitPoint, p);

	const Vector &localFixedDir = hitPoint.fromLight ? localLightDir : localEyeDir;
	const Vector &localSampledDir = hitPoint.fromLight ? localEyeDir : localLightDir;
	if (directPdfW)
		*directPdfW = PdfLobes(p, hitPoint.fromLight, localFixedDir, localSampledDir);
	if (reversePdfW)
		*reversePdfW = PdfLobes(p, !hitPoint.fromLight, localSampledDir, localFixedDir);
}

Spectrum DisneyMaterial::CalculateTint(const Spectrum &color) const {
	// The luminance with Disney's weights 0.3/0.6/0.1 (it was 0.3/0.6/1, which
	// dimmed the tint of blue colors); the weights are for Rec.709 colours
	const Spectrum color709 = workingColorSpace.ToRec709(color);
	const float luminance = color709.c[0] * .3f + color709.c[1] * .6f + color709.c[2] * .1f;

	return (luminance > 0.f) ? color * (1.f / luminance) : Spectrum(1.f);
}

PropertiesUPtr DisneyMaterial::ToProperties(const ImageMapCache &imgMapCache, const bool useRealFileName) const {
	auto props = std::make_unique<Properties>();

	const string name = GetName();
	props->Set(Property("scene.materials." + name + ".type")("disney"));
	props->Set(Property("scene.materials." + name + ".basecolor")(BaseColor->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".subsurface")(Subsurface->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".roughness")(Roughness->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".metallic")(Metallic->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".specular")(Specular->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".speculartint")(SpecularTint->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".clearcoat")(Clearcoat->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".clearcoatgloss")(ClearcoatGloss->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".anisotropic")(Anisotropic->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".sheen")(Sheen->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".sheentint")(SheenTint->GetSDLValue()));
	if (filmAmount)
		props->Set(Property("scene.materials." + name + ".filmamount")(filmAmount->GetSDLValue()));
	if (filmThickness)
		props->Set(Property("scene.materials." + name + ".filmthickness")(filmThickness->GetSDLValue()));
	if (filmIor)
		props->Set(Property("scene.materials." + name + ".filmior")(filmIor->GetSDLValue()));
	props->Set(Material::ToProperties(imgMapCache, useRealFileName));

	return props;
}

void DisneyMaterial::UpdateTextureReferences(TextureConstRef oldTex, TextureRef newTex) {
	Material::UpdateTextureReferences(oldTex, newTex);

	bool updateGlossiness = false;

	if (BaseColor == &oldTex) BaseColor = &newTex;
	if (Subsurface == &oldTex) Subsurface = &newTex;
	if (Roughness == &oldTex) {
		Roughness = &newTex;
		updateGlossiness = true;
	}
	if (Metallic == &oldTex) Metallic = &newTex;
	if (Specular == &oldTex) Specular = &newTex;
	if (SpecularTint == &oldTex) SpecularTint = &newTex;
	if (Clearcoat == &oldTex) Clearcoat = &newTex;
	if (ClearcoatGloss == &oldTex) ClearcoatGloss = &newTex;
	if (Anisotropic == &oldTex) Anisotropic = &newTex;
	if (Sheen == &oldTex) Sheen = &newTex;
	if (SheenTint == &oldTex) SheenTint = &newTex;
	if (filmAmount == &oldTex) filmAmount = &newTex;
	if (filmThickness == &oldTex) filmThickness = &newTex;
	if (filmIor == &oldTex) filmIor = &newTex;

	if (updateGlossiness)
		UpdateGlossiness();
}

void DisneyMaterial::AddReferencedTextures(std::unordered_set<const Texture *>  &referencedTexs) const {
	Material::AddReferencedTextures(referencedTexs);

	BaseColor->AddReferencedTextures(referencedTexs);
	Subsurface->AddReferencedTextures(referencedTexs);
	Roughness->AddReferencedTextures(referencedTexs);
	Metallic->AddReferencedTextures(referencedTexs);
	Specular->AddReferencedTextures(referencedTexs);
	SpecularTint->AddReferencedTextures(referencedTexs);
	Clearcoat->AddReferencedTextures(referencedTexs);
	ClearcoatGloss->AddReferencedTextures(referencedTexs);
	Anisotropic->AddReferencedTextures(referencedTexs);
	Sheen->AddReferencedTextures(referencedTexs);
	SheenTint->AddReferencedTextures(referencedTexs);
	if (filmAmount)
		filmAmount->AddReferencedTextures(referencedTexs);
	if (filmThickness)
		filmThickness->AddReferencedTextures(referencedTexs);
	if (filmIor)
		filmIor->AddReferencedTextures(referencedTexs);
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
