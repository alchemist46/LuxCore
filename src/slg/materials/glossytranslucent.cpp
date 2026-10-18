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
#include "slg/materials/glossytranslucent.h"

using namespace std;
using namespace luxrays;
using namespace slg;

//------------------------------------------------------------------------------
// Glossy Translucent material
//
// LuxRender GlossyTranslucent material porting.
//------------------------------------------------------------------------------

GlossyTranslucentMaterial::GlossyTranslucentMaterial(TextureConstPtr frontTransp, TextureConstPtr backTransp,
		TextureConstPtr emitted, TextureConstPtr bump,
		TextureConstPtr kd, TextureConstPtr kt, TextureConstPtr ks, TextureConstPtr ks2,
		TextureConstPtr u, TextureConstPtr u2, TextureConstPtr v, TextureConstPtr v2,
		TextureConstPtr ka, TextureConstPtr ka2, TextureConstPtr d, TextureConstPtr d2,
		TextureConstPtr i, TextureConstPtr i2, const bool mbounce, const bool mbounce2) :
			Material(frontTransp, backTransp, emitted, bump), Kd(kd), Kt(kt), Ks(ks), Ks_bf(ks2), nu(u), nu_bf(u2),
			nv(v), nv_bf(v2), Ka(ka), Ka_bf(ka2), depth(d), depth_bf(d2), index(i),
			index_bf(i2), multibounce(mbounce), multibounce_bf(mbounce2) {
	glossiness = Min(ComputeGlossiness(nu, nv), ComputeGlossiness(nu_bf, nv_bf));
}

Spectrum GlossyTranslucentMaterial::Albedo(const HitPoint &hitPoint) const {
	return Kd->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
}

//------------------------------------------------------------------------------
// Since 2026-10-18 the two coatings (front: ks, index, uroughness, vroughness,
// ka, d; back: the _bf ones) are layered over the diffuse slab by albedo
// scaling (see glossy2.cpp), with Ec_s the albedo of the energy-compensated
// coating of side s:
//
//  reflection on side s: f_coat_s + kd / pi (1 - Ec_s(mu_light)) (1 - Ec_s(mu_eye)) / (1 - Ec_s_avg)
//  transmission:         t / pi (1 - Ec_l(mu_light)) (1 - Ec_e(mu_eye)) / (1 - Ec_l_avg)
//
// with t = kt (1 - kd), l the side of the light direction and e the one of
// the eye direction, times the absorption of the coatings crossed. The albedo
// seen from the eye is Ec + (kd + t) (1 - Ec): 1 with kd + t = 1. With equal
// coatings on both sides the transmission is reciprocal. The old Schlick
// blend with the sqrt((1 - S1) (1 - S2)) transmission gained up to 26% at
// grazing angles. The multibounce options are ignored.
//------------------------------------------------------------------------------

namespace {

struct GlossyTranslucentSide {
	Spectrum ks, alpha;
	float u, v, d;
};

}

static GlossyTranslucentSide GlossyTranslucent_GetSide(const GlossyTranslucentMaterial &mat,
		const HitPoint &hitPoint, const bool front) {
	GlossyTranslucentSide side;
	side.ks = (front ? mat.GetKs() : mat.GetKs_bf())->GetSpectrumValue(hitPoint);
	const float i = (front ? mat.GetIndex() : mat.GetIndex_bf())->GetFloatValue(hitPoint);
	if (i > 0.f) {
		const float ti = (i - 1.f) / (i + 1.f);
		side.ks *= ti * ti;
	}
	side.ks = side.ks.Clamp(0.f, 1.f);
	side.u = Microfacet_Regularize(Clamp((front ? mat.GetNu() : mat.GetNu_bf())->GetFloatValue(hitPoint), 1e-9f, 1.f), hitPoint.regularization);
	side.v = Microfacet_Regularize(Clamp((front ? mat.GetNv() : mat.GetNv_bf())->GetFloatValue(hitPoint), 1e-9f, 1.f), hitPoint.regularization);
	side.alpha = (front ? mat.GetKa() : mat.GetKa_bf())->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
	side.d = (front ? mat.GetDepth() : mat.GetDepth_bf())->GetFloatValue(hitPoint);

	return side;
}

// The probabilities of the coating, the diffuse reflection and the diffuse
// transmission from a direction with the coating albedo Ec
static void GlossyTranslucent_LobeProbabilities(const Spectrum &Ec, const Spectrum &kd, const Spectrum &t,
		float *pCoat, float *pReflect, float *pTransmit) {
	const Spectrum reflectOut = (Spectrum(1.f) - Ec) * kd;
	const Spectrum transmitOut = (Spectrum(1.f) - Ec) * t;
	const float coat = Ec.Filter();
	const float reflect = reflectOut.Filter();
	const float transmit = transmitOut.Filter();
	const float sum = coat + reflect + transmit;
	if (sum > 0.f) {
		*pCoat = coat / sum;
		*pReflect = reflect / sum;
		*pTransmit = transmit / sum;
	} else {
		*pCoat = 0.f;
		*pReflect = 0.f;
		*pTransmit = 0.f;
	}
}

// The pdf of generating sampledDir from fixedDir
static float GlossyTranslucent_Pdf(const GlossyTranslucentSide &fixedSide, const Spectrum &kd, const Spectrum &t,
		const Vector &localFixedDir, const Vector &localSampledDir) {
	const Spectrum EcFixed = GGXEnergy_CoatE(fixedSide.u, fixedSide.v, fabsf(localFixedDir.z), fixedSide.ks);
	float pCoat, pReflect, pTransmit;
	GlossyTranslucent_LobeProbabilities(EcFixed, kd, t, &pCoat, &pReflect, &pTransmit);

	if (localFixedDir.z * localSampledDir.z > 0.f) {
		return pReflect * fabsf(localSampledDir.z) * INV_PI +
				pCoat * SchlickBSDF_CoatingPdf(fixedSide.u, fixedSide.v, localFixedDir, localSampledDir);
	} else
		return pTransmit * fabsf(localSampledDir.z) * INV_PI;
}

Spectrum GlossyTranslucentMaterial::Evaluate(const HitPoint &hitPoint,
	const Vector &localLightDir, const Vector &localEyeDir, BSDFEvent *event,
	float *directPdfW, float *reversePdfW) const {
	const Vector &localFixedDir = hitPoint.fromLight ? localLightDir : localEyeDir;
	const Vector &localSampledDir = hitPoint.fromLight ? localEyeDir : localLightDir;
	if (directPdfW)
		*directPdfW = 0.f;
	if (reversePdfW)
		*reversePdfW = 0.f;

	const float cosLight = fabsf(localLightDir.z);
	const float cosEye = fabsf(localEyeDir.z);
	if ((cosLight < DEFAULT_COS_EPSILON_STATIC) || (cosEye < DEFAULT_COS_EPSILON_STATIC))
		return Spectrum();

	const Spectrum kd = Kd->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
	const Spectrum t = Kt->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f) * (Spectrum(1.f) - kd);

	const GlossyTranslucentSide lightSide = GlossyTranslucent_GetSide(*this, hitPoint, localLightDir.z > 0.f);
	const GlossyTranslucentSide eyeSide = GlossyTranslucent_GetSide(*this, hitPoint, localEyeDir.z > 0.f);
	const GlossyTranslucentSide &fixedSide = hitPoint.fromLight ? lightSide : eyeSide;
	const GlossyTranslucentSide &sampledSide = hitPoint.fromLight ? eyeSide : lightSide;

	if (directPdfW)
		*directPdfW = GlossyTranslucent_Pdf(fixedSide, kd, t, localFixedDir, localSampledDir);
	if (reversePdfW)
		*reversePdfW = GlossyTranslucent_Pdf(sampledSide, kd, t, localSampledDir, localFixedDir);

	const Spectrum EcLight = GGXEnergy_CoatE(lightSide.u, lightSide.v, cosLight, lightSide.ks);
	const Spectrum EcEye = GGXEnergy_CoatE(eyeSide.u, eyeSide.v, cosEye, eyeSide.ks);
	const Spectrum EcLightAvg = GGXEnergy_CoatEavg(lightSide.u, lightSide.v, lightSide.ks);

	Spectrum layer;
	for (u_int c = 0; c < 3; ++c)
		layer.c[c] = (1.f - EcLight.c[c]) * (1.f - EcEye.c[c]) / Max(1.f - EcLightAvg.c[c], 1e-4f);

	if (localLightDir.z * localEyeDir.z < 0.f) {
		// Transmission: one crossing of each coating
		*event = DIFFUSE | TRANSMIT;

		const Spectrum absorption = Exp(lightSide.alpha * -(lightSide.d / cosLight) +
				eyeSide.alpha * -(eyeSide.d / cosEye));

		return absorption * layer * t * (INV_PI * cosLight);
	} else {
		// Reflection on the side of both directions
		*event = GLOSSY | REFLECT;

		const Spectrum absorption = CoatingAbsorption(cosLight, cosEye, eyeSide.alpha, eyeSide.d);
		const Spectrum coatingF = SchlickBSDF_CoatingF(hitPoint.fromLight, eyeSide.ks, eyeSide.u, eyeSide.v, localFixedDir, localSampledDir);

		return coatingF + absorption * layer * kd * (INV_PI * cosLight);
	}
}

Spectrum GlossyTranslucentMaterial::Sample(const HitPoint &hitPoint,
	const Vector &localFixedDir, Vector *localSampledDir,
	const float u0, const float u1, const float passThroughEvent,
	float *pdfW, BSDFEvent *event) const {
	if (fabsf(localFixedDir.z) < DEFAULT_COS_EPSILON_STATIC)
		return Spectrum();

	const Spectrum kd = Kd->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
	const Spectrum t = Kt->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f) * (Spectrum(1.f) - kd);
	const GlossyTranslucentSide fixedSide = GlossyTranslucent_GetSide(*this, hitPoint, localFixedDir.z > 0.f);

	float pCoat, pReflect, pTransmit;
	GlossyTranslucent_LobeProbabilities(GGXEnergy_CoatE(fixedSide.u, fixedSide.v, fabsf(localFixedDir.z), fixedSide.ks),
			kd, t, &pCoat, &pReflect, &pTransmit);
	if (pCoat + pReflect + pTransmit <= 0.f)
		return Spectrum();

	if (passThroughEvent < pCoat) {
		const float pdf = Microfacet_SampleReflect(fixedSide.u, fixedSide.v, localFixedDir, u0, u1, localSampledDir);
		if ((pdf <= 0.f) || (localFixedDir.z * localSampledDir->z <= 0.f))
			return Spectrum();
	} else {
		float pdf;
		*localSampledDir = CosineSampleHemisphere(u0, u1, &pdf);
		// A diffuse transmission keeps the azimuth: only z flips
		if (passThroughEvent < pCoat + pReflect)
			localSampledDir->z *= Sgn(localFixedDir.z);
		else
			localSampledDir->z *= -Sgn(localFixedDir.z);
	}
	if (fabsf(localSampledDir->z) < DEFAULT_COS_EPSILON_STATIC)
		return Spectrum();

	// The value and the pdf of the mixture from the evaluation
	const Vector &localLightDir = hitPoint.fromLight ? localFixedDir : *localSampledDir;
	const Vector &localEyeDir = hitPoint.fromLight ? *localSampledDir : localFixedDir;
	const Spectrum f = Evaluate(hitPoint, localLightDir, localEyeDir, event, pdfW, nullptr);
	if (f.Black() || !(*pdfW > 0.f))
		return Spectrum();

	return f / *pdfW;
}

void GlossyTranslucentMaterial::Pdf(const HitPoint &hitPoint,
		const Vector &localLightDir, const Vector &localEyeDir,
		float *directPdfW, float *reversePdfW) const {
	BSDFEvent event;
	Evaluate(hitPoint, localLightDir, localEyeDir, &event, directPdfW, reversePdfW);
}

void GlossyTranslucentMaterial::AddReferencedTextures(std::unordered_set<const Texture *>  &referencedTexs) const {
	Material::AddReferencedTextures(referencedTexs);

	Kd->AddReferencedTextures(referencedTexs);
	Kt->AddReferencedTextures(referencedTexs);
	Ks->AddReferencedTextures(referencedTexs);
	Ks_bf->AddReferencedTextures(referencedTexs);
	nu->AddReferencedTextures(referencedTexs);
	nu_bf->AddReferencedTextures(referencedTexs);
	nv->AddReferencedTextures(referencedTexs);
	nv_bf->AddReferencedTextures(referencedTexs);
	Ka->AddReferencedTextures(referencedTexs);
	Ka_bf->AddReferencedTextures(referencedTexs);
	depth->AddReferencedTextures(referencedTexs);
	depth_bf->AddReferencedTextures(referencedTexs);
	index->AddReferencedTextures(referencedTexs);
	index_bf->AddReferencedTextures(referencedTexs);
}

void GlossyTranslucentMaterial::UpdateTextureReferences(TextureConstRef oldTex, TextureRef newTex) {
	Material::UpdateTextureReferences(oldTex, newTex);

	bool updateGlossiness = false;
	if (Kd == &oldTex)
		Kd = &newTex;
	if (Kt == &oldTex)
		Kt = &newTex;
	if (Ks == &oldTex)
		Ks = &newTex;
	if (Ks_bf == &oldTex)
		Ks_bf = &newTex;
	if (nu == &oldTex) {
		nu = &newTex;
		updateGlossiness = true;
	}
	if (nu_bf == &oldTex) {
		nu_bf = &newTex;
		updateGlossiness = true;
	}
	if (nv == &oldTex) {
		nv = &newTex;
		updateGlossiness = true;
	}
	if (nv_bf == &oldTex) {
		nv_bf = &newTex;
		updateGlossiness = true;
	}
	if (Ka == &oldTex)
		Ka = &newTex;
	if (Ka_bf == &oldTex)
		Ka_bf = &newTex;
	if (depth == &oldTex)
		depth = &newTex;
	if (depth_bf == &oldTex)
		depth_bf = &newTex;
	if (index == &oldTex)
		index = &newTex;
	if (index_bf == &oldTex)
		index_bf = &newTex;

	if (updateGlossiness)
		glossiness = Min(ComputeGlossiness(nu, nv), ComputeGlossiness(nu_bf, nv_bf));
}

PropertiesUPtr GlossyTranslucentMaterial::ToProperties(const ImageMapCache &imgMapCache, const bool useRealFileName) const  {
	auto props = std::make_unique<Properties>();

	const string name = GetName();
	props->Set(Property("scene.materials." + name + ".type")("glossytranslucent"));
	props->Set(Property("scene.materials." + name + ".kd")(Kd->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".kt")(Kt->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".ks")(Ks->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".ks_bf")(Ks_bf->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".uroughness")(nu->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".uroughness_bf")(nu_bf->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".vroughness")(nv->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".vroughness_bf")(nv_bf->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".ka")(Ka->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".ka_bf")(Ka_bf->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".d")(depth->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".d_bf")(depth_bf->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".index")(index->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".index_bf")(index_bf->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".multibounce")(multibounce));
	props->Set(Property("scene.materials." + name + ".multibounce_bf")(multibounce_bf));
	props->Set(Material::ToProperties(imgMapCache, useRealFileName));

	return props;
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
