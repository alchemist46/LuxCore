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
#include "slg/materials/glossy2.h"

using namespace std;
using namespace luxrays;
using namespace slg;

//------------------------------------------------------------------------------
// Glossy2 material
//
// LuxRender Glossy2 material porting.
//------------------------------------------------------------------------------

Glossy2Material::Glossy2Material(TextureConstPtr frontTransp, TextureConstPtr backTransp,
		TextureConstPtr emitted, TextureConstPtr bump,
		TextureConstPtr kd, TextureConstPtr ks, TextureConstPtr u, TextureConstPtr v,
		TextureConstPtr ka, TextureConstPtr d, TextureConstPtr i, const bool mbounce, const bool doublesided) :
			Material(frontTransp, backTransp, emitted, bump), Kd(kd), Ks(ks), nu(u), nv(v),
			Ka(ka), depth(d), index(i), multibounce(mbounce), doublesided (doublesided) {
	glossiness = ComputeGlossiness(nu, nv);
}

Spectrum Glossy2Material::Albedo(const HitPoint &hitPoint) const {
	return Kd->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
}

//------------------------------------------------------------------------------
// Since 2026-10-18 the coating is layered over the Lambertian base by
// reciprocal albedo scaling: f = f_coat + kd / pi * (1 - Ec(mu_light))
// (1 - Ec(mu_eye)) / (1 - Ec_avg) times the absorption of the coating, with
// Ec the albedo of the energy-compensated coating lobe (GGX with the Schlick
// Fresnel of ks, per channel) and Ec_avg its cosine-weighted average. A white
// base under any non-absorbing coating has the albedo 1 at every angle. The
// old Schlick blend, coat + (1 - S(cos h)) base, lost or gained up to 20% at
// grazing angles. The multibounce option is ignored (the layering includes
// the energy it approximated).
//------------------------------------------------------------------------------

namespace {

// The probability of sampling the coating from the direction with the
// coating albedo Ec: the coating against the part of the base that comes out
float Glossy2_CoatProbability(const Spectrum &Ec, const Spectrum &kd) {
	const Spectrum baseOut = (Spectrum(1.f) - Ec) * kd;
	const float coat = Ec.Filter();
	const float base = baseOut.Filter();

	return ((coat + base) > 0.f) ? coat / (coat + base) : 1.f;
}

// The base scale of the albedo layering
Spectrum Glossy2_LayerScale(const Spectrum &EcLight, const Spectrum &EcEye, const Spectrum &EcAvg) {
	Spectrum result;
	for (u_int i = 0; i < 3; ++i)
		result.c[i] = (1.f - EcLight.c[i]) * (1.f - EcEye.c[i]) / Max(1.f - EcAvg.c[i], 1e-4f);

	return result;
}

}

Spectrum Glossy2Material::Evaluate(const HitPoint &hitPoint,
	const Vector &localLightDir, const Vector &localEyeDir, BSDFEvent *event,
	float *directPdfW, float *reversePdfW) const {
	const Vector &localFixedDir = hitPoint.fromLight ? localLightDir : localEyeDir;
	const Vector &localSampledDir = hitPoint.fromLight ? localEyeDir : localLightDir;

	if (directPdfW)
		*directPdfW = 0.f;
	if (reversePdfW)
		*reversePdfW = 0.f;

	const Spectrum kd = Kd->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
	const Spectrum baseF = kd * INV_PI * fabsf(localLightDir.z);

	if ((!doublesided) && (localFixedDir.z <= 0.f)) {
		// Back face: no coating
		if (directPdfW)
			*directPdfW = fabsf(localSampledDir.z * INV_PI);
		if (reversePdfW)
			*reversePdfW = fabsf(localFixedDir.z * INV_PI);

		*event = DIFFUSE | REFLECT;
		return baseF;
	}

	// Front face: coating + base
	*event = GLOSSY | REFLECT;

	Spectrum ks = Ks->GetSpectrumValue(hitPoint);
	const float i = index->GetFloatValue(hitPoint);
	if (i > 0.f) {
		const float ti = (i - 1.f) / (i + 1.f);
		ks *= ti * ti;
	}
	ks = ks.Clamp(0.f, 1.f);

	const float u = Microfacet_Regularize(Clamp(nu->GetFloatValue(hitPoint), MICROFACET_MIN_ROUGHNESS, 1.f), hitPoint.regularization);
	const float v = Microfacet_Regularize(Clamp(nv->GetFloatValue(hitPoint), MICROFACET_MIN_ROUGHNESS, 1.f), hitPoint.regularization);

	const Spectrum EcFixed = GGXEnergy_CoatE(u, v, fabsf(localFixedDir.z), ks);
	const Spectrum EcSampled = GGXEnergy_CoatE(u, v, fabsf(localSampledDir.z), ks);
	const Spectrum EcAvg = GGXEnergy_CoatEavg(u, v, ks);

	if (directPdfW) {
		const float wCoating = Glossy2_CoatProbability(EcFixed, kd);
		*directPdfW = (1.f - wCoating) * fabsf(localSampledDir.z * INV_PI) +
				wCoating * SchlickBSDF_CoatingPdf(u, v, localFixedDir, localSampledDir);
	}
	if (reversePdfW) {
		const float wCoatingR = Glossy2_CoatProbability(EcSampled, kd);
		*reversePdfW = (1.f - wCoatingR) * fabsf(localFixedDir.z * INV_PI) +
				wCoatingR * SchlickBSDF_CoatingPdf(u, v, localSampledDir, localFixedDir);
	}

	// Absorption
	const float cosi = fabsf(localSampledDir.z);
	const float coso = fabsf(localFixedDir.z);
	const Spectrum alpha = Ka->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
	const float d = depth->GetFloatValue(hitPoint);
	const Spectrum absorption = CoatingAbsorption(cosi, coso, alpha, d);

	const Spectrum coatingF = SchlickBSDF_CoatingF(hitPoint.fromLight, ks, u, v,
			localFixedDir, localSampledDir);

	const Spectrum &EcLight = hitPoint.fromLight ? EcFixed : EcSampled;
	const Spectrum &EcEye = hitPoint.fromLight ? EcSampled : EcFixed;

	return coatingF + absorption * Glossy2_LayerScale(EcLight, EcEye, EcAvg) * baseF;
}

Spectrum Glossy2Material::Sample(const HitPoint &hitPoint,
	const Vector &localFixedDir, Vector *localSampledDir,
	const float u0, const float u1, const float passThroughEvent,
	float *pdfW, BSDFEvent *event) const {
	if (fabsf(localFixedDir.z) < DEFAULT_COS_EPSILON_STATIC)
		return Spectrum();

	bool sampleCoating = false;
	float u = 0.f, v = 0.f;
	if (doublesided || (localFixedDir.z > 0.f)) {
		// The same lobe probability as Evaluate()
		Spectrum ks = Ks->GetSpectrumValue(hitPoint);
		const float i = index->GetFloatValue(hitPoint);
		if (i > 0.f) {
			const float ti = (i - 1.f) / (i + 1.f);
			ks *= ti * ti;
		}
		ks = ks.Clamp(0.f, 1.f);

		u = Microfacet_Regularize(Clamp(nu->GetFloatValue(hitPoint), MICROFACET_MIN_ROUGHNESS, 1.f), hitPoint.regularization);
		v = Microfacet_Regularize(Clamp(nv->GetFloatValue(hitPoint), MICROFACET_MIN_ROUGHNESS, 1.f), hitPoint.regularization);

		const Spectrum kd = Kd->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
		const float wCoating = Glossy2_CoatProbability(GGXEnergy_CoatE(u, v, fabsf(localFixedDir.z), ks), kd);
		sampleCoating = (passThroughEvent < wCoating);
	}

	if (sampleCoating) {
		const float pdf = Microfacet_SampleReflect(u, v, localFixedDir, u0, u1, localSampledDir);
		if ((pdf <= 0.f) || (localFixedDir.z * localSampledDir->z <= 0.f))
			return Spectrum();
	} else {
		float pdf;
		*localSampledDir = Sgn(localFixedDir.z) * CosineSampleHemisphere(u0, u1, &pdf);
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

void Glossy2Material::Pdf(const HitPoint &hitPoint,
		const Vector &localLightDir, const Vector &localEyeDir,
		float *directPdfW, float *reversePdfW) const {
	BSDFEvent event;
	Evaluate(hitPoint, localLightDir, localEyeDir, &event, directPdfW, reversePdfW);
}

void Glossy2Material::AddReferencedTextures(std::unordered_set<const Texture *>  &referencedTexs) const {
	Material::AddReferencedTextures(referencedTexs);

	Kd->AddReferencedTextures(referencedTexs);
	Ks->AddReferencedTextures(referencedTexs);
	nu->AddReferencedTextures(referencedTexs);
	nv->AddReferencedTextures(referencedTexs);
	Ka->AddReferencedTextures(referencedTexs);
	depth->AddReferencedTextures(referencedTexs);
	index->AddReferencedTextures(referencedTexs);
}

void Glossy2Material::UpdateTextureReferences(TextureConstRef oldTex, TextureRef newTex) {
	Material::UpdateTextureReferences(oldTex, newTex);

	bool updateGlossiness = false;
	if (Kd == &oldTex)
		Kd = &newTex;
	if (Ks == &oldTex)
		Ks = &newTex;
	if (nu == &oldTex) {
		nu = &newTex;
		updateGlossiness = true;
	}
	if (nv == &oldTex) {
		nv = &newTex;
		updateGlossiness = true;
	}
	if (Ka == &oldTex)
		Ka = &newTex;
	if (depth == &oldTex)
		depth = &newTex;
	if (index == &oldTex)
		index = &newTex;

	if (updateGlossiness)
		glossiness = ComputeGlossiness(nu, nv);
}

PropertiesUPtr Glossy2Material::ToProperties(const ImageMapCache &imgMapCache, const bool useRealFileName) const  {
	auto props = std::make_unique<Properties>();

	const string name = GetName();
	props->Set(Property("scene.materials." + name + ".type")("glossy2"));
	props->Set(Property("scene.materials." + name + ".kd")(Kd->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".ks")(Ks->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".uroughness")(nu->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".vroughness")(nv->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".ka")(Ka->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".d")(depth->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".index")(index->GetSDLValue()));
	props->Set(Property ("scene.materials." + name + ".multibounce")(multibounce));
	props->Set(Property ("scene.materials." + name + ".doublesided")(doublesided));
	props->Set(Material::ToProperties(imgMapCache, useRealFileName));

	return props;
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
