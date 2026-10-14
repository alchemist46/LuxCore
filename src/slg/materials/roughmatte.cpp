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

#include "slg/materials/roughmatte.h"

using namespace std;
using namespace luxrays;
using namespace slg;

//------------------------------------------------------------------------------
// Rough matte material
//------------------------------------------------------------------------------

RoughMatteMaterial::RoughMatteMaterial(TextureConstPtr frontTransp, TextureConstPtr backTransp,
		TextureConstPtr emitted, TextureConstPtr bump,
		TextureConstPtr col, TextureConstPtr s) :
			Material(frontTransp, backTransp, emitted, bump), Kd(col), sigma(s) {
}

//------------------------------------------------------------------------------
// EON, the energy-preserving Oren-Nayar lobe (Portsmouth, Kutz & Hill 2025,
// "EON: A practical energy-preserving rough diffuse BRDF", JCGT), used when
// the multiscattering property is enabled: Fujii's single-scattering lobe with
// its exact albedo and a Lambertian-like multiple-scattering term. The
// roughness r is sigma clamped to [0, 1].
//------------------------------------------------------------------------------

static const float EON_CONST1 = .5f - 2.f / (3.f * M_PI);
static const float EON_CONST2 = 2.f / 3.f - 28.f / (15.f * M_PI);

static float EON_E(const float mu, const float r) {
	const float AF = 1.f / (1.f + EON_CONST1 * r);
	const float Si = sqrtf(Max(0.f, 1.f - mu * mu));
	const float G = Si * (acosf(mu) - Si * mu) +
			(2.f / 3.f) * ((Si / mu) * (1.f - Si * Si * Si) - Si);

	return AF * (1.f + (r * INV_PI) * G);
}

// f_ss + f_ms without the cosine term, for two directions in the same hemisphere
static Spectrum EON_Evaluate(const Spectrum &rho, const float r, const Vector &wi, const Vector &wo) {
	const float mui = Max(fabsf(wi.z), 1e-4f);
	const float muo = Max(fabsf(wo.z), 1e-4f);
	const float AF = 1.f / (1.f + EON_CONST1 * r);

	// dot(wi, wo) - mui * muo
	const float s = wi.x * wo.x + wi.y * wo.y;
	const float stinv = (s > 0.f) ? s / Max(mui, muo) : s;
	const Spectrum fss = rho * (INV_PI * AF * (1.f + r * stinv));

	const float Eo = EON_E(muo, r);
	const float Ei = EON_E(mui, r);
	const float Eavg = AF * (1.f + EON_CONST2 * r);
	const float eps = .01f;
	Spectrum fms;
	for (u_int i = 0; i < 3; ++i) {
		const float rhoMS = rho.c[i] * rho.c[i] * Eavg / Max(1.f - rho.c[i] * (1.f - Eavg), eps);
		fms.c[i] = rhoMS * INV_PI * Max(eps, 1.f - Eo) * Max(eps, 1.f - Ei) / Max(eps, 1.f - Eavg);
	}

	return fss + fms;
}

Spectrum RoughMatteMaterial::Albedo(const HitPoint &hitPoint) const {
	return Kd->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
}

Spectrum RoughMatteMaterial::Evaluate(const HitPoint &hitPoint,
	const Vector &localLightDir, const Vector &localEyeDir, BSDFEvent *event,
	float *directPdfW, float *reversePdfW) const {
	if (directPdfW)
		*directPdfW = fabsf((hitPoint.fromLight ? localEyeDir.z : localLightDir.z) * INV_PI);

	if (reversePdfW)
		*reversePdfW = fabsf((hitPoint.fromLight ? localLightDir.z : localEyeDir.z) * INV_PI);

	*event = DIFFUSE | REFLECT;
	const float s = sigma->GetFloatValue(hitPoint);
	if (IsMultiscattering())
		return EON_Evaluate(Kd->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f), Clamp(s, 0.f, 1.f),
				localLightDir, localEyeDir) * fabsf(localLightDir.z);

	const float sigma2 = s * s;
	const float A = 1.f - (sigma2 / (2.f * (sigma2 + 0.33f)));
	const float B = 0.45f * sigma2 / (sigma2 + 0.09f);
	const float sinthetai = SinTheta(localEyeDir);
	const float sinthetao = SinTheta(localLightDir);
	float maxcos = 0.f;
	if (sinthetai > 1e-4f && sinthetao > 1e-4f) {
		const float dcos = CosPhi(localLightDir) * CosPhi(localEyeDir) +
			SinPhi(localLightDir) * SinPhi(localEyeDir);
		maxcos = max(0.f, dcos);
	}
	return Kd->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f) * (INV_PI * fabsf(localLightDir.z) *
		(A + B * maxcos * sinthetai * sinthetao / max(fabsf(CosTheta(localLightDir)), fabsf(CosTheta(localEyeDir)))));
}

Spectrum RoughMatteMaterial::Sample(const HitPoint &hitPoint,
	const Vector &localFixedDir, Vector *localSampledDir,
	const float u0, const float u1, const float passThroughEvent,
	float *pdfW, BSDFEvent *event) const {
	if (fabsf(localFixedDir.z) < DEFAULT_COS_EPSILON_STATIC)
		return Spectrum();

	*localSampledDir = Sgn(localFixedDir.z) * CosineSampleHemisphere(u0, u1, pdfW);
	if (fabsf(CosTheta(*localSampledDir)) < DEFAULT_COS_EPSILON_STATIC)
		return Spectrum();

	*event = DIFFUSE | REFLECT;
	const float s = sigma->GetFloatValue(hitPoint);
	if (IsMultiscattering()) {
		const Vector &localLightDir = hitPoint.fromLight ? localFixedDir : *localSampledDir;
		const Vector &localEyeDir = hitPoint.fromLight ? *localSampledDir : localFixedDir;

		return EON_Evaluate(Kd->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f), Clamp(s, 0.f, 1.f),
				localLightDir, localEyeDir) * (fabsf(localLightDir.z) / *pdfW);
	}

	const float sigma2 = s * s;
	const float A = 1.f - (sigma2 / (2.f * (sigma2 + 0.33f)));
	const float B = 0.45f * sigma2 / (sigma2 + 0.09f);
	const float sinthetai = SinTheta(localFixedDir);
	const float sinthetao = SinTheta(*localSampledDir);
	float maxcos = 0.f;
	if (sinthetai > 1e-4f && sinthetao > 1e-4f) {
		const float dcos = CosPhi(*localSampledDir) * CosPhi(localFixedDir) +
			SinPhi(*localSampledDir) * SinPhi(localFixedDir);
		maxcos = max(0.f, dcos);
	}
	const float coef = (A + B * maxcos * sinthetai * sinthetao / max(fabsf(CosTheta(*localSampledDir)), fabsf(CosTheta(localFixedDir))));
	if (hitPoint.fromLight)
		return Kd->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f) * (coef * fabsf(localFixedDir.z / localSampledDir->z));
	else
		return Kd->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f) * coef;
}

void RoughMatteMaterial::Pdf(const HitPoint &hitPoint,
		const Vector &localLightDir, const Vector &localEyeDir,
		float *directPdfW, float *reversePdfW) const {
	if (directPdfW)
		*directPdfW = fabsf((hitPoint.fromLight ? localEyeDir.z : localLightDir.z) * INV_PI);

	if (reversePdfW)
		*reversePdfW = fabsf((hitPoint.fromLight ? localLightDir.z : localEyeDir.z) * INV_PI);
}

void RoughMatteMaterial::AddReferencedTextures(std::unordered_set<const Texture *>  &referencedTexs) const {
	Material::AddReferencedTextures(referencedTexs);

	Kd->AddReferencedTextures(referencedTexs);
	sigma->AddReferencedTextures(referencedTexs);
}

void RoughMatteMaterial::UpdateTextureReferences(TextureConstRef oldTex, TextureRef newTex) {
	Material::UpdateTextureReferences(oldTex, newTex);

	if (Kd == &oldTex)
		Kd = &newTex;
	if (sigma == &oldTex)
		sigma = &newTex;
}

PropertiesUPtr RoughMatteMaterial::ToProperties(const ImageMapCache &imgMapCache, const bool useRealFileName) const  {
	auto props = std::make_unique<Properties>();

	const string name = GetName();
	props->Set(Property("scene.materials." + name + ".type")("roughmatte"));
	props->Set(Property("scene.materials." + name + ".kd")(Kd->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".sigma")(sigma->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".multiscattering")(IsMultiscattering()));
	props->Set(Material::ToProperties(imgMapCache, useRealFileName));

	return props;
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
