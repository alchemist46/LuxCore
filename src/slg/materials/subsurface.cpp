/***************************************************************************
 * Copyright 1998-2026 by authors (see AUTHORS.txt)                        *
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

#include "slg/materials/subsurface.h"

using namespace std;
using namespace luxrays;
using namespace slg;

//------------------------------------------------------------------------------
// Random-walk subsurface scattering material
//
// The roles of a vertex (HitPoint::subsurface):
//  HITPOINT_SSS_ENTRY (eye paths): f = f_coat on the front side; the sampled
//    base lobe is a cosine-weighted direction into the object with the weight
//    (1 - Ec(mu_eye)), the event DIFFUSE | TRANSMIT | SUBSURFACE, and the
//    path tracer replaces the ray with the walk. The entry lobe has no value
//    for a given direction (a BSSRDF, not a BSDF), so Evaluate() and the light
//    sampling see the coating only.
//  HITPOINT_SSS_EXIT: the exit point of a walk, a Lambertian transmission
//    (albedo 1) out of the object, as Cycles. The walk reaches it with the
//    probability of the transmittance of the coating's interface: the rest is
//    reflected back into the medium (SSSWalk_InternalReflection()).
//  HITPOINT_SSS_SURFACE (everything else): the coating over a Lambertian base
//    of albedo color with the reciprocal albedo scaling of Glossy2.
// The coating is the Glossy2 one: GGX (alpha = roughness^2) with the Schlick
// Fresnel of ior, energy compensated; ior <= 1 means no coating.
//------------------------------------------------------------------------------

SubsurfaceMaterial::SubsurfaceMaterial(TextureConstPtr frontTransp, TextureConstPtr backTransp,
		TextureConstPtr emitted, TextureConstPtr bump,
		TextureConstPtr col, TextureConstPtr rad, TextureConstPtr sc,
		TextureConstPtr aniso, TextureConstPtr i, TextureConstPtr rough) :
		Material(frontTransp, backTransp, emitted, bump),
		color(col), radius(rad), scale(sc), anisotropy(aniso), ior(i), roughness(rough) {
	glossiness = ComputeGlossiness(roughness);
}

Spectrum SubsurfaceMaterial::Albedo(const HitPoint &hitPoint) const {
	return color->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
}

Spectrum SubsurfaceMaterial::EvaluateTotal(const HitPoint &hitPoint) const {
	return color->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
}

void SubsurfaceMaterial::GetReflection(const HitPoint &hitPoint, float *alpha, float *F0) const {
	const float i = ior->GetFloatValue(hitPoint);
	if (i > 1.f) {
		const float t = (i - 1.f) / (i + 1.f);
		*F0 = t * t;
	} else
		*F0 = -1.f;

	const float r = Clamp(roughness->GetFloatValue(hitPoint), 0.f, 1.f);
	*alpha = Microfacet_Regularize(Clamp(r * r, 1e-9f, 1.f), hitPoint.regularization);
}

void SubsurfaceMaterial::GetSubsurfaceWalkParams(const HitPoint &hitPoint, Spectrum *col,
		Spectrum *rad, float *g, float *eta) const {
	*col = color->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
	*rad = (radius->GetSpectrumValue(hitPoint) * scale->GetFloatValue(hitPoint)).Clamp(0.f, INFINITY);
	*g = Clamp(anisotropy->GetFloatValue(hitPoint), -.9f, .9f);
	// The IOR of the coating (none at 1 or below)
	*eta = Max(ior->GetFloatValue(hitPoint), 1.f);
}

namespace {

// The probability of sampling the coating, as Glossy2
float Subsurface_CoatProbability(const Spectrum &Ec, const Spectrum &kd) {
	const Spectrum baseOut = (Spectrum(1.f) - Ec) * kd;
	const float coat = Ec.Filter();
	const float base = baseOut.Filter();

	return ((coat + base) > 0.f) ? coat / (coat + base) : 1.f;
}

}

Spectrum SubsurfaceMaterial::Evaluate(const HitPoint &hitPoint,
	const Vector &localLightDir, const Vector &localEyeDir, BSDFEvent *event,
	float *directPdfW, float *reversePdfW) const {
	const Vector &localFixedDir = hitPoint.fromLight ? localLightDir : localEyeDir;
	const Vector &localSampledDir = hitPoint.fromLight ? localEyeDir : localLightDir;

	if (directPdfW)
		*directPdfW = 0.f;
	if (reversePdfW)
		*reversePdfW = 0.f;

	if (hitPoint.subsurface == HITPOINT_SSS_EXIT) {
		// The exit of a walk: a Lambertian transmission
		if (localLightDir.z * localEyeDir.z >= 0.f)
			return Spectrum();

		if (directPdfW)
			*directPdfW = fabsf(localSampledDir.z * INV_PI);
		if (reversePdfW)
			*reversePdfW = fabsf(localFixedDir.z * INV_PI);

		*event = DIFFUSE | TRANSMIT;
		return Spectrum(INV_PI * fabsf(localLightDir.z));
	}

	// No transmission to evaluate
	if (localLightDir.z * localEyeDir.z <= 0.f)
		return Spectrum();

	const Spectrum kd = color->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
	const Spectrum baseF = kd * INV_PI * fabsf(localLightDir.z);

	float alpha, F0;
	GetReflection(hitPoint, &alpha, &F0);
	const bool entry = (hitPoint.subsurface == HITPOINT_SSS_ENTRY);

	if ((localFixedDir.z < 0.f) || (F0 < 0.f)) {
		// The back face, or no coating: the Lambertian base (nothing to
		// evaluate for an entry)
		if (entry && (localFixedDir.z > 0.f))
			return Spectrum();

		if (directPdfW)
			*directPdfW = fabsf(localSampledDir.z * INV_PI);
		if (reversePdfW)
			*reversePdfW = fabsf(localFixedDir.z * INV_PI);

		*event = DIFFUSE | REFLECT;
		return baseF;
	}

	const Spectrum ks(F0);
	const Spectrum EcFixed = GGXEnergy_CoatE(alpha, alpha, fabsf(localFixedDir.z), ks);
	const Spectrum EcSampled = GGXEnergy_CoatE(alpha, alpha, fabsf(localSampledDir.z), ks);

	const Spectrum coatingF = SchlickBSDF_CoatingF(hitPoint.fromLight, ks, alpha, alpha,
			localFixedDir, localSampledDir);
	const float wCoating = Subsurface_CoatProbability(EcFixed, kd);
	const float wCoatingR = Subsurface_CoatProbability(EcSampled, kd);
	const float coatingPdf = SchlickBSDF_CoatingPdf(alpha, alpha, localFixedDir, localSampledDir);
	const float coatingPdfR = SchlickBSDF_CoatingPdf(alpha, alpha, localSampledDir, localFixedDir);

	*event = GLOSSY | REFLECT;
	if (entry) {
		// The coating only
		if (directPdfW)
			*directPdfW = wCoating * coatingPdf;
		if (reversePdfW)
			*reversePdfW = wCoatingR * coatingPdfR;

		return coatingF;
	}

	if (directPdfW)
		*directPdfW = (1.f - wCoating) * fabsf(localSampledDir.z * INV_PI) + wCoating * coatingPdf;
	if (reversePdfW)
		*reversePdfW = (1.f - wCoatingR) * fabsf(localFixedDir.z * INV_PI) + wCoatingR * coatingPdfR;

	const Spectrum EcAvg = GGXEnergy_CoatEavg(alpha, alpha, ks);
	const Spectrum &EcLight = hitPoint.fromLight ? EcFixed : EcSampled;
	const Spectrum &EcEye = hitPoint.fromLight ? EcSampled : EcFixed;
	Spectrum layerScale;
	for (u_int c = 0; c < 3; ++c)
		layerScale.c[c] = (1.f - EcLight.c[c]) * (1.f - EcEye.c[c]) / Max(1.f - EcAvg.c[c], 1e-4f);

	return coatingF + layerScale * baseF;
}

Spectrum SubsurfaceMaterial::Sample(const HitPoint &hitPoint,
	const Vector &localFixedDir, Vector *localSampledDir,
	const float u0, const float u1, const float passThroughEvent,
	float *pdfW, BSDFEvent *event) const {
	if (fabsf(localFixedDir.z) < DEFAULT_COS_EPSILON_STATIC)
		return Spectrum();

	if (hitPoint.subsurface == HITPOINT_SSS_EXIT) {
		// The exit of a walk: a Lambertian transmission
		*localSampledDir = -Sgn(localFixedDir.z) * CosineSampleHemisphere(u0, u1, pdfW);
		if (fabsf(localSampledDir->z) < DEFAULT_COS_EPSILON_STATIC)
			return Spectrum();

		*event = DIFFUSE | TRANSMIT;
		return Spectrum(hitPoint.fromLight ? fabsf(localFixedDir.z / localSampledDir->z) : 1.f);
	}

	const Spectrum kd = color->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
	float alpha = 0.f, F0 = -1.f;
	if (localFixedDir.z > 0.f)
		GetReflection(hitPoint, &alpha, &F0);
	const Spectrum ks(Max(F0, 0.f));

	const Spectrum EcFixed = (F0 < 0.f) ? Spectrum() : GGXEnergy_CoatE(alpha, alpha, fabsf(localFixedDir.z), ks);
	const float wCoating = (F0 < 0.f) ? 0.f : Subsurface_CoatProbability(EcFixed, kd);

	if (passThroughEvent < wCoating) {
		const float pdf = Microfacet_SampleReflect(alpha, alpha, localFixedDir, u0, u1, localSampledDir);
		if ((pdf <= 0.f) || (localFixedDir.z * localSampledDir->z <= 0.f))
			return Spectrum();
	} else if ((hitPoint.subsurface == HITPOINT_SSS_ENTRY) && (localFixedDir.z > 0.f)) {
		// Into the object, for the walk
		float pdf;
		*localSampledDir = -CosineSampleHemisphere(u0, u1, &pdf);
		if (fabsf(localSampledDir->z) < DEFAULT_COS_EPSILON_STATIC)
			return Spectrum();

		*pdfW = (1.f - wCoating) * pdf;
		*event = DIFFUSE | TRANSMIT | SUBSURFACE;
		return (Spectrum(1.f) - EcFixed) / (1.f - wCoating);
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

void SubsurfaceMaterial::Pdf(const HitPoint &hitPoint,
		const Vector &localLightDir, const Vector &localEyeDir,
		float *directPdfW, float *reversePdfW) const {
	BSDFEvent event;
	Evaluate(hitPoint, localLightDir, localEyeDir, &event, directPdfW, reversePdfW);
}

void SubsurfaceMaterial::AddReferencedTextures(std::unordered_set<const Texture *>  &referencedTexs) const {
	Material::AddReferencedTextures(referencedTexs);

	color->AddReferencedTextures(referencedTexs);
	radius->AddReferencedTextures(referencedTexs);
	scale->AddReferencedTextures(referencedTexs);
	anisotropy->AddReferencedTextures(referencedTexs);
	ior->AddReferencedTextures(referencedTexs);
	roughness->AddReferencedTextures(referencedTexs);
}

void SubsurfaceMaterial::UpdateTextureReferences(TextureConstRef oldTex, TextureRef newTex) {
	Material::UpdateTextureReferences(oldTex, newTex);

	if (color == &oldTex)
		color = &newTex;
	if (radius == &oldTex)
		radius = &newTex;
	if (scale == &oldTex)
		scale = &newTex;
	if (anisotropy == &oldTex)
		anisotropy = &newTex;
	if (ior == &oldTex)
		ior = &newTex;
	if (roughness == &oldTex) {
		roughness = &newTex;
		glossiness = ComputeGlossiness(roughness);
	}
}

PropertiesUPtr SubsurfaceMaterial::ToProperties(const ImageMapCache &imgMapCache, const bool useRealFileName) const  {
	auto props = std::make_unique<Properties>();

	const string name = GetName();
	props->Set(Property("scene.materials." + name + ".type")("subsurface"));
	props->Set(Property("scene.materials." + name + ".color")(color->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".radius")(radius->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".scale")(scale->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".anisotropy")(anisotropy->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".ior")(ior->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".roughness")(roughness->GetSDLValue()));
	props->Set(Material::ToProperties(imgMapCache, useRealFileName));

	return props;
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
