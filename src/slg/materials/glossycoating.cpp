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
#include "slg/textures/constfloat.h"
#include "slg/materials/glossycoating.h"

using namespace std;
using namespace luxrays;
using namespace slg;

//------------------------------------------------------------------------------
// Since 2026-10-18 the coating is layered over its base by albedo scaling: a
// reflection of the base is scaled by 1 - Ec(mu_eye), with Ec the albedo of
// the energy-compensated coating (as Cycles and the OpenPBR specification do),
// so a base of albedo 1 under a non-absorbing coat has the albedo 1 whatever
// the base is. The reciprocal form of glossy2, (1 - Ec(mu_light))
// (1 - Ec(mu_eye)) / (1 - Ec_avg), is exact only for a Lambertian base (a
// metal base was off by up to 0.7%, a sharper one by several percent). The
// engines evaluate f with the light and eye roles fixed on every path, so the
// non-reciprocal form is consistent (the stochastic layering of P4 is the
// exact form of the physical layer). A transmission through the base crosses
// the coating once, (1 - Ec(mu)) of the direction on the coated side. The old
// Schlick blend lost or gained up to 25% at grazing angles. The multibounce
// option is ignored.
//
// The darkening (0 by default, as before 2026-10-19) is the one of OpenPBR's
// coat.darkening: the light the coating reflects back down from inside hits
// the base again, so the base is scaled by (1 - K) / (1 - E_base K), blended
// by the darkening amount. K = 1 - (1 - Favg) / eta^2 is the internal
// reflectance of the coating (eta from the Schlick F0 ks per channel, Favg
// its hemispherical average) and E_base the albedo of the base,
// Material::Albedo() (the diffuse color, the normal Fresnel of a metal, white
// for a glass). A white base keeps its albedo; a colored one gets darker and
// more saturated.
//------------------------------------------------------------------------------

namespace {

// The probability of sampling the coating from the direction with the
// coating albedo Ec (the base albedo is not known here: the GPU evaluates the
// base through its own ops)
float GlossyCoating_CoatProbability(const Spectrum &ks, const Spectrum &Ec) {
	return ks.Black() ? 0.f : Clamp(Ec.Filter(), .1f, .9f);
}

// The darkening (1 - K) / (1 - E_base K) blended by amount (see above)
Spectrum GlossyCoating_Darkening(const Spectrum &ks, const Spectrum &baseAlbedo, const float amount) {
	Spectrum result;
	for (u_int c = 0; c < 3; ++c) {
		const float F0 = Clamp(ks.c[c], 0.f, .95f);
		const float sqrtF0 = sqrtf(F0);
		const float eta = (1.f + sqrtF0) / (1.f - sqrtF0);
		const float Favg = F0 + (1.f - F0) / 21.f;
		const float K = 1.f - (1.f - Favg) / (eta * eta);
		const float delta = (1.f - K) / (1.f - Clamp(baseAlbedo.c[c], 0.f, 1.f) * K);
		result.c[c] = 1.f + amount * (delta - 1.f);
	}

	return result;
}

}

//------------------------------------------------------------------------------
// GlossyCoating material
//------------------------------------------------------------------------------

GlossyCoatingMaterial::GlossyCoatingMaterial(
	TextureConstPtr frontTransp, TextureConstPtr backTransp,
	TextureConstPtr emitted, TextureConstPtr bump,
	MaterialConstPtr mB,
	TextureConstPtr ks, TextureConstPtr u,
	TextureConstPtr v,
	TextureConstPtr ka, TextureConstPtr d,
	TextureConstPtr i,
	const bool mbounce,
	TextureConstPtr dark) :
			Material(frontTransp, backTransp, emitted, bump), matBase(mB), Ks(ks), nu(u), nv(v),
			Ka(ka), depth(d), index(i), multibounce(mbounce), darkening(dark) {
	glossiness = Min(ComputeGlossiness(nu, nv), matBase->GetGlossiness());
}

VolumeConstPtr GlossyCoatingMaterial::GetInteriorVolume(const HitPoint &hitPoint,
		const float passThroughEvent) const {
	if (interiorVolume)
		return interiorVolume;
	else
		return matBase->GetInteriorVolume(hitPoint, passThroughEvent);
}

VolumeConstPtr GlossyCoatingMaterial::GetExteriorVolume(const HitPoint &hitPoint,
		const float passThroughEvent) const {
	if (exteriorVolume)
		return exteriorVolume;
	else
		return matBase->GetExteriorVolume(hitPoint, passThroughEvent);
}

void GlossyCoatingMaterial::UpdateAvgPassThroughTransparency() {
	avgPassThroughTransparency = matBase->GetAvgPassThroughTransparency();
}

Spectrum GlossyCoatingMaterial::GetPassThroughTransparency(const HitPoint &hitPoint,
		const Vector &localFixedDir, const float passThroughEvent,
		const bool backTracing) const {
	return matBase->GetPassThroughTransparency(hitPoint, localFixedDir,
			passThroughEvent, backTracing);
}

float GlossyCoatingMaterial::GetEmittedRadianceY(const float oneOverPrimitiveArea) const {
	if (emittedTex)
		return Material::GetEmittedRadianceY(oneOverPrimitiveArea);
	else
		return matBase->GetEmittedRadianceY(oneOverPrimitiveArea);
}

Spectrum GlossyCoatingMaterial::GetEmittedRadiance(const HitPoint &hitPoint, const float oneOverPrimitiveArea) const {
	if (emittedTex)
		return Material::GetEmittedRadiance(hitPoint, oneOverPrimitiveArea);
	else
		return matBase->GetEmittedRadiance(hitPoint, oneOverPrimitiveArea);
}

Spectrum GlossyCoatingMaterial::Albedo(const HitPoint &hitPoint) const {
	return matBase->Albedo(hitPoint);
}

bool GlossyCoatingMaterial::HasDarkening() const {
	return !((darkening->GetType() == CONST_FLOAT) &&
			(static_cast<const ConstFloatTexture &>(*darkening).GetValue() <= 0.f));
}

Spectrum GlossyCoatingMaterial::DarkeningFactor(const HitPoint &hitPoint, const Spectrum &ks) const {
	const float amount = Clamp(darkening->GetFloatValue(hitPoint), 0.f, 1.f);
	if (!(amount > 0.f))
		return Spectrum(1.f);

	return GlossyCoating_Darkening(ks, matBase->Albedo(hitPoint), amount);
}

Spectrum GlossyCoatingMaterial::Evaluate(const HitPoint &hitPoint,
	const Vector &localLightDir, const Vector &localEyeDir, BSDFEvent *event,
	float *directPdfW, float *reversePdfW) const {
	const Frame frame(hitPoint.GetFrame());
	const float cosi = fabsf(localLightDir.z);
	const float coso = fabsf(localEyeDir.z);

	const float sideTest = Dot(frame.ToWorld(localEyeDir), hitPoint.geometryN) * Dot(frame.ToWorld(localLightDir), hitPoint.geometryN);
	if (sideTest > DEFAULT_COS_EPSILON_STATIC) {
		// Reflection

		HitPoint hitPointBase(hitPoint);
		matBase->Bump(&hitPointBase);
		const Frame frameBase(hitPointBase.GetFrame());
		const Vector lightDirBase = frameBase.ToLocal(frame.ToWorld(localLightDir));
		const Vector eyeDirBase = frameBase.ToLocal(frame.ToWorld(localEyeDir));

		const Spectrum baseF = matBase->Evaluate(hitPointBase, lightDirBase, eyeDirBase, event, directPdfW, reversePdfW);
		if (!(localLightDir.z > 0.f)) {
			// Back face: no coating
			return baseF;
		}

		// I have always to initialized baseF pdf because it is used below
		if (baseF.Black()) {
			if (directPdfW)
				*directPdfW = 0.f;
			if (reversePdfW)
				*reversePdfW = 0.f;
			
			*event = NONE;
		}

		// Front face: coating+base
		*event |= GLOSSY | REFLECT;
		Spectrum ks = Ks->GetSpectrumValue(hitPoint);
		const float i = index->GetFloatValue(hitPoint);
		if (i > 0.f) {
			const float ti = (i - 1.f) / (i + 1.f);
			ks *= ti * ti;
		}
		ks = ks.Clamp(0.f, 1.f);

		const float u = Microfacet_Regularize(Clamp(nu->GetFloatValue(hitPoint), 1e-9f, 1.f), hitPoint.regularization);
		const float v = Microfacet_Regularize(Clamp(nv->GetFloatValue(hitPoint), 1e-9f, 1.f), hitPoint.regularization);

		// The coating pdf is the one of sampling the generated direction from
		// the fixed one (SchlickBSDF_CoatingPdf(fixed, sampled), as Sample()
		// and the GPU use it): the bounded VNDF pdf is not symmetric
		const Vector &localFixedDir = hitPoint.fromLight ? localLightDir : localEyeDir;
		const Vector &localSampledDir = hitPoint.fromLight ? localEyeDir : localLightDir;

		const Spectrum EcLight = GGXEnergy_CoatE(u, v, fabsf(localLightDir.z), ks);
		const Spectrum EcEye = GGXEnergy_CoatE(u, v, fabsf(localEyeDir.z), ks);
		const Spectrum &EcFixed = hitPoint.fromLight ? EcLight : EcEye;
		const Spectrum &EcSampled = hitPoint.fromLight ? EcEye : EcLight;

		if (directPdfW) {
			const float wCoating = GlossyCoating_CoatProbability(ks, EcFixed);
			const float wBase = 1.f - wCoating;

			*directPdfW = wBase * *directPdfW +
				wCoating * SchlickBSDF_CoatingPdf(u, v, localFixedDir, localSampledDir);
		}

		if (reversePdfW) {
			const float wCoatingR = GlossyCoating_CoatProbability(ks, EcSampled);
			const float wBaseR = 1.f - wCoatingR;

			*reversePdfW = wBaseR * *reversePdfW +
				wCoatingR * SchlickBSDF_CoatingPdf(u, v, localSampledDir, localFixedDir);
		}

		// Absorption
		const Spectrum alpha = Ka->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
		const float d = depth->GetFloatValue(hitPoint);
		const Spectrum absorption = CoatingAbsorption(cosi, coso, alpha, d);

		const Spectrum coatingF = SchlickBSDF_CoatingF(true, ks, u, v,
			localLightDir, localEyeDir);

		return coatingF + absorption * (Spectrum(1.f) - EcEye) * DarkeningFactor(hitPoint, ks) * baseF;
	} else if (sideTest < -DEFAULT_COS_EPSILON_STATIC) {
		// Transmission

		HitPoint hitPointBase(hitPoint);
		matBase->Bump(&hitPointBase);
		const Frame frameBase(hitPointBase.GetFrame());
		const Vector lightDirBase = frameBase.ToLocal(frame.ToWorld(localLightDir));
		const Vector eyeDirBase = frameBase.ToLocal(frame.ToWorld(localEyeDir));

		const Spectrum baseF = matBase->Evaluate(hitPointBase, lightDirBase, eyeDirBase, event, directPdfW, reversePdfW);
		// I have always to initialized baseF pdf because it is used below
		if (baseF.Black()) {
			if (directPdfW)
				*directPdfW = 0.f;
			if (reversePdfW)
				*reversePdfW = 0.f;

			*event = NONE;
		}

		*event |= GLOSSY | TRANSMIT;

		Spectrum ks = Ks->GetSpectrumValue(hitPoint);
		const float i = index->GetFloatValue(hitPoint);
		if (i > 0.f) {
			const float ti = (i - 1.f) / (i + 1.f);
			ks *= ti * ti;
		}
		ks = ks.Clamp(0.f, 1.f);

		const float u = Microfacet_Regularize(Clamp(nu->GetFloatValue(hitPoint), 1e-9f, 1.f), hitPoint.regularization);
		const float v = Microfacet_Regularize(Clamp(nv->GetFloatValue(hitPoint), 1e-9f, 1.f), hitPoint.regularization);

		if (directPdfW) {
			const Vector localFixedDir = hitPoint.fromLight ? localLightDir : localEyeDir;
			const float wCoating = (localFixedDir.z > DEFAULT_COS_EPSILON_STATIC) ?
				GlossyCoating_CoatProbability(ks, GGXEnergy_CoatE(u, v, fabsf(localFixedDir.z), ks)) : 0.f;
			const float wBase = 1.f - wCoating;

			*directPdfW *= wBase;
		}

		if (reversePdfW) {
			const Vector localSampledDir = hitPoint.fromLight ? localEyeDir : localLightDir;
			const float wCoatingR = (localSampledDir.z > DEFAULT_COS_EPSILON_STATIC) ?
				GlossyCoating_CoatProbability(ks, GGXEnergy_CoatE(u, v, fabsf(localSampledDir.z), ks)) : 0.f;
			const float wBaseR = 1.f - wCoatingR;

			*reversePdfW *= wBaseR;
		}

		// Absorption
		const Spectrum alpha = Ka->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
		const float d = depth->GetFloatValue(hitPoint);
		const Spectrum absorption = CoatingAbsorption(cosi, coso, alpha, d);

		// One crossing of the coating, by the direction on the coated side.
		// The exit of a subsurface walk has none: the albedo scaling of its
		// entry covers the round trip, as the one of a reflection does.
		if (hitPoint.subsurface == HITPOINT_SSS_EXIT)
			return baseF;
		const float muCoated = (localLightDir.z > 0.f) ? localLightDir.z : localEyeDir.z;
		return absorption * (Spectrum(1.f) - GGXEnergy_CoatE(u, v, fabsf(muCoated), ks)) *
				DarkeningFactor(hitPoint, ks) * baseF;
	} else
		return Spectrum();
}

Spectrum GlossyCoatingMaterial::Sample(const HitPoint &hitPoint,
	const Vector &localFixedDir, Vector *localSampledDir,
	const float u0, const float u1, const float passThroughEvent,
	float *pdfW, BSDFEvent *event) const {
	if (fabsf(localFixedDir.z) < DEFAULT_COS_EPSILON_STATIC)
		return Spectrum();

	const Frame frame(hitPoint.GetFrame());
	Spectrum ks = Ks->GetSpectrumValue(hitPoint);
	const float i = index->GetFloatValue(hitPoint);
	if (i > 0.f) {
		const float ti = (i - 1.f) / (i + 1.f);
		ks *= ti * ti;
	}
	ks = ks.Clamp(0.f, 1.f);

	const float u = Microfacet_Regularize(Clamp(nu->GetFloatValue(hitPoint), 1e-9f, 1.f), hitPoint.regularization);
	const float v = Microfacet_Regularize(Clamp(nv->GetFloatValue(hitPoint), 1e-9f, 1.f), hitPoint.regularization);

	// The coating only on the front face
	const Spectrum EcFixed = GGXEnergy_CoatE(u, v, fabsf(localFixedDir.z), ks);
	const float wCoating = (localFixedDir.z > DEFAULT_COS_EPSILON_STATIC) ?
		GlossyCoating_CoatProbability(ks, EcFixed) : 0.f;
	const float wBase = 1.f - wCoating;

	float basePdf, coatingPdf;
	Spectrum baseF, coatingF;

	if (passThroughEvent < wBase) {
		HitPoint hitPointBase(hitPoint);
		matBase->Bump(&hitPointBase);
		const Frame frameBase(hitPointBase.GetFrame());
		const Vector fixedDirBase = frameBase.ToLocal(frame.ToWorld(localFixedDir));
		// Sample base layer
		baseF = matBase->Sample(hitPointBase, fixedDirBase, localSampledDir, u0, u1, passThroughEvent / wBase,
			&basePdf, event);
		assert (baseF.IsValid());

		if (baseF.Black())
			return Spectrum();

		baseF *= basePdf;
		*localSampledDir = frame.ToLocal(frameBase.ToWorld(*localSampledDir));

		// Don't add the coating scattering if the base sampled
		// component is specular	
		if (!(*event & SPECULAR)) {
			coatingF = SchlickBSDF_CoatingF(hitPoint.fromLight, ks, u, v,
				localFixedDir, *localSampledDir);
			assert (coatingF.IsValid());

			coatingPdf = SchlickBSDF_CoatingPdf(u, v, localFixedDir, *localSampledDir);
			assert (IsValid(coatingPdf));
		} else
			coatingPdf = 0.f;
	} else {
		// Sample coating BxDF
		coatingF = SchlickBSDF_CoatingSampleF(hitPoint.fromLight, ks, u, v,
			localFixedDir, localSampledDir, u0, u1, &coatingPdf);
		assert (coatingF.IsValid());

		if (coatingF.Black())
			return Spectrum();
		assert (IsValid(coatingPdf));

		if (fabsf(CosTheta(*localSampledDir)) < DEFAULT_COS_EPSILON_STATIC)
			return Spectrum();

		coatingF *= coatingPdf;

		HitPoint hitPointBase(hitPoint);
		matBase->Bump(&hitPointBase);
		const Frame frameBase(hitPointBase.GetFrame());
		const Vector localLightDir = frameBase.ToLocal(frame.ToWorld(hitPoint.fromLight ? localFixedDir : *localSampledDir));
		const Vector localEyeDir = frameBase.ToLocal(frame.ToWorld(hitPoint.fromLight ? *localSampledDir : localFixedDir));

		baseF = matBase->Evaluate(hitPointBase, localLightDir, localEyeDir, event, &basePdf);
		// I have always to initialized baseF pdf because it is used below
		if (baseF.Black())
			basePdf = 0.f;
		*event = GLOSSY | REFLECT;
	}

	// Absorption
	const float cosi = fabsf(localFixedDir.z);
	const float coso = fabsf(localSampledDir->z);
	const Spectrum alpha = Ka->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
	const float d = depth->GetFloatValue(hitPoint);
	const Spectrum absorption = CoatingAbsorption(cosi, coso, alpha, d);

	// If Dot(woW, ng) is too small, set sideTest to 0 to discard the result
	// and avoid numerical instability
	const float cosWo = Dot(frame.ToWorld(localFixedDir), hitPoint.geometryN);
	const float sideTest = (fabsf(cosWo) < DEFAULT_EPSILON_STATIC) ? 0.f : Dot(frame.ToWorld(*localSampledDir), hitPoint.geometryN) / cosWo;
	Spectrum result;
	if (sideTest > DEFAULT_COS_EPSILON_STATIC) {
		// Reflection
		if (!(cosWo > 0.f)) {
			// Back face reflection: no coating
			result = baseF;
			assert (!result.IsNaN() && !result.IsInf() && !result.IsNeg());
		} else {
			// Front face reflection: coating + base, albedo-scaled by the eye
			// direction
			const Spectrum EcEye = hitPoint.fromLight ?
				GGXEnergy_CoatE(u, v, fabsf(localSampledDir->z), ks) : EcFixed;

			result = coatingF + absorption * (Spectrum(1.f) - EcEye) * DarkeningFactor(hitPoint, ks) * baseF;
			assert (!result.IsNaN() && !result.IsInf() && !result.IsNeg());
		}
	} else if (sideTest < -DEFAULT_COS_EPSILON_STATIC) {
		// Transmission: one crossing of the coating, by the direction on the
		// coated side (none at the exit of a subsurface walk, see Evaluate())
		if (hitPoint.subsurface == HITPOINT_SSS_EXIT)
			result = baseF;
		else {
			const float muCoated = (localFixedDir.z > 0.f) ? localFixedDir.z : localSampledDir->z;
			result = absorption * (Spectrum(1.f) - GGXEnergy_CoatE(u, v, fabsf(muCoated), ks)) *
					DarkeningFactor(hitPoint, ks) * baseF;
		}
		assert (!result.IsNaN() && !result.IsInf() && !result.IsNeg());
	} else
		return Spectrum();

	*pdfW = coatingPdf * wCoating + basePdf * wBase;

	result /= *pdfW;
	assert (!result.IsNaN() && !result.IsInf() && !result.IsNeg());
	
	return result;
}

void GlossyCoatingMaterial::Pdf(const HitPoint &hitPoint,
		const Vector &localLightDir, const Vector &localEyeDir,
		float *directPdfW, float *reversePdfW) const {
	const Frame frame(hitPoint.GetFrame());
	HitPoint hitPointBase(hitPoint);
	matBase->Bump(&hitPointBase);
	const Frame frameBase(hitPointBase.GetFrame());
	const Vector lightDirBase = frameBase.ToLocal(frame.ToWorld(localLightDir));
	const Vector eyeDirBase = frameBase.ToLocal(frame.ToWorld(localEyeDir));
	matBase->Pdf(hitPointBase, lightDirBase, eyeDirBase, directPdfW, reversePdfW);
	const Vector localFixedDir = hitPoint.fromLight ? localLightDir : localEyeDir;
	if (!(localFixedDir.z > 0.f))
		return;

	Spectrum ks = Ks->GetSpectrumValue(hitPoint);
	const float i = index->GetFloatValue(hitPoint);
	if (i > 0.f) {
		const float ti = (i - 1.f) / (i + 1.f);
		ks *= ti * ti;
	}
	ks = ks.Clamp(0.f, 1.f);

	const float u = Microfacet_Regularize(Clamp(nu->GetFloatValue(hitPoint), 1e-9f, 1.f), hitPoint.regularization);
	const float v = Microfacet_Regularize(Clamp(nv->GetFloatValue(hitPoint), 1e-9f, 1.f), hitPoint.regularization);

	// The same pdfs as Evaluate() (the coating pdf is not symmetric)
	const Vector &localSampledDir = hitPoint.fromLight ? localEyeDir : localLightDir;
	const bool reflection = (localFixedDir.z * localSampledDir.z > 0.f);

	if (directPdfW) {
		const float wCoating = GlossyCoating_CoatProbability(ks, GGXEnergy_CoatE(u, v, fabsf(localFixedDir.z), ks));
		*directPdfW = (1.f - wCoating) * *directPdfW +
			(reflection ? wCoating * SchlickBSDF_CoatingPdf(u, v, localFixedDir, localSampledDir) : 0.f);
	}

	if (reversePdfW) {
		const float wCoatingR = (localSampledDir.z > DEFAULT_COS_EPSILON_STATIC) ?
			GlossyCoating_CoatProbability(ks, GGXEnergy_CoatE(u, v, fabsf(localSampledDir.z), ks)) : 0.f;
		*reversePdfW = (1.f - wCoatingR) * *reversePdfW +
			(reflection ? wCoatingR * SchlickBSDF_CoatingPdf(u, v, localSampledDir, localFixedDir) : 0.f);
	}
}

void GlossyCoatingMaterial::UpdateMaterialReferences(MaterialConstRef oldMat, MaterialRef newMat) {
	if (matBase == &oldMat)
		matBase = &newMat;

	// Update volumes too
	Material::UpdateMaterialReferences(oldMat, newMat);
}

bool GlossyCoatingMaterial::IsReferencing(MaterialConstRef mat) const {
	if (&mat == this)
		return true;

	return matBase->IsReferencing(mat);
}

void GlossyCoatingMaterial::AddReferencedMaterials(
	std::unordered_set<const Material *> &referencedMats
) const {
	Material::AddReferencedMaterials(referencedMats);

	matBase->AddReferencedMaterials(referencedMats);
}

void GlossyCoatingMaterial::AddReferencedTextures(std::unordered_set<const Texture *>  &referencedTexs) const {
	Material::AddReferencedTextures(referencedTexs);

	matBase->AddReferencedTextures(referencedTexs);
	Ks->AddReferencedTextures(referencedTexs);
	nu->AddReferencedTextures(referencedTexs);
	nv->AddReferencedTextures(referencedTexs);
	Ka->AddReferencedTextures(referencedTexs);
	depth->AddReferencedTextures(referencedTexs);
	index->AddReferencedTextures(referencedTexs);
	darkening->AddReferencedTextures(referencedTexs);
}

void GlossyCoatingMaterial::UpdateTextureReferences(TextureConstRef oldTex, TextureRef newTex) {
	Material::UpdateTextureReferences(oldTex, newTex);

	if (Ks == &oldTex)
		Ks = &newTex;
	if (nu == &oldTex)
		nu = &newTex;
	if (nv == &oldTex)
		nv = &newTex;
	if (Ka == &oldTex)
		Ka = &newTex;
	if (depth == &oldTex)
		depth = &newTex;
	if (index == &oldTex)
		index = &newTex;
	if (darkening == &oldTex)
		darkening = &newTex;

	// Always update glossiness just in case matBase->GetGlossiness() has changed
	const float coatGlossiness = ComputeGlossiness(nu, nv);
	if (matBase->GetEventTypes() & GLOSSY)
		glossiness = Min(coatGlossiness, matBase->GetGlossiness());
	else
		glossiness = coatGlossiness;
}

PropertiesUPtr GlossyCoatingMaterial::ToProperties(const ImageMapCache &imgMapCache, const bool useRealFileName) const  {
	auto props = std::make_unique<Properties>();

	const string name = GetName();
	props->Set(Property("scene.materials." + name + ".type")("glossycoating"));
	props->Set(Property("scene.materials." + name + ".base")(matBase->GetName()));
	props->Set(Property("scene.materials." + name + ".ks")(Ks->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".uroughness")(nu->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".vroughness")(nv->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".ka")(Ka->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".d")(depth->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".index")(index->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".multibounce")(multibounce));
	props->Set(Property("scene.materials." + name + ".darkening")(darkening->GetSDLValue()));
	props->Set(Material::ToProperties(imgMapCache, useRealFileName));

	return props;
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
