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

#include <boost/lexical_cast.hpp>

#include "luxrays/core/geometry/frame.h"
#include "slg/core/sphericalfunction/sphericalfunction.h"
#include "slg/materials/material.h"
#include "slg/bsdf/bsdf.h"
#include "slg/textures/fresnel/fresneltexture.h"

using namespace std;
using namespace luxrays;
using namespace slg;

//------------------------------------------------------------------------------
// Material
//------------------------------------------------------------------------------

Material::Material(TextureConstPtr frontTransp, TextureConstPtr backTransp,
		TextureConstPtr emitted, TextureConstPtr bump) :
		NamedObject("material"),
		matID(0), lightID(0),
		directLightSamplingType(DLS_AUTO), emittedImportance(1.f),
		emittedGain(1.f), emittedPower(0.f), emittedEfficiency(0.f),
		emittedPowerNormalize(true), emittedGainNormalize(false),
		emittedTemperature(-1.f), emittedNormalizeTemperature(false),
		frontTransparencyTex(frontTransp), backTransparencyTex(backTransp),
		emittedTex(emitted), bumpTex(bump), bumpSampleDistance(.001f),
		emissionMap(nullptr), emissionFunc(nullptr),
		interiorVolume(nullptr), exteriorVolume(nullptr),
		glossiness(0.f),
		isVisibleIndirectDiffuse(true), isVisibleIndirectGlossy(true), isVisibleIndirectSpecular(true),
		isShadowCatcher(false), isShadowCatcherOnlyInfiniteLights(false), isPhotonGIEnabled(true),
		isHoldout(false) {
	SetEmittedTheta(90.f);
	UpdateEmittedFactor();
	UpdateAvgPassThroughTransparency();
}

Material::~Material() = default;

void Material::SetEmittedTheta(const float theta) {
	if (theta <= 0.f) {
		emittedTheta = 0.f;
		emittedCosThetaMax = 1.f;
	} else if (theta <= 90.f) {
		emittedTheta = theta;

		const float radTheta = Radians(emittedTheta);
		emittedCosThetaMax = cosf(radTheta);
	} else {
		emittedTheta = 90.f;
		emittedCosThetaMax = 0.f;
	}
}

void Material::SetEmissionMap(ImageMapConstRef map) {
	emissionMap = &map;
	emissionFunc.reset();
	if (emissionMap)
		emissionFunc = std::make_unique<SampleableSphericalFunction>(new ImageMapSphericalFunction(emissionMap));
	else
		emissionFunc = nullptr;
}

// Interior volume
VolumeConstPtr Material::GetInteriorVolume(const HitPoint &hitPoint,
	const float passThroughEvent) const { return interiorVolume; }
VolumeConstPtr Material::GetInteriorVolume() const { return interiorVolume; }
void Material::SetInteriorVolume(VolumeConstPtr vol) { interiorVolume = vol; }
void Material::SetInteriorVolume(VolumeConstRef vol) {
	interiorVolume = std::addressof(vol);
}

// Exterior Volume
VolumeConstPtr Material::GetExteriorVolume(const HitPoint &hitPoint,
	const float passThroughEvent) const { return exteriorVolume; }
VolumeConstPtr Material::GetExteriorVolume() const { return exteriorVolume; }
void Material::SetExteriorVolume(VolumeConstPtr vol) { exteriorVolume = vol; }
void Material::SetExteriorVolume(VolumeConstRef vol) {
	exteriorVolume = std::addressof(vol);
}



Spectrum Material::GetPassThroughTransparency(const HitPoint &hitPoint,
		const luxrays::Vector &localFixedDir, const float passThroughEvent,
		const bool backTracing) const {
	TextureConstPtr transparencyTex = (hitPoint.intoObject != backTracing) ? frontTransparencyTex : backTransparencyTex;
	
	if (transparencyTex) {
		const float weight = Clamp(transparencyTex->GetFloatValue(hitPoint), 0.f, 1.f);

		return (passThroughEvent > weight) ? Spectrum(1.f) : Spectrum(0.f);
	} else
		return Spectrum(0.f);
}

Spectrum Material::GetEmittedRadiance(const HitPoint &hitPoint, const float oneOverPrimitiveArea) const {
	if (emittedTex) {
		return (emittedFactor * (usePrimitiveArea ? oneOverPrimitiveArea : 1.f)) *
				emittedTex->GetSpectrumValue(hitPoint).Clamp();
	} else
		return Spectrum();
}

float Material::GetEmittedRadianceY(const float oneOverPrimitiveArea) const {
	if (emittedTex)
		return emittedFactor.Y() * (usePrimitiveArea ? oneOverPrimitiveArea : 1.f) * Max(emittedTex->Y(), 0.f);
	else
		return 0.f;
}

void Material::Bump(HitPoint *hitPoint) const {
    if (bumpTex) {
		hitPoint->shadeN = bumpTex->Bump(*hitPoint, bumpSampleDistance);

		// Update dpdu and dpdv so they are still orthogonal to shadeN 
		hitPoint->dpdu = Cross(hitPoint->shadeN, Cross(hitPoint->dpdu, hitPoint->shadeN));
		hitPoint->dpdv = Cross(hitPoint->shadeN, Cross(hitPoint->dpdv, hitPoint->shadeN));
	}
}

Spectrum Material::Albedo(const HitPoint &hitPoint) const {
	return Spectrum(1.f);
}

Spectrum Material::EvaluateTotal(const HitPoint &hitPoint) const {
	// This is the generic implementation using Monte Carlo integration to
	// compute the result.
	//
	// Note: Matte material has a special fast path.

	assert (!hitPoint.fromLight);

	// Note: may be, this should become a configurable parameter
	const u_int samplesCount = 64;

	Spectrum result;
	for (u_int i = 0; i < samplesCount; ++i) {
		const float u1 = RadicalInverse(i, 3);
		const float u2 = RadicalInverse(i, 5);
		const float u3 = RadicalInverse(i, 7);
		
		const Vector fixedDir = UniformSampleHemisphere(u1, u2);
		const float fixedDirPdf = UniformHemispherePdf(u1, u2);

		BSDFEvent event;
		Vector sampledDir;
		float sampledDirPdf;
		Spectrum bsdfSample = Sample(hitPoint, fixedDir, &sampledDir,
					u1, u2,	u3, &sampledDirPdf, &event);

		if (!bsdfSample.Black() && (CosTheta(sampledDir) > 0.f))
				result += bsdfSample * CosTheta(fixedDir) / fixedDirPdf;
    }

    return result / (M_PI * samplesCount);
}

void Material::UpdateEmittedFactor() {
	const Spectrum emittedTemperatureScale = (emittedTemperature >= 0.f) ?
		workingColorSpace.FromRec709(TemperatureToWhitePoint(emittedTemperature, emittedNormalizeTemperature)) :
		Spectrum(1.f);

	if (emittedTex) {
		const float normalizePowerFactor = emittedPowerNormalize ? (1.f / Max(emittedTex->Y(), 0.f)) : 1.f;

		emittedFactor = emittedGain * (emittedPower * emittedEfficiency  * normalizePowerFactor);
		if (emittedFactor.Black() || emittedFactor.IsInf() || emittedFactor.IsNaN()) {
			const float normalizeGainFactor = emittedGainNormalize ? (1.f / Max(emittedTex->Y(), 0.f)) : 1.f;

			emittedFactor = emittedGain * normalizeGainFactor;
			usePrimitiveArea = false;
		} else {
			if (emittedTheta == 0.f) {
				// Nothing to do
			} else if (emittedTheta < 90.f)
				emittedFactor /= (1.f - emittedCosThetaMax) * (2.f * M_PI);
			else
				emittedFactor /= M_PI;
			usePrimitiveArea = true;
		}
	} else {
		const float normalizeGainFactor = emittedGainNormalize ? (1.f / Max(emittedTex->Y(), 0.f)) : 1.f;

		emittedFactor = emittedGain * normalizeGainFactor;
		usePrimitiveArea = false;
	}
	
	emittedFactor *= emittedTemperatureScale;
}

void Material::UpdateAvgPassThroughTransparency() {
	avgPassThroughTransparency = frontTransparencyTex ? Clamp(frontTransparencyTex->Filter(), 0.f, 1.f) : 1.f;
}

PropertiesUPtr Material::ToProperties(const ImageMapCache &imgMapCache, const bool useRealFileName) const {
	auto propsPtr = std::make_unique<Properties>();
	auto& props = *propsPtr;

	const string name = GetName();
	if (frontTransparencyTex)
		props.Set(Property("scene.materials." + name + ".transparency.front")(frontTransparencyTex->GetSDLValue()));
	if (backTransparencyTex)
		props.Set(Property("scene.materials." + name + ".transparency.back")(backTransparencyTex->GetSDLValue()));
	props.Set(Property("scene.materials." + name + ".transparency.shadow")(passThroughShadowTransparency));
	props.Set(Property("scene.materials." + name + ".transparency.shadowoverride")(passThroughShadowTransparencyOverride));
	props.Set(Property("scene.materials." + name + ".id")(matID));

	props.Set(Property("scene.materials." + name + ".emission.gain")(emittedGain));
	props.Set(Property("scene.materials." + name + ".emission.power")(emittedPower));
	props.Set(Property("scene.materials." + name + ".emission.normalizebycolor")(emittedPowerNormalize));
	props.Set(Property("scene.materials." + name + ".emission.efficiency")(emittedEfficiency));
	props.Set(Property("scene.materials." + name + ".emission.theta")(emittedTheta));
	props.Set(Property("scene.materials." + name + ".emission.id")(lightID));
	props.Set(Property("scene.materials." + name + ".emission.importance")(emittedImportance));
	if (emittedTex)
		props.Set(Property("scene.materials." + name + ".emission")(emittedTex->GetSDLValue()));
	if (emissionMap) {
		const string fileName = useRealFileName ?
			emissionMap->GetName() : imgMapCache.GetSequenceFileName(*emissionMap);
		props.Set(Property("scene.materials." + name + ".emission.mapfile")(fileName));
		props.Set(emissionMap->ToProperties("scene.materials." + name, false));
	}
	props.Set(Property("scene.materials." + name + ".emission.temperature")(emittedTemperature));
	props.Set(Property("scene.materials." + name + ".emission.temperature.normalize")(emittedNormalizeTemperature));

	switch (directLightSamplingType) {
		case DLS_ENABLED:
			props.Set(Property("scene.materials." + name + ".emission.directlightsampling.type")("ENABLED"));
			break;
		case DLS_DISABLED:
			props.Set(Property("scene.materials." + name + ".emission.directlightsampling.type")("DISABLED"));
			break;
		case DLS_AUTO:
			props.Set(Property("scene.materials." + name + ".emission.directlightsampling.type")("AUTO"));
			break;
		default:
			throw runtime_error("Unknown MaterialEmissionDLSType in Material::ToProperties(): " + ToString(directLightSamplingType));
	}

	if (bumpTex)
		props.Set(Property("scene.materials." + name + ".bumptex")(bumpTex->GetSDLValue()));
	props.Set(Property("scene.materials." + name + ".bumpsamplingdistance")(bumpSampleDistance));

	if (interiorVolume)
		props.Set(Property("scene.materials." + name + ".volume.interior")(interiorVolume->GetName()));
	if (exteriorVolume)
		props.Set(Property("scene.materials." + name + ".volume.exterior")(exteriorVolume->GetName()));
		
	props.Set(Property("scene.materials." + name + ".visibility.indirect.diffuse.enable")(isVisibleIndirectDiffuse));
	props.Set(Property("scene.materials." + name + ".visibility.indirect.glossy.enable")(isVisibleIndirectGlossy));
	props.Set(Property("scene.materials." + name + ".visibility.indirect.specular.enable")(isVisibleIndirectSpecular));

	props.Set(Property("scene.materials." + name + ".shadowcatcher.enable")(isShadowCatcher));
	props.Set(Property("scene.materials." + name + ".shadowcatcher.onlyinfinitelights")(isShadowCatcherOnlyInfiniteLights));

	props.Set(Property("scene.materials." + name + ".photongi.enable")(isPhotonGIEnabled));
	props.Set(Property("scene.materials." + name + ".holdout.enable")(isHoldout));

	return propsPtr;
}

void Material::UpdateMaterialReferences(MaterialConstRef oldMat, MaterialRef newMat) {
	updvol(interiorVolume, oldMat, newMat);
	updvol(exteriorVolume, oldMat, newMat);
}

void Material::AddReferencedMaterials(
	std::unordered_set<const Material *> &referencedMats
) const {
	referencedMats.insert(this);
	if (interiorVolume)
		referencedMats.insert(interiorVolume.get());
	if (exteriorVolume)
		referencedMats.insert(exteriorVolume.get());
}

void Material::AddReferencedTextures(std::unordered_set<const Texture *>  &referencedTexs) const {
	if (frontTransparencyTex)
		frontTransparencyTex->AddReferencedTextures(referencedTexs);
	if (backTransparencyTex)
		backTransparencyTex->AddReferencedTextures(referencedTexs);
	if (emittedTex)
		emittedTex->AddReferencedTextures(referencedTexs);
	if (bumpTex)
		bumpTex->AddReferencedTextures(referencedTexs);
}

void Material::AddReferencedImageMaps(
	std::unordered_set<const ImageMap *> &referencedImgMaps
) const {
	if (emissionMap)
		referencedImgMaps.insert(emissionMap.get());
}

// Update any reference to oldTex with newTex
void Material::UpdateTextureReferences(TextureConstRef oldTex, TextureRef newTex) {
	if (frontTransparencyTex == &oldTex) {
		frontTransparencyTex = &newTex;
		UpdateAvgPassThroughTransparency();
	}
	if (backTransparencyTex == &oldTex)
		backTransparencyTex = &newTex;
	if (emittedTex == &oldTex)
		emittedTex = &newTex;
	if (bumpTex == &oldTex)
		bumpTex = &newTex;
}

string Material::MaterialType2String(const MaterialType type) {
	switch (type) {
		case MATTE: return "MATTE";
		case MIRROR: return "MIRROR";
		case GLASS: return "GLASS";
		case ARCHGLASS: return "ARCHGLASS";
		case MIX: return "MIX";
		case NULLMAT: return "NULLMAT";
		case MATTETRANSLUCENT: return "MATTETRANSLUCENT";
		case GLOSSY2: return "GLOSSY2";
		case METAL2: return "METAL2";
		case ROUGHGLASS: return "ROUGHGLASS";
		case VELVET: return "VELVET";
		case CLOTH: return "CLOTH";
		case CARPAINT: return "CARPAINT";
		case ROUGHMATTE: return "ROUGHMATTE";
		case ROUGHMATTETRANSLUCENT: return "ROUGHMATTETRANSLUCENT";
		case GLOSSYTRANSLUCENT: return "GLOSSYTRANSLUCENT";
		case GLOSSYCOATING: return "GLOSSYCOATING";
		case DISNEY: return "DISNEY";
		case TWOSIDED: return "TWOSIDED";

		// Volumes
		case HOMOGENEOUS_VOL: return "HOMOGENEOUS_VOL";
		case CLEAR_VOL: return "CLEAR_VOL";
		case HETEROGENEOUS_VOL: return "HETEROGENEOUS_VOL";

		default:
			throw runtime_error("Unknown material type in Material::MaterialType2String(): " + ToString(type));
	}
}

float Material::ComputeGlossiness(TextureConstPtr t1, TextureConstPtr t2, TextureConstPtr t3) {
	const float glossinessT1 = t1 ? t1->Filter() : 1.f;
	const float glossinessT2 = t2 ? t2->Filter() : 1.f;
	const float glossinessT3 = t3 ? t3->Filter() : 1.f;

	return Min(glossinessT1, Min(glossinessT2, glossinessT3));
}

//------------------------------------------------------------------------------
// IOR utilities
//------------------------------------------------------------------------------

float slg::ExtractExteriorIors(const HitPoint &hitPoint, TextureConstPtr exteriorIor) {
	float nc = 1.f;
	if (exteriorIor)
		nc = exteriorIor->GetFloatValue(hitPoint);
	else if (hitPoint.exteriorVolume)
		nc = hitPoint.exteriorVolume->GetIOR(hitPoint);

	return nc;
}

float slg::ExtractInteriorIors(const HitPoint &hitPoint, TextureConstPtr interiorIor) {
	float nt = 1.f;
	if (interiorIor)
		nt = interiorIor->GetFloatValue(hitPoint);
	else if (hitPoint.interiorVolume)
		nt = hitPoint.interiorVolume->GetIOR(hitPoint);

	return nt;
}

//------------------------------------------------------------------------------
// Coating absorption
//------------------------------------------------------------------------------

Spectrum slg::CoatingAbsorption(const float cosi, const float coso,
	const Spectrum &alpha, const float depth) {
	if (depth > 0.f) {
		// 1/cosi+1/coso=(cosi+coso)/(cosi*coso)
		const float depthFactor = depth * (cosi + coso) / (cosi * coso);
		return Exp(alpha * -depthFactor);
	} else
		return Spectrum(1.f);
}

//------------------------------------------------------------------------------
// GGX distribution with the Smith masking (alpha = (u, v))
//------------------------------------------------------------------------------

float slg::GGXDistribution_D(const float alphaU, const float alphaV, const Vector &wh) {
	const float cosTheta2 = wh.z * wh.z;
	if (!(cosTheta2 > 0.f))
		return 0.f;

	const float ax = wh.x / alphaU;
	const float ay = wh.y / alphaV;
	const float e = ax * ax + ay * ay + cosTheta2;

	return 1.f / (M_PI * alphaU * alphaV * e * e);
}

float slg::GGXDistribution_Lambda(const float alphaU, const float alphaV, const Vector &w) {
	const float cosTheta2 = w.z * w.z;
	if (!(cosTheta2 > 0.f))
		return INFINITY;

	const float ax = alphaU * w.x;
	const float ay = alphaV * w.y;

	return (-1.f + sqrtf(1.f + (ax * ax + ay * ay) / cosTheta2)) * .5f;
}

float slg::GGXDistribution_G1(const float alphaU, const float alphaV, const Vector &w) {
	return 1.f / (1.f + GGXDistribution_Lambda(alphaU, alphaV, w));
}

float slg::GGXDistribution_G(const float alphaU, const float alphaV, const Vector &wo, const Vector &wi) {
	// Height-correlated Smith masking-shadowing
	return 1.f / (1.f + GGXDistribution_Lambda(alphaU, alphaV, wo) + GGXDistribution_Lambda(alphaU, alphaV, wi));
}

// The pdf of the visible normals of wo (wo.z > 0)
float slg::GGXDistribution_PdfVisibleH(const float alphaU, const float alphaV, const Vector &wo, const Vector &wh) {
	const float cosWH = Dot(wo, wh);
	if (!(cosWH > 0.f))
		return 0.f;

	return GGXDistribution_G1(alphaU, alphaV, wo) * GGXDistribution_D(alphaU, alphaV, wh) * cosWH / wo.z;
}

// Dupuy & Benyoub 2023, spherical caps (wo.z > 0, wh.z >= 0)
void slg::GGXDistribution_SampleVisibleH(const float alphaU, const float alphaV, const Vector &wo,
		const float u0, const float u1, Vector *wh) {
	// Warp to the hemisphere configuration
	const Vector woStd = Normalize(Vector(wo.x * alphaU, wo.y * alphaV, wo.z));

	// Sample a spherical cap in (-woStd.z, 1]
	const float phi = 2.f * M_PI * u0;
	const float z = (1.f - u1) * (1.f + woStd.z) - woStd.z;
	const float sinTheta = sqrtf(Clamp(1.f - z * z, 0.f, 1.f));
	const Vector h(sinTheta * cosf(phi) + woStd.x, sinTheta * sinf(phi) + woStd.y, z + woStd.z);

	// Warp back to the ellipsoid configuration
	*wh = Normalize(Vector(h.x * alphaU, h.y * alphaV, Max(0.f, h.z)));
}

// Eto & Tokuyoshi 2023, bounded VNDF sampling of the reflected direction (wo.z > 0)
static float GGXDistribution_BoundedK(const float alphaU, const float alphaV, const Vector &wo) {
	const float a = Clamp(Min(alphaU, alphaV), 0.f, 1.f);
	const float s = 1.f + sqrtf(wo.x * wo.x + wo.y * wo.y);
	const float a2 = a * a;
	const float s2 = s * s;

	return (1.f - a2) * s2 / (s2 + a2 * wo.z * wo.z);
}

void slg::GGXDistribution_SampleBoundedReflect(const float alphaU, const float alphaV, const Vector &wo,
		const float u0, const float u1, Vector *wi) {
	const Vector woStd = Normalize(Vector(wo.x * alphaU, wo.y * alphaV, wo.z));

	const float phi = 2.f * M_PI * u0;
	const float b = GGXDistribution_BoundedK(alphaU, alphaV, wo) * woStd.z;
	const float z = (1.f - u1) * (1.f + b) - b;
	const float sinTheta = sqrtf(Clamp(1.f - z * z, 0.f, 1.f));
	const Vector h(sinTheta * cosf(phi) + woStd.x, sinTheta * sinf(phi) + woStd.y, z + woStd.z);

	const Vector wh = Normalize(Vector(h.x * alphaU, h.y * alphaV, Max(0.f, h.z)));
	*wi = 2.f * Dot(wh, wo) * wh - wo;
}

float slg::GGXDistribution_PdfBoundedReflect(const float alphaU, const float alphaV, const Vector &wo, const Vector &wi) {
	if (!(wi.z > 0.f))
		return 0.f;

	const Vector wh = Normalize(wo + wi);
	const float ax = alphaU * wo.x;
	const float ay = alphaV * wo.y;
	const float t = sqrtf(ax * ax + ay * ay + wo.z * wo.z);
	const float k = GGXDistribution_BoundedK(alphaU, alphaV, wo);

	return GGXDistribution_D(alphaU, alphaV, wh) / (2.f * (k * wo.z + t));
}

//------------------------------------------------------------------------------
// Microfacet lobe
//------------------------------------------------------------------------------

float slg::Microfacet_D(const float u, const float v,
		const Vector &wh) {
	return GGXDistribution_D(u, v, wh);
}

float slg::Microfacet_G(const float u, const float v,
		const Vector &wo, const Vector &wi) {
	return GGXDistribution_G(u, v, wo, wi);
}

void slg::Microfacet_SampleH(const float u, const float v,
		const Vector &wo, const float u0, const float u1,
		Vector *wh, float *d, float *pdfH) {
	const Vector woUp = (wo.z < 0.f) ? -wo : wo;
	GGXDistribution_SampleVisibleH(u, v, woUp, u0, u1, wh);
	*d = GGXDistribution_D(u, v, *wh);
	*pdfH = GGXDistribution_PdfVisibleH(u, v, woUp, *wh);
}

float slg::Microfacet_PdfH(const float u, const float v,
		const Vector &wo, const Vector &wh) {
	const Vector woUp = (wo.z < 0.f) ? -wo : wo;
	return GGXDistribution_PdfVisibleH(u, v, woUp, wh);
}

float slg::Microfacet_SampleReflect(const float u, const float v,
		const Vector &wo, const float u0, const float u1, Vector *wi) {
	const bool flip = (wo.z < 0.f);
	const Vector woUp = flip ? -wo : wo;
	Vector wiUp;
	GGXDistribution_SampleBoundedReflect(u, v, woUp, u0, u1, &wiUp);
	*wi = flip ? -wiUp : wiUp;
	return GGXDistribution_PdfBoundedReflect(u, v, woUp, wiUp);
}

float slg::Microfacet_PdfReflect(const float u, const float v,
		const Vector &wo, const Vector &wi) {
	const bool flip = (wo.z < 0.f);
	return GGXDistribution_PdfBoundedReflect(u, v, flip ? -wo : wo, flip ? -wi : wi);
}

//------------------------------------------------------------------------------
// Schlick BSDF (the coating lobe)
//------------------------------------------------------------------------------

float slg::SchlickBSDF_CoatingWeight(const Spectrum &ks, const Vector &localFixedDir) {
	// Approximate H by using reflection direction for wi
	const float u = fabsf(localFixedDir.z);
	const Spectrum S = FresnelTexture::SchlickEvaluate(ks, u);

	// Ensures coating is never sampled less than half the time
	return .5f * (1.f + S.Filter());
}

Spectrum slg::SchlickBSDF_CoatingF(const bool fromLight, const Spectrum &ks,
		const float u, const float v, const bool mbounce,
		const bool multiscattering,
		const Vector &localFixedDir, const Vector &localSampledDir) {
	const float coso = fabsf(localFixedDir.z);
	const float cosi = fabsf(localSampledDir.z);

	const Vector wh(Normalize(localFixedDir + localSampledDir));
	const float cosi_Schlick = Clamp(AbsDot(localSampledDir, wh), 0.f, 1.f);
	const Spectrum S = FresnelTexture::SchlickEvaluate(ks, cosi_Schlick);
	assert (S.IsValid());

	const float G = Microfacet_G(u, v, localFixedDir, localSampledDir);

	// Multibounce - alternative with interreflection in the coating creases
	float factor = Microfacet_D(u, v, wh) * G;
	if (!fromLight)
		factor = factor / (4.f * coso) +
				(mbounce ? cosi * Clamp((1.f - G) / (4.f * coso * cosi), 0.f, 1.f) : 0.f);
	else
		factor = factor / (4.f * cosi) + 
				(mbounce ? coso * Clamp((1.f - G) / (4.f * cosi * coso), 0.f, 1.f) : 0.f);
	assert (IsValid(factor));

	// Multiple-scattering energy compensation
	const Vector &localEyeDir = fromLight ? localSampledDir : localFixedDir;
	const Spectrum msScale = Microfacet_MSScale(multiscattering, u, v, fabsf(localEyeDir.z),
			SchlickFresnel_Average(ks));

	return factor * S * msScale;
}

Spectrum slg::SchlickBSDF_CoatingSampleF(const bool fromLight, const Spectrum ks,
		const float u, const float v, const bool mbounce,
		const bool multiscattering,
		const Vector &localFixedDir, Vector *localSampledDir, float u0, float u1, float *pdf) {
	*pdf = Microfacet_SampleReflect(u, v, localFixedDir, u0, u1, localSampledDir);

	if ((fabsf(localSampledDir->z) < DEFAULT_COS_EPSILON_STATIC) || (localFixedDir.z * localSampledDir->z < 0.f))
		return Spectrum();

	if (*pdf <= 0.f)
		return Spectrum();

	const float coso = fabsf(localFixedDir.z);
	const float cosi = fabsf(localSampledDir->z);

	const Vector wh(Normalize(localFixedDir + *localSampledDir));
	const float cosWH = AbsDot(localFixedDir, wh);

	Spectrum S = FresnelTexture::SchlickEvaluate(ks, cosWH);

	const float d = Microfacet_D(u, v, wh);
	const float G = Microfacet_G(u, v, localFixedDir, *localSampledDir);
	if (!fromLight)
		//CoatingF(sw, *wi, wo, f_);
		S *= (d / *pdf) * G / (4.f * coso) + 
				(mbounce ? cosi * Clamp((1.f - G) / (4.f * coso * cosi), 0.f, 1.f) / *pdf : 0.f);
	else
		//CoatingF(sw, wo, *wi, f_);
		S *= (d / *pdf) * G / (4.f * cosi) + 
				(mbounce ? coso * Clamp((1.f - G) / (4.f * cosi * coso), 0.f, 1.f) / *pdf : 0.f);

	// Multiple-scattering energy compensation
	const Vector &localEyeDir = fromLight ? *localSampledDir : localFixedDir;
	S *= Microfacet_MSScale(multiscattering, u, v, fabsf(localEyeDir.z),
			SchlickFresnel_Average(ks));

	return S;
}

float slg::SchlickBSDF_CoatingPdf(const float u, const float v,
		const Vector &localFixedDir, const Vector &localSampledDir) {
	return Microfacet_PdfReflect(u, v, localFixedDir, localSampledDir);
}

//------------------------------------------------------------------------------
// EON, the energy-preserving Oren-Nayar lobe (Portsmouth, Kutz & Hill 2025,
// "EON: A practical energy-preserving rough diffuse BRDF", JCGT), the lobe of
// roughmatte and roughmattetranslucent: Fujii's single-scattering lobe with
// its exact albedo and a Lambertian-like multiple-scattering term. The
// roughness r is sigma clamped to [0, 1].
//------------------------------------------------------------------------------

static const float EON_CONST1 = .5f - 2.f / (3.f * M_PI);
static const float EON_CONST2 = 2.f / 3.f - 28.f / (15.f * M_PI);

float slg::EON_E(const float mu, const float r) {
	const float AF = 1.f / (1.f + EON_CONST1 * r);
	const float Si = sqrtf(Max(0.f, 1.f - mu * mu));
	const float G = Si * (acosf(mu) - Si * mu) +
			(2.f / 3.f) * ((Si / mu) * (1.f - Si * Si * Si) - Si);

	return AF * (1.f + (r * INV_PI) * G);
}

// f_ss + f_ms without the cosine term, for two directions in the same hemisphere
Spectrum slg::EON_Evaluate(const Spectrum &rho, const float r, const Vector &wi, const Vector &wo) {
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
	// Only the denominators are guarded: a floor on 1 - E (it was .01) added
	// a multiple-scattering term of about 1% at r = 0, where E = 1 and the
	// lobe must be exactly Lambertian. (1 - Eo) (1 - Ei) / (1 - Eavg) goes to
	// 0 with r.
	const float eps = 1e-6f;
	Spectrum fms;
	for (u_int i = 0; i < 3; ++i) {
		const float rhoMS = rho.c[i] * rho.c[i] * Eavg / Max(1.f - rho.c[i] * (1.f - Eavg), eps);
		fms.c[i] = rhoMS * INV_PI * Max(0.f, 1.f - Eo) * Max(0.f, 1.f - Ei) / Max(eps, 1.f - Eavg);
	}

	return fss + fms;
}

// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
