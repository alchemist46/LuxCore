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

// The OpenPBR material on the CPU: the textures at the hit point into the
// OpenPBRBSDF of the kernel code shared with the GPU (single-source, see
// luxrays/devices/cpudevice_oclemul.h)

#include <cmath>

#include "slg/bsdf/bsdf.h"
#include "slg/materials/openpbr.h"
#include "slg/textures/constfloat.h"
#include "slg/volumes/volume.h"

namespace slg { namespace ocl { namespace {
#include "luxrays/devices/cpudevice_oclemul.h"

// The definitions the kernel sources take from the OpenCL-only types files
#define BLACK MAKE_FLOAT3(0.f, 0.f, 0.f)
#define WHITE MAKE_FLOAT3(1.f, 1.f, 1.f)
#include "luxrays/core/epsilon_types.cl"
// mc_funcs.cl needs MachineEpsilon_PreviousFloat(), MachineEpsilon_E() is unused
#define PARAM_RAY_EPSILON_MIN DEFAULT_EPSILON_MIN
#define PARAM_RAY_EPSILON_MAX DEFAULT_EPSILON_MAX
#include "luxrays/core/epsilon_funcs.cl"

#include "luxrays/core/utils_funcs.cl"
#include "luxrays/core/color/color_funcs.cl"
#include "luxrays/utils/mc_funcs.cl"
#include "slg/materials/microfacet_energy_tables.cl"
#include "slg/materials/sheen_ltc_tables.cl"
#include "slg/materials/sheen_funcs.cl"
#include "slg/materials/materialdefs_funcs_generic.cl"
#include "slg/materials/materialdefs_funcs_thinfilmcoating.cl"
#include "slg/materials/materialdefs_funcs_openpbr_bsdf.cl"

#undef BLACK
#undef WHITE
#undef PARAM_RAY_EPSILON_MIN
#undef PARAM_RAY_EPSILON_MAX

float3 ToFloat3(const luxrays::Spectrum &s) {
	return MAKE_FLOAT3(s.c[0], s.c[1], s.c[2]);
}

float3 ToFloat3(const luxrays::Vector &v) {
	return MAKE_FLOAT3(v.x, v.y, v.z);
}

} } }

using namespace std;
using namespace luxrays;
using namespace slg;

namespace {

const char *OpenPBRInputNames[slg::ocl::OPENPBR_INPUT_COUNT] = {
	"base.weight", "base.color", "base.diffuseroughness", "base.metalness",
	"specular.weight", "specular.color", "specular.roughness", "specular.ior", "specular.anisotropy",
	"transmission.weight", "transmission.color", "transmission.depth", "transmission.scatter",
	"transmission.scatteranisotropy", "transmission.dispersionscale", "transmission.dispersionabbenumber",
	"subsurface.weight", "subsurface.color", "subsurface.radius", "subsurface.radiusscale",
	"subsurface.anisotropy",
	"fuzz.weight", "fuzz.color", "fuzz.roughness",
	"coat.weight", "coat.color", "coat.roughness", "coat.anisotropy", "coat.ior", "coat.darkening",
	"thinfilm.weight", "thinfilm.thickness", "thinfilm.ior",
	"tangent.rotation"
};

Spectrum ToSpectrum(const slg::ocl::float3 &v) {
	return Spectrum(v.x, v.y, v.z);
}

Vector ToVector(const slg::ocl::float3 &v) {
	return Vector(v.x, v.y, v.z);
}

void OpenPBR_GetBSDF(const OpenPBRMaterial &mat, const HitPoint &hitPoint,
		const WorkingColorSpace &workingColorSpace, slg::ocl::OpenPBRBSDF &b) {
	// The inputs of the lobes that are off (the material's or this hit
	// point's) are not read, with the same values as OpenPBRMaterial_GetBSDF()
	// on the GPU
	const u_int lobes = mat.GetLobes();
	const slg::ocl::float3 white = slg::ocl::make_float3(1.f, 1.f, 1.f);
	b.baseWeight = mat.GetFloat(slg::ocl::OPENPBR_BASE_WEIGHT, hitPoint);
	b.baseColor = slg::ocl::ToFloat3(mat.GetSpectrum(slg::ocl::OPENPBR_BASE_COLOR, hitPoint));
	b.baseDiffuseRoughness = (lobes & OPENPBR_LOBE_EON) ?
		mat.GetFloat(slg::ocl::OPENPBR_BASE_DIFFUSEROUGHNESS, hitPoint) : 0.f;
	b.baseMetalness = (lobes & OPENPBR_LOBE_METAL) ? mat.GetFloat(slg::ocl::OPENPBR_BASE_METALNESS, hitPoint) : 0.f;
	b.specularWeight = (lobes & OPENPBR_LOBE_SPECULAR) ? mat.GetFloat(slg::ocl::OPENPBR_SPECULAR_WEIGHT, hitPoint) : 0.f;
	if (lobes & OPENPBR_LOBES_BASE_SPECULAR) {
		b.specularColor = slg::ocl::ToFloat3(mat.GetSpectrum(slg::ocl::OPENPBR_SPECULAR_COLOR, hitPoint));
		b.specularRoughness = mat.GetFloat(slg::ocl::OPENPBR_SPECULAR_ROUGHNESS, hitPoint);
		b.specularIor = mat.GetFloat(slg::ocl::OPENPBR_SPECULAR_IOR, hitPoint);
		b.specularAnisotropy = mat.GetFloat(slg::ocl::OPENPBR_SPECULAR_ANISOTROPY, hitPoint);
	} else {
		b.specularColor = white;
		b.specularRoughness = .3f;
		b.specularIor = 1.5f;
		b.specularAnisotropy = 0.f;
	}
	b.thinWalled = mat.IsThinWalled();
	b.transmissionWeight = (lobes & OPENPBR_LOBE_TRANSMISSION) ?
		mat.GetFloat(slg::ocl::OPENPBR_TRANSMISSION_WEIGHT, hitPoint) : 0.f;
	if (b.transmissionWeight > 0.f) {
		// At depth, the interior volume does the colour
		b.transmissionColor = (!b.thinWalled &&
				(mat.GetFloat(slg::ocl::OPENPBR_TRANSMISSION_DEPTH, hitPoint) > 0.f)) ?
			white : slg::ocl::ToFloat3(mat.GetSpectrum(slg::ocl::OPENPBR_TRANSMISSION_COLOR, hitPoint));
	} else
		b.transmissionColor = white;
	if ((b.transmissionWeight > 0.f) && !b.thinWalled) {
		b.transmissionDispersionScale = mat.GetFloat(slg::ocl::OPENPBR_TRANSMISSION_DISPERSIONSCALE, hitPoint);
		b.transmissionDispersionAbbe = mat.GetFloat(slg::ocl::OPENPBR_TRANSMISSION_DISPERSIONABBENUMBER, hitPoint);
	} else {
		b.transmissionDispersionScale = 0.f;
		b.transmissionDispersionAbbe = 20.f;
	}
	b.subsurfaceWeight = (lobes & OPENPBR_LOBE_SUBSURFACE) ?
		mat.GetFloat(slg::ocl::OPENPBR_SUBSURFACE_WEIGHT, hitPoint) : 0.f;
	b.subsurfaceColor = (b.subsurfaceWeight > 0.f) ?
		slg::ocl::ToFloat3(mat.GetSpectrum(slg::ocl::OPENPBR_SUBSURFACE_COLOR, hitPoint)) : white;
	b.coatWeight = (lobes & OPENPBR_LOBE_COAT) ? mat.GetFloat(slg::ocl::OPENPBR_COAT_WEIGHT, hitPoint) : 0.f;
	if (b.coatWeight > 0.f) {
		b.coatColor = slg::ocl::ToFloat3(mat.GetSpectrum(slg::ocl::OPENPBR_COAT_COLOR, hitPoint));
		b.coatRoughness = mat.GetFloat(slg::ocl::OPENPBR_COAT_ROUGHNESS, hitPoint);
		b.coatAnisotropy = mat.GetFloat(slg::ocl::OPENPBR_COAT_ANISOTROPY, hitPoint);
		// coat.ior (OpenPBR_Init() makes it relative)
		b.coatEta = mat.GetFloat(slg::ocl::OPENPBR_COAT_IOR, hitPoint);
		b.coatDarkening = mat.GetFloat(slg::ocl::OPENPBR_COAT_DARKENING, hitPoint);
	} else {
		b.coatColor = white;
		b.coatRoughness = 0.f;
		b.coatAnisotropy = 0.f;
		b.coatEta = 1.6f;
		b.coatDarkening = 0.f;
	}
	b.fuzzWeight = (lobes & OPENPBR_LOBE_FUZZ) ? mat.GetFloat(slg::ocl::OPENPBR_FUZZ_WEIGHT, hitPoint) : 0.f;
	if (b.fuzzWeight > 0.f) {
		b.fuzzColor = slg::ocl::ToFloat3(mat.GetSpectrum(slg::ocl::OPENPBR_FUZZ_COLOR, hitPoint));
		b.fuzzRoughness = mat.GetFloat(slg::ocl::OPENPBR_FUZZ_ROUGHNESS, hitPoint);
	} else {
		b.fuzzColor = white;
		b.fuzzRoughness = .5f;
	}
	b.thinFilmWeight = (lobes & OPENPBR_LOBE_THINFILM) ?
		mat.GetFloat(slg::ocl::OPENPBR_THINFILM_WEIGHT, hitPoint) : 0.f;
	if (b.thinFilmWeight > 0.f) {
		b.thinFilmThickness = mat.GetFloat(slg::ocl::OPENPBR_THINFILM_THICKNESS, hitPoint);
		b.thinFilmIor = mat.GetFloat(slg::ocl::OPENPBR_THINFILM_IOR, hitPoint);
	} else {
		b.thinFilmThickness = 0.f;
		b.thinFilmIor = 1.4f;
	}
	b.tangentRotation = (lobes & OPENPBR_LOBE_ROTATION) ?
		mat.GetFloat(slg::ocl::OPENPBR_TANGENT_ROTATION, hitPoint) : 0.f;
	if (mat.HasCoatFrame() && (b.coatWeight > 0.f)) {
		// The coat's normal in the shading frame (as OpenPBRMaterial_GetBSDF()
		// on the GPU): the interpolated normal, or coat.bumptex bumped from it
		Normal n = hitPoint.interpolatedN;
		if (mat.GetCoatBump()) {
			HitPoint hp = hitPoint;
			hp.shadeN = n;
			hp.dpdu -= Dot(hp.dpdu, n) * Vector(n);
			hp.dpdv -= Dot(hp.dpdv, n) * Vector(n);
			n = mat.GetCoatBump()->Bump(hp, mat.GetBumpSampleDistance());

			// Made valid for the eye direction, as the shading normal in
			// BSDF_EnsureValidNormals()
			const float side = hitPoint.intoObject ? 1.f : -1.f;
			const Vector ng = side * Vector(hitPoint.geometryN);
			if (!hitPoint.fromLight && (Dot(hitPoint.fixedDir, ng) >= DEFAULT_COS_EPSILON_STATIC))
				n = Normal(side * BSDF_EnsureValidReflection(ng, hitPoint.fixedDir, side * Vector(n)));
		}
		b.coatN = slg::ocl::ToFloat3(hitPoint.GetFrame().ToLocal(Vector(n)));
	} else
		b.coatN = slg::ocl::make_float3(0.f, 0.f, 1.f);
	b.exteriorIor = (lobes & OPENPBR_LOBES_EXTERIOR_IOR) ? ExtractExteriorIors(hitPoint, nullptr) : 1.f;
	b.subsurfaceMode = hitPoint.subsurface;
	b.regularization = hitPoint.regularization;
	b.wavelengthU = hitPoint.passThroughEvent;
	b.wavelength = hitPoint.wavelength;
	b.dispersionTinted = hitPoint.dispersionTinted;
	b.lumWeights = slg::ocl::ToFloat3(workingColorSpace.GetLuminanceWeights());
	const float (&m)[3][3] = workingColorSpace.GetFromRec709Matrix();
	for (u_int i = 0; i < 3; ++i)
		b.fromRec709[i] = slg::ocl::make_float3(m[i][0], m[i][1], m[i][2]);
	const float (&mt)[3][3] = workingColorSpace.GetToRec709Matrix();
	for (u_int i = 0; i < 3; ++i)
		b.toRec709[i] = slg::ocl::make_float3(mt[i][0], mt[i][1], mt[i][2]);

	slg::ocl::OpenPBR_Init(&b);
}

// The albedo and the roughness of a material of the diffuse base only
// (OpenPBR_DiffuseOnly*()), clamped as OpenPBR_Init()
slg::ocl::float3 OpenPBR_DiffuseOnlyAlbedo(const OpenPBRMaterial &mat, const HitPoint &hitPoint,
		float *roughness) {
	*roughness = (mat.GetLobes() & OPENPBR_LOBE_EON) ?
		Clamp(mat.GetFloat(slg::ocl::OPENPBR_BASE_DIFFUSEROUGHNESS, hitPoint), 0.f, 1.f) : 0.f;

	return slg::ocl::ToFloat3(Clamp(mat.GetFloat(slg::ocl::OPENPBR_BASE_WEIGHT, hitPoint), 0.f, 1.f) *
			mat.GetSpectrum(slg::ocl::OPENPBR_BASE_COLOR, hitPoint).Clamp(0.f, 1.f));
}

}

OpenPBRMaterial::OpenPBRMaterial(TextureConstPtr frontTransp, TextureConstPtr backTransp,
		TextureConstPtr emitted, TextureConstPtr bump,
		const Inputs &ins, TextureConstPtr coatBump, const bool thin) :
		Material(frontTransp, backTransp, emitted, bump),
		inputs(ins), coatBumpTex(coatBump), thinWalled(thin) {
	Update();
}

const char *OpenPBRMaterial::GetInputName(const slg::ocl::OpenPBRInput input) {
	return OpenPBRInputNames[input];
}

Spectrum OpenPBRMaterial::GetInputDefault(const slg::ocl::OpenPBRInput input) {
	return Spectrum(slg::ocl::OpenPBR_Defaults[3 * input], slg::ocl::OpenPBR_Defaults[3 * input + 1],
			slg::ocl::OpenPBR_Defaults[3 * input + 2]);
}

float OpenPBRMaterial::GetFloat(const slg::ocl::OpenPBRInput input, const HitPoint &hitPoint) const {
	return inputs[input] ? inputs[input]->GetFloatValue(hitPoint) : slg::ocl::OpenPBR_Defaults[3 * input];
}

Spectrum OpenPBRMaterial::GetSpectrum(const slg::ocl::OpenPBRInput input, const HitPoint &hitPoint) const {
	return inputs[input] ? inputs[input]->GetSpectrumValue(hitPoint) : GetInputDefault(input);
}

bool OpenPBRMaterial::IsZero(const slg::ocl::OpenPBRInput input) const {
	const Texture *tex = inputs[input];
	if (!tex)
		return slg::ocl::OpenPBR_Defaults[3 * input] == 0.f;

	return (tex->GetType() == CONST_FLOAT) && (static_cast<const ConstFloatTexture *>(tex)->GetValue() == 0.f);
}

bool OpenPBRMaterial::GetConstant(const slg::ocl::OpenPBRInput input, float *value) const {
	const Texture *tex = inputs[input];
	if (!tex) {
		*value = slg::ocl::OpenPBR_Defaults[3 * input];
		return true;
	}
	if (tex->GetType() != CONST_FLOAT)
		return false;

	*value = static_cast<const ConstFloatTexture *>(tex)->GetValue();
	return true;
}

void OpenPBRMaterial::GetAlwaysDelta(bool *spec, bool *coat) const {
	*spec = false;
	*coat = false;

	// The coat: absent, or of constant delta alphas
	float cw, coatRoughness = 0.f;
	if (!GetConstant(slg::ocl::OPENPBR_COAT_WEIGHT, &cw))
		return;
	cw = Clamp(cw, 0.f, 1.f);
	if (cw > 0.f) {
		float coatAnisotropy;
		if (!GetConstant(slg::ocl::OPENPBR_COAT_ROUGHNESS, &coatRoughness) ||
				!GetConstant(slg::ocl::OPENPBR_COAT_ANISOTROPY, &coatAnisotropy))
			return;
		coatRoughness = Clamp(coatRoughness, 0.f, 1.f);
		float ax, ay;
		slg::ocl::OpenPBR_Alphas(coatRoughness, Clamp(coatAnisotropy, 0.f, 1.f), 0.f, &ax, &ay);
		*coat = (ax <= OPENPBR_MIN_ALPHA) && (ay <= OPENPBR_MIN_ALPHA);
		if (!*coat)
			return;
	} else
		*coat = true;

	// The base specular lobes, their roughness raised by the coat
	float roughness, anisotropy;
	if (!GetConstant(slg::ocl::OPENPBR_SPECULAR_ROUGHNESS, &roughness) ||
			!GetConstant(slg::ocl::OPENPBR_SPECULAR_ANISOTROPY, &anisotropy))
		return;
	const float r = Clamp(roughness, 0.f, 1.f);
	const float r2 = r * r;
	const float rc2 = coatRoughness * coatRoughness;
	const float rCoated = Min(1.f, sqrtf(sqrtf(r2 * r2 + 2.f * rc2 * rc2)));
	float ax, ay;
	slg::ocl::OpenPBR_Alphas(r + cw * (rCoated - r), Clamp(anisotropy, 0.f, 1.f), 0.f, &ax, &ay);
	*spec = (ax <= OPENPBR_MIN_ALPHA) && (ay <= OPENPBR_MIN_ALPHA);
}

bool OpenPBRMaterial::HasTransmissionDepth() const {
	return !thinWalled && !IsZero(slg::ocl::OPENPBR_TRANSMISSION_DEPTH);
}

void OpenPBRMaterial::Update() {
	const Texture *metalness = inputs[slg::ocl::OPENPBR_BASE_METALNESS];
	const bool dielectric = !metalness || (metalness->GetType() != CONST_FLOAT) ||
			(static_cast<const ConstFloatTexture *>(metalness)->GetValue() < 1.f);
	const bool transmission = dielectric && !IsZero(slg::ocl::OPENPBR_TRANSMISSION_WEIGHT);
	const bool subsurface = dielectric && !IsZero(slg::ocl::OPENPBR_SUBSURFACE_WEIGHT);

	// A roughness up to 0.01 makes a delta lobe (see OpenPBR_Init())
	auto canBeSmooth = [&](const slg::ocl::OpenPBRInput input) {
		const Texture *tex = inputs[input];
		if (!tex)
			return slg::ocl::OpenPBR_Defaults[3 * input] <= .01f;
		return (tex->GetType() != CONST_FLOAT) ||
				(static_cast<const ConstFloatTexture *>(tex)->GetValue() <= .01f);
	};

	// The kinds of events the material can have: the path depth checks the
	// budget of each (PathDepthInfo::IsLastPathVertex())
	bool specAlwaysDelta, coatAlwaysDelta;
	GetAlwaysDelta(&specAlwaysDelta, &coatAlwaysDelta);
	float value;
	const bool opaque = !(GetConstant(slg::ocl::OPENPBR_BASE_METALNESS, &value) && (value >= 1.f)) &&
			!(GetConstant(slg::ocl::OPENPBR_TRANSMISSION_WEIGHT, &value) && (value >= 1.f));
	const bool diffuse = opaque || !IsZero(slg::ocl::OPENPBR_FUZZ_WEIGHT);
	hasDiffuseLobe = opaque;
	isDelta = !diffuse && specAlwaysDelta && coatAlwaysDelta;

	// Path regularization roughens every lobe
	regularizedEventTypes = GLOSSY | REFLECT | (diffuse ? DIFFUSE : NONE) |
			((transmission || subsurface) ? TRANSMIT : NONE);

	eventTypes = REFLECT | (diffuse ? DIFFUSE : NONE) | ((transmission || subsurface) ? TRANSMIT : NONE);
	if (!specAlwaysDelta || !coatAlwaysDelta)
		eventTypes |= GLOSSY;
	if (canBeSmooth(slg::ocl::OPENPBR_SPECULAR_ROUGHNESS) ||
			(!IsZero(slg::ocl::OPENPBR_COAT_WEIGHT) && canBeSmooth(slg::ocl::OPENPBR_COAT_ROUGHNESS)))
		eventTypes |= SPECULAR;
	refractive = transmission && !thinWalled;
	// The walk enters through the opaque base
	subsurfaceWalk = subsurface && !thinWalled && opaque;
	// The EON diffuse (unless the subsurface weight is 1) and the fuzz are
	// diffuse lobes beside the walk
	diffuseBesideWalk = !(GetConstant(slg::ocl::OPENPBR_SUBSURFACE_WEIGHT, &value) && (value >= 1.f)) ||
			!IsZero(slg::ocl::OPENPBR_FUZZ_WEIGHT);
	// Without a bump of either, the coat has the shading frame
	coatFrame = !IsZero(slg::ocl::OPENPBR_COAT_WEIGHT) && (bumpTex || coatBumpTex);

	// A part is off where its input is a constant OpenPBR_Init() clamps to 0
	auto canBeOn = [&](const slg::ocl::OpenPBRInput input) {
		return !(GetConstant(input, &value) && !(value > 0.f));
	};
	lobes = (canBeOn(slg::ocl::OPENPBR_BASE_METALNESS) ? OPENPBR_LOBE_METAL : 0u) |
			(canBeOn(slg::ocl::OPENPBR_SPECULAR_WEIGHT) ? OPENPBR_LOBE_SPECULAR : 0u) |
			(canBeOn(slg::ocl::OPENPBR_TRANSMISSION_WEIGHT) ? OPENPBR_LOBE_TRANSMISSION : 0u) |
			(canBeOn(slg::ocl::OPENPBR_SUBSURFACE_WEIGHT) ? OPENPBR_LOBE_SUBSURFACE : 0u) |
			(canBeOn(slg::ocl::OPENPBR_FUZZ_WEIGHT) ? OPENPBR_LOBE_FUZZ : 0u) |
			(canBeOn(slg::ocl::OPENPBR_COAT_WEIGHT) ? OPENPBR_LOBE_COAT : 0u) |
			(canBeOn(slg::ocl::OPENPBR_THINFILM_WEIGHT) ? OPENPBR_LOBE_THINFILM : 0u) |
			(canBeOn(slg::ocl::OPENPBR_BASE_DIFFUSEROUGHNESS) ? OPENPBR_LOBE_EON : 0u) |
			// (not clamped: only 0 is no rotation)
			((GetConstant(slg::ocl::OPENPBR_TANGENT_ROTATION, &value) && (value == 0.f)) ?
				0u : OPENPBR_LOBE_ROTATION);

	UpdateGlossiness();
}

void OpenPBRMaterial::UpdateGlossiness() {
	// The glossiness (the GGX alpha of the specular lobes) only when they are
	// what the material looks like, a diffuse surface otherwise (as Disney)
	auto filter = [&](const slg::ocl::OpenPBRInput input) {
		return inputs[input] ? inputs[input]->Filter() : slg::ocl::OpenPBR_Defaults[3 * input];
	};

	if ((filter(slg::ocl::OPENPBR_BASE_METALNESS) >= .5f) ||
			(filter(slg::ocl::OPENPBR_TRANSMISSION_WEIGHT) >= .5f) ||
			(filter(slg::ocl::OPENPBR_BASE_WEIGHT) < .5f))
		glossiness = Sqr(Clamp(filter(slg::ocl::OPENPBR_SPECULAR_ROUGHNESS), 0.f, 1.f));
	else
		glossiness = 1.f;
}

Spectrum OpenPBRMaterial::Albedo(const HitPoint &hitPoint) const {
	if (OPENPBR_IS_DIFFUSE_ONLY(lobes)) {
		float roughness;
		return ToSpectrum(OpenPBR_DiffuseOnlyAlbedo(*this, hitPoint, &roughness));
	}

	slg::ocl::OpenPBRBSDF b;
	OpenPBR_GetBSDF(*this, hitPoint, workingColorSpace, b);

	return ToSpectrum(slg::ocl::OpenPBR_Albedo(&b));
}

Spectrum OpenPBRMaterial::Evaluate(const HitPoint &hitPoint,
		const Vector &localLightDir, const Vector &localEyeDir, BSDFEvent *event,
		float *directPdfW, float *reversePdfW) const {
	if (OPENPBR_IS_DIFFUSE_ONLY(lobes)) {
		float roughness;
		const slg::ocl::float3 albedo = OpenPBR_DiffuseOnlyAlbedo(*this, hitPoint, &roughness);
		return ToSpectrum(slg::ocl::OpenPBR_DiffuseOnlyEvaluate(albedo, roughness,
				slg::ocl::ToFloat3(localLightDir), slg::ocl::ToFloat3(localEyeDir),
				hitPoint.fromLight, event, directPdfW, reversePdfW));
	}

	slg::ocl::OpenPBRBSDF b;
	OpenPBR_GetBSDF(*this, hitPoint, workingColorSpace, b);

	return ToSpectrum(slg::ocl::OpenPBR_Disperse(&b, slg::ocl::OpenPBR_Evaluate(&b,
			slg::ocl::ToFloat3(localLightDir), slg::ocl::ToFloat3(localEyeDir),
			hitPoint.fromLight, event, directPdfW, reversePdfW), event));
}

Spectrum OpenPBRMaterial::Sample(const HitPoint &hitPoint,
		const Vector &localFixedDir, Vector *localSampledDir,
		const float u0, const float u1, const float passThroughEvent,
		float *pdfW, BSDFEvent *event) const {
	slg::ocl::float3 sampledDir;
	Spectrum result;
	if (OPENPBR_IS_DIFFUSE_ONLY(lobes)) {
		float roughness;
		const slg::ocl::float3 albedo = OpenPBR_DiffuseOnlyAlbedo(*this, hitPoint, &roughness);
		result = ToSpectrum(slg::ocl::OpenPBR_DiffuseOnlySample(albedo, roughness,
				slg::ocl::ToFloat3(localFixedDir), u0, u1, hitPoint.fromLight, &sampledDir, pdfW, event));
	} else {
		slg::ocl::OpenPBRBSDF b;
		OpenPBR_GetBSDF(*this, hitPoint, workingColorSpace, b);

		result = ToSpectrum(slg::ocl::OpenPBR_Disperse(&b, slg::ocl::OpenPBR_Sample(&b,
				slg::ocl::ToFloat3(localFixedDir), u0, u1, passThroughEvent, hitPoint.fromLight,
				&sampledDir, pdfW, event), event));
	}
	if (result.Black())
		return Spectrum();

	*localSampledDir = ToVector(sampledDir);

	return result;
}

void OpenPBRMaterial::Pdf(const HitPoint &hitPoint,
		const Vector &localLightDir, const Vector &localEyeDir,
		float *directPdfW, float *reversePdfW) const {
	BSDFEvent event;
	if (OPENPBR_IS_DIFFUSE_ONLY(lobes)) {
		float roughness;
		const slg::ocl::float3 albedo = OpenPBR_DiffuseOnlyAlbedo(*this, hitPoint, &roughness);
		const Spectrum f = ToSpectrum(slg::ocl::OpenPBR_DiffuseOnlyEvaluate(albedo, roughness,
				slg::ocl::ToFloat3(localLightDir), slg::ocl::ToFloat3(localEyeDir),
				hitPoint.fromLight, &event, directPdfW, reversePdfW));
		if (f.Black()) {
			if (directPdfW)
				*directPdfW = 0.f;
			if (reversePdfW)
				*reversePdfW = 0.f;
		}
		return;
	}

	slg::ocl::OpenPBRBSDF b;
	OpenPBR_GetBSDF(*this, hitPoint, workingColorSpace, b);

	const Spectrum f = ToSpectrum(slg::ocl::OpenPBR_Evaluate(&b, slg::ocl::ToFloat3(localLightDir), slg::ocl::ToFloat3(localEyeDir),
			hitPoint.fromLight, &event, directPdfW, reversePdfW));
	if (f.Black()) {
		if (directPdfW)
			*directPdfW = 0.f;
		if (reversePdfW)
			*reversePdfW = 0.f;
	}
}

bool OpenPBRMaterial::GetRefractionIors(const HitPoint &hitPoint, float *nc, float *nt) const {
	*nc = ExtractExteriorIors(hitPoint, nullptr);
	*nt = GetFloat(slg::ocl::OPENPBR_SPECULAR_IOR, hitPoint);

	return true;
}

void OpenPBRMaterial::GetSubsurfaceWalkParams(const HitPoint &hitPoint, Spectrum *color,
		Spectrum *radius, float *anisotropy, float *eta) const {
	*color = GetSpectrum(slg::ocl::OPENPBR_SUBSURFACE_COLOR, hitPoint).Clamp(0.f, 1.f);
	*radius = (GetSpectrum(slg::ocl::OPENPBR_SUBSURFACE_RADIUSSCALE, hitPoint) *
			GetFloat(slg::ocl::OPENPBR_SUBSURFACE_RADIUS, hitPoint)).Clamp(0.f, INFINITY);
	*anisotropy = Clamp(GetFloat(slg::ocl::OPENPBR_SUBSURFACE_ANISOTROPY, hitPoint), -.9f, .9f);
	*eta = slg::ocl::OpenPBR_WalkEta(GetFloat(slg::ocl::OPENPBR_SPECULAR_IOR, hitPoint),
			GetFloat(slg::ocl::OPENPBR_SPECULAR_WEIGHT, hitPoint), ExtractExteriorIors(hitPoint, nullptr),
			GetFloat(slg::ocl::OPENPBR_COAT_WEIGHT, hitPoint), GetFloat(slg::ocl::OPENPBR_COAT_IOR, hitPoint));
}

void OpenPBRMaterial::AddReferencedTextures(std::unordered_set<const Texture *> &referencedTexs) const {
	Material::AddReferencedTextures(referencedTexs);

	for (auto tex : inputs) {
		if (tex)
			tex->AddReferencedTextures(referencedTexs);
	}
	if (coatBumpTex)
		coatBumpTex->AddReferencedTextures(referencedTexs);
}

void OpenPBRMaterial::UpdateTextureReferences(TextureConstRef oldTex, TextureRef newTex) {
	Material::UpdateTextureReferences(oldTex, newTex);

	for (auto &tex : inputs) {
		if (tex == &oldTex)
			tex = &newTex;
	}
	if (coatBumpTex == &oldTex)
		coatBumpTex = &newTex;

	Update();
}

PropertiesUPtr OpenPBRMaterial::ToProperties(const ImageMapCache &imgMapCache, const bool useRealFileName) const {
	auto props = std::make_unique<Properties>();

	const string name = GetName();
	props->Set(Property("scene.materials." + name + ".type")("openpbr"));
	// Only the inputs that are set: the others have the default
	for (u_int i = 0; i < slg::ocl::OPENPBR_INPUT_COUNT; ++i) {
		if (inputs[i])
			props->Set(Property("scene.materials." + name + "." + OpenPBRInputNames[i])(inputs[i]->GetSDLValue()));
	}
	if (coatBumpTex)
		props->Set(Property("scene.materials." + name + ".coat.bumptex")(coatBumpTex->GetSDLValue()));
	if (thinWalled)
		props->Set(Property("scene.materials." + name + ".geometry.thinwalled")(true));
	props->Set(Material::ToProperties(imgMapCache, useRealFileName));
	// The interior volume of the transmission at depth is re-created
	if (interiorVolume && (interiorVolume->GetName() == GetInteriorVolumeName(name)))
		props->Delete("scene.materials." + name + ".volume.interior");

	return props;
}

Spectrum OpenPBRMaterial::EvaluateDiffuse(const HitPoint &hitPoint,
		const Vector &localLightDir, const Vector &localEyeDir, float *pdfW) const {
	if (OPENPBR_IS_DIFFUSE_ONLY(lobes)) {
		// The whole BSDF
		float roughness;
		const slg::ocl::float3 albedo = OpenPBR_DiffuseOnlyAlbedo(*this, hitPoint, &roughness);
		BSDFEvent event;
		*pdfW = 0.f;
		return ToSpectrum(slg::ocl::OpenPBR_DiffuseOnlyEvaluate(albedo, roughness,
				slg::ocl::ToFloat3(localLightDir), slg::ocl::ToFloat3(localEyeDir),
				false, &event, pdfW, nullptr));
	}

	slg::ocl::OpenPBRBSDF b;
	OpenPBR_GetBSDF(*this, hitPoint, workingColorSpace, b);

	BSDFEvent event = NONE;
	return ToSpectrum(slg::ocl::OpenPBR_Disperse(&b, slg::ocl::OpenPBR_EvaluateDiffuse(&b,
			slg::ocl::ToFloat3(localLightDir), slg::ocl::ToFloat3(localEyeDir), pdfW), &event));
}

// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
