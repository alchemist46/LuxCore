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

#include "slg/materials/openpbr.h"

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
#include "slg/materials/materialdefs_funcs_generic.cl"
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
	"transmission.scatteranisotropy",
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
	b.baseWeight = mat.GetFloat(slg::ocl::OPENPBR_BASE_WEIGHT, hitPoint);
	b.baseColor = slg::ocl::ToFloat3(mat.GetSpectrum(slg::ocl::OPENPBR_BASE_COLOR, hitPoint));
	b.baseDiffuseRoughness = mat.GetFloat(slg::ocl::OPENPBR_BASE_DIFFUSEROUGHNESS, hitPoint);
	b.baseMetalness = mat.GetFloat(slg::ocl::OPENPBR_BASE_METALNESS, hitPoint);
	b.specularWeight = mat.GetFloat(slg::ocl::OPENPBR_SPECULAR_WEIGHT, hitPoint);
	b.specularColor = slg::ocl::ToFloat3(mat.GetSpectrum(slg::ocl::OPENPBR_SPECULAR_COLOR, hitPoint));
	b.specularRoughness = mat.GetFloat(slg::ocl::OPENPBR_SPECULAR_ROUGHNESS, hitPoint);
	b.specularIor = mat.GetFloat(slg::ocl::OPENPBR_SPECULAR_IOR, hitPoint);
	b.specularAnisotropy = mat.GetFloat(slg::ocl::OPENPBR_SPECULAR_ANISOTROPY, hitPoint);
	b.tangentRotation = mat.GetFloat(slg::ocl::OPENPBR_TANGENT_ROTATION, hitPoint);
	b.regularization = hitPoint.regularization;
	b.lumWeights = slg::ocl::ToFloat3(workingColorSpace.GetLuminanceWeights());

	slg::ocl::OpenPBR_Init(&b);
}

}

OpenPBRMaterial::OpenPBRMaterial(TextureConstPtr frontTransp, TextureConstPtr backTransp,
		TextureConstPtr emitted, TextureConstPtr bump,
		const Inputs &ins, const bool thin) :
		Material(frontTransp, backTransp, emitted, bump),
		inputs(ins), thinWalled(thin) {
	UpdateGlossiness();
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
	slg::ocl::OpenPBRBSDF b;
	OpenPBR_GetBSDF(*this, hitPoint, workingColorSpace, b);

	return ToSpectrum(slg::ocl::OpenPBR_Albedo(&b));
}

Spectrum OpenPBRMaterial::Evaluate(const HitPoint &hitPoint,
		const Vector &localLightDir, const Vector &localEyeDir, BSDFEvent *event,
		float *directPdfW, float *reversePdfW) const {
	slg::ocl::OpenPBRBSDF b;
	OpenPBR_GetBSDF(*this, hitPoint, workingColorSpace, b);

	return ToSpectrum(slg::ocl::OpenPBR_Evaluate(&b, slg::ocl::ToFloat3(localLightDir), slg::ocl::ToFloat3(localEyeDir),
			hitPoint.fromLight, event, directPdfW, reversePdfW));
}

Spectrum OpenPBRMaterial::Sample(const HitPoint &hitPoint,
		const Vector &localFixedDir, Vector *localSampledDir,
		const float u0, const float u1, const float passThroughEvent,
		float *pdfW, BSDFEvent *event) const {
	slg::ocl::OpenPBRBSDF b;
	OpenPBR_GetBSDF(*this, hitPoint, workingColorSpace, b);

	slg::ocl::float3 sampledDir;
	const Spectrum result = ToSpectrum(slg::ocl::OpenPBR_Sample(&b, slg::ocl::ToFloat3(localFixedDir),
			u0, u1, passThroughEvent, hitPoint.fromLight, &sampledDir, pdfW, event));
	if (result.Black())
		return Spectrum();

	*localSampledDir = ToVector(sampledDir);

	return result;
}

void OpenPBRMaterial::Pdf(const HitPoint &hitPoint,
		const Vector &localLightDir, const Vector &localEyeDir,
		float *directPdfW, float *reversePdfW) const {
	slg::ocl::OpenPBRBSDF b;
	OpenPBR_GetBSDF(*this, hitPoint, workingColorSpace, b);

	BSDFEvent event;
	const Spectrum f = ToSpectrum(slg::ocl::OpenPBR_Evaluate(&b, slg::ocl::ToFloat3(localLightDir), slg::ocl::ToFloat3(localEyeDir),
			hitPoint.fromLight, &event, directPdfW, reversePdfW));
	if (f.Black()) {
		if (directPdfW)
			*directPdfW = 0.f;
		if (reversePdfW)
			*reversePdfW = 0.f;
	}
}

void OpenPBRMaterial::AddReferencedTextures(std::unordered_set<const Texture *> &referencedTexs) const {
	Material::AddReferencedTextures(referencedTexs);

	for (auto tex : inputs) {
		if (tex)
			tex->AddReferencedTextures(referencedTexs);
	}
}

void OpenPBRMaterial::UpdateTextureReferences(TextureConstRef oldTex, TextureRef newTex) {
	Material::UpdateTextureReferences(oldTex, newTex);

	for (auto &tex : inputs) {
		if (tex == &oldTex)
			tex = &newTex;
	}

	UpdateGlossiness();
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
	if (thinWalled)
		props->Set(Property("scene.materials." + name + ".geometry.thinwalled")(true));
	props->Set(Material::ToProperties(imgMapCache, useRealFileName));

	return props;
}

// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
