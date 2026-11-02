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

#include <cstring>

#include "slg/materials/toon.h"

// The steps of the direct light, shared with the OpenCL kernels (see
// luxrays/devices/cpudevice_oclemul.h)
namespace slg { namespace ocl { namespace {
#include "luxrays/devices/cpudevice_oclemul.h"
#include "slg/materials/materialdefs_funcs_toon_ramp.cl"
} } }

using namespace std;
using namespace luxrays;
using namespace slg;

static slg::ocl::float3 ToFloat3(const Vector &v) {
	return slg::ocl::make_float3(v.x, v.y, v.z);
}

static Spectrum ToSpectrum(const slg::ocl::float3 &c) {
	return Spectrum(c.x, c.y, c.z);
}

// Checks a ramp and packs it for the kernels
static void PackRamp(const string &name, const ToonRamp &ramp, slg::ocl::ToonRampParam &param) {
	const u_int count = ramp.positions.size();
	if (ramp.colors.size() != count)
		throw runtime_error("The toon material " + name + " has " + ToString(count) +
				" positions and " + ToString(ramp.colors.size()) + " colors");
	if (count > TOON_RAMP_MAX_STEPS)
		throw runtime_error("The toon material " + name + " has more than " + ToString(TOON_RAMP_MAX_STEPS) +
				" keys: " + ToString(count));
	if ((ramp.interpolations.size() > 1) && (ramp.interpolations.size() != count))
		throw runtime_error("The toon material " + name + " has " + ToString(count) +
				" keys and " + ToString(ramp.interpolations.size()) + " interpolations");
	for (u_int i = 1; i < count; ++i) {
		if (ramp.positions[i] < ramp.positions[i - 1])
			throw runtime_error("The toon material " + name + " positions must be in ascending order");
	}

	param.count = count;
	for (u_int i = 0; i < count; ++i) {
		param.positions[i] = ramp.positions[i];
		const Spectrum c = ramp.colors[i].Clamp(0.f, INFINITY);
		param.colors[3 * i] = c.c[0];
		param.colors[3 * i + 1] = c.c[1];
		param.colors[3 * i + 2] = c.c[2];
		const slg::ocl::ToonRampInterpolation mode = ramp.interpolations.empty() ? slg::ocl::TOON_RAMP_CONSTANT :
				ramp.interpolations[(ramp.interpolations.size() == 1) ? 0 : i];
		param.interpolations |= ((u_int)mode & 3) << (2 * i);
	}
	// A factor ramp is 0 below its first key, a colour ramp has the colour of
	// its first key there
	for (u_int c = 0; c < 3; ++c)
		param.base[c] = (ramp.shadowFill || (count == 0)) ? 0.f : param.colors[c];
	param.shadowFill = ramp.shadowFill;
	param.smooth = Max(ramp.smooth, 0.f);
}

// The SDL of a colour ramp
static void RampToProperties(const string &prefix, const slg::ocl::ToonRampParam &param, Properties &props) {
	static const char *interpolationNames[] = { "constant", "linear", "smooth" };
	Property interpolations(prefix + ".interpolation");
	for (u_int i = 0; i < param.count; ++i)
		interpolations.Add(string(interpolationNames[slg::ocl::ToonMaterial_RampInterpolation(&param, i)]));
	const vector<float> positions(param.positions, param.positions + param.count);
	const vector<float> colors(param.colors, param.colors + 3 * param.count);
	props.Set(Property(prefix + ".positions")(positions));
	props.Set(Property(prefix + ".colors")(colors));
	props.Set(interpolations);
	props.Set(Property(prefix + ".smooth")(param.smooth));
}

// The colour of the base and of the indirect light
static Spectrum BaseColor(const Spectrum &kd, const slg::ocl::ToonParam &params) {
	return (kd.Clamp(0.f, 1.f) * params.baseWeight).Clamp(0.f, 1.f);
}

//------------------------------------------------------------------------------
// Toon material
//------------------------------------------------------------------------------

ToonMaterial::ToonMaterial(TextureConstPtr frontTransp, TextureConstPtr backTransp,
		TextureConstPtr emitted, TextureConstPtr bump,
		TextureConstPtr kd, TextureConstPtr shadowCol,
		TextureConstPtr ks, TextureConstPtr rimCol, TextureConstPtr rough,
		const ToonOptions &options) :
		Material(frontTransp, backTransp, emitted, bump),
		Kd(kd), shadowColor(shadowCol), Ks(ks), rim(rimCol), roughness(rough),
		factorRamp(options.ramp.shadowFill) {
	if (options.specularRamp.shadowFill || options.rimRamp.shadowFill)
		throw runtime_error("The toon material specular and rim ramps are colour ramps");

	memset(&params, 0, sizeof(params));
	params.kdTexIndex = NULL_INDEX;
	params.shadowColorTexIndex = NULL_INDEX;
	params.ksTexIndex = NULL_INDEX;
	params.rimTexIndex = NULL_INDEX;
	params.roughnessTexIndex = NULL_INDEX;

	PackRamp("ramp", options.ramp, params.ramp);
	PackRamp("specular ramp", options.specularRamp, params.specularRamp);
	PackRamp("rim ramp", options.rimRamp, params.rimRamp);

	params.specularSmooth = Clamp(options.specularSmooth, 0.f, 1.f);
	params.specularStretchU = Clamp(options.specularStretchU, -.95f, 10.f);
	params.specularStretchV = Clamp(options.specularStretchV, -.95f, 10.f);
	params.rimSize = Clamp(options.rimSize, 0.f, 1.f);
	params.rimSmooth = Clamp(options.rimSmooth, 0.f, 1.f);
	params.baseWeight = Max(options.baseWeight, 0.f);
	params.specularWeight = Max(options.specularWeight, 0.f);
	params.rimWeight = Max(options.rimWeight, 0.f);
	params.indirectScale = Clamp(options.indirectScale, 0.f, 1.f);
	params.lightCenter = options.lightCenter;
}

Spectrum ToonMaterial::Albedo(const HitPoint &hitPoint) const {
	return BaseColor(Kd->GetSpectrumValue(hitPoint), params);
}

Spectrum ToonMaterial::EvaluateTotal(const HitPoint &hitPoint) const {
	return BaseColor(Kd->GetSpectrumValue(hitPoint), params);
}

Spectrum ToonMaterial::EvaluateToon(const HitPoint &hitPoint,
		const Vector &localLightDir, const Vector &localRampDir, const Vector &localEyeDir,
		const float lightRadius) const {
	if (CosTheta(localLightDir) * CosTheta(localEyeDir) <= 0.f)
		return Spectrum();

	// On the side of the eye direction
	const float side = Sgn(localEyeDir.z);
	const Vector rampDir = side * localRampDir;
	const Vector eyeDir = side * localEyeDir;

	const Spectrum ramp = ToSpectrum(slg::ocl::ToonMaterial_Ramp(&params.ramp, rampDir.z));
	Spectrum result = Kd->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f) * params.baseWeight * ramp;
	if (params.ramp.shadowFill)
		result += shadowColor->GetSpectrumValue(hitPoint).Clamp(0.f, INFINITY) *
				(Spectrum(1.f) - ramp).Clamp(0.f, INFINITY);

	const Spectrum ks = Ks->GetSpectrumValue(hitPoint).Clamp(0.f, INFINITY);
	if (!ks.Black())
		result += ks * ToSpectrum(slg::ocl::ToonMaterial_Highlight(&params, roughness->GetFloatValue(hitPoint),
				lightRadius, ToFloat3(rampDir), ToFloat3(eyeDir)));

	const Spectrum rimFactor = ToSpectrum(slg::ocl::ToonMaterial_Rim(&params, eyeDir.z));
	if (!rimFactor.Black())
		result += rim->GetSpectrumValue(hitPoint).Clamp(0.f, INFINITY) * rimFactor;

	return result * INV_PI;
}

Spectrum ToonMaterial::Evaluate(const HitPoint &hitPoint,
	const Vector &localLightDir, const Vector &localEyeDir, BSDFEvent *event,
	float *directPdfW, float *reversePdfW) const {
	const Spectrum result = EvaluateToon(hitPoint, localLightDir, localLightDir, localEyeDir, 0.f);
	if (result.Black())
		return result;

	*event = DIFFUSE | REFLECT;

	// The pdfs of Sample() (the path tracers leave the toon part out of the
	// MIS, see BSDF::IsDirectLightSamplingOnly())
	if (directPdfW)
		*directPdfW = fabsf((hitPoint.fromLight ? localEyeDir.z : localLightDir.z) * INV_PI);
	if (reversePdfW)
		*reversePdfW = fabsf((hitPoint.fromLight ? localLightDir.z : localEyeDir.z) * INV_PI);

	return result;
}

// The indirect light: a matte material of colour kd, times indirectScale
Spectrum ToonMaterial::Sample(const HitPoint &hitPoint,
	const Vector &localFixedDir, Vector *localSampledDir,
	const float u0, const float u1, const float passThroughEvent,
	float *pdfW, BSDFEvent *event) const {
	if ((fabsf(localFixedDir.z) < DEFAULT_COS_EPSILON_STATIC) || (params.indirectScale <= 0.f))
		return Spectrum();

	*localSampledDir = Sgn(localFixedDir.z) * CosineSampleHemisphere(u0, u1, pdfW);
	if (fabsf(CosTheta(*localSampledDir)) < DEFAULT_COS_EPSILON_STATIC)
		return Spectrum();

	*event = DIFFUSE | REFLECT;
	const Spectrum result = BaseColor(Kd->GetSpectrumValue(hitPoint), params) * params.indirectScale;
	if (hitPoint.fromLight)
		return result * fabsf(localFixedDir.z / localSampledDir->z);
	else
		return result;
}

void ToonMaterial::Pdf(const HitPoint &hitPoint,
		const Vector &localLightDir, const Vector &localEyeDir,
		float *directPdfW, float *reversePdfW) const {
	if (directPdfW)
		*directPdfW = fabsf((hitPoint.fromLight ? localEyeDir.z : localLightDir.z) * INV_PI);
	if (reversePdfW)
		*reversePdfW = fabsf((hitPoint.fromLight ? localLightDir.z : localEyeDir.z) * INV_PI);
}

void ToonMaterial::AddReferencedTextures(std::unordered_set<const Texture *>  &referencedTexs) const {
	Material::AddReferencedTextures(referencedTexs);

	Kd->AddReferencedTextures(referencedTexs);
	shadowColor->AddReferencedTextures(referencedTexs);
	Ks->AddReferencedTextures(referencedTexs);
	rim->AddReferencedTextures(referencedTexs);
	roughness->AddReferencedTextures(referencedTexs);
}

void ToonMaterial::UpdateTextureReferences(TextureConstRef oldTex, TextureRef newTex) {
	Material::UpdateTextureReferences(oldTex, newTex);

	if (Kd == &oldTex)
		Kd = &newTex;
	if (shadowColor == &oldTex)
		shadowColor = &newTex;
	if (Ks == &oldTex)
		Ks = &newTex;
	if (rim == &oldTex)
		rim = &newTex;
	if (roughness == &oldTex)
		roughness = &newTex;
}

PropertiesUPtr ToonMaterial::ToProperties(const ImageMapCache &imgMapCache, const bool useRealFileName) const  {
	auto props = std::make_unique<Properties>();

	const string prefix = "scene.materials." + GetName();
	props->Set(Property(prefix + ".type")("toon"));
	props->Set(Property(prefix + ".kd")(Kd->GetSDLValue()));
	props->Set(Property(prefix + ".shadowcolor")(shadowColor->GetSDLValue()));
	props->Set(Property(prefix + ".ks")(Ks->GetSDLValue()));
	props->Set(Property(prefix + ".rim")(rim->GetSDLValue()));

	if (factorRamp) {
		const vector<float> positions(params.ramp.positions, params.ramp.positions + params.ramp.count);
		vector<float> factors;
		for (u_int i = 0; i < params.ramp.count; ++i)
			factors.push_back(params.ramp.colors[3 * i]);
		props->Set(Property(prefix + ".ramp.thresholds")(positions));
		props->Set(Property(prefix + ".ramp.factors")(factors));
		props->Set(Property(prefix + ".ramp.smooth")(params.ramp.smooth));
	} else
		RampToProperties(prefix + ".ramp", params.ramp, *props);
	if (params.specularRamp.count > 0)
		RampToProperties(prefix + ".specular.ramp", params.specularRamp, *props);
	if (params.rimRamp.count > 0)
		RampToProperties(prefix + ".rim.ramp", params.rimRamp, *props);
	props->Set(Property(prefix + ".lightcenter")(params.lightCenter != 0));
	props->Set(Property(prefix + ".base.weight")(params.baseWeight));
	props->Set(Property(prefix + ".specular.weight")(params.specularWeight));
	props->Set(Property(prefix + ".specular.roughness")(roughness->GetSDLValue()));
	props->Set(Property(prefix + ".specular.smooth")(params.specularSmooth));
	props->Set(Property(prefix + ".specular.stretch.u")(params.specularStretchU));
	props->Set(Property(prefix + ".specular.stretch.v")(params.specularStretchV));
	props->Set(Property(prefix + ".rim.weight")(params.rimWeight));
	props->Set(Property(prefix + ".rim.size")(params.rimSize));
	props->Set(Property(prefix + ".rim.smooth")(params.rimSmooth));
	props->Set(Property(prefix + ".indirect.scale")(params.indirectScale));
	props->Set(Material::ToProperties(imgMapCache, useRealFileName));

	return props;
}

// The whole material is the diffuse lobe (for the PhotonGI cache)
Spectrum ToonMaterial::EvaluateDiffuse(const HitPoint &hitPoint,
		const Vector &localLightDir, const Vector &localEyeDir, float *pdfW) const {
	BSDFEvent event;
	return Evaluate(hitPoint, localLightDir, localEyeDir, &event, pdfW, nullptr);
}

// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
