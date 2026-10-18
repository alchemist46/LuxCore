#line 2 "materialdefs_funcs_glossy2.cl"

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

//------------------------------------------------------------------------------
// Glossy2 material
//
// LuxRender Glossy2 material porting.
//------------------------------------------------------------------------------

OPENCL_FORCE_INLINE void Glossy2Material_Albedo(__global const Material* restrict material,
		__global const HitPoint *hitPoint,
		__global float *evalStack, uint *evalStackOffset
		MATERIALS_PARAM_DECL) {
	const float3 albedo = Spectrum_Clamp(Texture_GetSpectrumValue(material->glossy2.kdTexIndex,
			hitPoint TEXTURES_PARAM));

	EvalStack_PushFloat3(albedo);
}

OPENCL_FORCE_INLINE void Glossy2Material_GetInteriorVolume(__global const Material* restrict material,
		__global const HitPoint *hitPoint,
		__global float *evalStack, uint *evalStackOffset
		MATERIALS_PARAM_DECL) {
	DefaultMaterial_GetInteriorVolume(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
}

OPENCL_FORCE_INLINE void Glossy2Material_GetExteriorVolume(__global const Material* restrict material,
		__global const HitPoint *hitPoint,
		__global float *evalStack, uint *evalStackOffset
		MATERIALS_PARAM_DECL) {
	DefaultMaterial_GetExteriorVolume(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
}

OPENCL_FORCE_INLINE void Glossy2Material_GetPassThroughTransparency(__global const Material* restrict material,
		__global const HitPoint *hitPoint,
		__global float *evalStack, uint *evalStackOffset
		MATERIALS_PARAM_DECL) {
	DefaultMaterial_GetPassThroughTransparency(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
}

OPENCL_FORCE_INLINE void Glossy2Material_GetEmittedRadiance(__global const Material* restrict material,
		__global const HitPoint *hitPoint,
		__global float *evalStack, uint *evalStackOffset
		MATERIALS_PARAM_DECL) {
	DefaultMaterial_GetEmittedRadiance(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
}

// The coating layered over the Lambertian base (see glossy2.cpp). The GPU
// traces eye paths only: the fixed direction is eyeDir.
OPENCL_FORCE_INLINE float3 Glossy2Material_GetKs(__global const Material* restrict material,
		__global const HitPoint *hitPoint
		TEXTURES_PARAM_DECL) {
	float3 ks = Texture_GetSpectrumValue(material->glossy2.ksTexIndex, hitPoint TEXTURES_PARAM);
	const float i = Texture_GetFloatValue(material->glossy2.indexTexIndex, hitPoint TEXTURES_PARAM);
	if (i > 0.f) {
		const float ti = (i - 1.f) / (i + 1.f);
		ks *= ti * ti;
	}

	return Spectrum_Clamp(ks);
}

OPENCL_FORCE_INLINE float3 Glossy2Material_EvaluateLayers(__global const Material* restrict material,
		__global const HitPoint *hitPoint, const float3 lightDir, const float3 eyeDir,
		BSDFEvent *event, float *directPdfW
		MATERIALS_PARAM_DECL) {
	const float3 fixedDir = eyeDir;
	const float3 sampledDir = lightDir;

	const float3 kd = Spectrum_Clamp(Texture_GetSpectrumValue(material->glossy2.kdTexIndex, hitPoint TEXTURES_PARAM));
	const float3 baseF = kd * M_1_PI_F * fabs(lightDir.z);

	if ((!material->glossy2.doublesided) && (fixedDir.z <= 0.f)) {
		// Back face: no coating
		*directPdfW = fabs(sampledDir.z * M_1_PI_F);
		*event = DIFFUSE | REFLECT;
		return baseF;
	}

	// Front face: coating + base
	*event = GLOSSY | REFLECT;

	const float3 ks = Glossy2Material_GetKs(material, hitPoint TEXTURES_PARAM);
	const float nuVal = Texture_GetFloatValue(material->glossy2.nuTexIndex, hitPoint TEXTURES_PARAM);
	const float nvVal = Texture_GetFloatValue(material->glossy2.nvTexIndex, hitPoint TEXTURES_PARAM);
	const float u = Microfacet_Regularize(clamp(nuVal, 1e-9f, 1.f), hitPoint->regularization);
	const float v = Microfacet_Regularize(clamp(nvVal, 1e-9f, 1.f), hitPoint->regularization);

	const float3 EcEye = GGXEnergy_CoatE3(u, v, fabs(eyeDir.z), ks);
	const float3 EcLight = GGXEnergy_CoatE3(u, v, fabs(lightDir.z), ks);
	const float3 EcAvg = GGXEnergy_CoatEavg3(u, v, ks);

	const float wCoating = Coating_Probability(EcEye, kd);
	*directPdfW = (1.f - wCoating) * fabs(sampledDir.z * M_1_PI_F) +
			wCoating * SchlickBSDF_CoatingPdf(u, v, fixedDir, sampledDir);

	// Absorption
	const float cosi = fabs(sampledDir.z);
	const float coso = fabs(fixedDir.z);
	const float3 alpha = Spectrum_Clamp(Texture_GetSpectrumValue(material->glossy2.kaTexIndex, hitPoint TEXTURES_PARAM));
	const float d = Texture_GetFloatValue(material->glossy2.depthTexIndex, hitPoint TEXTURES_PARAM);
	const float3 absorption = CoatingAbsorption(cosi, coso, alpha, d);

	const float3 coatingF = SchlickBSDF_CoatingF(ks, u, v, false, true, fixedDir, sampledDir);

	return coatingF + absorption * Coating_LayerScale(EcLight, EcEye, EcAvg) * baseF;
}

OPENCL_FORCE_INLINE void Glossy2Material_Evaluate(__global const Material* restrict material,
		__global const HitPoint *hitPoint,
		__global float *evalStack, uint *evalStackOffset
		MATERIALS_PARAM_DECL) {
	float3 lightDir, eyeDir;
	EvalStack_PopFloat3(eyeDir);
	EvalStack_PopFloat3(lightDir);

	BSDFEvent event;
	float directPdfW;
	const float3 result = Glossy2Material_EvaluateLayers(material, hitPoint, lightDir, eyeDir,
			&event, &directPdfW MATERIALS_PARAM);

	EvalStack_PushFloat3(result);
	EvalStack_PushBSDFEvent(event);
	EvalStack_PushFloat(directPdfW);
}

OPENCL_FORCE_INLINE void Glossy2Material_Sample(__global const Material* restrict material,
		__global const HitPoint *hitPoint,
		__global float *evalStack, uint *evalStackOffset
		MATERIALS_PARAM_DECL) {
	float u0, u1, passThroughEvent;
	EvalStack_PopFloat(passThroughEvent);
	EvalStack_PopFloat(u1);
	EvalStack_PopFloat(u0);
	float3 fixedDir;
	EvalStack_PopFloat3(fixedDir);

	if (fabs(fixedDir.z) < DEFAULT_COS_EPSILON_STATIC) {
		MATERIAL_SAMPLE_RETURN_BLACK;
	}

	bool sampleCoating = false;
	float u = 0.f, v = 0.f;
	if (material->glossy2.doublesided || (fixedDir.z > 0.f)) {
		// The same lobe probability as the evaluation
		const float3 ks = Glossy2Material_GetKs(material, hitPoint TEXTURES_PARAM);
		const float nuVal = Texture_GetFloatValue(material->glossy2.nuTexIndex, hitPoint TEXTURES_PARAM);
		const float nvVal = Texture_GetFloatValue(material->glossy2.nvTexIndex, hitPoint TEXTURES_PARAM);
		u = Microfacet_Regularize(clamp(nuVal, 1e-9f, 1.f), hitPoint->regularization);
		v = Microfacet_Regularize(clamp(nvVal, 1e-9f, 1.f), hitPoint->regularization);

		const float3 kd = Spectrum_Clamp(Texture_GetSpectrumValue(material->glossy2.kdTexIndex, hitPoint TEXTURES_PARAM));
		const float wCoating = Coating_Probability(GGXEnergy_CoatE3(u, v, fabs(fixedDir.z), ks), kd);
		sampleCoating = (passThroughEvent < wCoating);
	}

	float3 sampledDir;
	if (sampleCoating) {
		const float pdf = Microfacet_SampleReflect(u, v, fixedDir, u0, u1, &sampledDir);
		if ((pdf <= 0.f) || (fixedDir.z * sampledDir.z <= 0.f)) {
			MATERIAL_SAMPLE_RETURN_BLACK;
		}
	} else {
		float pdf;
		sampledDir = (signbit(fixedDir.z) ? -1.f : 1.f) * CosineSampleHemisphereWithPdf(u0, u1, &pdf);
	}
	if (fabs(sampledDir.z) < DEFAULT_COS_EPSILON_STATIC) {
		MATERIAL_SAMPLE_RETURN_BLACK;
	}

	BSDFEvent event;
	float pdfW;
	const float3 f = Glossy2Material_EvaluateLayers(material, hitPoint, sampledDir, fixedDir,
			&event, &pdfW MATERIALS_PARAM);
	if (Spectrum_IsBlack(f) || !(pdfW > 0.f)) {
		MATERIAL_SAMPLE_RETURN_BLACK;
	}

	const float3 result = f / pdfW;

	EvalStack_PushFloat3(result);
	EvalStack_PushFloat3(sampledDir);
	EvalStack_PushFloat(pdfW);
	EvalStack_PushBSDFEvent(event);
}

//------------------------------------------------------------------------------
// Material specific EvalOp
//------------------------------------------------------------------------------

OPENCL_FORCE_NOT_INLINE void Glossy2Material_EvalOp(
		__global const Material* restrict material,
		const MaterialEvalOpType evalType,
		__global float *evalStack,
		uint *evalStackOffset,
		__global const HitPoint *hitPoint
		MATERIALS_PARAM_DECL) {
	switch (evalType) {
		case EVAL_ALBEDO:
			Glossy2Material_Albedo(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		case EVAL_GET_INTERIOR_VOLUME:
			Glossy2Material_GetInteriorVolume(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		case EVAL_GET_EXTERIOR_VOLUME:
			Glossy2Material_GetExteriorVolume(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		case EVAL_GET_EMITTED_RADIANCE:
			Glossy2Material_GetEmittedRadiance(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		case EVAL_GET_PASS_TROUGH_TRANSPARENCY:
			Glossy2Material_GetPassThroughTransparency(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		case EVAL_EVALUATE:
			Glossy2Material_Evaluate(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		case EVAL_SAMPLE:
			Glossy2Material_Sample(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		default:
			// Something wrong here
			break;
	}
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
