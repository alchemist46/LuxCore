#line 2 "materialdefs_funcs_toon.cl"

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

//------------------------------------------------------------------------------
// Toon material (see toon.cpp)
//------------------------------------------------------------------------------

OPENCL_FORCE_INLINE void ToonMaterial_Albedo(__global const Material* restrict material,
		__global const HitPoint *hitPoint,
		__global float *evalStack, uint *evalStackOffset
		MATERIALS_PARAM_DECL) {
	const float3 albedo = Spectrum_Clamp(Texture_GetSpectrumValue(material->toon.kdTexIndex,
			hitPoint TEXTURES_PARAM));

	EvalStack_PushFloat3(albedo);
}

OPENCL_FORCE_INLINE void ToonMaterial_GetInteriorVolume(__global const Material* restrict material,
		__global const HitPoint *hitPoint,
		__global float *evalStack, uint *evalStackOffset
		MATERIALS_PARAM_DECL) {
	DefaultMaterial_GetInteriorVolume(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
}

OPENCL_FORCE_INLINE void ToonMaterial_GetExteriorVolume(__global const Material* restrict material,
		__global const HitPoint *hitPoint,
		__global float *evalStack, uint *evalStackOffset
		MATERIALS_PARAM_DECL) {
	DefaultMaterial_GetExteriorVolume(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
}

OPENCL_FORCE_INLINE void ToonMaterial_GetPassThroughTransparency(__global const Material* restrict material,
		__global const HitPoint *hitPoint,
		__global float *evalStack, uint *evalStackOffset
		MATERIALS_PARAM_DECL) {
	DefaultMaterial_GetPassThroughTransparency(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
}

OPENCL_FORCE_INLINE void ToonMaterial_GetEmittedRadiance(__global const Material* restrict material,
		__global const HitPoint *hitPoint,
		__global float *evalStack, uint *evalStackOffset
		MATERIALS_PARAM_DECL) {
	DefaultMaterial_GetEmittedRadiance(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
}

// The direct light of an eye path: the toon part of the ramp direction
// rampDir (lightDir, or the direction of the centre of the light, see
// ToonMaterial::EvaluateToon())
OPENCL_FORCE_NOT_INLINE float3 ToonMaterial_EvaluateToon(__global const Material* restrict material,
		__global const HitPoint *hitPoint,
		const float3 localLightDir, const float3 localRampDir, const float3 localEyeDir
		MATERIALS_PARAM_DECL) {
	if (CosTheta(localLightDir) * CosTheta(localEyeDir) <= 0.f)
		return BLACK;

	// On the side of the eye direction
	const float side = signbit(localEyeDir.z) ? -1.f : 1.f;
	const float3 rampDir = side * localRampDir;
	const float3 eyeDir = side * localEyeDir;

	__global const ToonParam *toon = &material->toon;

	const float3 ramp = ToonMaterial_Ramp(toon, rampDir.z);
	float3 result = Spectrum_Clamp(Texture_GetSpectrumValue(toon->kdTexIndex, hitPoint TEXTURES_PARAM)) * ramp;
	if (toon->rampShadowFill)
		result += fmax(Texture_GetSpectrumValue(toon->shadowColorTexIndex, hitPoint TEXTURES_PARAM), BLACK) *
				fmax(WHITE - ramp, BLACK);

	const float3 ks = fmax(Texture_GetSpectrumValue(toon->ksTexIndex, hitPoint TEXTURES_PARAM), BLACK);
	if (!Spectrum_IsBlack(ks)) {
		const float roughness = Texture_GetFloatValue(toon->roughnessTexIndex, hitPoint TEXTURES_PARAM);
		result += ks * ToonMaterial_Highlight(toon, roughness, rampDir, eyeDir);
	}

	const float rimStep = ToonMaterial_RimStep(toon, eyeDir.z);
	if (rimStep > 0.f)
		result += fmax(Texture_GetSpectrumValue(toon->rimTexIndex, hitPoint TEXTURES_PARAM), BLACK) * rimStep;

	return result * M_1_PI_F;
}

OPENCL_FORCE_INLINE void ToonMaterial_Evaluate(__global const Material* restrict material,
		__global const HitPoint *hitPoint,
		__global float *evalStack, uint *evalStackOffset
		MATERIALS_PARAM_DECL) {
	float3 lightDir, eyeDir;
	EvalStack_PopFloat3(eyeDir);
	EvalStack_PopFloat3(lightDir);

	const float3 result = ToonMaterial_EvaluateToon(material, hitPoint, lightDir, lightDir, eyeDir
			MATERIALS_PARAM);
	if (Spectrum_IsBlack(result)) {
		MATERIAL_EVALUATE_RETURN_BLACK;
	}

	// The pdf of the Sample op: the composite materials need it (the path
	// tracers leave the toon part out of the MIS, see
	// BSDF_IsDirectLightSamplingOnly())
	const float directPdfW = fabs(lightDir.z * M_1_PI_F);

	const BSDFEvent event = DIFFUSE | REFLECT;

	EvalStack_PushFloat3(result);
	EvalStack_PushBSDFEvent(event);
	EvalStack_PushFloat(directPdfW);
}

// The indirect light: a matte material of colour kd
OPENCL_FORCE_INLINE void ToonMaterial_Sample(__global const Material* restrict material,
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

	float pdfW;
	float3 sampledDir = CosineSampleHemisphereWithPdf(u0, u1, &pdfW);
	if (CosTheta(sampledDir) < DEFAULT_COS_EPSILON_STATIC) {
		MATERIAL_SAMPLE_RETURN_BLACK;
	}
	sampledDir *= signbit(fixedDir.z) ? -1.f : 1.f;

	const float3 result = Spectrum_Clamp(Texture_GetSpectrumValue(material->toon.kdTexIndex,
			hitPoint TEXTURES_PARAM));
	const BSDFEvent event = DIFFUSE | REFLECT;

	EvalStack_PushFloat3(result);
	EvalStack_PushFloat3(sampledDir);
	EvalStack_PushFloat(pdfW);
	EvalStack_PushBSDFEvent(event);
}

//------------------------------------------------------------------------------
// Material specific EvalOp
//------------------------------------------------------------------------------

OPENCL_FORCE_NOT_INLINE void ToonMaterial_EvalOp(
		__global const Material* restrict material,
		const MaterialEvalOpType evalType,
		__global float *evalStack,
		uint *evalStackOffset,
		__global const HitPoint *hitPoint
		MATERIALS_PARAM_DECL) {
	switch (evalType) {
		case EVAL_ALBEDO:
			ToonMaterial_Albedo(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		case EVAL_GET_INTERIOR_VOLUME:
			ToonMaterial_GetInteriorVolume(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		case EVAL_GET_EXTERIOR_VOLUME:
			ToonMaterial_GetExteriorVolume(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		case EVAL_GET_EMITTED_RADIANCE:
			ToonMaterial_GetEmittedRadiance(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		case EVAL_GET_PASS_TROUGH_TRANSPARENCY:
			ToonMaterial_GetPassThroughTransparency(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		case EVAL_EVALUATE:
			ToonMaterial_Evaluate(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		case EVAL_SAMPLE:
			ToonMaterial_Sample(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		default:
			// Something wrong here
			break;
	}
}

// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
