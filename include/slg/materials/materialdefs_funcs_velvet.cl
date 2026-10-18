#line 2 "materialdefs_funcs_velvet.cl"

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
// Velvet material
//------------------------------------------------------------------------------

OPENCL_FORCE_INLINE void VelvetMaterial_Albedo(__global const Material* restrict material,
		__global const HitPoint *hitPoint,
		__global float *evalStack, uint *evalStackOffset
		MATERIALS_PARAM_DECL) {
	const float3 albedo = Spectrum_Clamp(Texture_GetSpectrumValue(material->velvet.kdTexIndex,
			hitPoint TEXTURES_PARAM));

	EvalStack_PushFloat3(albedo);
}

OPENCL_FORCE_INLINE void VelvetMaterial_GetInteriorVolume(__global const Material* restrict material,
		__global const HitPoint *hitPoint,
		__global float *evalStack, uint *evalStackOffset
		MATERIALS_PARAM_DECL) {
	DefaultMaterial_GetInteriorVolume(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
}

OPENCL_FORCE_INLINE void VelvetMaterial_GetExteriorVolume(__global const Material* restrict material,
		__global const HitPoint *hitPoint,
		__global float *evalStack, uint *evalStackOffset
		MATERIALS_PARAM_DECL) {
	DefaultMaterial_GetExteriorVolume(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
}

OPENCL_FORCE_INLINE void VelvetMaterial_GetPassThroughTransparency(__global const Material* restrict material,
		__global const HitPoint *hitPoint,
		__global float *evalStack, uint *evalStackOffset
		MATERIALS_PARAM_DECL) {
	DefaultMaterial_GetPassThroughTransparency(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
}

OPENCL_FORCE_INLINE void VelvetMaterial_GetEmittedRadiance(__global const Material* restrict material,
		__global const HitPoint *hitPoint,
		__global float *evalStack, uint *evalStackOffset
		MATERIALS_PARAM_DECL) {
	DefaultMaterial_GetEmittedRadiance(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
}

// The microfiber sheen (see velvet.cpp, sheen_funcs.cl). The GPU traces eye
// paths only: the eye direction is the fixed one and the LTC is sampled.
OPENCL_FORCE_INLINE float3 VelvetMaterial_EvaluateSheen(__global const Material* restrict material,
		__global const HitPoint *hitPoint, const float3 localLightDir, const float3 localEyeDir,
		float *directPdfW
		MATERIALS_PARAM_DECL) {
	*directPdfW = 0.f;
	if (localLightDir.z * localEyeDir.z <= 0.f)
		return BLACK;

	// Both faces: the back face is the mirror image of the front one
	const float flip = (localEyeDir.z < 0.f) ? -1.f : 1.f;
	const float3 eyeDir = MAKE_FLOAT3(localEyeDir.x, localEyeDir.y, flip * localEyeDir.z);
	const float3 lightDir = MAKE_FLOAT3(localLightDir.x, localLightDir.y, flip * localLightDir.z);

	float a, b, albedo;
	SheenLTC_Lookup(eyeDir.z, Texture_GetFloatValue(material->velvet.roughnessTexIndex, hitPoint TEXTURES_PARAM),
			&a, &b, &albedo);
	if ((fabs(a) < 1e-5f) || (albedo < 1e-5f))
		return BLACK;

	const float ltc = Sheen_LTC(a, b, eyeDir, lightDir);
	if (!(ltc > 0.f))
		return BLACK;

	*directPdfW = ltc;
	const float3 kdVal = Texture_GetSpectrumValue(material->velvet.kdTexIndex, hitPoint TEXTURES_PARAM);
	return Spectrum_Clamp(kdVal) * (albedo * ltc);
}

OPENCL_FORCE_INLINE void VelvetMaterial_Evaluate(__global const Material* restrict material,
		__global const HitPoint *hitPoint,
		__global float *evalStack, uint *evalStackOffset
		MATERIALS_PARAM_DECL) {
	float3 lightDir, eyeDir;
	EvalStack_PopFloat3(eyeDir);
	EvalStack_PopFloat3(lightDir);

	float directPdfW;
	const float3 result = VelvetMaterial_EvaluateSheen(material, hitPoint, lightDir, eyeDir,
			&directPdfW MATERIALS_PARAM);
	if (Spectrum_IsBlack(result)) {
		MATERIAL_EVALUATE_RETURN_BLACK;
	}
	const BSDFEvent event = DIFFUSE | REFLECT;

	EvalStack_PushFloat3(result);
	EvalStack_PushBSDFEvent(event);
	EvalStack_PushFloat(directPdfW);
}

OPENCL_FORCE_INLINE void VelvetMaterial_Sample(__global const Material* restrict material,
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

	const float flip = (fixedDir.z < 0.f) ? -1.f : 1.f;
	const float3 eyeDir = MAKE_FLOAT3(fixedDir.x, fixedDir.y, flip * fixedDir.z);

	float a, b, albedo;
	SheenLTC_Lookup(eyeDir.z, Texture_GetFloatValue(material->velvet.roughnessTexIndex, hitPoint TEXTURES_PARAM),
			&a, &b, &albedo);
	if ((fabs(a) < 1e-5f) || (albedo < 1e-5f)) {
		MATERIAL_SAMPLE_RETURN_BLACK;
	}

	const float3 lightDir = Sheen_SampleLTC(a, b, eyeDir, u0, u1);
	if (lightDir.z < DEFAULT_COS_EPSILON_STATIC) {
		MATERIAL_SAMPLE_RETURN_BLACK;
	}
	const float3 sampledDir = MAKE_FLOAT3(lightDir.x, lightDir.y, flip * lightDir.z);

	float pdfW;
	const float3 f = VelvetMaterial_EvaluateSheen(material, hitPoint, sampledDir, fixedDir,
			&pdfW MATERIALS_PARAM);
	if (Spectrum_IsBlack(f) || !(pdfW > 0.f)) {
		MATERIAL_SAMPLE_RETURN_BLACK;
	}

	const float3 result = f / pdfW;
	const BSDFEvent event = DIFFUSE | REFLECT;

	EvalStack_PushFloat3(result);
	EvalStack_PushFloat3(sampledDir);
	EvalStack_PushFloat(pdfW);
	EvalStack_PushBSDFEvent(event);
}

//------------------------------------------------------------------------------
// Material specific EvalOp
//------------------------------------------------------------------------------

OPENCL_FORCE_NOT_INLINE void VelvetMaterial_EvalOp(
		__global const Material* restrict material,
		const MaterialEvalOpType evalType,
		__global float *evalStack,
		uint *evalStackOffset,
		__global const HitPoint *hitPoint
		MATERIALS_PARAM_DECL) {
	switch (evalType) {
		case EVAL_ALBEDO:
			VelvetMaterial_Albedo(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		case EVAL_GET_INTERIOR_VOLUME:
			VelvetMaterial_GetInteriorVolume(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		case EVAL_GET_EXTERIOR_VOLUME:
			VelvetMaterial_GetExteriorVolume(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		case EVAL_GET_EMITTED_RADIANCE:
			VelvetMaterial_GetEmittedRadiance(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		case EVAL_GET_PASS_TROUGH_TRANSPARENCY:
			VelvetMaterial_GetPassThroughTransparency(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		case EVAL_EVALUATE:
			VelvetMaterial_Evaluate(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		case EVAL_SAMPLE:
			VelvetMaterial_Sample(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		default:
			// Something wrong here
			break;
	}
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
