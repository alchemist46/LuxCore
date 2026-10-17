#line 2 "materialdefs_funcs_mirror.cl"

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
// Mirror material
//------------------------------------------------------------------------------

OPENCL_FORCE_INLINE void MirrorMaterial_Albedo(__global const Material* restrict material,
		__global const HitPoint *hitPoint,
		__global float *evalStack, uint *evalStackOffset
		MATERIALS_PARAM_DECL) {
    const float3 albedo = WHITE;

	EvalStack_PushFloat3(albedo);
}

OPENCL_FORCE_INLINE void MirrorMaterial_GetInteriorVolume(__global const Material* restrict material,
		__global const HitPoint *hitPoint,
		__global float *evalStack, uint *evalStackOffset
		MATERIALS_PARAM_DECL) {
	DefaultMaterial_GetInteriorVolume(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
}

OPENCL_FORCE_INLINE void MirrorMaterial_GetExteriorVolume(__global const Material* restrict material,
		__global const HitPoint *hitPoint,
		__global float *evalStack, uint *evalStackOffset
		MATERIALS_PARAM_DECL) {
	DefaultMaterial_GetExteriorVolume(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
}

OPENCL_FORCE_INLINE void MirrorMaterial_GetPassThroughTransparency(__global const Material* restrict material,
		__global const HitPoint *hitPoint,
		__global float *evalStack, uint *evalStackOffset
		MATERIALS_PARAM_DECL) {
	DefaultMaterial_GetPassThroughTransparency(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
}

OPENCL_FORCE_INLINE void MirrorMaterial_GetEmittedRadiance(__global const Material* restrict material,
		__global const HitPoint *hitPoint,
		__global float *evalStack, uint *evalStackOffset
		MATERIALS_PARAM_DECL) {
	DefaultMaterial_GetEmittedRadiance(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
}

OPENCL_FORCE_INLINE void MirrorMaterial_Evaluate(__global const Material* restrict material,
		__global const HitPoint *hitPoint,
		__global float *evalStack, uint *evalStackOffset
		MATERIALS_PARAM_DECL) {
	float3 lightDir, eyeDir;
	EvalStack_PopFloat3(eyeDir);
	EvalStack_PopFloat3(lightDir);

	// Path regularization (HitPoint::regularization > 0): a GGX reflection
	// with the alpha floor and Kr as its Fresnel term
	if ((hitPoint->regularization > 0.f) && (lightDir.z * eyeDir.z > 0.f)) {
		const float alpha = Microfacet_Regularize(0.f, hitPoint->regularization);
		const float3 wh = normalize(lightDir + eyeDir);
		const float directPdfW = Microfacet_PdfReflect(alpha, alpha, eyeDir, lightDir);
		const float3 kr = Spectrum_Clamp(Texture_GetSpectrumValue(material->mirror.krTexIndex, hitPoint TEXTURES_PARAM));
		const float3 result = (Microfacet_D(alpha, alpha, wh) * Microfacet_G(alpha, alpha, lightDir, eyeDir) /
				(4.f * fabs(eyeDir.z))) * kr;
		const BSDFEvent event = GLOSSY | REFLECT;

		EvalStack_PushFloat3(result);
		EvalStack_PushBSDFEvent(event);
		EvalStack_PushFloat(directPdfW);
		return;
	}

	MATERIAL_EVALUATE_RETURN_BLACK;
}

OPENCL_FORCE_INLINE void MirrorMaterial_Sample(__global const Material* restrict material,
		__global const HitPoint *hitPoint,
		__global float *evalStack, uint *evalStackOffset
		MATERIALS_PARAM_DECL) {
	float u0, u1, passThroughEvent;
	EvalStack_PopFloat(passThroughEvent);
	EvalStack_PopFloat(u1);
	EvalStack_PopFloat(u0);
	float3 fixedDir;
	EvalStack_PopFloat3(fixedDir);

	const float3 kr = Spectrum_Clamp(Texture_GetSpectrumValue(material->mirror.krTexIndex, hitPoint TEXTURES_PARAM));

	// Path regularization (HitPoint::regularization > 0): a GGX reflection
	// with the alpha floor and Kr as its Fresnel term
	if (hitPoint->regularization > 0.f) {
		if (fabs(fixedDir.z) < DEFAULT_COS_EPSILON_STATIC) {
			MATERIAL_SAMPLE_RETURN_BLACK;
		}

		const float alpha = Microfacet_Regularize(0.f, hitPoint->regularization);
		float3 sampledDir;
		const float pdfW = Microfacet_SampleReflect(alpha, alpha, fixedDir, u0, u1, &sampledDir);

		const float cosi = fabs(sampledDir.z);
		if ((cosi < DEFAULT_COS_EPSILON_STATIC) || (fixedDir.z * sampledDir.z < 0.f) || (pdfW <= 0.f)) {
			MATERIAL_SAMPLE_RETURN_BLACK;
		}

		const float3 wh = normalize(fixedDir + sampledDir);
		const float G = Microfacet_G(alpha, alpha, fixedDir, sampledDir);
		const float d = Microfacet_D(alpha, alpha, wh);

		// f * |cos(sampled)| / pdf
		const float3 result = (d * G / (4.f * pdfW * fabs(fixedDir.z))) * kr;
		const BSDFEvent event = GLOSSY | REFLECT;

		EvalStack_PushFloat3(result);
		EvalStack_PushFloat3(sampledDir);
		EvalStack_PushFloat(pdfW);
		EvalStack_PushBSDFEvent(event);
		return;
	}

	const BSDFEvent event = SPECULAR | REFLECT;

	const float3 sampledDir = MAKE_FLOAT3(-fixedDir.x, -fixedDir.y, fixedDir.z);
	const float pdfW = 1.f;

	const float3 result = kr;

	EvalStack_PushFloat3(result);
	EvalStack_PushFloat3(sampledDir);
	EvalStack_PushFloat(pdfW);
	EvalStack_PushBSDFEvent(event);
}

//------------------------------------------------------------------------------
// Material specific EvalOp
//------------------------------------------------------------------------------

OPENCL_FORCE_NOT_INLINE void MirrorMaterial_EvalOp(
		__global const Material* restrict material,
		const MaterialEvalOpType evalType,
		__global float *evalStack,
		uint *evalStackOffset,
		__global const HitPoint *hitPoint
		MATERIALS_PARAM_DECL) {
	switch (evalType) {
		case EVAL_ALBEDO:
			MirrorMaterial_Albedo(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		case EVAL_GET_INTERIOR_VOLUME:
			MirrorMaterial_GetInteriorVolume(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		case EVAL_GET_EXTERIOR_VOLUME:
			MirrorMaterial_GetExteriorVolume(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		case EVAL_GET_EMITTED_RADIANCE:
			MirrorMaterial_GetEmittedRadiance(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		case EVAL_GET_PASS_TROUGH_TRANSPARENCY:
			MirrorMaterial_GetPassThroughTransparency(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		case EVAL_EVALUATE:
			MirrorMaterial_Evaluate(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		case EVAL_SAMPLE:
			MirrorMaterial_Sample(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		default:
			// Something wrong here
			break;
	}
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
