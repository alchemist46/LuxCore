#line 2 "materialdefs_funcs_roughmatte_translucent.cl"

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
// MatteTranslucent material
//------------------------------------------------------------------------------

OPENCL_FORCE_INLINE void RoughMatteTranslucentMaterial_Albedo(__global const Material* restrict material,
		__global const HitPoint *hitPoint,
		__global float *evalStack, uint *evalStackOffset
		MATERIALS_PARAM_DECL) {
	const float3 r = Spectrum_Clamp(Texture_GetSpectrumValue(material->roughmatteTranslucent.krTexIndex, hitPoint TEXTURES_PARAM));
	const float3 t = Spectrum_Clamp(Texture_GetSpectrumValue(material->roughmatteTranslucent.ktTexIndex, hitPoint TEXTURES_PARAM)) * 
		// Energy conservation
		(1.f - r);

    const float3 albedo = r + t;

	EvalStack_PushFloat3(albedo);
}

OPENCL_FORCE_INLINE void RoughMatteTranslucentMaterial_GetInteriorVolume(__global const Material* restrict material,
		__global const HitPoint *hitPoint,
		__global float *evalStack, uint *evalStackOffset
		MATERIALS_PARAM_DECL) {
	DefaultMaterial_GetInteriorVolume(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
}

OPENCL_FORCE_INLINE void RoughMatteTranslucentMaterial_GetExteriorVolume(__global const Material* restrict material,
		__global const HitPoint *hitPoint,
		__global float *evalStack, uint *evalStackOffset
		MATERIALS_PARAM_DECL) {
	DefaultMaterial_GetExteriorVolume(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
}

OPENCL_FORCE_INLINE void RoughMatteTranslucentMaterial_GetPassThroughTransparency(__global const Material* restrict material,
		__global const HitPoint *hitPoint,
		__global float *evalStack, uint *evalStackOffset
		MATERIALS_PARAM_DECL) {
	DefaultMaterial_GetPassThroughTransparency(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
}

OPENCL_FORCE_INLINE void RoughMatteTranslucentMaterial_GetEmittedRadiance(__global const Material* restrict material,
		__global const HitPoint *hitPoint,
		__global float *evalStack, uint *evalStackOffset
		MATERIALS_PARAM_DECL) {
	DefaultMaterial_GetEmittedRadiance(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
}

// EON lobes of albedo 1 scaled by r and t = kt (1 - kr), the transmission one
// evaluated with the light direction mirrored through the surface (see
// roughmattetranslucent.cpp). The GPU traces eye paths only: the sampled
// direction is lightDir.
OPENCL_FORCE_INLINE float3 RoughMatteTranslucentMaterial_EvaluateMixture(__global const Material* restrict material,
		__global const HitPoint *hitPoint, const float3 lightDir, const float3 eyeDir,
		BSDFEvent *event, float *directPdfW
		MATERIALS_PARAM_DECL) {
	*directPdfW = 0.f;

	const float3 r = Spectrum_Clamp(Texture_GetSpectrumValue(material->roughmatteTranslucent.krTexIndex, hitPoint TEXTURES_PARAM));
	const float3 t = Spectrum_Clamp(Texture_GetSpectrumValue(material->roughmatteTranslucent.ktTexIndex, hitPoint TEXTURES_PARAM)) * 
		// Energy conservation
		(1.f - r);

	const float rf = Spectrum_Filter(r);
	const float tf = Spectrum_Filter(t);
	if (!(rf + tf > 0.f))
		return BLACK;
	const float pReflect = rf / (rf + tf);

	const bool reflected = (CosTheta(lightDir) * CosTheta(eyeDir) > 0.f);
	*directPdfW = (reflected ? pReflect : (1.f - pReflect)) * fabs(lightDir.z * M_1_PI_F);

	const float roughness = clamp(Texture_GetFloatValue(material->roughmatteTranslucent.sigmaTexIndex, hitPoint TEXTURES_PARAM), 0.f, 1.f);
	if (reflected) {
		*event = DIFFUSE | REFLECT;
		return r * EON_Evaluate(WHITE, roughness, lightDir, eyeDir) * fabs(lightDir.z);
	} else {
		*event = DIFFUSE | TRANSMIT;
		const float3 mirroredLightDir = MAKE_FLOAT3(lightDir.x, lightDir.y, -lightDir.z);
		return t * EON_Evaluate(WHITE, roughness, mirroredLightDir, eyeDir) * fabs(lightDir.z);
	}
}

OPENCL_FORCE_INLINE void RoughMatteTranslucentMaterial_Evaluate(__global const Material* restrict material,
		__global const HitPoint *hitPoint,
		__global float *evalStack, uint *evalStackOffset
		MATERIALS_PARAM_DECL) {
	float3 lightDir, eyeDir;
	EvalStack_PopFloat3(eyeDir);
	EvalStack_PopFloat3(lightDir);

	BSDFEvent event;
	float directPdfW;
	const float3 result = RoughMatteTranslucentMaterial_EvaluateMixture(material, hitPoint,
			lightDir, eyeDir, &event, &directPdfW MATERIALS_PARAM);
	if (Spectrum_IsBlack(result)) {
		MATERIAL_EVALUATE_RETURN_BLACK;
	}

	EvalStack_PushFloat3(result);
	EvalStack_PushBSDFEvent(event);
	EvalStack_PushFloat(directPdfW);
}

OPENCL_FORCE_INLINE void RoughMatteTranslucentMaterial_Sample(__global const Material* restrict material,
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

	const float3 r = Spectrum_Clamp(Texture_GetSpectrumValue(material->roughmatteTranslucent.krTexIndex, hitPoint TEXTURES_PARAM));
	const float3 t = Spectrum_Clamp(Texture_GetSpectrumValue(material->roughmatteTranslucent.ktTexIndex, hitPoint TEXTURES_PARAM)) * 
		(1.f - r);
	const float rf = Spectrum_Filter(r);
	const float tf = Spectrum_Filter(t);
	if (!(rf + tf > 0.f)) {
		MATERIAL_SAMPLE_RETURN_BLACK;
	}
	const float pReflect = rf / (rf + tf);

	float cosPdf;
	float3 sampledDir = CosineSampleHemisphereWithPdf(u0, u1, &cosPdf);
	if (fabs(CosTheta(sampledDir)) < DEFAULT_COS_EPSILON_STATIC) {
		MATERIAL_SAMPLE_RETURN_BLACK;
	}

	// Only the z flips: a diffuse transmission keeps the azimuth
	const float fixedSign = signbit(fixedDir.z) ? -1.f : 1.f;
	sampledDir.z *= (passThroughEvent < pReflect) ? fixedSign : -fixedSign;

	BSDFEvent event;
	float pdfW;
	const float3 f = RoughMatteTranslucentMaterial_EvaluateMixture(material, hitPoint,
			sampledDir, fixedDir, &event, &pdfW MATERIALS_PARAM);
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

OPENCL_FORCE_NOT_INLINE void RoughMatteTranslucentMaterial_EvalOp(
		__global const Material* restrict material,
		const MaterialEvalOpType evalType,
		__global float *evalStack,
		uint *evalStackOffset,
		__global const HitPoint *hitPoint
		MATERIALS_PARAM_DECL) {
	switch (evalType) {
		case EVAL_ALBEDO:
			RoughMatteTranslucentMaterial_Albedo(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		case EVAL_GET_INTERIOR_VOLUME:
			RoughMatteTranslucentMaterial_GetInteriorVolume(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		case EVAL_GET_EXTERIOR_VOLUME:
			RoughMatteTranslucentMaterial_GetExteriorVolume(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		case EVAL_GET_EMITTED_RADIANCE:
			RoughMatteTranslucentMaterial_GetEmittedRadiance(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		case EVAL_GET_PASS_TROUGH_TRANSPARENCY:
			RoughMatteTranslucentMaterial_GetPassThroughTransparency(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		case EVAL_EVALUATE:
			RoughMatteTranslucentMaterial_Evaluate(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		case EVAL_SAMPLE:
			RoughMatteTranslucentMaterial_Sample(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		default:
			// Something wrong here
			break;
	}
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
