#line 2 "materialdefs_funcs_metal2.cl"

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
// Metal2 material
//
// LuxRender Metal2 material porting.
//------------------------------------------------------------------------------

OPENCL_FORCE_INLINE void Metal2Material_GetNK(__global const Material* restrict material, __global const HitPoint *hitPoint,
		float3 *n, float3 *k
		MATERIALS_PARAM_DECL) {
	const uint fresnelTexIndex = material->metal2.fresnelTexIndex;
	if (fresnelTexIndex != NULL_INDEX) {
		__global const Texture* restrict fresnelTex = &texs[fresnelTexIndex];

		if (fresnelTex->type == FRESNELCOLOR_TEX) {
			const float3 f = Texture_GetSpectrumValue(fresnelTex->fresnelColor.krIndex, hitPoint TEXTURES_PARAM);
			*n = FresnelApproxN3(f);
			*k = FresnelApproxK3(f);
		} else {
			*n = VLOAD3F(&fresnelTex->fresnelConst.n.c[0]);
			*k = VLOAD3F(&fresnelTex->fresnelConst.k.c[0]);
		}
	} else {
		*n = clamp(Texture_GetSpectrumValue(material->metal2.nTexIndex, hitPoint TEXTURES_PARAM), .001f, INFINITY);
		*k = clamp(Texture_GetSpectrumValue(material->metal2.kTexIndex, hitPoint TEXTURES_PARAM), .001f, INFINITY);
	}
}

// The multiple-scattering scale of the lobe seen with the cosine cosEye: the
// average Fresnel reflectance 2 * integral F(mu) mu dmu by a 4-point
// Gauss-Legendre quadrature on [0, 1]
__constant float Metal2Material_GLNodes[4] = { .069431844f, .330009478f, .669990522f, .930568156f };
__constant float Metal2Material_GLWeights[4] = { .173927423f, .326072577f, .326072577f, .173927423f };

OPENCL_FORCE_INLINE float3 Metal2Material_MSScale(__global const Material* restrict material, __global const HitPoint *hitPoint,
		const float u, const float v, const float cosEye
		MATERIALS_PARAM_DECL) {
	float3 n, k;
	Metal2Material_GetNK(material, hitPoint,
			&n, &k
			MATERIALS_PARAM);

	float3 Fss = BLACK;
	for (uint i = 0; i < 4; ++i)
		Fss += (2.f * Metal2Material_GLWeights[i] * Metal2Material_GLNodes[i]) *
				Spectrum_Clamp(FresnelGeneral_Evaluate(n, k, Metal2Material_GLNodes[i]));

	return GGXEnergy_ReflectScale3(u, v, cosEye, Spectrum_Clamp(Fss), false);
}

OPENCL_FORCE_INLINE void Metal2Material_Albedo(__global const Material* restrict material,
		__global const HitPoint *hitPoint,
		__global float *evalStack, uint *evalStackOffset
		MATERIALS_PARAM_DECL) {
	float3 n, k;
	Metal2Material_GetNK(material, hitPoint,
			&n, &k
			MATERIALS_PARAM);

	const float3 albedo = Spectrum_Clamp(FresnelGeneral_Evaluate(n, k, 1.f));

	EvalStack_PushFloat3(albedo);
}

OPENCL_FORCE_INLINE void Metal2Material_GetInteriorVolume(__global const Material* restrict material,
		__global const HitPoint *hitPoint,
		__global float *evalStack, uint *evalStackOffset
		MATERIALS_PARAM_DECL) {
	DefaultMaterial_GetInteriorVolume(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
}

OPENCL_FORCE_INLINE void Metal2Material_GetExteriorVolume(__global const Material* restrict material,
		__global const HitPoint *hitPoint,
		__global float *evalStack, uint *evalStackOffset
		MATERIALS_PARAM_DECL) {
	DefaultMaterial_GetExteriorVolume(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
}

OPENCL_FORCE_INLINE void Metal2Material_GetPassThroughTransparency(__global const Material* restrict material,
		__global const HitPoint *hitPoint,
		__global float *evalStack, uint *evalStackOffset
		MATERIALS_PARAM_DECL) {
	DefaultMaterial_GetPassThroughTransparency(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
}

OPENCL_FORCE_INLINE void Metal2Material_GetEmittedRadiance(__global const Material* restrict material,
		__global const HitPoint *hitPoint,
		__global float *evalStack, uint *evalStackOffset
		MATERIALS_PARAM_DECL) {
	DefaultMaterial_GetEmittedRadiance(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
}

OPENCL_FORCE_INLINE void Metal2Material_Evaluate(__global const Material* restrict material,
		__global const HitPoint *hitPoint,
		__global float *evalStack, uint *evalStackOffset
		MATERIALS_PARAM_DECL) {
	float3 lightDir, eyeDir;
	EvalStack_PopFloat3(eyeDir);
	EvalStack_PopFloat3(lightDir);

	const float uVal = Texture_GetFloatValue(material->metal2.nuTexIndex, hitPoint TEXTURES_PARAM);
	const float vVal = Texture_GetFloatValue(material->metal2.nvTexIndex, hitPoint TEXTURES_PARAM);

	const float u = Microfacet_Regularize(clamp(uVal, MICROFACET_MIN_ROUGHNESS, 1.f), hitPoint->regularization);
	const float v = Microfacet_Regularize(clamp(vVal, MICROFACET_MIN_ROUGHNESS, 1.f), hitPoint->regularization);

	const float3 wh = normalize(lightDir + eyeDir);
	const float cosWH = dot(lightDir, wh);

	const float directPdfW = Microfacet_PdfReflect(u, v, eyeDir, lightDir);

	float3 nVal, kVal;
	Metal2Material_GetNK(material, hitPoint,
			&nVal, &kVal
			MATERIALS_PARAM);

	const float3 F = Spectrum_Clamp(FresnelGeneral_Evaluate(nVal, kVal, cosWH));

	const float G = Microfacet_G(u, v, lightDir, eyeDir);

	const BSDFEvent event = GLOSSY | REFLECT;
	const float3 result = (Microfacet_D(u, v, wh) * G / (4.f * fabs(eyeDir.z))) * F *
			Metal2Material_MSScale(material, hitPoint, u, v, fabs(eyeDir.z) MATERIALS_PARAM);
	
	EvalStack_PushFloat3(result);
	EvalStack_PushBSDFEvent(event);
	EvalStack_PushFloat(directPdfW);
}

OPENCL_FORCE_INLINE void Metal2Material_Sample(__global const Material* restrict material,
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

	const float uVal = Texture_GetFloatValue(material->metal2.nuTexIndex, hitPoint TEXTURES_PARAM);
	const float vVal = Texture_GetFloatValue(material->metal2.nvTexIndex, hitPoint TEXTURES_PARAM);

	const float u = Microfacet_Regularize(clamp(uVal, MICROFACET_MIN_ROUGHNESS, 1.f), hitPoint->regularization);
	const float v = Microfacet_Regularize(clamp(vVal, MICROFACET_MIN_ROUGHNESS, 1.f), hitPoint->regularization);

	float3 sampledDir;
	const float pdfW = Microfacet_SampleReflect(u, v, fixedDir, u0, u1, &sampledDir);

	const float coso = fabs(fixedDir.z);
	const float cosi = fabs(sampledDir.z);
	if ((cosi < DEFAULT_COS_EPSILON_STATIC) || (fixedDir.z * sampledDir.z < 0.f)) {
		MATERIAL_SAMPLE_RETURN_BLACK;
	}

	if (pdfW <= 0.f) {
		MATERIAL_SAMPLE_RETURN_BLACK;
	}

	const float3 wh = normalize(fixedDir + sampledDir);
	const float cosWH = fabs(dot(fixedDir, wh));

	const float G = Microfacet_G(u, v, fixedDir, sampledDir);
	const float d = Microfacet_D(u, v, wh);

	float3 nVal, kVal;
	Metal2Material_GetNK(material, hitPoint,
			&nVal, &kVal
			MATERIALS_PARAM);

	const float3 F = Spectrum_Clamp(FresnelGeneral_Evaluate(nVal, kVal, cosWH));

	// f * |cos(sampled)| / pdf
	float factor = d * G / (4.f * pdfW);
	//if (!fromLight)
		factor /= coso;
	//else
	//	factor /= cosi;

	const BSDFEvent event = GLOSSY | REFLECT;

	const float3 result = factor * F *
			Metal2Material_MSScale(material, hitPoint, u, v, coso MATERIALS_PARAM);

	EvalStack_PushFloat3(result);
	EvalStack_PushFloat3(sampledDir);
	EvalStack_PushFloat(pdfW);
	EvalStack_PushBSDFEvent(event);
}

//------------------------------------------------------------------------------
// Material specific EvalOp
//------------------------------------------------------------------------------

OPENCL_FORCE_NOT_INLINE void Metal2Material_EvalOp(
		__global const Material* restrict material,
		const MaterialEvalOpType evalType,
		__global float *evalStack,
		uint *evalStackOffset,
		__global const HitPoint *hitPoint
		MATERIALS_PARAM_DECL) {
	switch (evalType) {
		case EVAL_ALBEDO:
			Metal2Material_Albedo(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		case EVAL_GET_INTERIOR_VOLUME:
			Metal2Material_GetInteriorVolume(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		case EVAL_GET_EXTERIOR_VOLUME:
			Metal2Material_GetExteriorVolume(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		case EVAL_GET_EMITTED_RADIANCE:
			Metal2Material_GetEmittedRadiance(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		case EVAL_GET_PASS_TROUGH_TRANSPARENCY:
			Metal2Material_GetPassThroughTransparency(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		case EVAL_EVALUATE:
			Metal2Material_Evaluate(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		case EVAL_SAMPLE:
			Metal2Material_Sample(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		default:
			// Something wrong here
			break;
	}
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
