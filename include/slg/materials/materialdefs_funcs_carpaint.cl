#line 2 "materialdefs_funcs_carpaint.cl"

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
// CarPaint material
//
// LuxRender carpaint material porting.
//------------------------------------------------------------------------------

OPENCL_FORCE_INLINE void CarPaintMaterial_Albedo(__global const Material* restrict material,
		__global const HitPoint *hitPoint,
		__global float *evalStack, uint *evalStackOffset
		MATERIALS_PARAM_DECL) {
    const float3 albedo = Spectrum_Clamp(Texture_GetSpectrumValue(material->carpaint.KdTexIndex, hitPoint TEXTURES_PARAM));

	EvalStack_PushFloat3(albedo);
}

OPENCL_FORCE_INLINE void CarPaintMaterial_GetInteriorVolume(__global const Material* restrict material,
		__global const HitPoint *hitPoint,
		__global float *evalStack, uint *evalStackOffset
		MATERIALS_PARAM_DECL) {
	DefaultMaterial_GetInteriorVolume(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
}

OPENCL_FORCE_INLINE void CarPaintMaterial_GetExteriorVolume(__global const Material* restrict material,
		__global const HitPoint *hitPoint,
		__global float *evalStack, uint *evalStackOffset
		MATERIALS_PARAM_DECL) {
	DefaultMaterial_GetExteriorVolume(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
}

OPENCL_FORCE_INLINE void CarPaintMaterial_GetPassThroughTransparency(__global const Material* restrict material,
		__global const HitPoint *hitPoint,
		__global float *evalStack, uint *evalStackOffset
		MATERIALS_PARAM_DECL) {
	DefaultMaterial_GetPassThroughTransparency(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
}

OPENCL_FORCE_INLINE void CarPaintMaterial_GetEmittedRadiance(__global const Material* restrict material,
		__global const HitPoint *hitPoint,
		__global float *evalStack, uint *evalStackOffset
		MATERIALS_PARAM_DECL) {
	DefaultMaterial_GetEmittedRadiance(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
}

// One glossy layer: adds its f |cos light| and its pdf. The GPU traces eye
// paths only, so the fixed direction is always eyeDir.
OPENCL_FORCE_INLINE void CarPaintMaterial_AddLobe(const float3 ks, const float m, const float r,
		const int multiscattering, const float3 lightDir, const float3 eyeDir, const float3 H,
		float3 *result, float *pdf) {
	const float coso = fabs(eyeDir.z);
	*result += (Microfacet_D(m, m, H) * Microfacet_G(m, m, lightDir, eyeDir) / (4.f * coso)) *
			(ks * FresnelSchlick_Evaluate(TO_FLOAT3(r), fabs(dot(eyeDir, H)))) *
			Microfacet_MSScale(multiscattering, m, m, coso, SchlickFresnel_Average(TO_FLOAT3(r)));
	*pdf += Microfacet_PdfReflect(m, m, eyeDir, lightDir);
}

// The diffuse layer plus the glossy lobes, all of them evaluated (their sum is
// the BSDF) and sampled with the same probability each (the mixture pdf).
// The glossy lobes exist only on the front face (see carpaint.cpp).
OPENCL_FORCE_INLINE float3 CarPaintMaterial_EvaluateMixture(__global const Material* restrict material,
		__global const HitPoint *hitPoint, const float3 lightDir, const float3 eyeDir,
		float *directPdfW
		MATERIALS_PARAM_DECL) {
	// Absorption
	const float cosi = fabs(lightDir.z);
	const float coso = fabs(eyeDir.z);
	const float3 kaVal = Texture_GetSpectrumValue(material->carpaint.KaTexIndex, hitPoint TEXTURES_PARAM);
	const float3 alpha = Spectrum_Clamp(kaVal);
	const float d = Texture_GetFloatValue(material->carpaint.depthTexIndex, hitPoint TEXTURES_PARAM);
	const float3 absorption = CoatingAbsorption(cosi, coso, alpha, d);

	// Diffuse layer
	const float3 kdVal = Texture_GetSpectrumValue(material->carpaint.KdTexIndex, hitPoint TEXTURES_PARAM);
	float3 result = absorption * Spectrum_Clamp(kdVal) * M_1_PI_F * cosi;
	float pdf = cosi * M_1_PI_F;
	uint n = 1;

	// Glossy layers
	if ((lightDir.z > 0.f) && (eyeDir.z > 0.f)) {
		const float3 H = normalize(lightDir + eyeDir);
		const int ms = material->multiscattering;

		const float3 ks1 = Spectrum_Clamp(Texture_GetSpectrumValue(material->carpaint.Ks1TexIndex, hitPoint TEXTURES_PARAM));
		const float m1 = Microfacet_Regularize(Texture_GetFloatValue(material->carpaint.M1TexIndex, hitPoint TEXTURES_PARAM), hitPoint->regularization);
		if ((Spectrum_Filter(ks1) > 0.f) && (m1 > 0.f)) {
			const float r1 = Texture_GetFloatValue(material->carpaint.R1TexIndex, hitPoint TEXTURES_PARAM);
			CarPaintMaterial_AddLobe(ks1, m1, r1, ms, lightDir, eyeDir, H, &result, &pdf);
			++n;
		}

		const float3 ks2 = Spectrum_Clamp(Texture_GetSpectrumValue(material->carpaint.Ks2TexIndex, hitPoint TEXTURES_PARAM));
		const float m2 = Microfacet_Regularize(Texture_GetFloatValue(material->carpaint.M2TexIndex, hitPoint TEXTURES_PARAM), hitPoint->regularization);
		if ((Spectrum_Filter(ks2) > 0.f) && (m2 > 0.f)) {
			const float r2 = Texture_GetFloatValue(material->carpaint.R2TexIndex, hitPoint TEXTURES_PARAM);
			CarPaintMaterial_AddLobe(ks2, m2, r2, ms, lightDir, eyeDir, H, &result, &pdf);
			++n;
		}

		const float3 ks3 = Spectrum_Clamp(Texture_GetSpectrumValue(material->carpaint.Ks3TexIndex, hitPoint TEXTURES_PARAM));
		const float m3 = Microfacet_Regularize(Texture_GetFloatValue(material->carpaint.M3TexIndex, hitPoint TEXTURES_PARAM), hitPoint->regularization);
		if ((Spectrum_Filter(ks3) > 0.f) && (m3 > 0.f)) {
			const float r3 = Texture_GetFloatValue(material->carpaint.R3TexIndex, hitPoint TEXTURES_PARAM);
			CarPaintMaterial_AddLobe(ks3, m3, r3, ms, lightDir, eyeDir, H, &result, &pdf);
			++n;
		}
	}

	*directPdfW = pdf / n;
	return result;
}

OPENCL_FORCE_INLINE void CarPaintMaterial_Evaluate(__global const Material* restrict material,
		__global const HitPoint *hitPoint,
		__global float *evalStack, uint *evalStackOffset
		MATERIALS_PARAM_DECL) {
	float3 lightDir, eyeDir;
	EvalStack_PopFloat3(eyeDir);
	EvalStack_PopFloat3(lightDir);

	float directPdfW;
	const float3 result = CarPaintMaterial_EvaluateMixture(material, hitPoint, lightDir, eyeDir,
			&directPdfW MATERIALS_PARAM);
	const BSDFEvent event = GLOSSY | REFLECT;

	EvalStack_PushFloat3(result);
	EvalStack_PushBSDFEvent(event);
	EvalStack_PushFloat(directPdfW);
}

OPENCL_FORCE_INLINE void CarPaintMaterial_Sample(__global const Material* restrict material,
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

	// The active glossy layers (front face only)
	float m1 = 0.f, m2 = 0.f, m3 = 0.f;
	bool l1 = false, l2 = false, l3 = false;
	if (fixedDir.z > 0.f) {
		const float3 ks1 = Spectrum_Clamp(Texture_GetSpectrumValue(material->carpaint.Ks1TexIndex, hitPoint TEXTURES_PARAM));
		m1 = Microfacet_Regularize(Texture_GetFloatValue(material->carpaint.M1TexIndex, hitPoint TEXTURES_PARAM), hitPoint->regularization);
		l1 = (Spectrum_Filter(ks1) > 0.f) && (m1 > 0.f);

		const float3 ks2 = Spectrum_Clamp(Texture_GetSpectrumValue(material->carpaint.Ks2TexIndex, hitPoint TEXTURES_PARAM));
		m2 = Microfacet_Regularize(Texture_GetFloatValue(material->carpaint.M2TexIndex, hitPoint TEXTURES_PARAM), hitPoint->regularization);
		l2 = (Spectrum_Filter(ks2) > 0.f) && (m2 > 0.f);

		const float3 ks3 = Spectrum_Clamp(Texture_GetSpectrumValue(material->carpaint.Ks3TexIndex, hitPoint TEXTURES_PARAM));
		m3 = Microfacet_Regularize(Texture_GetFloatValue(material->carpaint.M3TexIndex, hitPoint TEXTURES_PARAM), hitPoint->regularization);
		l3 = (Spectrum_Filter(ks3) > 0.f) && (m3 > 0.f);
	}

	// One layer, each with the same probability (as the pdf of Evaluate)
	const uint n = 1 + (l1 ? 1 : 0) + (l2 ? 1 : 0) + (l3 ? 1 : 0);
	const uint layer = min((uint)(passThroughEvent * n), n - 1);

	float3 sampledDir;
	if (layer == 0) {
		// Diffuse layer
		float pdf;
		sampledDir = (signbit(fixedDir.z) ? -1.f : 1.f) * CosineSampleHemisphereWithPdf(u0, u1, &pdf);
	} else {
		// The layer-th active glossy layer
		float m = 0.f;
		uint index = 0;
		if (l1 && (++index == layer))
			m = m1;
		if (l2 && (++index == layer))
			m = m2;
		if (l3 && (++index == layer))
			m = m3;

		const float pdf = Microfacet_SampleReflect(m, m, fixedDir, u0, u1, &sampledDir);
		if ((pdf <= 0.f) || (sampledDir.z < DEFAULT_COS_EPSILON_STATIC)) {
			MATERIAL_SAMPLE_RETURN_BLACK;
		}
	}
	if (fabs(sampledDir.z) < DEFAULT_COS_EPSILON_STATIC) {
		MATERIAL_SAMPLE_RETURN_BLACK;
	}

	// The value and the pdf of the whole mixture (eye path: fixedDir is the eye)
	float pdfW;
	const float3 f = CarPaintMaterial_EvaluateMixture(material, hitPoint, sampledDir, fixedDir,
			&pdfW MATERIALS_PARAM);
	if (Spectrum_IsBlack(f) || (pdfW <= 0.f)) {
		MATERIAL_SAMPLE_RETURN_BLACK;
	}

	const float3 result = f / pdfW;
	const BSDFEvent event = GLOSSY | REFLECT;

	EvalStack_PushFloat3(result);
	EvalStack_PushFloat3(sampledDir);
	EvalStack_PushFloat(pdfW);
	EvalStack_PushBSDFEvent(event);
}

//------------------------------------------------------------------------------
// Material specific EvalOp
//------------------------------------------------------------------------------

OPENCL_FORCE_NOT_INLINE void CarPaintMaterial_EvalOp(
		__global const Material* restrict material,
		const MaterialEvalOpType evalType,
		__global float *evalStack,
		uint *evalStackOffset,
		__global const HitPoint *hitPoint
		MATERIALS_PARAM_DECL) {
	switch (evalType) {
		case EVAL_ALBEDO:
			CarPaintMaterial_Albedo(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		case EVAL_GET_INTERIOR_VOLUME:
			CarPaintMaterial_GetInteriorVolume(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		case EVAL_GET_EXTERIOR_VOLUME:
			CarPaintMaterial_GetExteriorVolume(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		case EVAL_GET_EMITTED_RADIANCE:
			CarPaintMaterial_GetEmittedRadiance(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		case EVAL_GET_PASS_TROUGH_TRANSPARENCY:
			CarPaintMaterial_GetPassThroughTransparency(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		case EVAL_EVALUATE:
			CarPaintMaterial_Evaluate(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		case EVAL_SAMPLE:
			CarPaintMaterial_Sample(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		default:
			// Something wrong here
			break;
	}
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
