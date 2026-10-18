#line 2 "materialdefs_funcs_glossytranslucent.cl"

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
// GlossyTranslucent material
//
// LuxRender GlossyTranslucent material porting.
//------------------------------------------------------------------------------

OPENCL_FORCE_INLINE void GlossyTranslucentMaterial_Albedo(__global const Material* restrict material,
		__global const HitPoint *hitPoint,
		__global float *evalStack, uint *evalStackOffset
		MATERIALS_PARAM_DECL) {
    const float3 albedo = Spectrum_Clamp(Texture_GetSpectrumValue(material->glossytranslucent.kdTexIndex, hitPoint TEXTURES_PARAM));

	EvalStack_PushFloat3(albedo);
}

OPENCL_FORCE_INLINE void GlossyTranslucentMaterial_GetInteriorVolume(__global const Material* restrict material,
		__global const HitPoint *hitPoint,
		__global float *evalStack, uint *evalStackOffset
		MATERIALS_PARAM_DECL) {
	DefaultMaterial_GetInteriorVolume(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
}

OPENCL_FORCE_INLINE void GlossyTranslucentMaterial_GetExteriorVolume(__global const Material* restrict material,
		__global const HitPoint *hitPoint,
		__global float *evalStack, uint *evalStackOffset
		MATERIALS_PARAM_DECL) {
	DefaultMaterial_GetExteriorVolume(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
}

OPENCL_FORCE_INLINE void GlossyTranslucentMaterial_GetPassThroughTransparency(__global const Material* restrict material,
		__global const HitPoint *hitPoint,
		__global float *evalStack, uint *evalStackOffset
		MATERIALS_PARAM_DECL) {
	DefaultMaterial_GetPassThroughTransparency(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
}

OPENCL_FORCE_INLINE void GlossyTranslucentMaterial_GetEmittedRadiance(__global const Material* restrict material,
		__global const HitPoint *hitPoint,
		__global float *evalStack, uint *evalStackOffset
		MATERIALS_PARAM_DECL) {
	DefaultMaterial_GetEmittedRadiance(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
}

// The two coatings layered over the diffuse slab (see glossytranslucent.cpp).
// The GPU traces eye paths only: the fixed direction is eyeDir.
OPENCL_FORCE_INLINE void GlossyTranslucentMaterial_GetSide(__global const Material* restrict material,
		__global const HitPoint *hitPoint, const bool front,
		float3 *ks, float3 *alpha, float *u, float *v, float *d
		TEXTURES_PARAM_DECL) {
	*ks = Texture_GetSpectrumValue(front ? material->glossytranslucent.ksTexIndex : material->glossytranslucent.ksbfTexIndex,
			hitPoint TEXTURES_PARAM);
	const float i = Texture_GetFloatValue(front ? material->glossytranslucent.indexTexIndex : material->glossytranslucent.indexbfTexIndex,
			hitPoint TEXTURES_PARAM);
	if (i > 0.f) {
		const float ti = (i - 1.f) / (i + 1.f);
		*ks *= ti * ti;
	}
	*ks = Spectrum_Clamp(*ks);

	const float nuVal = Texture_GetFloatValue(front ? material->glossytranslucent.nuTexIndex : material->glossytranslucent.nubfTexIndex,
			hitPoint TEXTURES_PARAM);
	const float nvVal = Texture_GetFloatValue(front ? material->glossytranslucent.nvTexIndex : material->glossytranslucent.nvbfTexIndex,
			hitPoint TEXTURES_PARAM);
	*u = Microfacet_Regularize(clamp(nuVal, 1e-9f, 1.f), hitPoint->regularization);
	*v = Microfacet_Regularize(clamp(nvVal, 1e-9f, 1.f), hitPoint->regularization);

	*alpha = Spectrum_Clamp(Texture_GetSpectrumValue(front ? material->glossytranslucent.kaTexIndex : material->glossytranslucent.kabfTexIndex,
			hitPoint TEXTURES_PARAM));
	*d = Texture_GetFloatValue(front ? material->glossytranslucent.depthTexIndex : material->glossytranslucent.depthbfTexIndex,
			hitPoint TEXTURES_PARAM);
}

OPENCL_FORCE_INLINE void GlossyTranslucentMaterial_LobeProbabilities(const float3 Ec, const float3 kd, const float3 t,
		float *pCoat, float *pReflect, float *pTransmit) {
	const float coat = Spectrum_Filter(Ec);
	const float reflect = Spectrum_Filter((WHITE - Ec) * kd);
	const float transmit = Spectrum_Filter((WHITE - Ec) * t);
	const float sum = coat + reflect + transmit;
	if (sum > 0.f) {
		*pCoat = coat / sum;
		*pReflect = reflect / sum;
		*pTransmit = transmit / sum;
	} else {
		*pCoat = 0.f;
		*pReflect = 0.f;
		*pTransmit = 0.f;
	}
}

OPENCL_FORCE_INLINE float3 GlossyTranslucentMaterial_EvaluateLayers(__global const Material* restrict material,
		__global const HitPoint *hitPoint, const float3 lightDir, const float3 eyeDir,
		BSDFEvent *event, float *directPdfW
		MATERIALS_PARAM_DECL) {
	*directPdfW = 0.f;

	const float cosLight = fabs(lightDir.z);
	const float cosEye = fabs(eyeDir.z);
	if ((cosLight < DEFAULT_COS_EPSILON_STATIC) || (cosEye < DEFAULT_COS_EPSILON_STATIC))
		return BLACK;

	const float3 kd = Spectrum_Clamp(Texture_GetSpectrumValue(material->glossytranslucent.kdTexIndex, hitPoint TEXTURES_PARAM));
	const float3 t = Spectrum_Clamp(Texture_GetSpectrumValue(material->glossytranslucent.ktTexIndex, hitPoint TEXTURES_PARAM)) *
			(WHITE - kd);

	float3 ksL, alphaL, ksE, alphaE;
	float uL, vL, dL, uE, vE, dE;
	GlossyTranslucentMaterial_GetSide(material, hitPoint, lightDir.z > 0.f, &ksL, &alphaL, &uL, &vL, &dL TEXTURES_PARAM);
	GlossyTranslucentMaterial_GetSide(material, hitPoint, eyeDir.z > 0.f, &ksE, &alphaE, &uE, &vE, &dE TEXTURES_PARAM);

	const float3 EcLight = GGXEnergy_CoatE3(uL, vL, cosLight, ksL);
	const float3 EcEye = GGXEnergy_CoatE3(uE, vE, cosEye, ksE);
	const float3 EcLightAvg = GGXEnergy_CoatEavg3(uL, vL, ksL);

	// The pdf of generating lightDir from eyeDir
	float pCoat, pReflect, pTransmit;
	GlossyTranslucentMaterial_LobeProbabilities(EcEye, kd, t, &pCoat, &pReflect, &pTransmit);

	const float3 layer = Coating_LayerScale(EcLight, EcEye, EcLightAvg);

	if (lightDir.z * eyeDir.z < 0.f) {
		// Transmission: one crossing of each coating
		*event = DIFFUSE | TRANSMIT;
		*directPdfW = pTransmit * cosLight * M_1_PI_F;

		const float3 absorption = Spectrum_Exp(alphaL * -(dL / cosLight) + alphaE * -(dE / cosEye));

		return absorption * layer * t * (M_1_PI_F * cosLight);
	} else {
		// Reflection on the side of both directions
		*event = GLOSSY | REFLECT;
		*directPdfW = pReflect * cosLight * M_1_PI_F + pCoat * SchlickBSDF_CoatingPdf(uE, vE, eyeDir, lightDir);

		const float3 absorption = CoatingAbsorption(cosLight, cosEye, alphaE, dE);
		const float3 coatingF = SchlickBSDF_CoatingF(ksE, uE, vE, eyeDir, lightDir);

		return coatingF + absorption * layer * kd * (M_1_PI_F * cosLight);
	}
}

OPENCL_FORCE_INLINE void GlossyTranslucentMaterial_Evaluate(__global const Material* restrict material,
		__global const HitPoint *hitPoint,
		__global float *evalStack, uint *evalStackOffset
		MATERIALS_PARAM_DECL) {
	float3 lightDir, eyeDir;
	EvalStack_PopFloat3(eyeDir);
	EvalStack_PopFloat3(lightDir);

	BSDFEvent event;
	float directPdfW;
	const float3 result = GlossyTranslucentMaterial_EvaluateLayers(material, hitPoint, lightDir, eyeDir,
			&event, &directPdfW MATERIALS_PARAM);
	if (Spectrum_IsBlack(result)) {
		MATERIAL_EVALUATE_RETURN_BLACK;
	}

	EvalStack_PushFloat3(result);
	EvalStack_PushBSDFEvent(event);
	EvalStack_PushFloat(directPdfW);
}

OPENCL_FORCE_INLINE void GlossyTranslucentMaterial_Sample(__global const Material* restrict material,
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

	const float3 kd = Spectrum_Clamp(Texture_GetSpectrumValue(material->glossytranslucent.kdTexIndex, hitPoint TEXTURES_PARAM));
	const float3 t = Spectrum_Clamp(Texture_GetSpectrumValue(material->glossytranslucent.ktTexIndex, hitPoint TEXTURES_PARAM)) *
			(WHITE - kd);

	float3 ks, alpha;
	float u, v, d;
	GlossyTranslucentMaterial_GetSide(material, hitPoint, fixedDir.z > 0.f, &ks, &alpha, &u, &v, &d TEXTURES_PARAM);

	float pCoat, pReflect, pTransmit;
	GlossyTranslucentMaterial_LobeProbabilities(GGXEnergy_CoatE3(u, v, fabs(fixedDir.z), ks), kd, t,
			&pCoat, &pReflect, &pTransmit);
	if (pCoat + pReflect + pTransmit <= 0.f) {
		MATERIAL_SAMPLE_RETURN_BLACK;
	}

	float3 sampledDir;
	if (passThroughEvent < pCoat) {
		const float pdf = Microfacet_SampleReflect(u, v, fixedDir, u0, u1, &sampledDir);
		if ((pdf <= 0.f) || (fixedDir.z * sampledDir.z <= 0.f)) {
			MATERIAL_SAMPLE_RETURN_BLACK;
		}
	} else {
		float pdf;
		sampledDir = CosineSampleHemisphereWithPdf(u0, u1, &pdf);
		// A diffuse transmission keeps the azimuth: only z flips
		const float fixedSign = signbit(fixedDir.z) ? -1.f : 1.f;
		sampledDir.z *= (passThroughEvent < pCoat + pReflect) ? fixedSign : -fixedSign;
	}
	if (fabs(sampledDir.z) < DEFAULT_COS_EPSILON_STATIC) {
		MATERIAL_SAMPLE_RETURN_BLACK;
	}

	BSDFEvent event;
	float pdfW;
	const float3 f = GlossyTranslucentMaterial_EvaluateLayers(material, hitPoint, sampledDir, fixedDir,
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

OPENCL_FORCE_NOT_INLINE void GlossyTranslucentMaterial_EvalOp(
		__global const Material* restrict material,
		const MaterialEvalOpType evalType,
		__global float *evalStack,
		uint *evalStackOffset,
		__global const HitPoint *hitPoint
		MATERIALS_PARAM_DECL) {
	switch (evalType) {
		case EVAL_ALBEDO:
			GlossyTranslucentMaterial_Albedo(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		case EVAL_GET_INTERIOR_VOLUME:
			GlossyTranslucentMaterial_GetInteriorVolume(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		case EVAL_GET_EXTERIOR_VOLUME:
			GlossyTranslucentMaterial_GetExteriorVolume(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		case EVAL_GET_EMITTED_RADIANCE:
			GlossyTranslucentMaterial_GetEmittedRadiance(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		case EVAL_GET_PASS_TROUGH_TRANSPARENCY:
			GlossyTranslucentMaterial_GetPassThroughTransparency(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		case EVAL_EVALUATE:
			GlossyTranslucentMaterial_Evaluate(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		case EVAL_SAMPLE:
			GlossyTranslucentMaterial_Sample(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		default:
			// Something wrong here
			break;
	}
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
