#line 2 "materialdefs_funcs_subsurface.cl"

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
// Random-walk subsurface scattering material (see subsurface.cpp). The GPU
// traces eye paths only: the fixed direction is eyeDir.
//------------------------------------------------------------------------------

OPENCL_FORCE_INLINE void SubsurfaceMaterial_Albedo(__global const Material* restrict material,
		__global const HitPoint *hitPoint,
		__global float *evalStack, uint *evalStackOffset
		MATERIALS_PARAM_DECL) {
	const float3 albedo = Spectrum_Clamp(Texture_GetSpectrumValue(material->subsurface.colorTexIndex,
			hitPoint TEXTURES_PARAM));

	EvalStack_PushFloat3(albedo);
}

OPENCL_FORCE_INLINE void SubsurfaceMaterial_GetInteriorVolume(__global const Material* restrict material,
		__global const HitPoint *hitPoint,
		__global float *evalStack, uint *evalStackOffset
		MATERIALS_PARAM_DECL) {
	DefaultMaterial_GetInteriorVolume(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
}

OPENCL_FORCE_INLINE void SubsurfaceMaterial_GetExteriorVolume(__global const Material* restrict material,
		__global const HitPoint *hitPoint,
		__global float *evalStack, uint *evalStackOffset
		MATERIALS_PARAM_DECL) {
	DefaultMaterial_GetExteriorVolume(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
}

OPENCL_FORCE_INLINE void SubsurfaceMaterial_GetPassThroughTransparency(__global const Material* restrict material,
		__global const HitPoint *hitPoint,
		__global float *evalStack, uint *evalStackOffset
		MATERIALS_PARAM_DECL) {
	DefaultMaterial_GetPassThroughTransparency(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
}

OPENCL_FORCE_INLINE void SubsurfaceMaterial_GetEmittedRadiance(__global const Material* restrict material,
		__global const HitPoint *hitPoint,
		__global float *evalStack, uint *evalStackOffset
		MATERIALS_PARAM_DECL) {
	DefaultMaterial_GetEmittedRadiance(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
}

// The GGX alpha and the Schlick F0 of the reflection (F0 < 0: none)
OPENCL_FORCE_INLINE void SubsurfaceMaterial_GetReflection(__global const Material* restrict material,
		__global const HitPoint *hitPoint, float *alpha, float *F0
		TEXTURES_PARAM_DECL) {
	const float i = Texture_GetFloatValue(material->subsurface.iorTexIndex, hitPoint TEXTURES_PARAM);
	if (i > 1.f) {
		const float t = (i - 1.f) / (i + 1.f);
		*F0 = t * t;
	} else
		*F0 = -1.f;

	const float r = clamp(Texture_GetFloatValue(material->subsurface.roughnessTexIndex, hitPoint TEXTURES_PARAM), 0.f, 1.f);
	*alpha = Microfacet_Regularize(clamp(r * r, MICROFACET_MIN_ROUGHNESS, 1.f), hitPoint->regularization);
}

OPENCL_FORCE_INLINE float3 SubsurfaceMaterial_EvaluateLayers(__global const Material* restrict material,
		__global const HitPoint *hitPoint, const float3 lightDir, const float3 eyeDir,
		BSDFEvent *event, float *directPdfW
		MATERIALS_PARAM_DECL) {
	const float3 fixedDir = eyeDir;
	const float3 sampledDir = lightDir;
	*directPdfW = 0.f;

	if (hitPoint->subsurface == HITPOINT_SSS_EXIT) {
		// The exit of a walk: a Lambertian transmission
		if (lightDir.z * eyeDir.z >= 0.f)
			return BLACK;

		*directPdfW = fabs(sampledDir.z * M_1_PI_F);
		*event = DIFFUSE | TRANSMIT;
		return TO_FLOAT3(M_1_PI_F * fabs(lightDir.z));
	}

	// No transmission to evaluate
	if (lightDir.z * eyeDir.z <= 0.f)
		return BLACK;

	const float3 kd = Spectrum_Clamp(Texture_GetSpectrumValue(material->subsurface.colorTexIndex, hitPoint TEXTURES_PARAM));
	const float3 baseF = kd * M_1_PI_F * fabs(lightDir.z);

	float alpha, F0;
	SubsurfaceMaterial_GetReflection(material, hitPoint, &alpha, &F0 TEXTURES_PARAM);
	const bool entry = (hitPoint->subsurface == HITPOINT_SSS_ENTRY);

	if ((fixedDir.z < 0.f) || (F0 < 0.f)) {
		// The back face, or no coating: the Lambertian base (nothing to
		// evaluate for an entry)
		if (entry && (fixedDir.z > 0.f))
			return BLACK;

		*directPdfW = fabs(sampledDir.z * M_1_PI_F);
		*event = DIFFUSE | REFLECT;
		return baseF;
	}

	const float3 ks = TO_FLOAT3(F0);
	const float3 EcEye = GGXEnergy_CoatE3(alpha, alpha, fabs(eyeDir.z), ks);
	const float3 coatingF = SchlickBSDF_CoatingF(ks, alpha, alpha, fixedDir, sampledDir);
	const float wCoating = Coating_Probability(EcEye, kd);
	const float coatingPdf = SchlickBSDF_CoatingPdf(alpha, alpha, fixedDir, sampledDir);

	*event = GLOSSY | REFLECT;
	if (entry) {
		// The coating only
		*directPdfW = wCoating * coatingPdf;
		return coatingF;
	}

	*directPdfW = (1.f - wCoating) * fabs(sampledDir.z * M_1_PI_F) + wCoating * coatingPdf;

	const float3 EcLight = GGXEnergy_CoatE3(alpha, alpha, fabs(lightDir.z), ks);
	const float3 EcAvg = GGXEnergy_CoatEavg3(alpha, alpha, ks);

	return coatingF + Coating_LayerScale(EcLight, EcEye, EcAvg) * baseF;
}

OPENCL_FORCE_INLINE void SubsurfaceMaterial_Evaluate(__global const Material* restrict material,
		__global const HitPoint *hitPoint,
		__global float *evalStack, uint *evalStackOffset
		MATERIALS_PARAM_DECL) {
	float3 lightDir, eyeDir;
	EvalStack_PopFloat3(eyeDir);
	EvalStack_PopFloat3(lightDir);

	BSDFEvent event;
	float directPdfW;
	const float3 result = SubsurfaceMaterial_EvaluateLayers(material, hitPoint, lightDir, eyeDir,
			&event, &directPdfW MATERIALS_PARAM);
	if (Spectrum_IsBlack(result)) {
		MATERIAL_EVALUATE_RETURN_BLACK;
	}

	EvalStack_PushFloat3(result);
	EvalStack_PushBSDFEvent(event);
	EvalStack_PushFloat(directPdfW);
}

OPENCL_FORCE_INLINE void SubsurfaceMaterial_Sample(__global const Material* restrict material,
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

	float3 sampledDir;
	float pdfW;
	BSDFEvent event;
	float3 result;
	if (hitPoint->subsurface == HITPOINT_SSS_EXIT) {
		// The exit of a walk: a Lambertian transmission
		sampledDir = (signbit(fixedDir.z) ? 1.f : -1.f) * CosineSampleHemisphereWithPdf(u0, u1, &pdfW);
		if (fabs(sampledDir.z) < DEFAULT_COS_EPSILON_STATIC) {
			MATERIAL_SAMPLE_RETURN_BLACK;
		}

		event = DIFFUSE | TRANSMIT;
		result = WHITE;
	} else {
		const float3 kd = Spectrum_Clamp(Texture_GetSpectrumValue(material->subsurface.colorTexIndex, hitPoint TEXTURES_PARAM));
		float alpha = 0.f, F0 = -1.f;
		if (fixedDir.z > 0.f)
			SubsurfaceMaterial_GetReflection(material, hitPoint, &alpha, &F0 TEXTURES_PARAM);
		const float3 ks = TO_FLOAT3(fmax(F0, 0.f));

		const float3 EcEye = (F0 < 0.f) ? BLACK : GGXEnergy_CoatE3(alpha, alpha, fabs(fixedDir.z), ks);
		const float wCoating = (F0 < 0.f) ? 0.f : Coating_Probability(EcEye, kd);

		if ((passThroughEvent >= wCoating) && (hitPoint->subsurface == HITPOINT_SSS_ENTRY) && (fixedDir.z > 0.f)) {
			// Into the object, for the walk
			float pdf;
			sampledDir = -CosineSampleHemisphereWithPdf(u0, u1, &pdf);
			if (fabs(sampledDir.z) < DEFAULT_COS_EPSILON_STATIC) {
				MATERIAL_SAMPLE_RETURN_BLACK;
			}

			pdfW = (1.f - wCoating) * pdf;
			event = DIFFUSE | TRANSMIT | SUBSURFACE;
			result = (WHITE - EcEye) / (1.f - wCoating);
		} else {
			if (passThroughEvent < wCoating) {
				const float pdf = Microfacet_SampleReflect(alpha, alpha, fixedDir, u0, u1, &sampledDir);
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

			// The value and the pdf of the mixture from the evaluation
			const float3 f = SubsurfaceMaterial_EvaluateLayers(material, hitPoint, sampledDir, fixedDir,
					&event, &pdfW MATERIALS_PARAM);
			if (Spectrum_IsBlack(f) || !(pdfW > 0.f)) {
				MATERIAL_SAMPLE_RETURN_BLACK;
			}

			result = f / pdfW;
		}
	}

	EvalStack_PushFloat3(result);
	EvalStack_PushFloat3(sampledDir);
	EvalStack_PushFloat(pdfW);
	EvalStack_PushBSDFEvent(event);
}

//------------------------------------------------------------------------------
// Material specific EvalOp
//------------------------------------------------------------------------------

OPENCL_FORCE_NOT_INLINE void SubsurfaceMaterial_EvalOp(
		__global const Material* restrict material,
		const MaterialEvalOpType evalType,
		__global float *evalStack,
		uint *evalStackOffset,
		__global const HitPoint *hitPoint
		MATERIALS_PARAM_DECL) {
	switch (evalType) {
		case EVAL_ALBEDO:
			SubsurfaceMaterial_Albedo(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		case EVAL_GET_INTERIOR_VOLUME:
			SubsurfaceMaterial_GetInteriorVolume(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		case EVAL_GET_EXTERIOR_VOLUME:
			SubsurfaceMaterial_GetExteriorVolume(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		case EVAL_GET_EMITTED_RADIANCE:
			SubsurfaceMaterial_GetEmittedRadiance(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		case EVAL_GET_PASS_TROUGH_TRANSPARENCY:
			SubsurfaceMaterial_GetPassThroughTransparency(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		case EVAL_EVALUATE:
			SubsurfaceMaterial_Evaluate(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		case EVAL_SAMPLE:
			SubsurfaceMaterial_Sample(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		default:
			// Something wrong here
			break;
	}
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
