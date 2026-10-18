#line 2 "materialdefs_funcs_disney.cl"

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
// Disney BRDF material
//------------------------------------------------------------------------------

OPENCL_FORCE_INLINE void DisneyMaterial_Albedo(__global const Material* restrict material,
		__global const HitPoint *hitPoint,
		__global float *evalStack, uint *evalStackOffset
		MATERIALS_PARAM_DECL) {
    const float3 albedo = Spectrum_Clamp(Texture_GetSpectrumValue(material->disney.baseColorTexIndex, hitPoint TEXTURES_PARAM));

	EvalStack_PushFloat3(albedo);
}

OPENCL_FORCE_INLINE void DisneyMaterial_GetInteriorVolume(__global const Material* restrict material,
		__global const HitPoint *hitPoint,
		__global float *evalStack, uint *evalStackOffset
		MATERIALS_PARAM_DECL) {
	DefaultMaterial_GetInteriorVolume(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
}

OPENCL_FORCE_INLINE void DisneyMaterial_GetExteriorVolume(__global const Material* restrict material,
		__global const HitPoint *hitPoint,
		__global float *evalStack, uint *evalStackOffset
		MATERIALS_PARAM_DECL) {
	DefaultMaterial_GetExteriorVolume(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
}

OPENCL_FORCE_INLINE void DisneyMaterial_GetPassThroughTransparency(__global const Material* restrict material,
		__global const HitPoint *hitPoint,
		__global float *evalStack, uint *evalStackOffset
		MATERIALS_PARAM_DECL) {
	DefaultMaterial_GetPassThroughTransparency(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
}

OPENCL_FORCE_INLINE void DisneyMaterial_GetEmittedRadiance(__global const Material* restrict material,
		__global const HitPoint *hitPoint,
		__global float *evalStack, uint *evalStackOffset
		MATERIALS_PARAM_DECL) {
	DefaultMaterial_GetEmittedRadiance(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
}

//------------------------------------------------------------------------------

// The energy-conserving lobe stack of the Disney parameters (see disney.cpp).
// The GPU traces eye paths only: the fixed direction is the eye one.

#define DISNEY_SHEEN_ROUGHNESS .5f
#define DISNEY_CLEARCOAT_F0 .04f

typedef struct {
	float3 color;
	float subsurface, roughness, metallic;
	float ax, ay;
	float3 F0d;
	float clearcoat, alphaCC;
	float sheen;
	float3 sheenColor;
	float filmAmount, filmThickness, filmIor;
} DisneyLobes;

OPENCL_FORCE_INLINE float3 DisneyMaterial_CalculateTint(const float3 color) {
	// Disney's luminance weights 0.3/0.6/0.1 (see disney.cpp), for Rec.709 colours
	const float3 color709 = WorkingColorSpace_ToRec709(color);
	const float luminance = color709.x * .3f + color709.y * .6f + color709.z * .1f;
	return (luminance > 0.f) ? color * (1.f / luminance) : WHITE;
}

OPENCL_FORCE_INLINE void DisneyMaterial_GetLobes(__global const Material* restrict material,
		__global const HitPoint *hitPoint, DisneyLobes *p
		TEXTURES_PARAM_DECL) {
	p->color = Spectrum_Clamp(Texture_GetSpectrumValue(material->disney.baseColorTexIndex, hitPoint TEXTURES_PARAM));
	p->subsurface = clamp(Texture_GetFloatValue(material->disney.subsurfaceTexIndex, hitPoint TEXTURES_PARAM), 0.f, 1.f);
	p->roughness = Disney_RegularizeRoughness(clamp(Texture_GetFloatValue(material->disney.roughnessTexIndex, hitPoint TEXTURES_PARAM), 0.f, 1.f),
			hitPoint->regularization);
	p->metallic = clamp(Texture_GetFloatValue(material->disney.metallicTexIndex, hitPoint TEXTURES_PARAM), 0.f, 1.f);

	const float anisotropic = clamp(Texture_GetFloatValue(material->disney.anisotropicTexIndex, hitPoint TEXTURES_PARAM), 0.f, 1.f);
	const float aspect = sqrt(1.f - .9f * anisotropic);
	p->ax = fmax(.001f, p->roughness * p->roughness / aspect);
	p->ay = fmax(.001f, p->roughness * p->roughness * aspect);

	const float3 tint = DisneyMaterial_CalculateTint(p->color);
	const float specular = clamp(Texture_GetFloatValue(material->disney.specularTexIndex, hitPoint TEXTURES_PARAM), 0.f, 1.f);
	const float specularTint = clamp(Texture_GetFloatValue(material->disney.specularTintTexIndex, hitPoint TEXTURES_PARAM), 0.f, 1.f);
	p->F0d = Spectrum_Clamp(specular * .08f * mix(WHITE, tint, specularTint));

	p->clearcoat = clamp(Texture_GetFloatValue(material->disney.clearcoatTexIndex, hitPoint TEXTURES_PARAM), 0.f, 1.f);
	const float clearcoatGloss = Disney_RegularizeClearcoatGloss(clamp(Texture_GetFloatValue(material->disney.clearcoatGlossTexIndex, hitPoint TEXTURES_PARAM), 0.f, 1.f),
			hitPoint->regularization);
	p->alphaCC = mix(.1f, .001f, clearcoatGloss);

	p->sheen = clamp(Texture_GetFloatValue(material->disney.sheenTexIndex, hitPoint TEXTURES_PARAM), 0.f, 1.f);
	const float sheenTint = clamp(Texture_GetFloatValue(material->disney.sheenTintTexIndex, hitPoint TEXTURES_PARAM), 0.f, 1.f);
	// Clamped in Rec.709, so that the colour doesn't depend on the working space
	p->sheenColor = WorkingColorSpace_FromRec709(Spectrum_Clamp(WorkingColorSpace_ToRec709(mix(WHITE, tint, sheenTint))));

	p->filmAmount = (material->disney.filmAmountTexIndex != NULL_INDEX)
			? clamp(Texture_GetFloatValue(material->disney.filmAmountTexIndex, hitPoint TEXTURES_PARAM), 0.f, 1.f) : 1.f;
	p->filmThickness = (material->disney.filmThicknessTexIndex != NULL_INDEX)
			? Texture_GetFloatValue(material->disney.filmThicknessTexIndex, hitPoint TEXTURES_PARAM) : 0.f;
	p->filmIor = (p->filmThickness > 0.f && material->disney.filmIorTexIndex != NULL_INDEX)
			? Texture_GetFloatValue(material->disney.filmIorTexIndex, hitPoint TEXTURES_PARAM) : 1.f;
}

// The probabilities of the lobes from the direction with the cosine mu
OPENCL_FORCE_INLINE void DisneyMaterial_LobeWeights(const DisneyLobes *p, const float mu,
		float *wCC, float *wSpec, float *wSheen, float *wDiff, float *sheenA, float *sheenB) {
	const float ccE = p->clearcoat * GGXEnergy_CoatE(p->alphaCC, p->alphaCC, mu, DISNEY_CLEARCOAT_F0);
	const float rest = 1.f - ccE;

	const float3 EspecM = GGXEnergy_CoatE3(p->ax, p->ay, mu, p->color);
	const float3 EspecD = GGXEnergy_CoatE3(p->ax, p->ay, mu, p->F0d);

	float albedoSheen = 0.f;
	*sheenA = 0.f;
	*sheenB = 0.f;
	if (p->sheen > 0.f) {
		SheenLTC_Lookup(mu, DISNEY_SHEEN_ROUGHNESS, sheenA, sheenB, &albedoSheen);
		if ((fabs(*sheenA) < 1e-5f) || (albedoSheen < 1e-5f))
			albedoSheen = 0.f;
	}
	const float sheenW = p->sheen * albedoSheen;

	*wCC = ccE;
	// By luminance: the same choices in every working colour space (the
	// diffuse weight is (1 - Y(Espec)) Y(color), not Y((1 - Espec) color))
	*wSpec = rest * (p->metallic * fmax(0.f, Spectrum_Y(EspecM)) + (1.f - p->metallic) * (1.f - sheenW) * fmax(0.f, Spectrum_Y(EspecD)));
	*wSheen = rest * (1.f - p->metallic) * sheenW * fmax(0.f, Spectrum_Y(p->sheenColor));
	*wDiff = rest * (1.f - p->metallic) * (1.f - sheenW) * fmax(0.f, 1.f - Spectrum_Y(EspecD)) * fmax(0.f, Spectrum_Y(p->color));

	const float sum = *wCC + *wSpec + *wSheen + *wDiff;
	if (sum > 0.f) {
		const float invSum = 1.f / sum;
		*wCC *= invSum;
		*wSpec *= invSum;
		*wSheen *= invSum;
		*wDiff *= invSum;
	}
}

// f |cos light| of the stack, both directions above the surface (the caller
// mirrors the back face)
OPENCL_FORCE_INLINE float3 DisneyMaterial_EvaluateLayers(const DisneyLobes *p,
		const float3 lightDir, const float3 eyeDir) {
	const float muL = lightDir.z;
	const float muE = eyeDir.z;
	if ((muL < DEFAULT_COS_EPSILON_STATIC) || (muE < DEFAULT_COS_EPSILON_STATIC))
		return BLACK;

	// The specular lobes with the thin film (eye path: fixed = eye)
	float3 specM = SchlickBSDF_CoatingF(p->color, p->ax, p->ay, eyeDir, lightDir);
	float3 specD = SchlickBSDF_CoatingF(p->F0d, p->ax, p->ay, eyeDir, lightDir);
	if (p->filmThickness > 0.f) {
		const float3 filmColor = CalcFilmColor(eyeDir, p->filmThickness, p->filmIor);
		specM = mix(specM, specM * filmColor, p->filmAmount);
		specD = mix(specD, specD * filmColor, p->filmAmount);
	}

	// The diffuse lobe under the dielectric specular
	float3 diffuse = EON_Evaluate(p->color, p->roughness, lightDir, eyeDir) * muL;
	if (p->subsurface > 0.f) {
		const float3 H = normalize(lightDir + eyeDir);
		const float LdotH = dot(lightDir, H);
		const float FL = pow(1.f - muL, 5.f);
		const float FV = pow(1.f - muE, 5.f);
		const float Fss90 = LdotH * LdotH * p->roughness;
		const float Fss = mix(1.f, Fss90, FL) * mix(1.f, Fss90, FV);
		const float ss = 1.25f * (Fss * (1.f / (muL + muE) - .5f) + .5f);
		diffuse = mix(diffuse, (M_1_PI_F * ss * muL) * p->color, p->subsurface);
	}
	const float3 layerD = Coating_LayerScale(GGXEnergy_CoatE3(p->ax, p->ay, muL, p->F0d),
			GGXEnergy_CoatE3(p->ax, p->ay, muE, p->F0d), GGXEnergy_CoatEavg3(p->ax, p->ay, p->F0d));

	// The sheen above the dielectric part
	float3 sheenF = BLACK;
	float sheenScale = 1.f;
	if (p->sheen > 0.f) {
		float a, b, albedo;
		SheenLTC_Lookup(muE, DISNEY_SHEEN_ROUGHNESS, &a, &b, &albedo);
		if ((fabs(a) >= 1e-5f) && (albedo >= 1e-5f)) {
			sheenF = (p->sheen * albedo * Sheen_LTC(a, b, eyeDir, lightDir)) * p->sheenColor;
			sheenScale = 1.f - p->sheen * albedo;
		}
	}

	const float3 dielectric = sheenF + sheenScale * (specD + layerD * diffuse);
	const float3 base = p->metallic * specM + (1.f - p->metallic) * dielectric;

	// The clearcoat above everything
	if (p->clearcoat > 0.f) {
		const float3 F0cc = TO_FLOAT3(DISNEY_CLEARCOAT_F0);
		const float3 ccF = p->clearcoat * SchlickBSDF_CoatingF(F0cc, p->alphaCC, p->alphaCC, eyeDir, lightDir);
		// Albedo scaling by the eye direction (see disney.cpp)
		const float EccEye = p->clearcoat * GGXEnergy_CoatE(p->alphaCC, p->alphaCC, muE, DISNEY_CLEARCOAT_F0);

		return ccF + (1.f - EccEye) * base;
	}

	return base;
}

// The pdf of generating lightDir from eyeDir (eye path), both above the surface
OPENCL_FORCE_INLINE float DisneyMaterial_PdfLobes(const DisneyLobes *p, const float3 eyeDir, const float3 lightDir) {
	float wCC, wSpec, wSheen, wDiff, sheenA, sheenB;
	DisneyMaterial_LobeWeights(p, eyeDir.z, &wCC, &wSpec, &wSheen, &wDiff, &sheenA, &sheenB);

	float pdf = wDiff * lightDir.z * M_1_PI_F;
	if (wSpec > 0.f)
		pdf += wSpec * Microfacet_PdfReflect(p->ax, p->ay, eyeDir, lightDir);
	if (wCC > 0.f)
		pdf += wCC * Microfacet_PdfReflect(p->alphaCC, p->alphaCC, eyeDir, lightDir);
	if (wSheen > 0.f)
		pdf += wSheen * Sheen_LTC(sheenA, sheenB, eyeDir, lightDir);

	return pdf;
}

OPENCL_FORCE_INLINE void DisneyMaterial_Evaluate(__global const Material* restrict material,
		__global const HitPoint *hitPoint,
		__global float *evalStack, uint *evalStackOffset
		MATERIALS_PARAM_DECL) {
	float3 lightDir, eyeDir;
	EvalStack_PopFloat3(eyeDir);
	EvalStack_PopFloat3(lightDir);

	if (lightDir.z * eyeDir.z <= 0.f) {
		MATERIAL_EVALUATE_RETURN_BLACK;
	}

	// Both faces: the back face is the mirror image of the front one
	const float flip = (eyeDir.z < 0.f) ? -1.f : 1.f;
	const float3 eye = MAKE_FLOAT3(eyeDir.x, eyeDir.y, flip * eyeDir.z);
	const float3 light = MAKE_FLOAT3(lightDir.x, lightDir.y, flip * lightDir.z);

	DisneyLobes p;
	DisneyMaterial_GetLobes(material, hitPoint, &p TEXTURES_PARAM);

	const float3 result = DisneyMaterial_EvaluateLayers(&p, light, eye);
	if (Spectrum_IsBlack(result)) {
		MATERIAL_EVALUATE_RETURN_BLACK;
	}
	const float directPdfW = DisneyMaterial_PdfLobes(&p, eye, light);
	const BSDFEvent event = GLOSSY | REFLECT;

	EvalStack_PushFloat3(result);
	EvalStack_PushBSDFEvent(event);
	EvalStack_PushFloat(directPdfW);
}

OPENCL_FORCE_INLINE void DisneyMaterial_Sample(__global const Material* restrict material,
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

	DisneyLobes p;
	DisneyMaterial_GetLobes(material, hitPoint, &p TEXTURES_PARAM);

	const float flip = (fixedDir.z < 0.f) ? -1.f : 1.f;
	const float3 eye = MAKE_FLOAT3(fixedDir.x, fixedDir.y, flip * fixedDir.z);

	float wCC, wSpec, wSheen, wDiff, sheenA, sheenB;
	DisneyMaterial_LobeWeights(&p, eye.z, &wCC, &wSpec, &wSheen, &wDiff, &sheenA, &sheenB);
	if (wCC + wSpec + wSheen + wDiff <= 0.f) {
		MATERIAL_SAMPLE_RETURN_BLACK;
	}

	float3 light;
	if (passThroughEvent < wCC) {
		if (Microfacet_SampleReflect(p.alphaCC, p.alphaCC, eye, u0, u1, &light) <= 0.f) {
			MATERIAL_SAMPLE_RETURN_BLACK;
		}
	} else if (passThroughEvent < wCC + wSpec) {
		if (Microfacet_SampleReflect(p.ax, p.ay, eye, u0, u1, &light) <= 0.f) {
			MATERIAL_SAMPLE_RETURN_BLACK;
		}
	} else if (passThroughEvent < wCC + wSpec + wSheen)
		light = Sheen_SampleLTC(sheenA, sheenB, eye, u0, u1);
	else {
		float pdf;
		light = CosineSampleHemisphereWithPdf(u0, u1, &pdf);
	}
	if (light.z < DEFAULT_COS_EPSILON_STATIC) {
		MATERIAL_SAMPLE_RETURN_BLACK;
	}

	const float pdfW = DisneyMaterial_PdfLobes(&p, eye, light);
	const float3 f = DisneyMaterial_EvaluateLayers(&p, light, eye);
	if (Spectrum_IsBlack(f) || !(pdfW > 0.f)) {
		MATERIAL_SAMPLE_RETURN_BLACK;
	}

	const float3 sampledDir = MAKE_FLOAT3(light.x, light.y, flip * light.z);
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

OPENCL_FORCE_NOT_INLINE void DisneyMaterial_EvalOp(
		__global const Material* restrict material,
		const MaterialEvalOpType evalType,
		__global float *evalStack,
		uint *evalStackOffset,
		__global const HitPoint *hitPoint
		MATERIALS_PARAM_DECL) {
	switch (evalType) {
		case EVAL_ALBEDO:
			DisneyMaterial_Albedo(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		case EVAL_GET_INTERIOR_VOLUME:
			DisneyMaterial_GetInteriorVolume(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		case EVAL_GET_EXTERIOR_VOLUME:
			DisneyMaterial_GetExteriorVolume(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		case EVAL_GET_EMITTED_RADIANCE:
			DisneyMaterial_GetEmittedRadiance(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		case EVAL_GET_PASS_TROUGH_TRANSPARENCY:
			DisneyMaterial_GetPassThroughTransparency(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		case EVAL_EVALUATE:
			DisneyMaterial_Evaluate(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		case EVAL_SAMPLE:
			DisneyMaterial_Sample(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		default:
			// Something wrong here
			break;
	}
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
