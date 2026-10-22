#line 2 "materialdefs_funcs_openpbr.cl"

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
// OpenPBR material: the textures of the hit point into OpenPBRBSDF, the BSDF
// itself is in materialdefs_funcs_openpbr_bsdf.cl. The GPU traces eye paths
// only (fromLight = false). The random walk of the subsurface part starts in
// AdvancePaths_StartSubsurfaceWalk() with the parameters of
// OpenPBRMaterial_GetWalkParams().
//------------------------------------------------------------------------------

OPENCL_FORCE_INLINE float OpenPBRMaterial_GetFloat(__global const Material* restrict material,
		const OpenPBRInput input, __global const HitPoint *hitPoint
		TEXTURES_PARAM_DECL) {
	const uint texIndex = material->openpbr.texIndex[input];

	return (texIndex != NULL_INDEX) ?
		Texture_GetFloatValue(texIndex, hitPoint TEXTURES_PARAM) :
		OpenPBR_Defaults[3 * input];
}

OPENCL_FORCE_INLINE float3 OpenPBRMaterial_GetSpectrum(__global const Material* restrict material,
		const OpenPBRInput input, __global const HitPoint *hitPoint
		TEXTURES_PARAM_DECL) {
	const uint texIndex = material->openpbr.texIndex[input];

	return (texIndex != NULL_INDEX) ?
		Texture_GetSpectrumValue(texIndex, hitPoint TEXTURES_PARAM) :
		MAKE_FLOAT3(OpenPBR_Defaults[3 * input], OpenPBR_Defaults[3 * input + 1], OpenPBR_Defaults[3 * input + 2]);
}

OPENCL_FORCE_NOT_INLINE void OpenPBRMaterial_GetBSDF(__global const Material* restrict material,
		__global const HitPoint *hitPoint, OpenPBRBSDF *b
		TEXTURES_PARAM_DECL) {
	b->baseWeight = OpenPBRMaterial_GetFloat(material, OPENPBR_BASE_WEIGHT, hitPoint TEXTURES_PARAM);
	b->baseColor = OpenPBRMaterial_GetSpectrum(material, OPENPBR_BASE_COLOR, hitPoint TEXTURES_PARAM);
	b->baseDiffuseRoughness = OpenPBRMaterial_GetFloat(material, OPENPBR_BASE_DIFFUSEROUGHNESS, hitPoint TEXTURES_PARAM);
	b->baseMetalness = OpenPBRMaterial_GetFloat(material, OPENPBR_BASE_METALNESS, hitPoint TEXTURES_PARAM);
	b->specularWeight = OpenPBRMaterial_GetFloat(material, OPENPBR_SPECULAR_WEIGHT, hitPoint TEXTURES_PARAM);
	b->specularColor = OpenPBRMaterial_GetSpectrum(material, OPENPBR_SPECULAR_COLOR, hitPoint TEXTURES_PARAM);
	b->specularRoughness = OpenPBRMaterial_GetFloat(material, OPENPBR_SPECULAR_ROUGHNESS, hitPoint TEXTURES_PARAM);
	b->specularIor = OpenPBRMaterial_GetFloat(material, OPENPBR_SPECULAR_IOR, hitPoint TEXTURES_PARAM);
	b->specularAnisotropy = OpenPBRMaterial_GetFloat(material, OPENPBR_SPECULAR_ANISOTROPY, hitPoint TEXTURES_PARAM);
	b->thinWalled = material->openpbr.thinWalled;
	b->transmissionWeight = OpenPBRMaterial_GetFloat(material, OPENPBR_TRANSMISSION_WEIGHT, hitPoint TEXTURES_PARAM);
	if (b->transmissionWeight > 0.f) {
		// At depth, the interior volume does the colour
		b->transmissionColor = (!b->thinWalled &&
				(OpenPBRMaterial_GetFloat(material, OPENPBR_TRANSMISSION_DEPTH, hitPoint TEXTURES_PARAM) > 0.f)) ?
			WHITE : OpenPBRMaterial_GetSpectrum(material, OPENPBR_TRANSMISSION_COLOR, hitPoint TEXTURES_PARAM);
	} else
		b->transmissionColor = WHITE;
	if ((b->transmissionWeight > 0.f) && !b->thinWalled) {
		b->transmissionDispersionScale = OpenPBRMaterial_GetFloat(material, OPENPBR_TRANSMISSION_DISPERSIONSCALE, hitPoint TEXTURES_PARAM);
		b->transmissionDispersionAbbe = OpenPBRMaterial_GetFloat(material, OPENPBR_TRANSMISSION_DISPERSIONABBENUMBER, hitPoint TEXTURES_PARAM);
	} else {
		b->transmissionDispersionScale = 0.f;
		b->transmissionDispersionAbbe = 20.f;
	}
	b->subsurfaceWeight = OpenPBRMaterial_GetFloat(material, OPENPBR_SUBSURFACE_WEIGHT, hitPoint TEXTURES_PARAM);
	b->subsurfaceColor = (b->subsurfaceWeight > 0.f) ?
		OpenPBRMaterial_GetSpectrum(material, OPENPBR_SUBSURFACE_COLOR, hitPoint TEXTURES_PARAM) : WHITE;
	b->coatWeight = OpenPBRMaterial_GetFloat(material, OPENPBR_COAT_WEIGHT, hitPoint TEXTURES_PARAM);
	if (b->coatWeight > 0.f) {
		b->coatColor = OpenPBRMaterial_GetSpectrum(material, OPENPBR_COAT_COLOR, hitPoint TEXTURES_PARAM);
		b->coatRoughness = OpenPBRMaterial_GetFloat(material, OPENPBR_COAT_ROUGHNESS, hitPoint TEXTURES_PARAM);
		b->coatAnisotropy = OpenPBRMaterial_GetFloat(material, OPENPBR_COAT_ANISOTROPY, hitPoint TEXTURES_PARAM);
		b->coatIor = OpenPBRMaterial_GetFloat(material, OPENPBR_COAT_IOR, hitPoint TEXTURES_PARAM);
		b->coatDarkening = OpenPBRMaterial_GetFloat(material, OPENPBR_COAT_DARKENING, hitPoint TEXTURES_PARAM);
	} else {
		b->coatColor = WHITE;
		b->coatRoughness = 0.f;
		b->coatAnisotropy = 0.f;
		b->coatIor = 1.6f;
		b->coatDarkening = 0.f;
	}
	b->fuzzWeight = OpenPBRMaterial_GetFloat(material, OPENPBR_FUZZ_WEIGHT, hitPoint TEXTURES_PARAM);
	if (b->fuzzWeight > 0.f) {
		b->fuzzColor = OpenPBRMaterial_GetSpectrum(material, OPENPBR_FUZZ_COLOR, hitPoint TEXTURES_PARAM);
		b->fuzzRoughness = OpenPBRMaterial_GetFloat(material, OPENPBR_FUZZ_ROUGHNESS, hitPoint TEXTURES_PARAM);
	} else {
		b->fuzzColor = WHITE;
		b->fuzzRoughness = .5f;
	}
	b->thinFilmWeight = OpenPBRMaterial_GetFloat(material, OPENPBR_THINFILM_WEIGHT, hitPoint TEXTURES_PARAM);
	if (b->thinFilmWeight > 0.f) {
		b->thinFilmThickness = OpenPBRMaterial_GetFloat(material, OPENPBR_THINFILM_THICKNESS, hitPoint TEXTURES_PARAM);
		b->thinFilmIor = OpenPBRMaterial_GetFloat(material, OPENPBR_THINFILM_IOR, hitPoint TEXTURES_PARAM);
	} else {
		b->thinFilmThickness = 0.f;
		b->thinFilmIor = 1.4f;
	}
	b->tangentRotation = OpenPBRMaterial_GetFloat(material, OPENPBR_TANGENT_ROTATION, hitPoint TEXTURES_PARAM);
	b->exteriorIor = ExtractExteriorIors(hitPoint, NULL_INDEX TEXTURES_PARAM);
	b->subsurfaceMode = hitPoint->subsurface;
	b->regularization = hitPoint->regularization;
	b->wavelengthU = hitPoint->passThroughEvent;
	b->wavelength = hitPoint->wavelength;
	b->dispersionTinted = hitPoint->dispersionTinted;

	OpenPBR_Init(b);
}

// The medium of the subsurface random walk at the entry point
OPENCL_FORCE_INLINE void OpenPBRMaterial_GetWalkParams(__global const Material* restrict material,
		__global const HitPoint *hitPoint, float3 *color, float3 *radius, float *anisotropy, float *eta
		TEXTURES_PARAM_DECL) {
	*color = Spectrum_Clamp(OpenPBRMaterial_GetSpectrum(material, OPENPBR_SUBSURFACE_COLOR, hitPoint TEXTURES_PARAM));
	*radius = OpenPBRMaterial_GetFloat(material, OPENPBR_SUBSURFACE_RADIUS, hitPoint TEXTURES_PARAM) *
			OpenPBRMaterial_GetSpectrum(material, OPENPBR_SUBSURFACE_RADIUSSCALE, hitPoint TEXTURES_PARAM);
	*anisotropy = clamp(OpenPBRMaterial_GetFloat(material, OPENPBR_SUBSURFACE_ANISOTROPY, hitPoint TEXTURES_PARAM), -.9f, .9f);
	*eta = OpenPBR_WalkEta(OpenPBRMaterial_GetFloat(material, OPENPBR_SPECULAR_IOR, hitPoint TEXTURES_PARAM),
			OpenPBRMaterial_GetFloat(material, OPENPBR_SPECULAR_WEIGHT, hitPoint TEXTURES_PARAM),
			ExtractExteriorIors(hitPoint, NULL_INDEX TEXTURES_PARAM),
			OpenPBRMaterial_GetFloat(material, OPENPBR_COAT_WEIGHT, hitPoint TEXTURES_PARAM),
			OpenPBRMaterial_GetFloat(material, OPENPBR_COAT_IOR, hitPoint TEXTURES_PARAM));
}

OPENCL_FORCE_INLINE void OpenPBRMaterial_Albedo(__global const Material* restrict material,
		__global const HitPoint *hitPoint,
		__global float *evalStack, uint *evalStackOffset
		MATERIALS_PARAM_DECL) {
	OpenPBRBSDF b;
	OpenPBRMaterial_GetBSDF(material, hitPoint, &b TEXTURES_PARAM);

	const float3 albedo = OpenPBR_Albedo(&b);

	EvalStack_PushFloat3(albedo);
}

OPENCL_FORCE_INLINE void OpenPBRMaterial_GetInteriorVolume(__global const Material* restrict material,
		__global const HitPoint *hitPoint,
		__global float *evalStack, uint *evalStackOffset
		MATERIALS_PARAM_DECL) {
	DefaultMaterial_GetInteriorVolume(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
}

OPENCL_FORCE_INLINE void OpenPBRMaterial_GetExteriorVolume(__global const Material* restrict material,
		__global const HitPoint *hitPoint,
		__global float *evalStack, uint *evalStackOffset
		MATERIALS_PARAM_DECL) {
	DefaultMaterial_GetExteriorVolume(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
}

OPENCL_FORCE_INLINE void OpenPBRMaterial_GetPassThroughTransparency(__global const Material* restrict material,
		__global const HitPoint *hitPoint,
		__global float *evalStack, uint *evalStackOffset
		MATERIALS_PARAM_DECL) {
	DefaultMaterial_GetPassThroughTransparency(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
}

OPENCL_FORCE_INLINE void OpenPBRMaterial_GetEmittedRadiance(__global const Material* restrict material,
		__global const HitPoint *hitPoint,
		__global float *evalStack, uint *evalStackOffset
		MATERIALS_PARAM_DECL) {
	DefaultMaterial_GetEmittedRadiance(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
}

OPENCL_FORCE_INLINE void OpenPBRMaterial_Evaluate(__global const Material* restrict material,
		__global const HitPoint *hitPoint,
		__global float *evalStack, uint *evalStackOffset
		MATERIALS_PARAM_DECL) {
	float3 lightDir, eyeDir;
	EvalStack_PopFloat3(eyeDir);
	EvalStack_PopFloat3(lightDir);

	OpenPBRBSDF b;
	OpenPBRMaterial_GetBSDF(material, hitPoint, &b TEXTURES_PARAM);

	BSDFEvent event;
	float directPdfW;
	const float3 result = OpenPBR_Disperse(&b, OpenPBR_Evaluate(&b, lightDir, eyeDir, false,
			&event, &directPdfW, NULL), &event);
	if (Spectrum_IsBlack(result)) {
		MATERIAL_EVALUATE_RETURN_BLACK;
	}

	EvalStack_PushFloat3(result);
	EvalStack_PushBSDFEvent(event);
	EvalStack_PushFloat(directPdfW);
}

OPENCL_FORCE_INLINE void OpenPBRMaterial_Sample(__global const Material* restrict material,
		__global const HitPoint *hitPoint,
		__global float *evalStack, uint *evalStackOffset
		MATERIALS_PARAM_DECL) {
	float u0, u1, passThroughEvent;
	EvalStack_PopFloat(passThroughEvent);
	EvalStack_PopFloat(u1);
	EvalStack_PopFloat(u0);
	float3 fixedDir;
	EvalStack_PopFloat3(fixedDir);

	OpenPBRBSDF b;
	OpenPBRMaterial_GetBSDF(material, hitPoint, &b TEXTURES_PARAM);

	float3 sampledDir;
	float pdfW;
	BSDFEvent event;
	const float3 result = OpenPBR_Disperse(&b, OpenPBR_Sample(&b, fixedDir, u0, u1, passThroughEvent, false,
			&sampledDir, &pdfW, &event), &event);
	if (Spectrum_IsBlack(result)) {
		MATERIAL_SAMPLE_RETURN_BLACK;
	}

	EvalStack_PushFloat3(result);
	EvalStack_PushFloat3(sampledDir);
	EvalStack_PushFloat(pdfW);
	EvalStack_PushBSDFEvent(event);
}

//------------------------------------------------------------------------------
// Material specific EvalOp
//------------------------------------------------------------------------------

OPENCL_FORCE_NOT_INLINE void OpenPBRMaterial_EvalOp(
		__global const Material* restrict material,
		const MaterialEvalOpType evalType,
		__global float *evalStack,
		uint *evalStackOffset,
		__global const HitPoint *hitPoint
		MATERIALS_PARAM_DECL) {
	switch (evalType) {
		case EVAL_ALBEDO:
			OpenPBRMaterial_Albedo(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		case EVAL_GET_INTERIOR_VOLUME:
			OpenPBRMaterial_GetInteriorVolume(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		case EVAL_GET_EXTERIOR_VOLUME:
			OpenPBRMaterial_GetExteriorVolume(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		case EVAL_GET_EMITTED_RADIANCE:
			OpenPBRMaterial_GetEmittedRadiance(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		case EVAL_GET_PASS_TROUGH_TRANSPARENCY:
			OpenPBRMaterial_GetPassThroughTransparency(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		case EVAL_EVALUATE:
			OpenPBRMaterial_Evaluate(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		case EVAL_SAMPLE:
			OpenPBRMaterial_Sample(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		default:
			// Something wrong here
			break;
	}
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
