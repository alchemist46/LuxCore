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

#if defined(PARAM_ENABLE_OPENPBR_COAT_BUMP)
// The coat's normal of coat.bumptex, bumped from the frame before the
// material's bump: the interpolated normal, with dpdu and dpdv projected on
// its plane. Not inlined: only the materials with coat.bumptex run it (a
// Texture_Bump() call site changes how the kernel is compiled). The hit point
// is left as it was.
OPENCL_FORCE_NOT_INLINE float3 OpenPBRMaterial_CoatBump(__global const Material* restrict material,
		__global const HitPoint *hitPointConst
		TEXTURES_PARAM_DECL) {
	__global HitPoint *hitPoint = (__global HitPoint *)hitPointConst;
	const float3 shadeN = VLOAD3F(&hitPoint->shadeN.x);
	const float3 dpdu = VLOAD3F(&hitPoint->dpdu.x);
	const float3 dpdv = VLOAD3F(&hitPoint->dpdv.x);

	const float3 n = VLOAD3F(&hitPoint->interpolatedN.x);
	VSTORE3F(n, &hitPoint->shadeN.x);
	VSTORE3F(dpdu - dot(dpdu, n) * n, &hitPoint->dpdu.x);
	VSTORE3F(dpdv - dot(dpdv, n) * n, &hitPoint->dpdv.x);
	const float3 coatN = Texture_Bump(material->openpbr.coatBumpTexIndex, hitPoint, material->bumpSampleDistance
			TEXTURES_PARAM);

	VSTORE3F(shadeN, &hitPoint->shadeN.x);
	VSTORE3F(dpdu, &hitPoint->dpdu.x);
	VSTORE3F(dpdv, &hitPoint->dpdv.x);

	// Made valid for the eye direction, as the shading normal in
	// BSDF_EnsureValidNormals()
	const float3 fixedDir = VLOAD3F(&hitPoint->fixedDir.x);
	const float side = hitPoint->intoObject ? 1.f : -1.f;
	const float3 ng = side * VLOAD3F(&hitPoint->geometryN.x);
	return (dot(fixedDir, ng) < DEFAULT_COS_EPSILON_STATIC) ? coatN :
		(side * BSDF_EnsureValidReflection(ng, fixedDir, side * coatN));
}
#endif

// The normal of coat.bumptex (world space) in b->coatN, for
// OpenPBRMaterial_GetBSDF(): its callers compute it, a Texture_Bump() under
// OpenPBRMaterial_GetBSDF() is one more level on the deepest call chain of
// the kernel (400 bytes of scratch per lane, every scene 7% slower on HIP).
// Only the scenes with coat.bumptex compile it
// (PARAM_ENABLE_OPENPBR_COAT_BUMP, CompiledScene::hasOpenPBRCoatBump): even
// so, its code costs the other scenes 2.5-4.5% on HIP.
OPENCL_FORCE_INLINE void OpenPBRMaterial_CoatBumpNormal(__global const Material* restrict material,
		__global const HitPoint *hitPoint, OpenPBRBSDF *b
		TEXTURES_PARAM_DECL) {
#if defined(PARAM_ENABLE_OPENPBR_COAT_BUMP)
	if (material->openpbr.coatBumpTexIndex != NULL_INDEX)
		b->coatN = OpenPBRMaterial_CoatBump(material, hitPoint TEXTURES_PARAM);
#endif
}

// With coat.bumptex, b->coatN is OpenPBRMaterial_CoatBumpNormal(). Inlined, as
// OpenPBR_EvaluateFront(): passed to a call, OpenPBRBSDF goes through scratch
// memory (both not inlined: OpenPBR plastic, metal and coat 11% slower on
// HIP, the native scenes of the other materials within 2%)
OPENCL_FORCE_INLINE void OpenPBRMaterial_GetBSDF(__global const Material* restrict material,
		__global const HitPoint *hitPoint, OpenPBRBSDF *b
		TEXTURES_PARAM_DECL) {
	// The inputs of the lobes that are off (the material's or this hit
	// point's) are not read
	const uint lobes = material->openpbr.lobes;
	b->baseWeight = OpenPBRMaterial_GetFloat(material, OPENPBR_BASE_WEIGHT, hitPoint TEXTURES_PARAM);
	b->baseColor = OpenPBRMaterial_GetSpectrum(material, OPENPBR_BASE_COLOR, hitPoint TEXTURES_PARAM);
	b->baseDiffuseRoughness = (lobes & OPENPBR_LOBE_EON) ?
		OpenPBRMaterial_GetFloat(material, OPENPBR_BASE_DIFFUSEROUGHNESS, hitPoint TEXTURES_PARAM) : 0.f;
	b->baseMetalness = (lobes & OPENPBR_LOBE_METAL) ?
		OpenPBRMaterial_GetFloat(material, OPENPBR_BASE_METALNESS, hitPoint TEXTURES_PARAM) : 0.f;
	b->specularWeight = (lobes & OPENPBR_LOBE_SPECULAR) ?
		OpenPBRMaterial_GetFloat(material, OPENPBR_SPECULAR_WEIGHT, hitPoint TEXTURES_PARAM) : 0.f;
	if (lobes & OPENPBR_LOBES_BASE_SPECULAR) {
		b->specularColor = OpenPBRMaterial_GetSpectrum(material, OPENPBR_SPECULAR_COLOR, hitPoint TEXTURES_PARAM);
		b->specularRoughness = OpenPBRMaterial_GetFloat(material, OPENPBR_SPECULAR_ROUGHNESS, hitPoint TEXTURES_PARAM);
		b->specularIor = OpenPBRMaterial_GetFloat(material, OPENPBR_SPECULAR_IOR, hitPoint TEXTURES_PARAM);
		b->specularAnisotropy = OpenPBRMaterial_GetFloat(material, OPENPBR_SPECULAR_ANISOTROPY, hitPoint TEXTURES_PARAM);
	} else {
		b->specularColor = WHITE;
		b->specularRoughness = .3f;
		b->specularIor = 1.5f;
		b->specularAnisotropy = 0.f;
	}
	b->thinWalled = material->openpbr.thinWalled;
	b->transmissionWeight = (lobes & OPENPBR_LOBE_TRANSMISSION) ?
		OpenPBRMaterial_GetFloat(material, OPENPBR_TRANSMISSION_WEIGHT, hitPoint TEXTURES_PARAM) : 0.f;
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
	b->subsurfaceWeight = (lobes & OPENPBR_LOBE_SUBSURFACE) ?
		OpenPBRMaterial_GetFloat(material, OPENPBR_SUBSURFACE_WEIGHT, hitPoint TEXTURES_PARAM) : 0.f;
	b->subsurfaceColor = (b->subsurfaceWeight > 0.f) ?
		OpenPBRMaterial_GetSpectrum(material, OPENPBR_SUBSURFACE_COLOR, hitPoint TEXTURES_PARAM) : WHITE;
	b->coatWeight = (lobes & OPENPBR_LOBE_COAT) ?
		OpenPBRMaterial_GetFloat(material, OPENPBR_COAT_WEIGHT, hitPoint TEXTURES_PARAM) : 0.f;
	if (b->coatWeight > 0.f) {
		b->coatColor = OpenPBRMaterial_GetSpectrum(material, OPENPBR_COAT_COLOR, hitPoint TEXTURES_PARAM);
		b->coatRoughness = OpenPBRMaterial_GetFloat(material, OPENPBR_COAT_ROUGHNESS, hitPoint TEXTURES_PARAM);
		b->coatAnisotropy = OpenPBRMaterial_GetFloat(material, OPENPBR_COAT_ANISOTROPY, hitPoint TEXTURES_PARAM);
		// coat.ior (OpenPBR_Init() makes it relative)
		b->coatEta = OpenPBRMaterial_GetFloat(material, OPENPBR_COAT_IOR, hitPoint TEXTURES_PARAM);
		b->coatDarkening = OpenPBRMaterial_GetFloat(material, OPENPBR_COAT_DARKENING, hitPoint TEXTURES_PARAM);
	} else {
		b->coatColor = WHITE;
		b->coatRoughness = 0.f;
		b->coatAnisotropy = 0.f;
		b->coatEta = 1.6f;
		b->coatDarkening = 0.f;
	}
	b->fuzzWeight = (lobes & OPENPBR_LOBE_FUZZ) ?
		OpenPBRMaterial_GetFloat(material, OPENPBR_FUZZ_WEIGHT, hitPoint TEXTURES_PARAM) : 0.f;
	if (b->fuzzWeight > 0.f) {
		b->fuzzColor = OpenPBRMaterial_GetSpectrum(material, OPENPBR_FUZZ_COLOR, hitPoint TEXTURES_PARAM);
		b->fuzzRoughness = OpenPBRMaterial_GetFloat(material, OPENPBR_FUZZ_ROUGHNESS, hitPoint TEXTURES_PARAM);
	} else {
		b->fuzzColor = WHITE;
		b->fuzzRoughness = .5f;
	}
	b->thinFilmWeight = (lobes & OPENPBR_LOBE_THINFILM) ?
		OpenPBRMaterial_GetFloat(material, OPENPBR_THINFILM_WEIGHT, hitPoint TEXTURES_PARAM) : 0.f;
	if (b->thinFilmWeight > 0.f) {
		b->thinFilmThickness = OpenPBRMaterial_GetFloat(material, OPENPBR_THINFILM_THICKNESS, hitPoint TEXTURES_PARAM);
		b->thinFilmIor = OpenPBRMaterial_GetFloat(material, OPENPBR_THINFILM_IOR, hitPoint TEXTURES_PARAM);
	} else {
		b->thinFilmThickness = 0.f;
		b->thinFilmIor = 1.4f;
	}
	b->tangentRotation = (lobes & OPENPBR_LOBE_ROTATION) ?
		OpenPBRMaterial_GetFloat(material, OPENPBR_TANGENT_ROTATION, hitPoint TEXTURES_PARAM) : 0.f;
	if (material->openpbr.coatFrame && (b->coatWeight > 0.f)) {
		// The coat's normal in the shading frame of the hit point (as
		// Frame_Set() builds it in BSDF_Init())
#if defined(PARAM_ENABLE_OPENPBR_COAT_BUMP)
		const float3 n = (material->openpbr.coatBumpTexIndex != NULL_INDEX) ?
			b->coatN : VLOAD3F(&hitPoint->interpolatedN.x);
#else
		const float3 n = VLOAD3F(&hitPoint->interpolatedN.x);
#endif
		const float3 shadeN = VLOAD3F(&hitPoint->shadeN.x);
		const float3 Y = normalize(cross(shadeN, VLOAD3F(&hitPoint->dpdu.x)));
		const float3 X = cross(Y, shadeN);
		b->coatN = MAKE_FLOAT3(dot(n, X), dot(n, Y), dot(n, shadeN));
	} else
		b->coatN = MAKE_FLOAT3(0.f, 0.f, 1.f);
	b->exteriorIor = (lobes & OPENPBR_LOBES_EXTERIOR_IOR) ?
		ExtractExteriorIors(hitPoint, NULL_INDEX TEXTURES_PARAM) : 1.f;
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
	*radius = Spectrum_Clamp(fmax(OpenPBRMaterial_GetFloat(material, OPENPBR_SUBSURFACE_RADIUS, hitPoint TEXTURES_PARAM), 0.f) *
			OpenPBRMaterial_GetSpectrum(material, OPENPBR_SUBSURFACE_RADIUSSCALE, hitPoint TEXTURES_PARAM));
	*anisotropy = clamp(OpenPBRMaterial_GetFloat(material, OPENPBR_SUBSURFACE_ANISOTROPY, hitPoint TEXTURES_PARAM), -.9f, .9f);
	*eta = OpenPBR_WalkEta(OpenPBRMaterial_GetFloat(material, OPENPBR_SPECULAR_IOR, hitPoint TEXTURES_PARAM),
			OpenPBRMaterial_GetFloat(material, OPENPBR_SPECULAR_WEIGHT, hitPoint TEXTURES_PARAM),
			ExtractExteriorIors(hitPoint, NULL_INDEX TEXTURES_PARAM),
			OpenPBRMaterial_GetFloat(material, OPENPBR_COAT_WEIGHT, hitPoint TEXTURES_PARAM),
			OpenPBRMaterial_GetFloat(material, OPENPBR_COAT_IOR, hitPoint TEXTURES_PARAM));
}

// The albedo and the roughness of a material of the diffuse base only
// (OpenPBR_DiffuseOnly*()), clamped as OpenPBR_Init()
OPENCL_FORCE_INLINE float3 OpenPBRMaterial_DiffuseOnlyAlbedo(__global const Material* restrict material,
		__global const HitPoint *hitPoint, float *roughness
		TEXTURES_PARAM_DECL) {
	*roughness = (material->openpbr.lobes & OPENPBR_LOBE_EON) ?
		clamp(OpenPBRMaterial_GetFloat(material, OPENPBR_BASE_DIFFUSEROUGHNESS, hitPoint TEXTURES_PARAM), 0.f, 1.f) : 0.f;

	return clamp(OpenPBRMaterial_GetFloat(material, OPENPBR_BASE_WEIGHT, hitPoint TEXTURES_PARAM), 0.f, 1.f) *
			Spectrum_Clamp(OpenPBRMaterial_GetSpectrum(material, OPENPBR_BASE_COLOR, hitPoint TEXTURES_PARAM));
}

OPENCL_FORCE_INLINE void OpenPBRMaterial_Albedo(__global const Material* restrict material,
		__global const HitPoint *hitPoint,
		__global float *evalStack, uint *evalStackOffset
		MATERIALS_PARAM_DECL) {
	float3 albedo;
	if (OPENPBR_IS_DIFFUSE_ONLY(material->openpbr.lobes)) {
		float roughness;
		albedo = OpenPBRMaterial_DiffuseOnlyAlbedo(material, hitPoint, &roughness TEXTURES_PARAM);
	} else {
		OpenPBRBSDF b;
		OpenPBRMaterial_CoatBumpNormal(material, hitPoint, &b TEXTURES_PARAM);
		OpenPBRMaterial_GetBSDF(material, hitPoint, &b TEXTURES_PARAM);

		albedo = OpenPBR_Albedo(&b);
	}

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

	BSDFEvent event;
	float directPdfW;
	float3 result;
	if (OPENPBR_IS_DIFFUSE_ONLY(material->openpbr.lobes)) {
		float roughness;
		const float3 albedo = OpenPBRMaterial_DiffuseOnlyAlbedo(material, hitPoint, &roughness TEXTURES_PARAM);
		result = OpenPBR_DiffuseOnlyEvaluate(albedo, roughness, lightDir, eyeDir, false,
				&event, &directPdfW, NULL);
	} else {
		OpenPBRBSDF b;
		OpenPBRMaterial_CoatBumpNormal(material, hitPoint, &b TEXTURES_PARAM);
		OpenPBRMaterial_GetBSDF(material, hitPoint, &b TEXTURES_PARAM);

		result = OpenPBR_Disperse(&b, OpenPBR_Evaluate(&b, lightDir, eyeDir, false,
				&event, &directPdfW, NULL), &event);
	}
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

	float3 sampledDir;
	float pdfW;
	BSDFEvent event;
	float3 result;
	if (OPENPBR_IS_DIFFUSE_ONLY(material->openpbr.lobes)) {
		float roughness;
		const float3 albedo = OpenPBRMaterial_DiffuseOnlyAlbedo(material, hitPoint, &roughness TEXTURES_PARAM);
		result = OpenPBR_DiffuseOnlySample(albedo, roughness, fixedDir, u0, u1, false,
				&sampledDir, &pdfW, &event);
	} else {
		OpenPBRBSDF b;
		OpenPBRMaterial_CoatBumpNormal(material, hitPoint, &b TEXTURES_PARAM);
		OpenPBRMaterial_GetBSDF(material, hitPoint, &b TEXTURES_PARAM);

		result = OpenPBR_Disperse(&b, OpenPBR_Sample(&b, fixedDir, u0, u1, passThroughEvent, false,
				&sampledDir, &pdfW, &event), &event);
	}
	if (Spectrum_IsBlack(result)) {
		MATERIAL_SAMPLE_RETURN_BLACK;
	}

	EvalStack_PushFloat3(result);
	EvalStack_PushFloat3(sampledDir);
	EvalStack_PushFloat(pdfW);
	EvalStack_PushBSDFEvent(event);
}

// The diffuse reflection lobe (Material::EvaluateDiffuse())
OPENCL_FORCE_INLINE void OpenPBRMaterial_EvaluateDiffuse(__global const Material* restrict material,
		__global const HitPoint *hitPoint,
		__global float *evalStack, uint *evalStackOffset
		MATERIALS_PARAM_DECL) {
	float3 lightDir, eyeDir;
	EvalStack_PopFloat3(eyeDir);
	EvalStack_PopFloat3(lightDir);

	BSDFEvent event = DIFFUSE | REFLECT;
	float directPdfW;
	float3 result;
	if (OPENPBR_IS_DIFFUSE_ONLY(material->openpbr.lobes)) {
		// The whole BSDF
		float roughness;
		const float3 albedo = OpenPBRMaterial_DiffuseOnlyAlbedo(material, hitPoint, &roughness TEXTURES_PARAM);
		BSDFEvent diffuseEvent;
		directPdfW = 0.f;
		result = OpenPBR_DiffuseOnlyEvaluate(albedo, roughness, lightDir, eyeDir, false,
				&diffuseEvent, &directPdfW, NULL);
	} else {
		OpenPBRBSDF b;
		OpenPBRMaterial_CoatBumpNormal(material, hitPoint, &b TEXTURES_PARAM);
		OpenPBRMaterial_GetBSDF(material, hitPoint, &b TEXTURES_PARAM);

		result = OpenPBR_Disperse(&b, OpenPBR_EvaluateDiffuse(&b, lightDir, eyeDir, &directPdfW), &event);
	}

	EvalStack_PushFloat3(result);
	EvalStack_PushBSDFEvent(event);
	EvalStack_PushFloat(directPdfW);
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
