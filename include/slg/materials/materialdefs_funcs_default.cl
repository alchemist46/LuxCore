#line 2 "material_funcs_default.cl"

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
// Default material functions
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// EVAL_EVALUATE_DIFFUSE (Material::EvaluateDiffuse()) of a material without a
// diffuse lobe, and the reflection only of an EVAL_EVALUATE result
//------------------------------------------------------------------------------

OPENCL_FORCE_INLINE void DefaultMaterial_EvaluateNoDiffuse(__global float *evalStack, uint *evalStackOffset) {
	float3 lightDir, eyeDir;
	EvalStack_PopFloat3(eyeDir);
	EvalStack_PopFloat3(lightDir);

	MATERIAL_EVALUATE_RETURN_BLACK;
}

OPENCL_FORCE_INLINE void DefaultMaterial_KeepDiffuseReflection(__global float *evalStack, uint *evalStackOffset) {
	float directPdfW;
	BSDFEvent event;
	float3 result;
	EvalStack_PopFloat(directPdfW);
	EvalStack_PopBSDFEvent(event);
	EvalStack_PopFloat3(result);

	if (event & TRANSMIT) {
		MATERIAL_EVALUATE_RETURN_BLACK;
	}

	EvalStack_PushFloat3(result);
	EvalStack_PushBSDFEvent(event);
	EvalStack_PushFloat(directPdfW);
}

//------------------------------------------------------------------------------
// DefaultMaterial_GetInteriorVolume
//------------------------------------------------------------------------------

OPENCL_FORCE_INLINE void DefaultMaterial_GetInteriorVolume(__global const Material* restrict material,
		__global const HitPoint *hitPoint,
		__global float *evalStack, uint *evalStackOffset
		MATERIALS_PARAM_DECL) {
	float passThroughEvent;
	EvalStack_PopFloat(passThroughEvent);

	const uint volIndex = material->interiorVolumeIndex;

	EvalStack_PushUInt(volIndex);
}

//------------------------------------------------------------------------------
// DefaultMaterial_GetExteriorVolume
//------------------------------------------------------------------------------

OPENCL_FORCE_INLINE void DefaultMaterial_GetExteriorVolume(__global const Material* restrict material,
		__global const HitPoint *hitPoint,
		__global float *evalStack, uint *evalStackOffset
		MATERIALS_PARAM_DECL) {
	float passThroughEvent;
	EvalStack_PopFloat(passThroughEvent);

	const uint volIndex = material->exteriorVolumeIndex;

	EvalStack_PushUInt(volIndex);
}

//------------------------------------------------------------------------------
// DefaultMaterial_GetPassThroughTransparency
//------------------------------------------------------------------------------

OPENCL_FORCE_INLINE void DefaultMaterial_GetPassThroughTransparency(__global const Material* restrict material,
		__global const HitPoint *hitPoint,
		__global float *evalStack, uint *evalStackOffset
		MATERIALS_PARAM_DECL) {
	bool backTracing;
	EvalStack_PopUInt(backTracing);
	float passThroughEvent;
	EvalStack_PopFloat(passThroughEvent);
	float3 localFixedDir;
	EvalStack_PopFloat3(localFixedDir);

	const uint transpTexIndex = (hitPoint->intoObject != backTracing) ?
		material->frontTranspTexIndex : material->backTranspTexIndex;

	float3 transp;
	// The pass-through event at a vertex (see
	// Material::GetVertexPassThroughEvent()): u <= weight there, u / weight
	// is uniform again; -1 if it stays
	float vertexPassThroughEvent = -1.f;
	if (transpTexIndex != NULL_INDEX) {
		const float weight = clamp(
				Texture_GetFloatValue(transpTexIndex, hitPoint TEXTURES_PARAM),
				0.f, 1.f);

		if (passThroughEvent > weight)
			transp = WHITE;
		else {
			transp = BLACK;
			if (weight > 0.f)
				vertexPassThroughEvent = fmin(passThroughEvent / weight, .99999994f);
		}
	} else
		transp = BLACK;

	EvalStack_PushFloat3(transp);
	EvalStack_PushFloat(vertexPassThroughEvent);
}

//------------------------------------------------------------------------------
// DefaultMaterial_GetEmittedRadiance
//------------------------------------------------------------------------------

OPENCL_FORCE_INLINE void DefaultMaterial_GetEmittedRadiance(__global const Material* restrict material,
		__global const HitPoint *hitPoint,
		__global float *evalStack, uint *evalStackOffset
		MATERIALS_PARAM_DECL) {
	float oneOverPrimitiveArea;
	EvalStack_PopFloat(oneOverPrimitiveArea);

	const uint emitTexIndex = material->emitTexIndex;
	const float3 emittedRadiance = (emitTexIndex == NULL_INDEX) ?
		BLACK :
		(VLOAD3F(material->emittedFactor.c) *
		(material->usePrimitiveArea ? oneOverPrimitiveArea : 1.f) *
		clamp(Texture_GetSpectrumValue(emitTexIndex, hitPoint
				TEXTURES_PARAM), BLACK, MAKE_FLOAT3(INFINITY, INFINITY, INFINITY)));

	EvalStack_PushFloat3(emittedRadiance);
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
