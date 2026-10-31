#line 2 "material_eval_funcs.cl"

/***************************************************************************
 * Copyright 1998-2020 by authors (see AUTHORS.txt)                        *
 *                                                                         *
 *   This file is part of LuxCoreRender.                                   *
 *                                                                         *
 * Licensed under the Apache License, Version 2.0 (the License);         *
 * you may not use this file except in compliance with the License.        *
 * You may obtain a copy of the License at                                 *
 *                                                                         *
 *     http://www.apache.org/licenses/LICENSE-2.0                          *
 *                                                                         *
 * Unless required by applicable law or agreed to in writing, software     *
 * distributed under the License is distributed on an AS IS BASIS,       *
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.*
 * See the License for the specific language governing permissions and     *
 * limitations under the License.                                          *
 ***************************************************************************/

//#define DEBUG_PRINTF_KERNEL_NAME 1
//#define DEBUG_PRINTF_MATERIAL_EVAL 1

//------------------------------------------------------------------------------
// Material evaluation functions
//------------------------------------------------------------------------------

OPENCL_FORCE_NOT_INLINE uint Material_EvalOp(
		__global const MaterialEvalOp* restrict evalOp,
		__global float *evalStack,
		uint *evalStackOffset,
		__global const HitPoint *hitPoint
		MATERIALS_PARAM_DECL) {
	const uint matIndex = evalOp->matIndex;
	__global const Material* restrict material = &mats[matIndex];

#if defined(DEBUG_PRINTF_MATERIAL_EVAL)
	printf("EvalOp material index=%d type=%d evalType=%d *evalStackOffset=%d\n", evalOp->matIndex, material->type, evalOp->evalType, *evalStackOffset);
#endif

	const MaterialEvalOpType evalType = evalOp->evalType;

	//--------------------------------------------------------------------------
	// The support for generic ops
	//--------------------------------------------------------------------------

	switch (evalType) {
		case EVAL_CONDITIONAL_GOTO: {
				bool condition;
				EvalStack_PopInt(condition);

				if (condition)
					return evalOp->opData.opsCount;
				break;
		}
		case EVAL_UNCONDITIONAL_GOTO:
			return evalOp->opData.opsCount;
		default:
			// It will be handled by the following switch
			break;
	}

	//--------------------------------------------------------------------------
	// The support for material specific ops
	//--------------------------------------------------------------------------

	switch (material->type) {
		//----------------------------------------------------------------------
		// MATTE
		//----------------------------------------------------------------------
		case MATTE:
			MatteMaterial_EvalOp(material, evalType, evalStack, evalStackOffset, hitPoint MATERIALS_PARAM);
			break;
		//----------------------------------------------------------------------
		// MATTETRANSLUCENT
		//----------------------------------------------------------------------
		case MATTETRANSLUCENT:
			MatteTranslucentMaterial_EvalOp(material, evalType, evalStack, evalStackOffset, hitPoint MATERIALS_PARAM);
			break;
		//----------------------------------------------------------------------
		// GLOSSY2
		//----------------------------------------------------------------------
		case GLOSSY2:
			Glossy2Material_EvalOp(material, evalType, evalStack, evalStackOffset, hitPoint MATERIALS_PARAM);
			break;			
		//----------------------------------------------------------------------
		// METAL2
		//----------------------------------------------------------------------
		case METAL2:
			Metal2Material_EvalOp(material, evalType, evalStack, evalStackOffset, hitPoint MATERIALS_PARAM);
			break;			
		//----------------------------------------------------------------------
		// VELVET
		//----------------------------------------------------------------------
		case VELVET:
			VelvetMaterial_EvalOp(material, evalType, evalStack, evalStackOffset, hitPoint MATERIALS_PARAM);
			break;	
		//----------------------------------------------------------------------
		// CLOTH
		//----------------------------------------------------------------------
		case CLOTH:
			ClothMaterial_EvalOp(material, evalType, evalStack, evalStackOffset, hitPoint MATERIALS_PARAM);
			break;
		//----------------------------------------------------------------------
		// CARPAINT
		//----------------------------------------------------------------------
		case CARPAINT:
			CarPaintMaterial_EvalOp(material, evalType, evalStack, evalStackOffset, hitPoint MATERIALS_PARAM);
			break;
		//----------------------------------------------------------------------
		// ROUGHMATTE
		//----------------------------------------------------------------------
		case ROUGHMATTE:
			RoughMatteMaterial_EvalOp(material, evalType, evalStack, evalStackOffset, hitPoint MATERIALS_PARAM);
			break;
		//----------------------------------------------------------------------
		// ROUGHMATTETRANSLUCENT
		//----------------------------------------------------------------------
		case ROUGHMATTETRANSLUCENT:
			RoughMatteTranslucentMaterial_EvalOp(material, evalType, evalStack, evalStackOffset, hitPoint MATERIALS_PARAM);
			break;
		//----------------------------------------------------------------------
		// GLOSSYTRANSLUCENT
		//----------------------------------------------------------------------
		case GLOSSYTRANSLUCENT:
			GlossyTranslucentMaterial_EvalOp(material, evalType, evalStack, evalStackOffset, hitPoint MATERIALS_PARAM);
			break;
		//----------------------------------------------------------------------
		// DISNEY
		//----------------------------------------------------------------------
		case DISNEY:
			DisneyMaterial_EvalOp(material, evalType, evalStack, evalStackOffset, hitPoint MATERIALS_PARAM);
			break;
		//----------------------------------------------------------------------
		// OPENPBR
		//----------------------------------------------------------------------
		case OPENPBR:
			OpenPBRMaterial_EvalOp(material, evalType, evalStack, evalStackOffset, hitPoint MATERIALS_PARAM);
			break;
		//----------------------------------------------------------------------
		// SUBSURFACE
		//----------------------------------------------------------------------
#if defined(PARAM_ENABLE_SUBSURFACE)
		case SUBSURFACEMAT:
			SubsurfaceMaterial_EvalOp(material, evalType, evalStack, evalStackOffset, hitPoint MATERIALS_PARAM);
			break;
#endif
		//----------------------------------------------------------------------
		// NULLMAT
		//----------------------------------------------------------------------
		case NULLMAT:
			NullMaterial_EvalOp(material, evalType, evalStack, evalStackOffset, hitPoint MATERIALS_PARAM);
			break;
		//----------------------------------------------------------------------
		// MIX
		//----------------------------------------------------------------------
		case MIX:
			MixMaterial_EvalOp(material, evalType, evalOp->opData.bumpSlots,
					evalStack, evalStackOffset, hitPoint MATERIALS_PARAM);
			break;
		//----------------------------------------------------------------------
		// GLOSSYCOATING
		//----------------------------------------------------------------------
		case GLOSSYCOATING:
			GlossyCoatingMaterial_EvalOp(material, evalType, evalOp->opData.bumpSlots,
					evalStack, evalStackOffset, hitPoint MATERIALS_PARAM);
			break;
		//----------------------------------------------------------------------
		// MIRROR
		//----------------------------------------------------------------------
		case MIRROR:
			MirrorMaterial_EvalOp(material, evalType, evalStack, evalStackOffset, hitPoint MATERIALS_PARAM);
			break;
		//----------------------------------------------------------------------
		// GLASS
		//----------------------------------------------------------------------
		case GLASS:
			GlassMaterial_EvalOp(material, evalType, evalStack, evalStackOffset, hitPoint MATERIALS_PARAM);
			break;
		//----------------------------------------------------------------------
		// ARCHGLASS
		//----------------------------------------------------------------------
		case ARCHGLASS:
			ArchGlassMaterial_EvalOp(material, evalType, evalStack, evalStackOffset, hitPoint MATERIALS_PARAM);
			break;
		//----------------------------------------------------------------------
		// ROUGHGLASS
		//----------------------------------------------------------------------
		case ROUGHGLASS:
			RoughGlassMaterial_EvalOp(material, evalType, evalStack, evalStackOffset, hitPoint MATERIALS_PARAM);
			break;
		//----------------------------------------------------------------------
		// TWOSIDED
		//----------------------------------------------------------------------
		case TWOSIDED:
			TwoSidedMaterial_EvalOp(material, evalType, evalStack, evalStackOffset, hitPoint MATERIALS_PARAM);
			break;
		//----------------------------------------------------------------------
		// CLEAR_VOL
		//----------------------------------------------------------------------
		case CLEAR_VOL:
			ClearVolMaterial_EvalOp(material, evalType, evalStack, evalStackOffset, hitPoint MATERIALS_PARAM);
			break;
		//----------------------------------------------------------------------
		// HOMOGENEOUS_VOL
		//----------------------------------------------------------------------
		case HOMOGENEOUS_VOL:
			HomogeneousVolMaterial_EvalOp(material, evalType, evalStack, evalStackOffset, hitPoint MATERIALS_PARAM);
			break;
		//----------------------------------------------------------------------
		// HETEROGENEOUS_VOL
		//----------------------------------------------------------------------
		case HETEROGENEOUS_VOL:
			HeterogeneousVolMaterial_EvalOp(material, evalType, evalStack, evalStackOffset, hitPoint MATERIALS_PARAM);
			break;
		default:
			// Something wrong here
			break;
	}

	return 0;
}

//------------------------------------------------------------------------------
// The EVAL_EVALUATE_DIFFUSE ops of Material_EvaluateDiffuse() (the PhotonGI
// indirect cache); its other ops (the set-ups and gotos) go to Material_EvalOp().
// Neither cases of the material EvalOp switches nor a call down to
// Material_EvalOp() from here: either one made the path kernel slower with the
// cache off (2026-10-31, gfx1030: kitchen 64.5 -> 69.6 ns/sample, and the cases
// luxball-hdr-vol 43.9 -> 46.5), as any code under Material_EvalOp() or a
// deeper call chain does.
//------------------------------------------------------------------------------

OPENCL_FORCE_NOT_INLINE void MaterialDiffuse_EvalOp(
		__global const MaterialEvalOp* restrict evalOp,
		__global float *evalStack,
		uint *evalStackOffset,
		__global const HitPoint *hitPoint
		MATERIALS_PARAM_DECL) {
	__global const Material* restrict material = &mats[evalOp->matIndex];

	switch (material->type) {
		case MATTE:
			// The whole material is the diffuse lobe
			MatteMaterial_Evaluate(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		case ROUGHMATTE:
			// The whole material is the diffuse lobe
			RoughMatteMaterial_Evaluate(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		case VELVET:
			// The whole material is the diffuse lobe
			VelvetMaterial_Evaluate(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		case MATTETRANSLUCENT:
			MatteTranslucentMaterial_Evaluate(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			// The diffuse lobe is the reflection
			DefaultMaterial_KeepDiffuseReflection(evalStack, evalStackOffset);
			break;
		case ROUGHMATTETRANSLUCENT:
			RoughMatteTranslucentMaterial_Evaluate(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			// The diffuse lobe is the reflection
			DefaultMaterial_KeepDiffuseReflection(evalStack, evalStackOffset);
			break;
		case GLOSSY2:
			Glossy2Material_EvaluateDiffuse(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		case GLOSSYTRANSLUCENT:
			GlossyTranslucentMaterial_EvaluateDiffuse(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		case CARPAINT:
			CarPaintMaterial_EvaluateDiffuse(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		case DISNEY:
			DisneyMaterial_EvaluateDiffuse(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		case OPENPBR:
			OpenPBRMaterial_EvaluateDiffuse(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
#if defined(PARAM_ENABLE_SUBSURFACE)
		case SUBSURFACEMAT:
			SubsurfaceMaterial_EvaluateDiffuse(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
#endif
		case MIX:
			// The diffuse lobes of the sub-materials (their EVAL_EVALUATE_DIFFUSE
			// ops), weighted as in EVAL_EVALUATE
			MixMaterial_Evaluate(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		case GLOSSYCOATING:
			// The diffuse lobe of the base (its EVAL_EVALUATE_DIFFUSE ops) under the coating
			GlossyCoatingMaterial_EvaluateDiffuse(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		default:
			// No diffuse lobe (the glasses, mirror, metal, cloth, null and the volumes)
			DefaultMaterial_EvaluateNoDiffuse(evalStack, evalStackOffset);
			break;
	}
}

// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
