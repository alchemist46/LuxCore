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

#if !defined(LUXRAYS_DISABLE_OPENCL)

#include <iosfwd>
#include <limits>

#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>

#include "slg/engines/pathoclbase/compiledscene.h"
#include "slg/kernels/kernels.h"

#include "slg/textures/constfloat.h"
#include "slg/textures/constfloat3.h"

#include "slg/materials/archglass.h"
#include "slg/materials/carpaint.h"
#include "slg/materials/cloth.h"
#include "slg/materials/glass.h"
#include "slg/materials/glossy2.h"
#include "slg/materials/glossycoating.h"
#include "slg/materials/glossytranslucent.h"
#include "slg/materials/matte.h"
#include "slg/materials/mattetranslucent.h"
#include "slg/materials/metal2.h"
#include "slg/materials/mirror.h"
#include "slg/materials/mix.h"
#include "slg/materials/null.h"
#include "slg/materials/roughglass.h"
#include "slg/materials/roughmatte.h"
#include "slg/materials/roughmattetranslucent.h"
#include "slg/materials/twosided.h"
#include "slg/materials/velvet.h"
#include "slg/materials/subsurface.h"

#include "slg/volumes/clear.h"
#include "slg/volumes/heterogenous.h"
#include "slg/volumes/homogenous.h"
#include "slg/materials/disney.h"
#include "slg/materials/openpbr.h"

using namespace std;
using namespace luxrays;
using namespace slg;

u_int CompiledScene::CompileMaterialConditionalOps(const u_int matIndex,
		const vector<slg::ocl::MaterialEvalOp> &evalOpsA, const u_int evalOpStackSizeA,
		const vector<slg::ocl::MaterialEvalOp> &evalOpsB, const u_int evalOpStackSizeB,
		vector<slg::ocl::MaterialEvalOp> &evalOps) const {
	u_int evalOpStackSize = 0;
	
	// Add the conditional goto
	slg::ocl::MaterialEvalOp opGotoA;
	opGotoA.matIndex = matIndex;
	opGotoA.evalType = slg::ocl::EVAL_CONDITIONAL_GOTO;
	// The +1 is for the unconditional goto
	opGotoA.opData.opsCount = evalOpsA.size() + 1;
	evalOps.push_back(opGotoA);

	// Add compiled Ops of material A
	evalOps.insert(evalOps.end(), evalOpsA.begin(), evalOpsA.end());
	evalOpStackSize += evalOpStackSizeA;

	// Add the unconditional goto
	slg::ocl::MaterialEvalOp opGotoB;
	opGotoB.matIndex = matIndex;
	opGotoB.evalType = slg::ocl::EVAL_UNCONDITIONAL_GOTO;
	opGotoB.opData.opsCount = evalOpsB.size();
	evalOps.push_back(opGotoB);

	// Add compiled Ops of material B
	evalOps.insert(evalOps.end(), evalOpsB.begin(), evalOpsB.end());
	evalOpStackSize += evalOpStackSizeB;

	return evalOpStackSize;
}

u_int CompiledScene::CompileMaterialConditionalOps(const u_int matIndex,
		const u_int matAIndex, const slg::ocl::MaterialEvalOpType opTypeA, const u_int bumpSlotBaseA,
		const u_int matBIndex, const slg::ocl::MaterialEvalOpType opTypeB, const u_int bumpSlotBaseB,
		vector<slg::ocl::MaterialEvalOp> &evalOps) const {
	// Compile opType of material A
	vector<slg::ocl::MaterialEvalOp> evalOpsA;
	const u_int evalOpStackSizeA = CompileMaterialOps(matAIndex, opTypeA,
			bumpSlotBaseA, evalOpsA);

	// Compile opType of material B
	vector<slg::ocl::MaterialEvalOp> evalOpsB;
	const u_int evalOpStackSizeB = CompileMaterialOps(matBIndex, opTypeB,
			bumpSlotBaseB, evalOpsB);

	return CompileMaterialConditionalOps(matIndex,
			evalOpsA, evalOpStackSizeA,
			evalOpsB, evalOpStackSizeB,
			evalOps);
}

// The materials without sub-nodes (the first case of CompileMaterialOps()):
// their volume ops are DefaultMaterial_GetInteriorVolume() and
// DefaultMaterial_GetExteriorVolume(), and the pass through transparency op is
// DefaultMaterial_GetPassThroughTransparency() except for NULLMAT and ARCHGLASS
static bool IsLeafMaterial(const slg::ocl::MaterialType type) {
	switch (type) {
		case slg::ocl::MATTE:
		case slg::ocl::MIRROR:
		case slg::ocl::GLASS:
		case slg::ocl::ARCHGLASS:
		case slg::ocl::NULLMAT:
		case slg::ocl::MATTETRANSLUCENT:
		case slg::ocl::GLOSSY2:
		case slg::ocl::METAL2:
		case slg::ocl::ROUGHGLASS:
		case slg::ocl::VELVET:
		case slg::ocl::CLOTH:
		case slg::ocl::CARPAINT:
		case slg::ocl::ROUGHMATTE:
		case slg::ocl::ROUGHMATTETRANSLUCENT:
		case slg::ocl::GLOSSYTRANSLUCENT:
		case slg::ocl::DISNEY:
		case slg::ocl::OPENPBR:
		case slg::ocl::SUBSURFACEMAT:
		case slg::ocl::HOMOGENEOUS_VOL:
		case slg::ocl::CLEAR_VOL:
		case slg::ocl::HETEROGENEOUS_VOL:
			return true;
		default:
			return false;
	}
}

// True if Material_Bump() of the material can change the hit point
bool CompiledScene::MaterialHasBump(const u_int matIndex) const {
	const slg::ocl::Material *mat = &mats[matIndex];

	// As Material_GetBumpMaterial(): a TWOSIDED one, down its nested ones
	if (mat->type == slg::ocl::TWOSIDED)
		return MaterialHasBump(mat->twosided.frontMatIndex) ||
				MaterialHasBump(mat->twosided.backMatIndex);
	else
		return (mat->bumpTexIndex != NULL_INDEX);
}

// False if the pass through transparency op of the material always returns
// black with no new vertex pass-through event (as
// Material::HasPassThroughTransparency()): no transparency texture that the
// op reads, no NULLMAT or ARCHGLASS, down the tree
bool CompiledScene::MaterialHasPassThroughTransparency(const u_int matIndex) const {
	const slg::ocl::Material *mat = &mats[matIndex];
	const bool ownTransparency = (mat->frontTranspTexIndex != NULL_INDEX) ||
			(mat->backTranspTexIndex != NULL_INDEX);

	switch (mat->type) {
		case slg::ocl::NULLMAT:
		case slg::ocl::ARCHGLASS:
			return true;
		case slg::ocl::MIX:
			return ownTransparency ||
					MaterialHasPassThroughTransparency(mat->mix.matAIndex) ||
					MaterialHasPassThroughTransparency(mat->mix.matBIndex);
		case slg::ocl::GLOSSYCOATING:
			// Its own transparency textures are not read
			// (GlossyCoatingMaterial_GetPassThroughTransparency())
			return MaterialHasPassThroughTransparency(mat->glossycoating.matBaseIndex);
		case slg::ocl::TWOSIDED:
			return ownTransparency ||
					MaterialHasPassThroughTransparency(mat->twosided.frontMatIndex) ||
					MaterialHasPassThroughTransparency(mat->twosided.backMatIndex);
		default:
			return ownTransparency;
	}
}

//------------------------------------------------------------------------------
// The weights of the Mix materials at a vertex
//------------------------------------------------------------------------------

u_int CompiledScene::AddVertexFactorTexture(const slg::ocl::Texture &tex) {
	texs.push_back(tex);

	return texs.size() - 1;
}

u_int CompiledScene::AddVertexFactorConstTexture(const float value) {
	slg::ocl::Texture tex;
	memset(&tex, 0, sizeof(tex));
	tex.type = slg::ocl::CONST_FLOAT;
	tex.constFloat.value = value;

	return AddVertexFactorTexture(tex);
}

// Material::GetVertexProbability() as a texture, for a material in the tree of
// a Mix with vertex weights: it depends only on the hit point
// (Material::HasPointVertexProbability())
u_int CompiledScene::CompileVertexProbabilityTexture(const Material &material) {
	if (!material.HasPassThroughTransparency())
		return AddVertexFactorConstTexture(1.f);

	const bool ownTransparency = material.GetFrontTransparencyTexture() ||
			material.GetBackTransparencyTexture();
	switch (material.GetType()) {
		case NULLMAT:
			return AddVertexFactorConstTexture(0.f);
		case MIX:
			if (!ownTransparency) {
				// w1 o_A + w2 o_B (MixMaterial::GetVertexProbability()): the
				// mix texture clamps its amount as the Mix clamps its factor
				auto &mm = dynamic_cast<const MixMaterial &>(material);
				slg::ocl::Texture tex;
				memset(&tex, 0, sizeof(tex));
				tex.type = slg::ocl::MIX_TEX;
				tex.mixTex.amountTexIndex = scene.GetTextures().GetTextureIndex(mm.GetMixFactor());
				tex.mixTex.tex1Index = CompileVertexProbabilityTexture(mm.GetMaterialA());
				tex.mixTex.tex2Index = CompileVertexProbabilityTexture(mm.GetMaterialB());

				return AddVertexFactorTexture(tex);
			}
			break;
		case GLOSSYCOATING:
			return CompileVertexProbabilityTexture(
					*dynamic_cast<const GlossyCoatingMaterial &>(material).GetMaterialBase());
		default:
			break;
	}

	// The material's own transparency texture, the same on both sides
	auto transpTex = material.GetFrontTransparencyTexture();
	if (!transpTex)
		return AddVertexFactorConstTexture(1.f);

	const u_int transpTexIndex = scene.GetTextures().GetTextureIndex(transpTex);
	if (texs[transpTexIndex].type == slg::ocl::CONST_FLOAT)
		return AddVertexFactorConstTexture(Clamp(texs[transpTexIndex].constFloat.value, 0.f, 1.f));

	slg::ocl::Texture tex;
	memset(&tex, 0, sizeof(tex));
	tex.type = slg::ocl::CLAMP_TEX;
	tex.clampTex.texIndex = transpTexIndex;
	tex.clampTex.minVal = 0.f;
	tex.clampTex.maxVal = 1.f;

	return AddVertexFactorTexture(tex);
}

// The weight of material B at a vertex (MixMaterial::GetVertexWeight2()) of
// each Mix, in MixParam::vertexFactorTexIndex: the mix factor or, for a Mix with
// vertex weights, a texture added to the scene's ones, w2 o_B / (w1 o_A +
// w2 o_B) with o the vertex probability of a sub-material
// (Material::GetVertexProbability()). A texture instead of a computation in
// the Mix ops, and the pass-through op as it was (MixMaterial::Sample() picks
// the sub-material with the mix factor): the speed of the path kernel on
// every scene moves with any code under Material_EvalOp() (2026-10-26, gfx1030:
// a walk of the sub-materials in the Mix ops took 3-16% more time per sample,
// on scenes with or without such a Mix, and the vertex weights in the
// pass-through op, which runs in the intersection loop, 4-11% more on the
// scenes with one).
void CompiledScene::CompileMixVertexFactors() {
	const size_t texsCount = texs.size();

	for (u_int i = 0; i < mats.size(); ++i) {
		slg::ocl::Material *mat = &mats[i];
		if (mat->type != slg::ocl::MIX)
			continue;

		mat->mix.vertexFactorTexIndex = mat->mix.mixFactorTexIndex;

		auto &mm = dynamic_cast<const MixMaterial &>(scene.GetMaterials().GetMaterial(i));
		if (!mm.HasVertexWeights())
			continue;

		const u_int probA = CompileVertexProbabilityTexture(mm.GetMaterialA());
		const u_int probB = CompileVertexProbabilityTexture(mm.GetMaterialB());
		const bool constA = (texs[probA].type == slg::ocl::CONST_FLOAT);
		const bool constB = (texs[probB].type == slg::ocl::CONST_FLOAT);
		const float valueA = constA ? texs[probA].constFloat.value : -1.f;
		const float valueB = constB ? texs[probB].constFloat.value : -1.f;

		if (constA && constB && (valueA == valueB)) {
			// The weights stay the factor's
			continue;
		} else if (constA && (valueA == 0.f)) {
			// Only material B makes a vertex. (Compiling the Mix as material B
			// for the evaluations at a vertex is not the same: the Mix's ops
			// repeat the reflection/transmission test of BSDF_Evaluate() with
			// the event types of the sub-material.)
			mat->mix.vertexFactorTexIndex = AddVertexFactorConstTexture(1.f);
		} else if (constB && (valueB == 0.f)) {
			// Only material A
			mat->mix.vertexFactorTexIndex = AddVertexFactorConstTexture(0.f);
		} else {
			// w2, clamped as the Mix clamps it
			slg::ocl::Texture weight2Tex;
			memset(&weight2Tex, 0, sizeof(weight2Tex));
			weight2Tex.type = slg::ocl::CLAMP_TEX;
			weight2Tex.clampTex.texIndex = mat->mix.mixFactorTexIndex;
			weight2Tex.clampTex.minVal = 0.f;
			weight2Tex.clampTex.maxVal = 1.f;
			const u_int weight2 = AddVertexFactorTexture(weight2Tex);

			// w2 o_B
			u_int v2 = weight2;
			if (!constB || (valueB != 1.f)) {
				slg::ocl::Texture tex;
				memset(&tex, 0, sizeof(tex));
				tex.type = slg::ocl::SCALE_TEX;
				tex.scaleTex.tex1Index = weight2;
				tex.scaleTex.tex2Index = probB;
				v2 = AddVertexFactorTexture(tex);
			}

			// w1 o_A + w2 o_B
			slg::ocl::Texture vTex;
			memset(&vTex, 0, sizeof(vTex));
			vTex.type = slg::ocl::MIX_TEX;
			vTex.mixTex.amountTexIndex = weight2;
			vTex.mixTex.tex1Index = probA;
			vTex.mixTex.tex2Index = probB;
			const u_int v = AddVertexFactorTexture(vTex);

			// The quotient, 0 where v = 0 (never a vertex)
			slg::ocl::Texture tex;
			memset(&tex, 0, sizeof(tex));
			tex.type = slg::ocl::DIVIDE_TEX;
			tex.divideTex.tex1Index = v2;
			tex.divideTex.tex2Index = v;
			mat->mix.vertexFactorTexIndex = AddVertexFactorTexture(tex);
		}
	}

	if (texs.size() != texsCount) {
		SLG_LOG("Mix vertex weights: " << (texs.size() - texsCount) << " textures");
		CompileTextureOps();
	}
}

// The bump cache slots of the tree of a material: one per bumped sub-material
// of a MIX or GLOSSYCOATING node (a TWOSIDED node has none of its own: its
// parent bumps it with the front or back bump), in the order of
// CompileMaterialOps(), the same for every op type
u_int CompiledScene::MaterialBumpSlotCount(const u_int matIndex) const {
	const slg::ocl::Material *mat = &mats[matIndex];

	switch (mat->type) {
		case MIX:
			return (MaterialHasBump(mat->mix.matAIndex) ? 1 : 0) +
					(MaterialHasBump(mat->mix.matBIndex) ? 1 : 0) +
					MaterialBumpSlotCount(mat->mix.matAIndex) +
					MaterialBumpSlotCount(mat->mix.matBIndex);
		case GLOSSYCOATING:
			return (MaterialHasBump(mat->glossycoating.matBaseIndex) ? 1 : 0) +
					MaterialBumpSlotCount(mat->glossycoating.matBaseIndex);
		case TWOSIDED:
			return MaterialBumpSlotCount(mat->twosided.frontMatIndex) +
					MaterialBumpSlotCount(mat->twosided.backMatIndex);
		default:
			return 0;
	}
}

static u_int BumpCacheSlot(const u_int slot) {
	return (slot < MATERIAL_BUMP_CACHE_MAX_SLOTS) ? slot : MATERIAL_BUMP_CACHE_NO_SLOT;
}

// bumpSlotBase is the first bump cache slot of the tree of the material
u_int CompiledScene::CompileMaterialOps(const u_int matIndex,
		const slg::ocl::MaterialEvalOpType opType, const u_int bumpSlotBase,
		vector<slg::ocl::MaterialEvalOp> &evalOps) const {
	// Translate materials to material evaluate ops

	const slg::ocl::Material *mat = &mats[matIndex];
	u_int evalOpStackSize = 0;
	bool addDefaultOp = true;

	// The bump cache slots of the sub-materials the setup ops bump (A or the
	// base, and B) and the first slots of the trees of the sub-materials (A,
	// the base or the front, and B or the back)
	u_int bumpSlotA = MATERIAL_BUMP_CACHE_NO_SLOT;
	u_int bumpSlotB = MATERIAL_BUMP_CACHE_NO_SLOT;
	u_int bumpSlotBaseA = bumpSlotBase;
	u_int bumpSlotBaseB = bumpSlotBase;
	switch (mat->type) {
		case MIX:
			if (MaterialHasBump(mat->mix.matAIndex))
				bumpSlotA = BumpCacheSlot(bumpSlotBaseA++);
			if (MaterialHasBump(mat->mix.matBIndex))
				bumpSlotB = BumpCacheSlot(bumpSlotBaseA++);
			bumpSlotBaseB = bumpSlotBaseA + MaterialBumpSlotCount(mat->mix.matAIndex);
			break;
		case GLOSSYCOATING:
			if (MaterialHasBump(mat->glossycoating.matBaseIndex))
				bumpSlotA = BumpCacheSlot(bumpSlotBaseA++);
			break;
		case TWOSIDED:
			bumpSlotBaseB = bumpSlotBaseA + MaterialBumpSlotCount(mat->twosided.frontMatIndex);
			break;
		default:
			break;
	}

	switch (mat->type) {
		//----------------------------------------------------------------------
		// Materials without sub-nodes
		//----------------------------------------------------------------------
		case MATTE:
		case MIRROR:
		case GLASS:
		case ARCHGLASS:
		case NULLMAT:
		case MATTETRANSLUCENT:
		case GLOSSY2:
		case METAL2:
		case ROUGHGLASS:
		case VELVET:
		case CLOTH:
		case CARPAINT:
		case ROUGHMATTE:
		case ROUGHMATTETRANSLUCENT:
		case GLOSSYTRANSLUCENT:
		case DISNEY:
		case OPENPBR:
		case SUBSURFACEMAT:
		case HOMOGENEOUS_VOL:
		case CLEAR_VOL:
		case HETEROGENEOUS_VOL:
			switch (opType) {
				case slg::ocl::EVAL_ALBEDO:
					// 1 x parameter and 3 x results
					evalOpStackSize += 3;
					break;
				case slg::ocl::EVAL_GET_INTERIOR_VOLUME:
					// 1 x parameter and 1 x result
					evalOpStackSize += 1;
					break;
				case slg::ocl::EVAL_GET_EXTERIOR_VOLUME:
					// 1 x parameter and 1 x result
					evalOpStackSize += 1;
					break;
				case slg::ocl::EVAL_GET_EMITTED_RADIANCE:
					// 1 x parameter and 3 x results
					evalOpStackSize += 3;
					break;
				case slg::ocl::EVAL_GET_PASS_TROUGH_TRANSPARENCY:
					// 5 x parameters and 4 x results (the transparency and the vertex pass-through event)
					evalOpStackSize += 5;
					break;
				case slg::ocl::EVAL_EVALUATE:
					// 6 x parameters and 5 x results
					evalOpStackSize += 6;
					break;
				case slg::ocl::EVAL_SAMPLE:
					// 6 x parameters and 8 x results
					evalOpStackSize += 8;
					break;
				default:
					throw runtime_error("Unknown eval. type in CompiledScene::CompileMaterialOps(" + ToString(mat->type) + "): " + ToString(opType));
			}
			break;
		//----------------------------------------------------------------------
		// Materials with sub-nodes
		//----------------------------------------------------------------------
		case MIX:
			switch (opType) {
				case slg::ocl::EVAL_ALBEDO:
					evalOpStackSize += CompileMaterialOps(mat->mix.matAIndex, slg::ocl::EVAL_ALBEDO, bumpSlotBaseA, evalOps);
					evalOpStackSize += CompileMaterialOps(mat->mix.matBIndex, slg::ocl::EVAL_ALBEDO, bumpSlotBaseB, evalOps);
					break;
				case slg::ocl::EVAL_GET_VOLUME_MIX_SETUP1:
					evalOpStackSize += 3;
					break;
				case slg::ocl::EVAL_GET_VOLUME_MIX_SETUP2:
					evalOpStackSize += 4;
					break;
				case slg::ocl::EVAL_GET_INTERIOR_VOLUME:
					if (mat->interiorVolumeIndex == NULL_INDEX) {
						evalOpStackSize += CompileMaterialOps(matIndex, slg::ocl::EVAL_GET_VOLUME_MIX_SETUP1, bumpSlotBase, evalOps);
						evalOpStackSize += CompileMaterialOps(mat->mix.matAIndex, slg::ocl::EVAL_GET_INTERIOR_VOLUME, bumpSlotBaseA, evalOps);
						evalOpStackSize += CompileMaterialOps(matIndex, slg::ocl::EVAL_GET_VOLUME_MIX_SETUP2, bumpSlotBase, evalOps);
						evalOpStackSize += CompileMaterialOps(mat->mix.matBIndex, slg::ocl::EVAL_GET_INTERIOR_VOLUME, bumpSlotBaseB, evalOps);
					}

					// 1 x parameter and 1 x result
					evalOpStackSize += 1;	
					break;
				case slg::ocl::EVAL_GET_EXTERIOR_VOLUME:
					if (mat->exteriorVolumeIndex == NULL_INDEX) {
						evalOpStackSize += CompileMaterialOps(matIndex, slg::ocl::EVAL_GET_VOLUME_MIX_SETUP1, bumpSlotBase, evalOps);
						evalOpStackSize += CompileMaterialOps(mat->mix.matAIndex, slg::ocl::EVAL_GET_EXTERIOR_VOLUME, bumpSlotBaseA, evalOps);
						evalOpStackSize += CompileMaterialOps(matIndex, slg::ocl::EVAL_GET_VOLUME_MIX_SETUP2, bumpSlotBase, evalOps);
						evalOpStackSize += CompileMaterialOps(mat->mix.matBIndex, slg::ocl::EVAL_GET_EXTERIOR_VOLUME, bumpSlotBaseB, evalOps);
					}

					// 1 x parameter and 1 x result
					evalOpStackSize += 1;	
					break;
				case slg::ocl::EVAL_GET_EMITTED_RADIANCE_MIX_SETUP1:
					// 1 x parameter and 2 x results
					evalOpStackSize += 2;
					break;
				case slg::ocl::EVAL_GET_EMITTED_RADIANCE_MIX_SETUP2:
					// 1 x parameter and 5 x results
					evalOpStackSize += 5;
					break;
				case slg::ocl::EVAL_GET_EMITTED_RADIANCE:
					if (mat->emitTexIndex == NULL_INDEX) {
						evalOpStackSize += CompileMaterialOps(matIndex, slg::ocl::EVAL_GET_EMITTED_RADIANCE_MIX_SETUP1, bumpSlotBase, evalOps);
						evalOpStackSize += CompileMaterialOps(mat->mix.matAIndex, slg::ocl::EVAL_GET_EMITTED_RADIANCE, bumpSlotBaseA, evalOps);
						evalOpStackSize += CompileMaterialOps(matIndex, slg::ocl::EVAL_GET_EMITTED_RADIANCE_MIX_SETUP2, bumpSlotBase, evalOps);
						evalOpStackSize += CompileMaterialOps(mat->mix.matBIndex, slg::ocl::EVAL_GET_EMITTED_RADIANCE, bumpSlotBaseB, evalOps);
					}

					// 1 x parameter and 3 x result
					evalOpStackSize += 3;
					break;
				case slg::ocl::EVAL_GET_PASS_TROUGH_TRANSPARENCY_MIX_SETUP1:
					// 5 x parameters and 4 x results (the transparency and the vertex pass-through event)
					evalOpStackSize += 5;
					break;
				case slg::ocl::EVAL_GET_PASS_TROUGH_TRANSPARENCY_MIX_SETUP2:
					// 5 x parameters and 4 x results (the transparency and the vertex pass-through event)
					evalOpStackSize += 5;
					break;
				case slg::ocl::EVAL_GET_PASS_TROUGH_TRANSPARENCY:
					evalOpStackSize += CompileMaterialOps(matIndex, slg::ocl::EVAL_GET_PASS_TROUGH_TRANSPARENCY_MIX_SETUP1, bumpSlotBase, evalOps);
					evalOpStackSize += CompileMaterialOps(mat->mix.matAIndex, slg::ocl::EVAL_GET_PASS_TROUGH_TRANSPARENCY, bumpSlotBaseA, evalOps);
					evalOpStackSize += CompileMaterialOps(matIndex, slg::ocl::EVAL_GET_PASS_TROUGH_TRANSPARENCY_MIX_SETUP2, bumpSlotBase, evalOps);
					evalOpStackSize += CompileMaterialOps(mat->mix.matBIndex, slg::ocl::EVAL_GET_PASS_TROUGH_TRANSPARENCY, bumpSlotBaseB, evalOps);

					// 5 x parameters and 4 x results (the transparency and the vertex pass-through event)
					evalOpStackSize += 5;
					break;
				case slg::ocl::EVAL_EVALUATE_MIX_SETUP1:
					// 6 x parameters and 21 x results
					evalOpStackSize += 21;
					break;
				case slg::ocl::EVAL_EVALUATE_MIX_SETUP2:
					// 12 x parameters and 26 x results
					evalOpStackSize += 26;
					break;
				case slg::ocl::EVAL_EVALUATE:
					evalOpStackSize += CompileMaterialOps(matIndex, slg::ocl::EVAL_EVALUATE_MIX_SETUP1, bumpSlotBase, evalOps);
					evalOpStackSize += CompileMaterialOps(mat->mix.matAIndex, slg::ocl::EVAL_EVALUATE, bumpSlotBaseA, evalOps);
					evalOpStackSize += CompileMaterialOps(matIndex, slg::ocl::EVAL_EVALUATE_MIX_SETUP2, bumpSlotBase, evalOps);
					evalOpStackSize += CompileMaterialOps(mat->mix.matBIndex, slg::ocl::EVAL_EVALUATE, bumpSlotBaseB, evalOps);
					break;
				case slg::ocl::EVAL_SAMPLE_MIX_SETUP1:
					// 6 x parameters and 25 x results
					evalOpStackSize += 25;
					break;
				case slg::ocl::EVAL_SAMPLE_MIX_SETUP2:
					// 16 x parameters and 33 x results
					evalOpStackSize += 33;
					break;
				case slg::ocl::EVAL_SAMPLE: {
					evalOpStackSize += CompileMaterialOps(matIndex, slg::ocl::EVAL_SAMPLE_MIX_SETUP1, bumpSlotBase, evalOps);
					evalOpStackSize += CompileMaterialConditionalOps(matIndex,
							mat->mix.matAIndex, slg::ocl::EVAL_SAMPLE, bumpSlotBaseA,
							mat->mix.matBIndex, slg::ocl::EVAL_SAMPLE, bumpSlotBaseB,
							evalOps);
					evalOpStackSize += CompileMaterialOps(matIndex, slg::ocl::EVAL_SAMPLE_MIX_SETUP2, bumpSlotBase, evalOps);
					evalOpStackSize += CompileMaterialConditionalOps(matIndex,
							mat->mix.matAIndex, slg::ocl::EVAL_EVALUATE, bumpSlotBaseA,
							mat->mix.matBIndex, slg::ocl::EVAL_EVALUATE, bumpSlotBaseB,
							evalOps);
					break;
				}
				default:
					throw runtime_error("Unknown eval. type in CompiledScene::CompileMaterialOps(" + ToString(mat->type) + "): " + ToString(opType));
			}
			break;
		case GLOSSYCOATING:
			switch (opType) {
				case slg::ocl::EVAL_ALBEDO:
					evalOpStackSize += CompileMaterialOps(mat->glossycoating.matBaseIndex, slg::ocl::EVAL_ALBEDO, bumpSlotBaseA, evalOps);
					break;
				case slg::ocl::EVAL_GET_INTERIOR_VOLUME:
					if (mat->interiorVolumeIndex == NULL_INDEX)
						evalOpStackSize += CompileMaterialOps(mat->glossycoating.matBaseIndex, slg::ocl::EVAL_GET_INTERIOR_VOLUME, bumpSlotBaseA, evalOps);

					// 1 x parameter and 1 x result
					evalOpStackSize += 1;					
					break;
				case slg::ocl::EVAL_GET_EXTERIOR_VOLUME:
					if (mat->exteriorVolumeIndex == NULL_INDEX)
						evalOpStackSize += CompileMaterialOps(mat->glossycoating.matBaseIndex, slg::ocl::EVAL_GET_EXTERIOR_VOLUME, bumpSlotBaseA, evalOps);

					// 1 x parameter and 1 x result
					evalOpStackSize += 1;
					break;
				case slg::ocl::EVAL_GET_EMITTED_RADIANCE:
					if (mat->emitTexIndex == NULL_INDEX)
						evalOpStackSize += CompileMaterialOps(mat->glossycoating.matBaseIndex, slg::ocl::EVAL_GET_EMITTED_RADIANCE, bumpSlotBaseA, evalOps);

					// 1 x parameter and 3 x results
					evalOpStackSize += 3;
					break;
				case slg::ocl::EVAL_GET_PASS_TROUGH_TRANSPARENCY:
					evalOpStackSize += CompileMaterialOps(mat->glossycoating.matBaseIndex, slg::ocl::EVAL_GET_PASS_TROUGH_TRANSPARENCY, bumpSlotBaseA, evalOps);

					// 5 x parameters and 4 x results (the transparency and the vertex pass-through event)
					evalOpStackSize += 5;
					break;
				case slg::ocl::EVAL_EVALUATE_GLOSSYCOATING_SETUP:
					// 6 x parameters and 21 x results
					evalOpStackSize += 21;
					break;
				case slg::ocl::EVAL_EVALUATE:
					evalOpStackSize += CompileMaterialOps(matIndex, slg::ocl::EVAL_EVALUATE_GLOSSYCOATING_SETUP, bumpSlotBase, evalOps);
					evalOpStackSize += CompileMaterialOps(mat->glossycoating.matBaseIndex, slg::ocl::EVAL_EVALUATE, bumpSlotBaseA, evalOps);
					// The albedo of the base for the darkening, on top of the
					// stack for the glossy coating evaluation
					if (mat->glossycoating.darkening)
						evalOpStackSize += CompileMaterialOps(mat->glossycoating.matBaseIndex, slg::ocl::EVAL_ALBEDO, bumpSlotBaseA, evalOps);
					break;
				case slg::ocl::EVAL_SAMPLE_GLOSSYCOATING_SETUP:
					// 6 x parameters and 35 x results
					evalOpStackSize += 35;
					break;
				case slg::ocl::EVAL_SAMPLE_GLOSSYCOATING_CLOSE_SAMPLE_BASE:
					// 26 x parameters and 8 x results
					evalOpStackSize += 8;
					break;
				case slg::ocl::EVAL_SAMPLE_GLOSSYCOATING_CLOSE_EVALUATE_BASE:
					// 26 x parameters and 8 x results
					evalOpStackSize += 8;
					break;
				case slg::ocl::EVAL_SAMPLE: {
					evalOpStackSize += CompileMaterialOps(matIndex, slg::ocl::EVAL_SAMPLE_GLOSSYCOATING_SETUP, bumpSlotBase, evalOps);
					
					vector<slg::ocl::MaterialEvalOp> evalOpsSampleBase;
					u_int evalOpStackSizeSampleBase = CompileMaterialOps(mat->glossycoating.matBaseIndex, slg::ocl::EVAL_SAMPLE, bumpSlotBaseA, evalOpsSampleBase);
					if (mat->glossycoating.darkening)
						evalOpStackSizeSampleBase += CompileMaterialOps(mat->glossycoating.matBaseIndex, slg::ocl::EVAL_ALBEDO, bumpSlotBaseA, evalOpsSampleBase);
					evalOpStackSizeSampleBase += CompileMaterialOps(matIndex, slg::ocl::EVAL_SAMPLE_GLOSSYCOATING_CLOSE_SAMPLE_BASE, bumpSlotBase, evalOpsSampleBase);

					vector<slg::ocl::MaterialEvalOp> evalOpsEvaluateBase;
					u_int evalOpStackSizeEvaluateBase = CompileMaterialOps(mat->glossycoating.matBaseIndex, slg::ocl::EVAL_EVALUATE, bumpSlotBaseA, evalOpsEvaluateBase);
					if (mat->glossycoating.darkening)
						evalOpStackSizeEvaluateBase += CompileMaterialOps(mat->glossycoating.matBaseIndex, slg::ocl::EVAL_ALBEDO, bumpSlotBaseA, evalOpsEvaluateBase);
					evalOpStackSizeEvaluateBase += CompileMaterialOps(matIndex, slg::ocl::EVAL_SAMPLE_GLOSSYCOATING_CLOSE_EVALUATE_BASE, bumpSlotBase, evalOpsEvaluateBase);
							
					evalOpStackSize += CompileMaterialConditionalOps(matIndex,
							evalOpsSampleBase, evalOpStackSizeSampleBase,
							evalOpsEvaluateBase, evalOpStackSizeEvaluateBase,
							evalOps);
					
					addDefaultOp = false;
					break;
				}
				default:
					throw runtime_error("Unknown eval. type in CompiledScene::CompileMaterialOps(" + ToString(mat->type) + "): " + ToString(opType));
			}
			break;
		case TWOSIDED:
			switch (opType) {
				case slg::ocl::EVAL_ALBEDO:
					evalOpStackSize += CompileMaterialOps(matIndex, slg::ocl::EVAL_TWOSIDED_SETUP, bumpSlotBase, evalOps);
					evalOpStackSize += CompileMaterialConditionalOps(matIndex,
							mat->twosided.frontMatIndex, opType, bumpSlotBaseA,
							mat->twosided.backMatIndex, opType, bumpSlotBaseB,
							evalOps);

					addDefaultOp = false;
					break;
				case slg::ocl::EVAL_GET_INTERIOR_VOLUME:
					if (mat->interiorVolumeIndex == NULL_INDEX) {
						evalOpStackSize += CompileMaterialOps(matIndex, slg::ocl::EVAL_TWOSIDED_SETUP, bumpSlotBase, evalOps);
						evalOpStackSize += CompileMaterialConditionalOps(matIndex,
								mat->twosided.frontMatIndex, opType, bumpSlotBaseA,
								mat->twosided.backMatIndex, opType, bumpSlotBaseB,
								evalOps);

						addDefaultOp = false;
					} else {
						// 1 x parameter and 1 x result
						evalOpStackSize += 1;
					}
					break;
				case slg::ocl::EVAL_GET_EXTERIOR_VOLUME:
					if (mat->exteriorVolumeIndex == NULL_INDEX) {
						evalOpStackSize += CompileMaterialOps(matIndex, slg::ocl::EVAL_TWOSIDED_SETUP, bumpSlotBase, evalOps);
						evalOpStackSize += CompileMaterialConditionalOps(matIndex,
								mat->twosided.frontMatIndex, opType, bumpSlotBaseA,
								mat->twosided.backMatIndex, opType, bumpSlotBaseB,
								evalOps);

						addDefaultOp = false;
					} else {
						// 1 x parameter and 1 x result
						evalOpStackSize += 1;
					}
					break;
				case slg::ocl::EVAL_GET_EMITTED_RADIANCE:
					if (mat->emitTexIndex == NULL_INDEX) {
						evalOpStackSize += CompileMaterialOps(matIndex, slg::ocl::EVAL_TWOSIDED_SETUP, bumpSlotBase, evalOps);
						evalOpStackSize += CompileMaterialConditionalOps(matIndex,
								mat->twosided.frontMatIndex, opType, bumpSlotBaseA,
								mat->twosided.backMatIndex, opType, bumpSlotBaseB,
								evalOps);

						addDefaultOp = false;
					} else {
						// 1 x parameter and 3 x results
						evalOpStackSize += 3;
					}
					break;
				case slg::ocl::EVAL_GET_PASS_TROUGH_TRANSPARENCY:
					if ((mat->frontTranspTexIndex == NULL_INDEX) && (mat->backTranspTexIndex == NULL_INDEX)) {
						evalOpStackSize += CompileMaterialOps(matIndex, slg::ocl::EVAL_TWOSIDED_SETUP, bumpSlotBase, evalOps);
						evalOpStackSize += CompileMaterialConditionalOps(matIndex,
								mat->twosided.frontMatIndex, opType, bumpSlotBaseA,
								mat->twosided.backMatIndex, opType, bumpSlotBaseB,
								evalOps);

						addDefaultOp = false;
					} else {
						// 5 x parameters and 4 x results (the transparency and the vertex pass-through event)
						evalOpStackSize += 5;
					}
					break;
				case slg::ocl::EVAL_EVALUATE:
					evalOpStackSize += CompileMaterialOps(matIndex, slg::ocl::EVAL_TWOSIDED_SETUP, bumpSlotBase, evalOps);
					evalOpStackSize += CompileMaterialConditionalOps(matIndex,
							mat->twosided.frontMatIndex, opType, bumpSlotBaseA,
							mat->twosided.backMatIndex, opType, bumpSlotBaseB,
							evalOps);

					addDefaultOp = false;
					break;
				case slg::ocl::EVAL_SAMPLE:
					evalOpStackSize += CompileMaterialOps(matIndex, slg::ocl::EVAL_TWOSIDED_SETUP, bumpSlotBase, evalOps);
					evalOpStackSize += CompileMaterialConditionalOps(matIndex,
							mat->twosided.frontMatIndex, opType, bumpSlotBaseA,
							mat->twosided.backMatIndex, opType, bumpSlotBaseB,
							evalOps);

					addDefaultOp = false;
					break;
				case slg::ocl::EVAL_TWOSIDED_SETUP:
					// 1 x parameter and 1 x results
					evalOpStackSize += 1;
					break;
				default:
					throw runtime_error("Unknown eval. type in CompiledScene::CompileMaterialOps(" + ToString(mat->type) + "): " + ToString(opType));
			}
			break;
		default:
			throw runtime_error("Unknown material in CompiledScene::CompileMaterialOps(" + ToString(opType) + "): " + ToString(mat->type));
	}

	if (addDefaultOp) {
		slg::ocl::MaterialEvalOp op;

		op.matIndex = matIndex;
		op.evalType = opType;
		op.opData.bumpSlots = MATERIAL_BUMP_CACHE_SLOTS(bumpSlotA, bumpSlotB);

		evalOps.push_back(op);
	}

	return evalOpStackSize;
}

void CompiledScene::CompileMaterialOps() {
	// Translate materials to material evaluate ops

	matEvalOps.clear();
	maxMaterialEvalStackSize = 0;
	u_int maxBumpSlotCount = 0;

	for (u_int i = 0; i < mats.size(); ++i) {
		slg::ocl::Material *mat = &mats[i];

		// The queries the kernels answer without the interpreter: a leaf
		// material's volumes are its own, and the pass through transparency
		// of a tree without a transparency texture, NULLMAT or ARCHGLASS is
		// black (DefaultMaterial_*(), the Mix ops)
		const bool leaf = IsLeafMaterial(mat->type);
		mat->constVolumes = leaf;
		mat->noPassThroughTransparency = !MaterialHasPassThroughTransparency(i);
		mat->nullPassThrough = (mat->type == slg::ocl::NULLMAT) &&
				(mat->frontTranspTexIndex == NULL_INDEX) && (mat->backTranspTexIndex == NULL_INDEX);
		mat->defaultEmittedRadiance = leaf;

		maxBumpSlotCount = Max(maxBumpSlotCount, MaterialBumpSlotCount(i));
		mat->constEmittedRadiance = leaf && ((mat->emitTexIndex == NULL_INDEX) ||
				(texs[mat->emitTexIndex].type == slg::ocl::CONST_FLOAT) ||
				(texs[mat->emitTexIndex].type == slg::ocl::CONST_FLOAT3));

		//----------------------------------------------------------------------
		// EVAL_ALBEDO
		//----------------------------------------------------------------------
		
		mat->evalAlbedoOpStartIndex = matEvalOps.size();
		const u_int evalAlbedoOpsStackSizeFloat = CompileMaterialOps(i,
				slg::ocl::MaterialEvalOpType::EVAL_ALBEDO, 0, matEvalOps);
		mat->evalAlbedoOpLength = matEvalOps.size() - mat->evalAlbedoOpStartIndex;

		maxMaterialEvalStackSize = Max(maxMaterialEvalStackSize, evalAlbedoOpsStackSizeFloat);

		//----------------------------------------------------------------------
		// EVAL_GET_INTERIOR_VOLUME
		//----------------------------------------------------------------------
		
		mat->evalGetInteriorVolumeOpStartIndex = matEvalOps.size();
		const u_int evalGetInteriorVolumeOpsStackSizeFloat = CompileMaterialOps(i,
				slg::ocl::MaterialEvalOpType::EVAL_GET_INTERIOR_VOLUME, 0, matEvalOps);
		mat->evalGetInteriorVolumeOpLength = matEvalOps.size() - mat->evalGetInteriorVolumeOpStartIndex;

		maxMaterialEvalStackSize = Max(maxMaterialEvalStackSize, evalGetInteriorVolumeOpsStackSizeFloat);

		//----------------------------------------------------------------------
		// EVAL_GET_EXTERIOR_VOLUME
		//----------------------------------------------------------------------
		
		mat->evalGetExteriorVolumeOpStartIndex = matEvalOps.size();
		const u_int evalGetExteriorVolumeOpsStackSizeFloat = CompileMaterialOps(i,
				slg::ocl::MaterialEvalOpType::EVAL_GET_EXTERIOR_VOLUME, 0, matEvalOps);
		mat->evalGetExteriorVolumeOpLength = matEvalOps.size() - mat->evalGetExteriorVolumeOpStartIndex;

		maxMaterialEvalStackSize = Max(maxMaterialEvalStackSize, evalGetExteriorVolumeOpsStackSizeFloat);

		//----------------------------------------------------------------------
		// EVAL_GET_EMITTED_RADIANCE
		//----------------------------------------------------------------------
		
		mat->evalGetEmittedRadianceOpStartIndex = matEvalOps.size();
		const u_int evalGetEmittedRadianceOpsStackSizeFloat = CompileMaterialOps(i,
				slg::ocl::MaterialEvalOpType::EVAL_GET_EMITTED_RADIANCE, 0, matEvalOps);
		mat->evalGetEmittedRadianceOpLength = matEvalOps.size() - mat->evalGetEmittedRadianceOpStartIndex;

		maxMaterialEvalStackSize = Max(maxMaterialEvalStackSize, evalGetEmittedRadianceOpsStackSizeFloat);

		//----------------------------------------------------------------------
		// EVAL_GET_PASS_TROUGH_TRANSPARENCY
		//----------------------------------------------------------------------
		
		mat->evalGetPassThroughTransparencyOpStartIndex = matEvalOps.size();
		const u_int evalGetPassThroughTransparencyOpsStackSizeFloat = CompileMaterialOps(i,
				slg::ocl::MaterialEvalOpType::EVAL_GET_PASS_TROUGH_TRANSPARENCY, 0, matEvalOps);
		mat->evalGetPassThroughTransparencyOpLength = matEvalOps.size() - mat->evalGetPassThroughTransparencyOpStartIndex;

		maxMaterialEvalStackSize = Max(maxMaterialEvalStackSize, evalGetPassThroughTransparencyOpsStackSizeFloat);

		//----------------------------------------------------------------------
		// EVAL_EVALUATE
		//----------------------------------------------------------------------
		
		mat->evalEvaluateOpStartIndex = matEvalOps.size();
		const u_int evalEvaluateOpsStackSizeFloat = CompileMaterialOps(i,
				slg::ocl::MaterialEvalOpType::EVAL_EVALUATE, 0, matEvalOps);
		mat->evalEvaluateOpLength = matEvalOps.size() - mat->evalEvaluateOpStartIndex;

		maxMaterialEvalStackSize = Max(maxMaterialEvalStackSize, evalEvaluateOpsStackSizeFloat);

		//----------------------------------------------------------------------
		// EVAL_SAMPLE
		//----------------------------------------------------------------------
		
		mat->evalSampleOpStartIndex = matEvalOps.size();
		const u_int evalSampleOpsStackSizeFloat = CompileMaterialOps(i,
				slg::ocl::MaterialEvalOpType::EVAL_SAMPLE, 0, matEvalOps);
		mat->evalSampleOpLength = matEvalOps.size() - mat->evalSampleOpStartIndex;

		maxMaterialEvalStackSize = Max(maxMaterialEvalStackSize, evalSampleOpsStackSizeFloat);
	}

	materialBumpCacheSize = 3 * Min<u_int>(maxBumpSlotCount, MATERIAL_BUMP_CACHE_MAX_SLOTS);

	SLG_LOG("Material evaluation ops count: " << matEvalOps.size());
	SLG_LOG("Material evaluation max. stack size: " << maxMaterialEvalStackSize);
	SLG_LOG("Material bump cache slots: " << maxBumpSlotCount);
}

void CompiledScene::CompileMaterials() {
	wasMaterialsCompiled = true;

	CompileTextures();

	const u_int materialsCount = scene.GetMaterials().GetSize();
	SLG_LOG("Compile " << materialsCount << " Materials");

	//--------------------------------------------------------------------------
	// Translate material definitions
	//--------------------------------------------------------------------------

	const double tStart = WallClockTime();

	mats.resize(materialsCount);

	for (u_int i = 0; i < materialsCount; ++i) {
		auto& m = scene.GetMaterials().GetMaterial(i);
		slg::ocl::Material *mat = &mats[i];
		//SLG_LOG(" Type: " << m.GetType());

		mat->matID = m.GetID();
		mat->lightID = m.GetLightID();
        mat->bumpSampleDistance = m.GetBumpSampleDistance();

		// Material transparency
		auto frontTranspTex = m.GetFrontTransparencyTexture();
		if (frontTranspTex) {
			mat->frontTranspTexIndex = scene.GetTextures().GetTextureIndex(frontTranspTex);
		} else
			mat->frontTranspTexIndex = NULL_INDEX;

		auto backTranspTex = m.GetBackTransparencyTexture();
		if (backTranspTex) {
			mat->backTranspTexIndex = scene.GetTextures().GetTextureIndex(backTranspTex);
		} else
			mat->backTranspTexIndex = NULL_INDEX;

		ASSIGN_SPECTRUM(mat->passThroughShadowTransparency, m.GetPassThroughShadowTransparency());

		// Material emission
		auto emitTex = m.GetEmitTexture();
		if (emitTex)
			mat->emitTexIndex = scene.GetTextures().GetTextureIndex(emitTex);
		else
			mat->emitTexIndex = NULL_INDEX;
		ASSIGN_SPECTRUM(mat->emittedFactor, m.GetEmittedFactor());
		mat->emittedCosThetaMax = m.GetEmittedCosThetaMax();
		mat->usePrimitiveArea = m.IsUsingPrimitiveArea();

		// Material bump mapping
		auto bumpTex = m.GetBumpTexture();
		if (bumpTex)
			mat->bumpTexIndex = scene.GetTextures().GetTextureIndex(bumpTex);
		else
			mat->bumpTexIndex = NULL_INDEX;

		mat->visibility =
				(m.IsVisibleIndirectDiffuse() ? DIFFUSE : NONE) |
				(m.IsVisibleIndirectGlossy() ? GLOSSY : NONE) |
				(m.IsVisibleIndirectSpecular() ? SPECULAR : NONE);

		// Material volumes
		auto interiorVol = m.GetInteriorVolume();
		mat->interiorVolumeIndex = interiorVol ? scene.GetMaterials().GetMaterialIndex(interiorVol) : NULL_INDEX;
		auto exteriorVol = m.GetExteriorVolume();
		mat->exteriorVolumeIndex = exteriorVol ? scene.GetMaterials().GetMaterialIndex(exteriorVol) : NULL_INDEX;
		mat->glossiness = m.GetGlossiness();
		mat->avgPassThroughTransparency = m.GetAvgPassThroughTransparency();
		mat->isShadowCatcher = m.IsShadowCatcher();
		mat->isShadowCatcherOnlyInfiniteLights = m.IsShadowCatcherOnlyInfiniteLights();
		mat->isPhotonGIEnabled = m.IsPhotonGIEnabled();
		mat->isHoldout = m.IsHoldout();

		// Bake Material::GetEventTypes() and Material::IsDelta()
		mat->eventTypes = m.GetEventTypes();
		mat->isDelta = m.IsDelta();
		mat->regularizedEventTypes = m.GetRegularizedEventTypes();
		mat->isRegularizedDelta = m.IsRegularizedDelta();

		// Bake Material::GetRefractiveMaterial()
		const Material *refractive = m.GetRefractiveMaterial();
		mat->isRefractive = refractive ? 1 : 0;
		mat->refractionExteriorIorTexIndex = NULL_INDEX;
		mat->refractionInteriorIorTexIndex = NULL_INDEX;
		if (refractive) {
			TextureConstPtr extIor, intIor;
			if (refractive->GetType() == GLASS) {
				auto &gm = dynamic_cast<const GlassMaterial &>(*refractive);
				extIor = gm.GetExteriorIOR();
				intIor = gm.GetInteriorIOR();
			} else if (refractive->GetType() == ROUGHGLASS) {
				auto &gm = dynamic_cast<const RoughGlassMaterial &>(*refractive);
				extIor = gm.GetExteriorIOR();
				intIor = gm.GetInteriorIOR();
			} else if (refractive->GetType() == OPENPBR) {
				// Not set: the default of the specification (see
				// BSDF_GetRefractionEtaScale())
				auto &opm = dynamic_cast<const OpenPBRMaterial &>(*refractive);
				intIor = opm.GetInput(slg::ocl::OPENPBR_SPECULAR_IOR);
			}
			if (extIor)
				mat->refractionExteriorIorTexIndex = scene.GetTextures().GetTextureIndex(extIor);
			if (intIor)
				mat->refractionInteriorIorTexIndex = scene.GetTextures().GetTextureIndex(intIor);
		}

		// Bake Material::GetSubsurfaceWalkMaterial()
		const Material *walkMaterial = m.GetSubsurfaceWalkMaterial();
		mat->subsurfaceWalkMatIndex = walkMaterial ?
			scene.GetMaterials().GetMaterialIndex(*walkMaterial) : NULL_INDEX;
		mat->hasDiffuseBesideSubsurfaceWalk = m.HasDiffuseBesideSubsurfaceWalk() ? 1 : 0;

		// Material specific parameters
		switch (m.GetType()) {
			case MATTE: {
				auto& mm = dynamic_cast<const MatteMaterial &>(m);

				mat->type = slg::ocl::MATTE;
				mat->matte.kdTexIndex = scene.GetTextures().GetTextureIndex(mm.GetKd());
				break;
			}
			case ROUGHMATTE: {
				auto& mm = dynamic_cast<const RoughMatteMaterial &>(m);

				mat->type = slg::ocl::ROUGHMATTE;
				mat->roughmatte.kdTexIndex = scene.GetTextures().GetTextureIndex(mm.GetKd());
				mat->roughmatte.sigmaTexIndex = scene.GetTextures().GetTextureIndex(mm.GetSigma());
				break;
			}
			case MIRROR: {
				auto& mm = dynamic_cast<const MirrorMaterial &>(m);

				mat->type = slg::ocl::MIRROR;
				mat->mirror.krTexIndex = scene.GetTextures().GetTextureIndex(mm.GetKr());
				break;
			}
			case GLASS: {
				auto& gm = dynamic_cast<const GlassMaterial &>(m);

				mat->type = slg::ocl::GLASS;
				mat->glass.krTexIndex = scene.GetTextures().GetTextureIndex(gm.GetKr());
				mat->glass.ktTexIndex = scene.GetTextures().GetTextureIndex(gm.GetKt());
				if (gm.GetExteriorIOR())
					mat->glass.exteriorIorTexIndex = scene.GetTextures().GetTextureIndex(gm.GetExteriorIOR());
				else
					mat->glass.exteriorIorTexIndex = NULL_INDEX;
				if (gm.GetInteriorIOR())
					mat->glass.interiorIorTexIndex = scene.GetTextures().GetTextureIndex(gm.GetInteriorIOR());
				else
					mat->glass.interiorIorTexIndex = NULL_INDEX;
				if (gm.GetCauchyB())
					mat->glass.cauchyBTex = scene.GetTextures().GetTextureIndex(gm.GetCauchyB());
				else
					mat->glass.cauchyBTex = NULL_INDEX;
				if (gm.GetFilmThickness())
					mat->glass.filmThicknessTexIndex = scene.GetTextures().GetTextureIndex(gm.GetFilmThickness());
				else
					mat->glass.filmThicknessTexIndex = NULL_INDEX;
				if (gm.GetFilmIOR())
					mat->glass.filmIorTexIndex = scene.GetTextures().GetTextureIndex(gm.GetFilmIOR());
				else
					mat->glass.filmIorTexIndex = NULL_INDEX;
				break;
			}
			case ARCHGLASS: {
				auto& am = dynamic_cast<const ArchGlassMaterial &>(m);

				mat->type = slg::ocl::ARCHGLASS;
				mat->archglass.krTexIndex = scene.GetTextures().GetTextureIndex(am.GetKr());
				mat->archglass.ktTexIndex = scene.GetTextures().GetTextureIndex(am.GetKt());
				if (am.GetExteriorIOR())
					mat->archglass.exteriorIorTexIndex = scene.GetTextures().GetTextureIndex(am.GetExteriorIOR());
				else
					mat->archglass.exteriorIorTexIndex = NULL_INDEX;
				if (am.GetInteriorIOR())
					mat->archglass.interiorIorTexIndex = scene.GetTextures().GetTextureIndex(am.GetInteriorIOR());
				else
					mat->archglass.interiorIorTexIndex = NULL_INDEX;
				if (am.GetFilmThickness())
					mat->archglass.filmThicknessTexIndex = scene.GetTextures().GetTextureIndex(am.GetFilmThickness());
				else
					mat->archglass.filmThicknessTexIndex = NULL_INDEX;
				if (am.GetFilmIOR())
					mat->archglass.filmIorTexIndex = scene.GetTextures().GetTextureIndex(am.GetFilmIOR());
				else
					mat->archglass.filmIorTexIndex = NULL_INDEX;
				break;
			}
			case MIX: {
				auto& mm = dynamic_cast<const MixMaterial &>(m);

				mat->type = slg::ocl::MIX;
				mat->mix.matAIndex = scene.GetMaterials().GetMaterialIndex(mm.GetMaterialA());
				mat->mix.matBIndex = scene.GetMaterials().GetMaterialIndex(mm.GetMaterialB());
				mat->mix.mixFactorTexIndex = scene.GetTextures().GetTextureIndex(mm.GetMixFactor());
				// Set by CompileMixVertexFactors()
				mat->mix.vertexFactorTexIndex = mat->mix.mixFactorTexIndex;
				break;
			}
			case NULLMAT: {
				mat->type = slg::ocl::NULLMAT;
				break;
			}
			case MATTETRANSLUCENT: {
				auto& mt = dynamic_cast<const MatteTranslucentMaterial &>(m);

				mat->type = slg::ocl::MATTETRANSLUCENT;
				mat->matteTranslucent.krTexIndex = scene.GetTextures().GetTextureIndex(mt.GetKr());
				mat->matteTranslucent.ktTexIndex = scene.GetTextures().GetTextureIndex(mt.GetKt());
				break;
			}
			case ROUGHMATTETRANSLUCENT: {
				auto& rmt = dynamic_cast<const RoughMatteTranslucentMaterial &>(m);

				mat->type = slg::ocl::ROUGHMATTETRANSLUCENT;
				mat->roughmatteTranslucent.krTexIndex = scene.GetTextures().GetTextureIndex(rmt.GetKr());
				mat->roughmatteTranslucent.ktTexIndex = scene.GetTextures().GetTextureIndex(rmt.GetKt());
				mat->roughmatteTranslucent.sigmaTexIndex = scene.GetTextures().GetTextureIndex(rmt.GetSigma());
				break;
			}
			case GLOSSY2: {
				auto& g2m = dynamic_cast<const Glossy2Material &>(m);

				mat->type = slg::ocl::GLOSSY2;
				mat->glossy2.kdTexIndex = scene.GetTextures().GetTextureIndex(g2m.GetKd());
				mat->glossy2.ksTexIndex = scene.GetTextures().GetTextureIndex(g2m.GetKs());

				auto nuTex = g2m.GetNu();
				auto nvTex = g2m.GetNv();
				mat->glossy2.nuTexIndex = scene.GetTextures().GetTextureIndex(nuTex);
				mat->glossy2.nvTexIndex = scene.GetTextures().GetTextureIndex(nvTex);

				auto depthTex = g2m.GetDepth();
				mat->glossy2.kaTexIndex = scene.GetTextures().GetTextureIndex(g2m.GetKa());
				mat->glossy2.depthTexIndex = scene.GetTextures().GetTextureIndex(depthTex);

				auto indexTex = g2m.GetIndex();
				mat->glossy2.indexTexIndex = scene.GetTextures().GetTextureIndex(indexTex);
				mat->glossy2.multibounce = g2m.IsMultibounce () ? 1 : 0;
				mat->glossy2.doublesided = g2m.IsDoubleSided () ? 1 : 0;
				break;
			}
			case METAL2: {
				auto& m2m = dynamic_cast<const Metal2Material &>(m);

				mat->type = slg::ocl::METAL2;
				if (m2m.GetFresnel())
					mat->metal2.fresnelTexIndex = scene.GetTextures().GetTextureIndex(m2m.GetFresnel());
				else
					mat->metal2.fresnelTexIndex = NULL_INDEX;
				if (m2m.GetN())
					mat->metal2.nTexIndex = scene.GetTextures().GetTextureIndex(m2m.GetN());
				else
					mat->metal2.nTexIndex = NULL_INDEX;
				if (m2m.GetK())
					mat->metal2.kTexIndex = scene.GetTextures().GetTextureIndex(m2m.GetK());
				else
					mat->metal2.kTexIndex = NULL_INDEX;

				auto nuTex = m2m.GetNu();
				auto nvTex = m2m.GetNv();
				mat->metal2.nuTexIndex = scene.GetTextures().GetTextureIndex(nuTex);
				mat->metal2.nvTexIndex = scene.GetTextures().GetTextureIndex(nvTex);
				break;
			}
			case ROUGHGLASS: {
				auto& rgm = dynamic_cast<const RoughGlassMaterial &>(m);

				mat->type = slg::ocl::ROUGHGLASS;
				mat->roughglass.krTexIndex = scene.GetTextures().GetTextureIndex(rgm.GetKr());
				mat->roughglass.ktTexIndex = scene.GetTextures().GetTextureIndex(rgm.GetKt());
				if (rgm.GetExteriorIOR())
					mat->roughglass.exteriorIorTexIndex = scene.GetTextures().GetTextureIndex(rgm.GetExteriorIOR());
				else
					mat->roughglass.exteriorIorTexIndex = NULL_INDEX;
				if (rgm.GetInteriorIOR())
					mat->roughglass.interiorIorTexIndex = scene.GetTextures().GetTextureIndex(rgm.GetInteriorIOR());
				else
					mat->roughglass.interiorIorTexIndex = NULL_INDEX;

				auto nuTex = rgm.GetNu();
				auto nvTex = rgm.GetNv();
				mat->roughglass.nuTexIndex = scene.GetTextures().GetTextureIndex(nuTex);
				mat->roughglass.nvTexIndex = scene.GetTextures().GetTextureIndex(nvTex);
				
				if (rgm.GetFilmThickness())
					mat->roughglass.filmThicknessTexIndex = scene.GetTextures().GetTextureIndex(rgm.GetFilmThickness());
				else
					mat->roughglass.filmThicknessTexIndex = NULL_INDEX;
				if (rgm.GetFilmIOR())
					mat->roughglass.filmIorTexIndex = scene.GetTextures().GetTextureIndex(rgm.GetFilmIOR());
				else
					mat->roughglass.filmIorTexIndex = NULL_INDEX;
				break;
			}
			case VELVET: {
				auto& vm = dynamic_cast<const VelvetMaterial &>(m);

				mat->type = slg::ocl::VELVET;
				mat->velvet.kdTexIndex = scene.GetTextures().GetTextureIndex(vm.GetKd());
				mat->velvet.roughnessTexIndex = scene.GetTextures().GetTextureIndex(vm.GetRoughness());
				break;
			}
			case CLOTH: {
				auto& cm = dynamic_cast<const ClothMaterial &>(m);

				mat->type = slg::ocl::CLOTH;
				mat->cloth.Preset = cm.GetPreset();
				mat->cloth.Weft_KdIndex = scene.GetTextures().GetTextureIndex(cm.GetWeftKd());
				mat->cloth.Weft_KsIndex = scene.GetTextures().GetTextureIndex(cm.GetWeftKs());
				mat->cloth.Warp_KdIndex = scene.GetTextures().GetTextureIndex(cm.GetWarpKd());
				mat->cloth.Warp_KsIndex = scene.GetTextures().GetTextureIndex(cm.GetWarpKs());
				mat->cloth.Repeat_U = cm.GetRepeatU();
				mat->cloth.Repeat_V = cm.GetRepeatV();
				mat->cloth.specularNormalization = cm.GetSpecularNormalization();
				break;
			}
			case CARPAINT: {
				auto& cm = dynamic_cast<const CarPaintMaterial &>(m);
				mat->type = slg::ocl::CARPAINT;
				mat->carpaint.KdTexIndex = scene.GetTextures().GetTextureIndex(cm.Kd);
				mat->carpaint.Ks1TexIndex = scene.GetTextures().GetTextureIndex(cm.Ks1);
				mat->carpaint.Ks2TexIndex = scene.GetTextures().GetTextureIndex(cm.Ks2);
				mat->carpaint.Ks3TexIndex = scene.GetTextures().GetTextureIndex(cm.Ks3);
				mat->carpaint.M1TexIndex = scene.GetTextures().GetTextureIndex(cm.M1);
				mat->carpaint.M2TexIndex = scene.GetTextures().GetTextureIndex(cm.M2);
				mat->carpaint.M3TexIndex = scene.GetTextures().GetTextureIndex(cm.M3);
				mat->carpaint.R1TexIndex = scene.GetTextures().GetTextureIndex(cm.R1);
				mat->carpaint.R2TexIndex = scene.GetTextures().GetTextureIndex(cm.R2);
				mat->carpaint.R3TexIndex = scene.GetTextures().GetTextureIndex(cm.R3);
				mat->carpaint.KaTexIndex = scene.GetTextures().GetTextureIndex(cm.Ka);
				mat->carpaint.depthTexIndex = scene.GetTextures().GetTextureIndex(cm.depth);
				break;
			}
			case GLOSSYTRANSLUCENT: {
				auto& gtm = dynamic_cast<const GlossyTranslucentMaterial &>(m);

				mat->type = slg::ocl::GLOSSYTRANSLUCENT;
				mat->glossytranslucent.kdTexIndex = scene.GetTextures().GetTextureIndex(gtm.GetKd());
				mat->glossytranslucent.ktTexIndex = scene.GetTextures().GetTextureIndex(gtm.GetKt());
				mat->glossytranslucent.ksTexIndex = scene.GetTextures().GetTextureIndex(gtm.GetKs());
				mat->glossytranslucent.ksbfTexIndex = scene.GetTextures().GetTextureIndex(gtm.GetKs_bf());

				auto nuTex = gtm.GetNu();
				auto nvTex = gtm.GetNv();
				mat->glossytranslucent.nuTexIndex = scene.GetTextures().GetTextureIndex(nuTex);
				mat->glossytranslucent.nvTexIndex = scene.GetTextures().GetTextureIndex(nvTex);
				auto nubfTex = gtm.GetNu_bf();
				auto nvbfTex = gtm.GetNv_bf();
				mat->glossytranslucent.nubfTexIndex = scene.GetTextures().GetTextureIndex(nubfTex);
				mat->glossytranslucent.nvbfTexIndex = scene.GetTextures().GetTextureIndex(nvbfTex);

				mat->glossytranslucent.kaTexIndex = scene.GetTextures().GetTextureIndex(gtm.GetKa());
				mat->glossytranslucent.depthTexIndex = scene.GetTextures().GetTextureIndex(gtm.GetDepth());
				mat->glossytranslucent.kabfTexIndex = scene.GetTextures().GetTextureIndex(gtm.GetKa_bf());
				mat->glossytranslucent.depthbfTexIndex = scene.GetTextures().GetTextureIndex(gtm.GetDepth_bf());

				mat->glossytranslucent.indexTexIndex = scene.GetTextures().GetTextureIndex(gtm.GetIndex());
				mat->glossytranslucent.indexbfTexIndex = scene.GetTextures().GetTextureIndex(gtm.GetIndex_bf());

				mat->glossytranslucent.multibounce = gtm.IsMultibounce() ? 1 : 0;
				mat->glossytranslucent.multibouncebf = gtm.IsMultibounce_bf() ? 1 : 0;
				break;
			}
			case GLOSSYCOATING: {
				auto& gcm = dynamic_cast<const GlossyCoatingMaterial &>(m);

				mat->type = slg::ocl::GLOSSYCOATING;
				mat->glossycoating.matBaseIndex = scene.GetMaterials().GetMaterialIndex(gcm.GetMaterialBase());
				mat->glossycoating.ksTexIndex = scene.GetTextures().GetTextureIndex(gcm.GetKs());

				auto nuTex = gcm.GetNu();
				auto nvTex = gcm.GetNv();
				mat->glossycoating.nuTexIndex = scene.GetTextures().GetTextureIndex(nuTex);
				mat->glossycoating.nvTexIndex = scene.GetTextures().GetTextureIndex(nvTex);

				auto depthTex = gcm.GetDepth();
				mat->glossycoating.kaTexIndex = scene.GetTextures().GetTextureIndex(gcm.GetKa());
				mat->glossycoating.depthTexIndex = scene.GetTextures().GetTextureIndex(depthTex);

				auto indexTex = gcm.GetIndex();
				mat->glossycoating.indexTexIndex = scene.GetTextures().GetTextureIndex(indexTex);
				mat->glossycoating.multibounce = gcm.IsMultibounce() ? 1 : 0;
				mat->glossycoating.darkeningTexIndex = scene.GetTextures().GetTextureIndex(gcm.GetDarkening());
				mat->glossycoating.darkening = gcm.HasDarkening() ? 1 : 0;
				break;
			}
			case DISNEY: {
				auto& dm = dynamic_cast<const DisneyMaterial &>(m);

				mat->type = slg::ocl::DISNEY;
				mat->disney.baseColorTexIndex = scene.GetTextures().GetTextureIndex(dm.GetBaseColor());
				mat->disney.subsurfaceTexIndex = scene.GetTextures().GetTextureIndex(dm.GetSubsurface());
				mat->disney.roughnessTexIndex = scene.GetTextures().GetTextureIndex(dm.GetRoughness());
				mat->disney.metallicTexIndex = scene.GetTextures().GetTextureIndex(dm.GetMetallic());
				mat->disney.specularTexIndex = scene.GetTextures().GetTextureIndex(dm.GetSpecular());
				mat->disney.specularTintTexIndex = scene.GetTextures().GetTextureIndex(dm.GetSpecularTint());
				mat->disney.clearcoatTexIndex = scene.GetTextures().GetTextureIndex(dm.GetClearcoat());
				mat->disney.clearcoatGlossTexIndex = scene.GetTextures().GetTextureIndex(dm.GetClearcoatGloss());
				mat->disney.anisotropicTexIndex = scene.GetTextures().GetTextureIndex(dm.GetAnisotropic());
				mat->disney.sheenTexIndex = scene.GetTextures().GetTextureIndex(dm.GetSheen());
				mat->disney.sheenTintTexIndex = scene.GetTextures().GetTextureIndex(dm.GetSheenTint());
				if (dm.GetFilmAmount())
					mat->disney.filmAmountTexIndex = scene.GetTextures().GetTextureIndex(dm.GetFilmAmount());
				else
					mat->disney.filmAmountTexIndex = NULL_INDEX;
				if (dm.GetFilmThickness())
					mat->disney.filmThicknessTexIndex = scene.GetTextures().GetTextureIndex(dm.GetFilmThickness());
				else
					mat->disney.filmThicknessTexIndex = NULL_INDEX;
				if (dm.GetFilmIOR())
					mat->disney.filmIorTexIndex = scene.GetTextures().GetTextureIndex(dm.GetFilmIOR());
				else
					mat->disney.filmIorTexIndex = NULL_INDEX;
				break;
			}
			case OPENPBR: {
				auto& opm = dynamic_cast<const OpenPBRMaterial &>(m);

				mat->type = slg::ocl::OPENPBR;
				for (u_int i = 0; i < slg::ocl::OPENPBR_INPUT_COUNT; ++i) {
					auto tex = opm.GetInput(static_cast<slg::ocl::OpenPBRInput>(i));
					mat->openpbr.texIndex[i] = tex ? scene.GetTextures().GetTextureIndex(tex) : NULL_INDEX;
				}
				mat->openpbr.thinWalled = opm.IsThinWalled() ? 1 : 0;
				mat->openpbr.subsurfaceWalk = opm.GetSubsurfaceWalkMaterial() ? 1 : 0;
				break;
			}
			case SUBSURFACEMAT: {
				auto& ssm = dynamic_cast<const SubsurfaceMaterial &>(m);

				mat->type = slg::ocl::SUBSURFACEMAT;
				mat->subsurface.colorTexIndex = scene.GetTextures().GetTextureIndex(ssm.GetColor());
				mat->subsurface.radiusTexIndex = scene.GetTextures().GetTextureIndex(ssm.GetRadius());
				mat->subsurface.scaleTexIndex = scene.GetTextures().GetTextureIndex(ssm.GetScale());
				mat->subsurface.anisotropyTexIndex = scene.GetTextures().GetTextureIndex(ssm.GetAnisotropy());
				mat->subsurface.iorTexIndex = scene.GetTextures().GetTextureIndex(ssm.GetIOR());
				mat->subsurface.roughnessTexIndex = scene.GetTextures().GetTextureIndex(ssm.GetRoughness());
				break;
			}
			case TWOSIDED: {
				auto& tsm = dynamic_cast<const TwoSidedMaterial &>(m);

				mat->type = slg::ocl::TWOSIDED;
				mat->twosided.frontMatIndex = scene.GetMaterials().GetMaterialIndex(tsm.GetFrontMaterial());
				mat->twosided.backMatIndex = scene.GetMaterials().GetMaterialIndex(tsm.GetBackMaterial());
				break;
			}
			//------------------------------------------------------------------
			// Volumes
			//------------------------------------------------------------------
			case CLEAR_VOL:
			case HOMOGENEOUS_VOL:
			case HETEROGENEOUS_VOL: {
				auto& v = dynamic_cast<const Volume &>(m);
				mat->volume.iorTexIndex = v.HasIORTexture() ?
					scene.GetTextures().GetTextureIndex(v.GetIORTexture()) :
					NULL_INDEX;
				mat->volume.volumeEmissionTexIndex = v.HasVolumeEmissionTexture() ?
					scene.GetTextures().GetTextureIndex(v.GetVolumeEmissionTexture()) :
					NULL_INDEX;
				mat->volume.volumeLightID = v.GetVolumeLightID();
				mat->volume.priority = v.GetPriority();

				switch (m.GetType()) {
					case CLEAR_VOL: {
						auto& cv = dynamic_cast<const ClearVolume &>(m);
						mat->type = slg::ocl::CLEAR_VOL;
						mat->volume.clear.sigmaATexIndex = scene.GetTextures().GetTextureIndex(cv.GetSigmaA());
						break;
					}
					case HOMOGENEOUS_VOL: {
						auto& hv = dynamic_cast<const HomogeneousVolume &>(m);
						mat->type = slg::ocl::HOMOGENEOUS_VOL;
						mat->volume.homogenous.sigmaATexIndex = scene.GetTextures().GetTextureIndex(hv.GetSigmaA());
						mat->volume.homogenous.sigmaSTexIndex = scene.GetTextures().GetTextureIndex(hv.GetSigmaS());
						mat->volume.homogenous.gTexIndex = scene.GetTextures().GetTextureIndex(hv.GetG());
						mat->volume.homogenous.multiScattering = hv.IsMultiScattering();
						break;
					}
					case HETEROGENEOUS_VOL: {
						auto& hv = dynamic_cast<const HeterogeneousVolume &>(m);
						mat->type = slg::ocl::HETEROGENEOUS_VOL;
						mat->volume.heterogenous.sigmaATexIndex = scene.GetTextures().GetTextureIndex(hv.GetSigmaA());
						mat->volume.heterogenous.sigmaSTexIndex = scene.GetTextures().GetTextureIndex(hv.GetSigmaS());
						mat->volume.heterogenous.gTexIndex = scene.GetTextures().GetTextureIndex(hv.GetG());
						mat->volume.heterogenous.stepSize = hv.GetStepSize();
						mat->volume.heterogenous.maxStepsCount = hv.GetMaxStepsCount();
						mat->volume.heterogenous.multiScattering = hv.IsMultiScattering();
						break;
					}
					default:
						throw runtime_error("Unknown volume in CompiledScene::CompileMaterials(): " + ToString(m.GetType()));
				}
				break;
			}
			default:
				throw runtime_error("Unknown material in CompiledScene::CompileMaterials(): " + ToString(m.GetType()));
		}
	}

	//--------------------------------------------------------------------------
	// Material evaluation ops
	//--------------------------------------------------------------------------

	// The weights of the Mix materials at a vertex: textures, compiled with
	// the others
	CompileMixVertexFactors();

	CompileMaterialOps();

	//--------------------------------------------------------------------------
	// Default scene volume
	//--------------------------------------------------------------------------

	defaultWorldVolumeIndex = scene.HasDefaultWorldVolume() ?
		scene.GetMaterials().GetMaterialIndex(scene.GetDefaultWorldVolume()) : NULL_INDEX;

	//--------------------------------------------------------------------------
	// Shadow rays need the closest hit and its BSDF only if something can let
	// them through (see Scene_Intersect()): a pass-through transparency (null
	// and archglass materials, transparency textures), a shadow transparency
	// or a volume (the volume priority system and the transmittance)
	//--------------------------------------------------------------------------

	shadowRaysNeedBSDF = (defaultWorldVolumeIndex != NULL_INDEX);
	for (const slg::ocl::Material &mat : mats) {
		if ((mat.type == slg::ocl::NULLMAT) || (mat.type == slg::ocl::ARCHGLASS) ||
				(mat.frontTranspTexIndex != NULL_INDEX) || (mat.backTranspTexIndex != NULL_INDEX) ||
				(mat.passThroughShadowTransparency.c[0] != 0.f) ||
				(mat.passThroughShadowTransparency.c[1] != 0.f) ||
				(mat.passThroughShadowTransparency.c[2] != 0.f) ||
				(mat.interiorVolumeIndex != NULL_INDEX) || (mat.exteriorVolumeIndex != NULL_INDEX)) {
			shadowRaysNeedBSDF = true;
			break;
		}
	}
	SLG_LOG("Shadow rays any-hit: " << (shadowRaysNeedBSDF ? "no (transparency or volumes)" : "yes"));

	hasSubsurface = false;
	for (const slg::ocl::Material &mat : mats) {
		if (mat.subsurfaceWalkMatIndex != NULL_INDEX) {
			hasSubsurface = true;
			break;
		}
	}

	const double tEnd = WallClockTime();
	SLG_LOG("Material compilation time: " << int((tEnd - tStart) * 1000.0) << "ms");
}

#endif
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
