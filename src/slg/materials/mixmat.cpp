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

#include "slg/materials/mix.h"
#include "slg/materials/glossycoating.h"
#include "slg/textures/constfloat.h"

using namespace std;
using namespace luxrays;
using namespace slg;

//------------------------------------------------------------------------------
// Mix material
//------------------------------------------------------------------------------

MixMaterial::MixMaterial(
	TextureConstPtr frontTransp,
	TextureConstPtr backTransp,
	TextureConstPtr emitted,
	TextureConstPtr bump,
	MaterialConstRef mA,
	MaterialConstRef mB,
	TextureConstPtr mix
) :
	Material(frontTransp, backTransp, emitted, bump),
	matA(&mA),
	matB(&mB),
	mixFactor(mix)
{
	Preprocess();
}

BSDFEvent MixMaterial::GetEventTypesImpl() const {
	return (matA->GetEventTypes() | matB->GetEventTypes());
}

bool MixMaterial::IsLightSourceImpl() const {
	return (Material::IsLightSource() || matA->IsLightSource() || matB->IsLightSource());
}

bool MixMaterial::IsDeltaImpl() const {
	return (matA->IsDelta() && matB->IsDelta());
}

void MixMaterial::Preprocess() {
	// Cache values for performance with very large material node trees

	eventTypes = GetEventTypesImpl();

	if (matA->GetEventTypes() & GLOSSY) {
		if (matB->GetEventTypes() & GLOSSY)
			glossiness = Min(matA->GetGlossiness(), matB->GetGlossiness());
		else
			glossiness = matA->GetGlossiness();
	} else {
		if (matB->GetEventTypes() & GLOSSY)
			glossiness = matB->GetGlossiness();
		else
			glossiness = 0.f;
	}

	isLightSource = IsLightSourceImpl();
	isDelta = IsDeltaImpl();

	UpdateVertexWeights();
}

// Material::GetVertexProbability() when it is the same at every hit point,
// -1 otherwise (as CompiledScene::CompileVertexProbabilityTexture())
static float ConstVertexProbability(const Material &material) {
	if (!material.HasPassThroughTransparency())
		return 1.f;

	auto constValue = [](const Texture *tex) {
		return (tex->GetType() == CONST_FLOAT) ?
			Clamp(static_cast<const ConstFloatTexture *>(tex)->GetValue(), 0.f, 1.f) : -1.f;
	};

	auto frontTex = material.GetFrontTransparencyTexture();
	auto backTex = material.GetBackTransparencyTexture();
	switch (material.GetType()) {
		case NULLMAT:
			return 0.f;
		case MIX:
			if (!frontTex && !backTex) {
				auto &mm = static_cast<const MixMaterial &>(material);
				const float probA = ConstVertexProbability(mm.GetMaterialA());
				const float probB = ConstVertexProbability(mm.GetMaterialB());
				const float weight2 = constValue(&mm.GetMixFactor());
				return ((probA >= 0.f) && (probB >= 0.f) && (weight2 >= 0.f)) ?
					Lerp(weight2, probA, probB) : -1.f;
			}
			break;
		case GLOSSYCOATING:
			return ConstVertexProbability(*static_cast<const GlossyCoatingMaterial &>(material).GetMaterialBase());
		default:
			break;
	}

	if (frontTex != backTex)
		return -1.f;
	return frontTex ? constValue(frontTex.get()) : 1.f;
}

void MixMaterial::UpdateVertexWeights() {
	vertexWeights = passThroughDecidesVertex &&
			!frontTransparencyTex && !backTransparencyTex &&
			(matA->HasPassThroughTransparency() || matB->HasPassThroughTransparency()) &&
			matA->HasPointVertexProbability() && matB->HasPointVertexProbability();

	// The constant vertex probabilities spare GetVertexWeight2() the calls; two
	// equal ones leave the weights as they are (as on the GPU,
	// CompiledScene::CompileMixVertexFactors())
	constVertexProbA = vertexWeights ? ConstVertexProbability(*matA) : -1.f;
	constVertexProbB = vertexWeights ? ConstVertexProbability(*matB) : -1.f;
	if ((constVertexProbA >= 0.f) && (constVertexProbA == constVertexProbB))
		vertexWeights = false;
}

void MixMaterial::SetPassThroughDecidesVertex(const bool decides) {
	passThroughDecidesVertex = decides;
	UpdateVertexWeights();
}

// A Mix without a transparency texture of its own lets a ray through with the
// sub-material its pass-through event picks (GetPassThroughTransparency()):
// sub-material i, of weight w_i and vertex probability o_i, makes the vertex
// with probability w_i o_i, and the Mix with o = w_A o_A + w_B o_B. The
// surface's BSDF without the pass-through part is w_A o_A f_A + w_B o_B f_B
// (f_i the BSDF of sub-material i at its vertex, its Evaluate()), so at a
// vertex, which the path reaches with probability o, the BSDF is that divided
// by o: w'_A f_A + w'_B f_B with w'_i = w_i o_i / o. Evaluate(), Sample() and
// Pdf() use these weights, and so do GetEmittedRadiance() and Albedo().
// Sample() picks the sub-material with the plain weights from the vertex
// event, which lies in the interval of the sub-material that made the vertex
// (GetVertexPassThroughEvent()): sub-material i with probability w'_i, as the
// weights need (the volumes are picked the same way). The plain weights would
// count w_i twice: Mix(null, X, w) is a vertex with probability w and its
// light sampling estimated w f_X there. Only when every o_i depends on the hit
// point alone (HasPointVertexProbability(): not an archglass, which lets its
// transmission through, nor a side-dependent transparency), and only where
// the Mix's own pass-through decides its vertices: under a Mix or two-sided
// material with a transparency texture of its own, the vertex is that
// material's and the Mix a plain mix whose null lets the path through as a
// BSDF lobe (SetPassThroughDecidesVertex(),
// MaterialDefinitions::UpdateMixVertexWeights()). The other Mixes keep the
// plain weights (BUGS.md). On the GPU, w'_B is a texture
// (CompiledScene::CompileMixVertexFactors()).
float MixMaterial::GetVertexWeight2(const HitPoint &hitPoint, const float weight2) const {
	if (!vertexWeights)
		return weight2;

	const float probA = (constVertexProbA >= 0.f) ? constVertexProbA : matA->GetVertexProbability(hitPoint);
	const float probB = (constVertexProbB >= 0.f) ? constVertexProbB : matB->GetVertexProbability(hitPoint);
	const float v1 = (1.f - weight2) * probA;
	const float v2 = weight2 * probB;
	const float v = v1 + v2;

	// v = 0: never a vertex
	return (v > 0.f) ? (v2 / v) : weight2;
}

float MixMaterial::GetVertexProbability(const HitPoint &hitPoint) const {
	if (frontTransparencyTex || backTransparencyTex)
		return Material::GetVertexProbability(hitPoint);
	if (!HasPassThroughTransparency())
		return 1.f;

	const float weight2 = Clamp(mixFactor->GetFloatValue(hitPoint), 0.f, 1.f);
	const float weight1 = 1.f - weight2;

	return weight1 * matA->GetVertexProbability(hitPoint) +
			weight2 * matB->GetVertexProbability(hitPoint);
}

VolumeConstPtr MixMaterial::GetInteriorVolume(const HitPoint &hitPoint,
		const float passThroughEvent) const {
	if (interiorVolume)
		return interiorVolume;
	else {
		const float weight2 = Clamp(mixFactor->GetFloatValue(hitPoint), 0.f, 1.f);
		const float weight1 = 1.f - weight2;

		if (passThroughEvent < weight1)
			return matA->GetInteriorVolume(hitPoint, passThroughEvent / weight1);
		else
			return matB->GetInteriorVolume(hitPoint, (passThroughEvent - weight1) / weight2);
	}
}

VolumeConstPtr MixMaterial::GetExteriorVolume(const HitPoint &hitPoint,
		const float passThroughEvent) const {
	if (exteriorVolume)
		return exteriorVolume;
	else {
		const float weight2 = Clamp(mixFactor->GetFloatValue(hitPoint), 0.f, 1.f);
		const float weight1 = 1.f - weight2;

		if (passThroughEvent < weight1)
			return matA->GetExteriorVolume(hitPoint, passThroughEvent / weight1);
		else
			return matB->GetExteriorVolume(hitPoint, (passThroughEvent - weight1) / weight2);
	}
}

void MixMaterial::UpdateAvgPassThroughTransparency() {
	if (frontTransparencyTex || backTransparencyTex)
		Material::UpdateAvgPassThroughTransparency();
	else {
		const float avgMix = mixFactor->Filter();

		const float weight2 = Clamp(avgMix, 0.f, 1.f);
		const float weight1 = 1.f - weight2;

		avgPassThroughTransparency = weight1 * matA->GetAvgPassThroughTransparency() +
				weight2 * matB->GetAvgPassThroughTransparency();
	}
}

Spectrum MixMaterial::GetPassThroughTransparency(const HitPoint &hitPoint,
		const Vector &localFixedDir, const float passThroughEvent,
		const bool backTracing) const {
	if (frontTransparencyTex || backTransparencyTex) {
		return Material::GetPassThroughTransparency(hitPoint, localFixedDir,
				passThroughEvent, backTracing);
	} else {
		const float weight2 = Clamp(mixFactor->GetFloatValue(hitPoint), 0.f, 1.f);
		const float weight1 = 1.f - weight2;

		if (passThroughEvent < weight1) {
			return matA->GetPassThroughTransparency(hitPoint, localFixedDir,
					passThroughEvent / weight1, backTracing);
		} else {
			return matB->GetPassThroughTransparency(hitPoint, localFixedDir,
					(passThroughEvent - weight1) / weight2, backTracing);
		}
	}
}

float MixMaterial::GetVertexPassThroughEvent(const HitPoint &hitPoint,
		const float passThroughEvent, const bool backTracing) const {
	if (frontTransparencyTex || backTransparencyTex)
		return Material::GetVertexPassThroughEvent(hitPoint, passThroughEvent, backTracing);

	const float weight2 = Clamp(mixFactor->GetFloatValue(hitPoint), 0.f, 1.f);
	const float weight1 = 1.f - weight2;

	// u stays in the interval of the sub-material that decided the vertex
	// (GetPassThroughTransparency()), remapped inside it as the sub-material
	// remaps its own
	if (passThroughEvent < weight1) {
		const float u = passThroughEvent / weight1;
		const float vertexU = matA->GetVertexPassThroughEvent(hitPoint, u, backTracing);

		return (vertexU != u) ? (weight1 * vertexU) : passThroughEvent;
	} else {
		const float u = (passThroughEvent - weight1) / weight2;
		const float vertexU = matB->GetVertexPassThroughEvent(hitPoint, u, backTracing);

		return (vertexU != u) ? Min(weight1 + weight2 * vertexU, .99999994f) : passThroughEvent;
	}
}

float MixMaterial::GetEmittedRadianceY(const float oneOverPrimitiveArea) const {
	if (emittedTex)
		return Material::GetEmittedRadianceY(oneOverPrimitiveArea);
	else
		return luxrays::Lerp(mixFactor->Y(), matA->GetEmittedRadianceY(oneOverPrimitiveArea), matB->GetEmittedRadianceY(oneOverPrimitiveArea));
}

Spectrum MixMaterial::GetEmittedRadiance(const HitPoint &hitPoint, const float oneOverPrimitiveArea) const {
	if (emittedTex)
		return Material::GetEmittedRadiance(hitPoint, oneOverPrimitiveArea);
	else {
		Spectrum result;

		const float weight2 = GetVertexWeight2(hitPoint, Clamp(mixFactor->GetFloatValue(hitPoint), 0.f, 1.f));
		const float weight1 = 1.f - weight2;

		if (matA->IsLightSource() && (weight1 > 0.f))
			result += weight1 * matA->GetEmittedRadiance(hitPoint, oneOverPrimitiveArea);
		if (matB->IsLightSource() && (weight2 > 0.f))
			result += weight2 * matB->GetEmittedRadiance(hitPoint, oneOverPrimitiveArea);

		return result;
	}
}

Spectrum MixMaterial::Albedo(const HitPoint &hitPoint) const {
	const float weight2 = GetVertexWeight2(hitPoint, Clamp(mixFactor->GetFloatValue(hitPoint), 0.f, 1.f));
	const float weight1 = 1.f - weight2;

	return weight1 * matA->Albedo(hitPoint) + weight2 * matB->Albedo(hitPoint);
}

Spectrum MixMaterial::Evaluate(const HitPoint &hitPoint,
	const Vector &localLightDir, const Vector &localEyeDir, BSDFEvent *event,
	float *directPdfW, float *reversePdfW) const {
	const Frame frame(hitPoint.GetFrame());
	Spectrum result;

	// This test is usually done by BSDF::Evaluate() and must be repeated in
	// material referencing other materials
	const float isTransmitEval = (Sgn(localLightDir.z) != Sgn(localEyeDir.z));

	const float weight2 = GetVertexWeight2(hitPoint, Clamp(mixFactor->GetFloatValue(hitPoint), 0.f, 1.f));
	const float weight1 = 1.f - weight2;

	if (directPdfW)
		*directPdfW = 0.f;
	if (reversePdfW)
		*reversePdfW = 0.f;

	*event = NONE;
	
	BSDFEvent eventTypesMatA = matA->GetEventTypes();
	if ((weight1 > 0.f) &&
			((!isTransmitEval && (eventTypesMatA & REFLECT)) ||
			(isTransmitEval && (eventTypesMatA & TRANSMIT)))) {
		HitPoint hitPointA(hitPoint);
		matA->Bump(&hitPointA);
		const Frame frameA(hitPointA.GetFrame());
		const Vector lightDirA = frameA.ToLocal(frame.ToWorld(localLightDir));
		const Vector eyeDirA = frameA.ToLocal(frame.ToWorld(localEyeDir));

		BSDFEvent eventMatA;
		float directPdfWMatA, reversePdfWMatA;
		const Spectrum matAResult = matA->Evaluate(hitPointA, lightDirA, eyeDirA, &eventMatA, &directPdfWMatA, &reversePdfWMatA);
		if (!matAResult.Black()) {
			result += weight1 * matAResult;

			if (directPdfW)
				*directPdfW += weight1 * directPdfWMatA;
			if (reversePdfW)
				*reversePdfW += weight1 * reversePdfWMatA;
			
			*event |= eventMatA;
		}
	}

	BSDFEvent eventTypesMatB = matB->GetEventTypes();
	if ((weight2 > 0.f) &&
			((!isTransmitEval && (eventTypesMatB & REFLECT)) ||
			(isTransmitEval && (eventTypesMatB & TRANSMIT)))) {
		HitPoint hitPointB(hitPoint);
		matB->Bump(&hitPointB);
		const Frame frameB(hitPointB.GetFrame());
		const Vector lightDirB = frameB.ToLocal(frame.ToWorld(localLightDir));
		const Vector eyeDirB = frameB.ToLocal(frame.ToWorld(localEyeDir));

		BSDFEvent eventMatB;
		float directPdfWMatB, reversePdfWMatB;
		const Spectrum matBResult = matB->Evaluate(hitPointB, lightDirB, eyeDirB, &eventMatB, &directPdfWMatB, &reversePdfWMatB);
		if (!matBResult.Black()) {
			result += weight2 * matBResult;

			if (directPdfW)
				*directPdfW += weight2 * directPdfWMatB;
			if (reversePdfW)
				*reversePdfW += weight2 * reversePdfWMatB;
			
			*event |= eventMatB;
		}
	}

	return result;
}

Spectrum MixMaterial::Sample(const HitPoint &hitPoint,
	const Vector &localFixedDir, Vector *localSampledDir,
	const float u0, const float u1, const float passThroughEvent,
	float *pdfW, BSDFEvent *event) const {
	const Frame frame(hitPoint.GetFrame());

	HitPoint hitPointA(hitPoint);
	matA->Bump(&hitPointA);
	const Frame frameA(hitPointA.GetFrame());
	const Vector fixedDirA = frameA.ToLocal(frame.ToWorld(localFixedDir));

	HitPoint hitPointB(hitPoint);
	matB->Bump(&hitPointB);
	const Frame frameB(hitPointB.GetFrame());
	const Vector fixedDirB = frameB.ToLocal(frame.ToWorld(localFixedDir));

	const float weight2 = Clamp(mixFactor->GetFloatValue(hitPoint), 0.f, 1.f);
	const float weight1 = 1.f - weight2;

	// The sub-material is picked with the mix factor's weights: at a vertex u
	// lies in the interval of the sub-material that decided it
	// (GetVertexPassThroughEvent()), which is sub-material i with probability
	// w'_i (GetVertexWeight2()), the weight its BSDF has here
	const bool sampleMatA = (passThroughEvent < weight1);

	const float vertexWeight2 = GetVertexWeight2(hitPoint, weight2);
	const float vertexWeight1 = 1.f - vertexWeight2;
	const float weightFirst = sampleMatA ? vertexWeight1 : vertexWeight2;
	const float weightSecond = sampleMatA ? vertexWeight2 : vertexWeight1;
	// Only where the sub-material can't make a vertex (u = 0 on a
	// transparency of 0)
	if (weightFirst <= 0.f)
		return Spectrum();

	const float passThroughEventFirst = sampleMatA ? (passThroughEvent / weight1) : (passThroughEvent - weight1) / weight2;

	// Sample the first material, evaluate the second
	auto matFirst = sampleMatA ? matA : matB;
	auto matSecond = sampleMatA ? matB : matA;

	HitPoint &hitPoint1 = sampleMatA ? hitPointA : hitPointB;
	HitPoint &hitPoint2 = sampleMatA ? hitPointB : hitPointA;
	const Frame &frame1 = sampleMatA ? frameA : frameB;
	const Frame &frame2 = sampleMatA ? frameB : frameA;
	const Vector &fixedDir1 = sampleMatA ? fixedDirA : fixedDirB;
	const Vector &fixedDir2 = sampleMatA ? fixedDirB : fixedDirA;

	// Sample the first material
	Spectrum result = matFirst->Sample(hitPoint1, fixedDir1, localSampledDir,
			u0, u1, passThroughEventFirst, pdfW, event);
	if (result.Black())
		return Spectrum();

	*localSampledDir = frame1.ToWorld(*localSampledDir);
	const Vector sampledDir2 = frame2.ToLocal(*localSampledDir);
	*localSampledDir = frame.ToLocal(*localSampledDir);

	*pdfW *= weightFirst;

	if ((*event) & SPECULAR)
		return result;
	
	result *= *pdfW;

	// Evaluate the second material
	const Vector &localLightDir = (hitPoint2.fromLight) ? fixedDir2 : sampledDir2;
	const Vector &localEyeDir = (hitPoint2.fromLight) ? sampledDir2 : fixedDir2;
	
	// This test is usually done by BSDF::Evaluate() and must be repeated in
	// material referencing other materials
	const float isTransmitEval = (Sgn(localLightDir.z) != Sgn(localEyeDir.z));

	const BSDFEvent eventSecondMat = matSecond->GetEventTypes();
	if ((!isTransmitEval && (eventSecondMat & REFLECT)) ||
			(isTransmitEval && (eventSecondMat & TRANSMIT))) {
		BSDFEvent eventSecond;
		float pdfWSecond;
		Spectrum evalSecond = matSecond->Evaluate(hitPoint2, localLightDir, localEyeDir, &eventSecond, &pdfWSecond);
		if (!evalSecond.Black()) {
			result += weightSecond * evalSecond;
			*pdfW += weightSecond * pdfWSecond;
		}
	}

	return result / *pdfW;
}

void MixMaterial::Pdf(const HitPoint &hitPoint,
		const Vector &localLightDir, const Vector &localEyeDir,
		float *directPdfW, float *reversePdfW) const {
	const Frame frame(hitPoint.GetFrame());
	const float weight2 = GetVertexWeight2(hitPoint, Clamp(mixFactor->GetFloatValue(hitPoint), 0.f, 1.f));
	const float weight1 = 1.f - weight2;

	// As in Evaluate(): a sub-material without an event on the side of the two
	// directions can't sample one of them, so it adds nothing to the pdfs
	const float isTransmitEval = (Sgn(localLightDir.z) != Sgn(localEyeDir.z));

	float directPdfWMatA = 0.f;
	float reversePdfWMatA = 0.f;
	const BSDFEvent eventTypesMatA = matA->GetEventTypes();
	if ((weight1 > 0.f) &&
			((!isTransmitEval && (eventTypesMatA & REFLECT)) ||
			(isTransmitEval && (eventTypesMatA & TRANSMIT)))) {
		HitPoint hitPointA(hitPoint);
		matA->Bump(&hitPointA);
		const Frame frameA(hitPointA.GetFrame());
		const Vector lightDirA = frameA.ToLocal(frame.ToWorld(localLightDir));
		const Vector eyeDirA = frameA.ToLocal(frame.ToWorld(localEyeDir));
		matA->Pdf(hitPointA, lightDirA, eyeDirA, &directPdfWMatA, &reversePdfWMatA);
	}

	float directPdfWMatB = 0.f;
	float reversePdfWMatB = 0.f;
	const BSDFEvent eventTypesMatB = matB->GetEventTypes();
	if ((weight2 > 0.f) &&
			((!isTransmitEval && (eventTypesMatB & REFLECT)) ||
			(isTransmitEval && (eventTypesMatB & TRANSMIT)))) {
		HitPoint hitPointB(hitPoint);
		matB->Bump(&hitPointB);
		const Frame frameB(hitPointB.GetFrame());
		const Vector lightDirB = frameB.ToLocal(frame.ToWorld(localLightDir));
		const Vector eyeDirB = frameB.ToLocal(frame.ToWorld(localEyeDir));
		matB->Pdf(hitPointB, lightDirB, eyeDirB, &directPdfWMatB, &reversePdfWMatB);
	}

	if (directPdfW)
		*directPdfW = weight1 * directPdfWMatA + weight2 *directPdfWMatB;
	if (reversePdfW)
		*reversePdfW = weight1 * reversePdfWMatA + weight2 * reversePdfWMatB;
}

void MixMaterial::UpdateMaterialReferences(MaterialConstRef oldMat, MaterialRef newMat) {
	if (matA == &oldMat)
		matA = &newMat;

	if (matB == &oldMat)
		matB = &newMat;
	
	// Update volumes too
	Material::UpdateMaterialReferences(oldMat, newMat);
	
	Preprocess();
}

bool MixMaterial::IsReferencing(MaterialConstRef mat) const {
	return matA == &mat || matA->IsReferencing(mat) ||
		matB == &mat || matB->IsReferencing(mat);
}

void MixMaterial::AddReferencedMaterials(
	std::unordered_set<const Material *> &referencedMats
) const {
	Material::AddReferencedMaterials(referencedMats);

	referencedMats.insert(matA.get());
	matA->AddReferencedMaterials(referencedMats);

	referencedMats.insert(matB.get());
	matB->AddReferencedMaterials(referencedMats);
}

void MixMaterial::AddReferencedTextures(std::unordered_set<const Texture *>  &referencedTexs) const {
	Material::AddReferencedTextures(referencedTexs);

	matA->AddReferencedTextures(referencedTexs);
	matB->AddReferencedTextures(referencedTexs);
	mixFactor->AddReferencedTextures(referencedTexs);
}

void MixMaterial::UpdateTextureReferences(TextureConstRef oldTex, TextureRef newTex) {
	if (mixFactor == &oldTex)
		mixFactor = &newTex;

	Material::UpdateTextureReferences(oldTex, newTex);

	Preprocess();
}

PropertiesUPtr MixMaterial::ToProperties(const ImageMapCache &imgMapCache, const bool useRealFileName) const  {
	auto props = std::make_unique<Properties>();

	const string name = GetName();
	props->Set(Property("scene.materials." + name + ".type")("mix"));
	props->Set(Property("scene.materials." + name + ".material1")(matA->GetName()));
	props->Set(Property("scene.materials." + name + ".material2")(matB->GetName()));
	props->Set(Property("scene.materials." + name + ".amount")(mixFactor->GetSDLValue()));
	props->Set(Material::ToProperties(imgMapCache, useRealFileName));

	return props;
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
