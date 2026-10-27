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

#ifndef _SLG_MIXMAT_H
#define	_SLG_MIXMAT_H

#include "slg/materials/material.h"
#include "slg/volumes/volume.h"

namespace slg {

//------------------------------------------------------------------------------
// Mix material
//------------------------------------------------------------------------------

class MixMaterial : public Material {
public:
	MixMaterial(TextureConstPtr frontTransp, TextureConstPtr backTransp,
			TextureConstPtr emitted, TextureConstPtr bump,
			MaterialConstRef mA, MaterialConstRef mB, TextureConstPtr mix);

	virtual MaterialType GetType() const { return MIX; }
	virtual BSDFEvent GetEventTypes() const { return eventTypes; };

	virtual bool IsLightSource() const { return isLightSource; }
	virtual bool IsDelta() const { return isDelta; }
	virtual const Material *GetRefractiveMaterial() const {
		const Material *m = matA->GetRefractiveMaterial();
		return m ? m : matB->GetRefractiveMaterial();
	}
	virtual bool IsRegularizedDelta() const { return matA->IsRegularizedDelta() && matB->IsRegularizedDelta(); }
	virtual BSDFEvent GetRegularizedEventTypes() const { return matA->GetRegularizedEventTypes() | matB->GetRegularizedEventTypes(); }

	virtual luxrays::Spectrum GetPassThroughTransparency(const HitPoint &hitPoint,
		const luxrays::Vector &localFixedDir, const float passThroughEvent,
		const bool backTracing) const;
	virtual float GetVertexPassThroughEvent(const HitPoint &hitPoint,
		const float passThroughEvent, const bool backTracing) const;
	virtual float GetVertexProbability(const HitPoint &hitPoint) const;
	virtual bool HasPointVertexProbability() const {
		return (frontTransparencyTex || backTransparencyTex) ? Material::HasPointVertexProbability() :
				(matA->HasPointVertexProbability() && matB->HasPointVertexProbability());
	}
	// True if it weighs its sub-materials at a vertex with their vertex
	// probabilities (GetVertexWeight2())
	bool HasVertexWeights() const { return vertexWeights; }
	// False when the Mix is under a material whose own transparency texture
	// decides the pass-through (MaterialDefinitions::UpdateMixVertexWeights())
	void SetPassThroughDecidesVertex(const bool decides);
	virtual bool HasPassThroughTransparency() const {
		return Material::HasPassThroughTransparency() ||
				matA->HasPassThroughTransparency() || matB->HasPassThroughTransparency();
	}

	virtual VolumeConstPtr GetInteriorVolume(const HitPoint &hitPoint,
		const float passThroughEvent) const;
	virtual VolumeConstPtr GetExteriorVolume(const HitPoint &hitPoint,
		const float passThroughEvent) const;

	virtual float GetEmittedRadianceY(const float oneOverPrimitiveArea) const;
	virtual luxrays::Spectrum GetEmittedRadiance(const HitPoint &hitPoint,
		const float oneOverPrimitiveArea) const;

	virtual luxrays::Spectrum Albedo(const HitPoint &hitPoint) const;

	virtual luxrays::Spectrum Evaluate(const HitPoint &hitPoint,
		const luxrays::Vector &localLightDir, const luxrays::Vector &localEyeDir, BSDFEvent *event,
		float *directPdfW = NULL, float *reversePdfW = NULL) const;
	virtual luxrays::Spectrum Sample(const HitPoint &hitPoint,
		const luxrays::Vector &localFixedDir, luxrays::Vector *localSampledDir,
		const float u0, const float u1, const float passThroughEvent,
		float *pdfW, BSDFEvent *event) const;
	void Pdf(const HitPoint &hitPoint,
		const luxrays::Vector &localLightDir, const luxrays::Vector &localEyeDir,
		float *directPdfW, float *reversePdfW) const;

	virtual void UpdateMaterialReferences(MaterialConstRef oldMat, MaterialRef newMat);
	virtual bool IsReferencing(MaterialConstRef mat) const;
	virtual void AddReferencedMaterials(
		std::unordered_set<const Material *> &referencedMats
	) const;
	virtual void AddReferencedTextures(std::unordered_set<const Texture *>  &referencedTexsreferencedTexs) const;
	virtual void UpdateTextureReferences(TextureConstRef oldTex, TextureRef newTex);

	virtual luxrays::PropertiesUPtr ToProperties(const ImageMapCache &imgMapCache, const bool useRealFileName) const;

	MaterialConstRef GetMaterialA() const { return *matA; }
	MaterialConstRef GetMaterialB() const { return *matB; }
	TextureConstRef GetMixFactor() const { return *mixFactor; }

protected:
	virtual void UpdateAvgPassThroughTransparency();

private:
	// Used by Preprocess()
	BSDFEvent GetEventTypesImpl() const;
	bool IsLightSourceImpl() const;
	bool IsDeltaImpl() const;

	void Preprocess();
	void UpdateVertexWeights();

	// The weight of material B at a vertex, from the clamped mix factor
	// weight2: weight2 itself or, when the sub-materials decide the
	// pass-through (vertexWeights), the weight conditioned on the vertex,
	// w2 o_B / (w1 o_A + w2 o_B) with o the GetVertexProbability() of each
	float GetVertexWeight2(const HitPoint &hitPoint, const float weight2) const;
	MaterialConstPtr matA;
	MaterialConstPtr matB;
	TextureConstPtr mixFactor;

	// Cached values for performance with very large material node trees
	BSDFEvent eventTypes;
	bool isLightSource, isDelta;
	// No transparency texture of its own and a sub-material that can let a
	// ray through: a vertex is reached through one of the sub-materials only
	// with its vertex probability. Only when the vertex probabilities depend
	// only on the hit point (HasPointVertexProbability()) and its own
	// pass-through decides its vertices (passThroughDecidesVertex).
	bool vertexWeights;
	bool passThroughDecidesVertex = true;
	// The vertex probabilities of the sub-materials when they are constant,
	// -1 otherwise
	float constVertexProbA, constVertexProbB;

};

}

#endif	/* _SLG_MIXMAT_H */
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
