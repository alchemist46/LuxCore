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

#ifndef _SLG_LIGHTSTRATEGY_LIGHTTREE_H
#define	_SLG_LIGHTSTRATEGY_LIGHTTREE_H

#include <atomic>
#include <memory>
#include <vector>

#include "slg/slg.h"
#include "slg/lights/strategies/lightstrategy.h"
#include "slg/lights/strategies/logpower.h"

namespace slg {

// OpenCL data types
namespace ocl {
#include "slg/lights/strategies/lighttree_types.cl"
}

//------------------------------------------------------------------------------
// LightStrategyLightTree
//
// The direct light sampling picks a light by its importance at the shading
// point, with a BVH over the lights (see lighttree_funcs.cl). The light
// emission (light tracing, BiDir, PhotonGI) and the shadow catchers that only
// see the infinite lights have no shading point to use: they pick by log power.
//------------------------------------------------------------------------------

class LightStrategyLightTree : public LightStrategy {
public:
	LightStrategyLightTree(const bool learnEnable, const u_int learnDepth,
			const float learnMaxWeight, const float learnCellSize, const float learnSamples);
	virtual ~LightStrategyLightTree();

	virtual void Preprocess(SceneConstRef scene, const LightStrategyTask taskType,
			const bool useRTMode);

	// Used for direct light sampling
	virtual LightSourcePtr SampleLights(
			SceneConstRef scene,
			const float u,
			const luxrays::Point &p, const luxrays::Normal &n,
			const bool isVolume,
			float *pdf) const;

	virtual float SampleLightPdf(LightSourceConstRef light,
			const luxrays::Point &p, const luxrays::Normal &n,
			const bool isVolume) const;

	virtual void AddSample(const luxrays::Point &p, const luxrays::Normal &n,
			const bool isVolume, const u_int lightIndex, const float pickPdf,
			const float contribution) const;

	// Used for light emission
	virtual LightSourcePtr SampleLights(
		SceneConstRef scene, const float u, float *pdf
	) const;

	virtual LightStrategyType GetType() const { return GetObjectType(); }
	virtual std::string GetTag() const { return GetObjectTag(); }

	virtual luxrays::PropertiesUPtr ToProperties() const;

	// Used for OpenCL data translation
	const luxrays::Distribution1DRPtr GetLightsDistribution() const { return distributionStrategy->GetLightsDistribution(); }
	// True if the direct light sampling uses the tree
	bool UseTree() const { return taskType == TASK_ILLUMINATE; }
	const std::vector<ocl::LightTreeNode> &GetNodes() const { return nodes; }
	const std::vector<ocl::LightTreeDistantLight> &GetDistantLights() const { return distantLights; }
	const std::vector<u_int> &GetLightIndex() const { return lightIndex; }
	// True if the sampling uses the learned slots (see LightTreeLearn)
	bool UseLearning() const { return UseTree() && (learnCellCount > 0); }
	const ocl::LightTreeLearn &GetLearn() const { return learn; }
	u_int GetLearnCellCount() const { return learnCellCount; }
	float GetLearnMaxWeight() const { return learnMaxWeight; }
	float GetLearnSamples() const { return learnSamples; }

	// The cells of the learned slots from the sums of the samples (sums and
	// boundsSums have cellCount * slotCount entries, counts cellCount, cells
	// cellCount * (slotCount + 1)); the device threads use it too
	template <class Sums, class Counts>
	static void BuildLearnedCells(const Sums &sums, const Sums &boundsSums, const Counts &counts,
			const u_int cellCount, const u_int slotCount,
			const float maxWeight, const float samples, float *cells) {
		for (size_t c = 0; c < cellCount; ++c) {
			float *cell = &cells[c * (slotCount + 1)];
			const float count = counts[c];

			// The ratio of the contribution of each slot to its bounds
			// probability (what the sampling scales the bounds probabilities
			// with; their normalization makes the units irrelevant)
			float total = 0.f;
			for (u_int j = 0; j < slotCount; ++j) {
				const float sum = sums[c * slotCount + j];
				const float boundsSum = boundsSums[c * slotCount + j];
				cell[1 + j] = (boundsSum > 0.f) ? (sum / boundsSum) : 0.f;
				total += sum;
			}
			if (!(count > 0.f) || !(total > 0.f)) {
				cell[0] = 0.f;
				continue;
			}

			// The weight of what the cell has learned grows with its samples
			cell[0] = luxrays::Min(maxWeight, count / (count + samples));
		}
	}
	// The cells of the learned slots, as the sampling reads them now
	const std::vector<float> &GetLearnCells() const { return learnCells[learnLiveIndex.load()]; }

	//--------------------------------------------------------------------------
	// Static methods used by LightStrategyRegistry
	//--------------------------------------------------------------------------

	static LightStrategyType GetObjectType() { return TYPE_LIGHT_TREE; }
	static std::string GetObjectTag() { return "LIGHT_TREE"; }
	static luxrays::PropertiesUPtr ToProperties(const luxrays::Properties &cfg);
	static LightStrategyUPtr FromProperties(const luxrays::Properties &cfg);

protected:
	static luxrays::PropertiesUPtr GetDefaultProps();

	void Build(SceneConstRef scene);
	void BuildLearning(SceneConstRef scene, const std::vector<u_int> &nodeDepths);
	void RebuildLearnedCells() const;
	// The slot of a light (NULL_INDEX if the strategy never picks it)
	u_int SlotOf(const u_int lightIndex) const;
	const float *LiveLearnCells() const { return learnCells[learnLiveIndex.load()].data(); }

	LightStrategyTask taskType;
	LightStrategyLogPowerUPtr distributionStrategy = std::make_unique<LightStrategyLogPower>();

	// The tree, the lights outside of it and, for each light, its entry (see
	// LIGHTTREE_DISTANT_LIGHT)
	std::vector<ocl::LightTreeNode> nodes;
	std::vector<ocl::LightTreeDistantLight> distantLights;
	std::vector<u_int> lightIndex;

	// The learned slots (lighttree_funcs.cl, LightTreeLearn). The sampling
	// reads one of two copies of the cells; the other is rebuilt from the
	// sums of the samples at each epoch (a doubling of the sample count),
	// then the copies are swapped.
	bool learnEnable;
	u_int learnDepth;
	float learnMaxWeight, learnCellSize, learnSamples;
	ocl::LightTreeLearn learn;
	u_int learnCellCount;
	mutable std::vector<float> learnCells[2];
	mutable std::atomic<u_int> learnLiveIndex;
	// Per cell and slot: the sum of the contribution estimates and the sum
	// of the bounds probabilities
	mutable std::unique_ptr<std::atomic<float>[]> learnSums, learnBoundsSums;
	mutable std::unique_ptr<std::atomic<u_int>[]> learnCounts;
	mutable std::atomic<unsigned long long> learnSampleCount, learnNextEpoch;
	mutable std::atomic<bool> learnRebuilding;
};

}

#endif	/* _SLG_LIGHTSTRATEGY_LIGHTTREE_H */
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
