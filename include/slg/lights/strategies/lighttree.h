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
	LightStrategyLightTree();
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

	LightStrategyTask taskType;
	LightStrategyLogPowerUPtr distributionStrategy = std::make_unique<LightStrategyLogPower>();

	// The tree, the lights outside of it and, for each light, its entry (see
	// LIGHTTREE_DISTANT_LIGHT)
	std::vector<ocl::LightTreeNode> nodes;
	std::vector<ocl::LightTreeDistantLight> distantLights;
	std::vector<u_int> lightIndex;
};

}

#endif	/* _SLG_LIGHTSTRATEGY_LIGHTTREE_H */
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
