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

#include "oneapi/tbb.h"

#include "slg/lights/strategies/logpower.h"
#include "luxrays/utils/properties.h"
#include "slg/scene/scene.h"
#include "slg/lights/trianglelight.h"
#include <memory>

using namespace std;
using namespace luxrays;
using namespace slg;

//------------------------------------------------------------------------------
// LightStrategyLogPower
//------------------------------------------------------------------------------

void LightStrategyLogPower::Preprocess(SceneConstRef scene, const LightStrategyTask taskType,
			const bool useRTMode) {
	// Delete old lightsDistribution
	lightsDistribution = nullptr;

	DistributionLightStrategy::Preprocess(scene, taskType);

	const u_int lightCount = scene.GetLightSources().GetSize();
	if (lightCount == 0)
		return;

	// The lights in parallel (a scene can have millions of triangle lights)
	vector<float> lightPower(lightCount);
	tbb::parallel_for(tbb::blocked_range<u_int>(0, lightCount, 4096),
		[&](const tbb::blocked_range<u_int> &r) {
			for (u_int i = r.begin(); i < r.end(); ++i) {
				auto& l = scene.GetLightSources().GetLightSource(i);
				const float power = logf(1.f + l.GetPower(scene)) * l.GetImportance();

				switch (taskType) {
					case TASK_EMIT: {
						lightPower[i] = power;
						break;
					}
					case TASK_ILLUMINATE: {
						if (l.IsDirectLightSamplingEnabled()){
							// A quad of two triangle lights is one light
							lightPower[i] = logf(1.f + GetIlluminatePower(scene, l)) * l.GetImportance();
						} else
							lightPower[i] = 0.f;
						break;
					}
					case TASK_INFINITE_ONLY: {
						if (l.IsInfinite())
							lightPower[i] = power;
						else
							lightPower[i] = 0.f;
						break;
					}
					default:
						throw runtime_error("Unknown task in LightStrategyLogPower::Preprocess(): " + ToString(taskType));
				}
			}
		});

	// Build the data to power based light sampling
	lightsDistribution = std::make_unique<Distribution1D>(lightPower);
}

// Static methods used by LightStrategyRegistry

PropertiesUPtr LightStrategyLogPower::ToProperties(const Properties &cfg) {
	PropertiesUPtr props = std::make_unique<Properties>();
	
	*props <<
				cfg.Get(GetDefaultProps()->Get("lightstrategy.type"));
	
	return props;
}

LightStrategyUPtr LightStrategyLogPower::FromProperties(const Properties &cfg) {
	return std::make_unique<LightStrategyLogPower>();
}

PropertiesUPtr LightStrategyLogPower::GetDefaultProps() {
	auto props = std::make_unique<Properties>();
	*props <<
			LightStrategy::GetDefaultProps() <<
			Property("lightstrategy.type")(GetObjectTag());

	return props;
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
