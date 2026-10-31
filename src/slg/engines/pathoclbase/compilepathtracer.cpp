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

#include "slg/engines/pathoclbase/compiledscene.h"

using namespace std;
using namespace luxrays;
using namespace slg;

bool CompiledScene::HasVolumeSegmentLightBuffer() const {
	return pathTracer->volumeSegmentLight && hasHomogeneousVolumes && !pathTracer->GetPhotonGICache();
}

void CompiledScene::CompilePathTracer() {
	compiledPathTracer.eyeSampleBootSize = pathTracer->eyeSampleBootSize;
	compiledPathTracer.eyeSampleStepSize = pathTracer->eyeSampleStepSize;
	compiledPathTracer.eyeSampleSize = pathTracer->eyeSampleSize;

	compiledPathTracer.maxPathDepth.depth = pathTracer->maxPathDepth.depth;
	compiledPathTracer.maxPathDepth.diffuseDepth = pathTracer->maxPathDepth.diffuseDepth;
	compiledPathTracer.maxPathDepth.glossyDepth = pathTracer->maxPathDepth.glossyDepth;
	compiledPathTracer.maxPathDepth.subsurfaceDepth = pathTracer->maxPathDepth.subsurfaceDepth;
	compiledPathTracer.maxPathDepth.transmissionDepth = pathTracer->maxPathDepth.transmissionDepth;
	
	compiledPathTracer.rrType = pathTracer->rrType;
	compiledPathTracer.rrDepth = pathTracer->rrDepth;
	compiledPathTracer.rrImportanceCap = pathTracer->rrImportanceCap;
	compiledPathTracer.regularization = pathTracer->regularizationEnable ? pathTracer->regularizationRoughness : 0.f;
	
	compiledPathTracer.sqrtVarianceClampMaxValue = pathTracer->sqrtVarianceClampMaxValue;

	compiledPathTracer.hybridBackForward.enabled = pathTracer->hybridBackForwardEnable;
	compiledPathTracer.hybridBackForward.glossinessThreshold = pathTracer->hybridBackForwardGlossinessThreshold;
	
	compiledPathTracer.forceBlackBackground = pathTracer->forceBlackBackground;

	CompilePhotonGI();

	// The direct light along the volume segments of the vertex rays (as
	// PathTracer::RenderEyePath(): not with the PhotonGI cache, and it needs
	// a light that isn't infinite). The buffer of PathOCLBaseOCLRenderThread
	// is allocated without the lights' condition, which an edit can change.
	compiledPathTracer.volumeSegmentLight = HasVolumeSegmentLightBuffer() &&
			scene.GetLightSources().HasFiniteLights();
	compiledPathTracer.volumeSegmentLightBounds = scene.GetLightSources().HasFiniteLightsBounds();
	const BBox &lightBounds = scene.GetLightSources().GetFiniteLightsBounds();
	for (u_int i = 0; i < 3; ++i) {
		compiledPathTracer.volumeSegmentLightBoundsMin[i] = lightBounds.pMin[i];
		compiledPathTracer.volumeSegmentLightBoundsMax[i] = lightBounds.pMax[i];
	}

	compiledPathTracer.albedo.specularSetting = (slg::ocl::AlbedoSpecularSetting)pathTracer->albedoSpecularSetting;
	compiledPathTracer.albedo.specularGlossinessThreshold = pathTracer->albedoSpecularGlossinessThreshold;
}

#endif
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
