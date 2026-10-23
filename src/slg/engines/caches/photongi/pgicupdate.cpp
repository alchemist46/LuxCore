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

#include "slg/engines/caches/photongi/photongicache.h"

using namespace std;
using namespace luxrays;
using namespace slg;

void PhotonGICache::InitUpdate(const u_int threadCnt) {
	threadCount = threadCnt;
	threadsSyncBarrier.reset(new std::barrier(threadCount, completion_t()));
	finishUpdateFlag = false;
}

bool PhotonGICache::EndSceneEdit(PhotonGICache *&cache, SceneConstRef scn,
		const Properties &cfg, const EditActionList &editActions,
		const u_int threadCount) {
	if (!cache)
		return false;

	// The photons and the cache entries depend on the geometry, the materials
	// (and their image maps) and the lights. The camera edits keep the cache:
	// the visibility entries stay where the first camera saw, the indirect
	// cache falls back to path tracing elsewhere.
	if (editActions.Has(GEOMETRY_EDIT) || editActions.Has(GEOMETRY_TRANS_EDIT) ||
			editActions.Has(MATERIALS_EDIT) || editActions.Has(MATERIAL_TYPES_EDIT) ||
			editActions.Has(LIGHTS_EDIT) || editActions.Has(LIGHT_TYPES_EDIT) ||
			editActions.Has(IMAGEMAPS_EDIT)) {
		SLG_LOG("Computing the PhotonGI cache again after the scene edit");

		delete cache;
		cache = nullptr;
		cache = FromProperties(scn, cfg);
		if (cache)
			cache->Preprocess(threadCount);

		return true;
	} else {
		// The film starts again from 0 samples/pixel
		cache->InitUpdate(threadCount);
		cache->lastUpdateSpp = 0;

		return false;
	}
}

bool PhotonGICache::Update(const u_int threadIndex, const u_int filmSPP,
		const std::function<void()> &threadZeroCallback) {
	if (!params.caustic.enabled || (params.caustic.updateSpp == 0) || finishUpdateFlag)
		return false;

	// Check if it is time to update the caustic cache (the film may have
	// started again with fewer samples)
	const u_int deltaSpp = (filmSPP > lastUpdateSpp) ? (filmSPP - lastUpdateSpp) : 0;
	if (deltaSpp > params.caustic.updateSpp) {
		// Time to update the caustic cache

		threadsSyncBarrier->arrive_and_wait();

		bool result = false;
		if ((threadIndex == 0) && !finishUpdateFlag) {
			const double startTime = WallClockTime();

			SLG_LOG("Updating PhotonGI caustic cache after " << filmSPP << " samples/pixel (Pass " << causticPhotonPass << ")");

			// A safety check to avoid the update if visibility map has been deallocated
			if (visibilityParticles.size() == 0) {
				SLG_LOG("ERROR: Updating PhotonGI caustic cache is not possible without visibility information");
				lastUpdateSpp = filmSPP;
			} else {
				// Drop previous cache
				delete causticPhotonsBVH;
				causticPhotonsBVH = nullptr;
				causticPhotons.clear();

				// Reduce the look up radius
				params.caustic.lookUpRadius = params.caustic.lookUpRadius /
						powf(float(causticPhotonPass + 1), .5f * (1.f - params.caustic.radiusReduction));
				// Place a cap to radius reduction
				params.caustic.lookUpRadius = Max(params.caustic.lookUpRadius, params.caustic.minLookUpRadius);
				params.caustic.lookUpRadius2 = Sqr(params.caustic.lookUpRadius);
				SLG_LOG("New PhotonGI caustic cache lookup radius: " << params.caustic.lookUpRadius);
				++causticPhotonPass;

				// Trace the photons for a new one
				TracePhotons(false, params.caustic.enabled);

				if (causticPhotons.size() > 0) {
					// Build a new BVH
					SLG_LOG("PhotonGI building caustic photons BVH");
					causticPhotonsBVH = new PGICPhotonBvh(&causticPhotons, causticPhotonTracedCount,
							params.caustic.lookUpRadius, params.caustic.lookUpNormalAngle);
				}

				lastUpdateSpp = filmSPP;
				++updateGeneration;

				if (threadZeroCallback)
					threadZeroCallback();

				result = true;
			}

			const float dt = WallClockTime() - startTime;
			SLG_LOG("Updating PhotonGI caustic cache done in: " << std::setprecision(3) << dt << " secs");
		}

		threadsSyncBarrier->arrive_and_wait();

		return result;
	} else
		return false;
}

void PhotonGICache::FinishUpdate(const u_int threadIndex) {
	for (;;) {
		if (finishUpdateFlag)
			return;

		threadsSyncBarrier->arrive_and_wait();
		finishUpdateFlag = true;
		threadsSyncBarrier->arrive_and_wait();
	}
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
