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

#include <atomic>

#include "oneapi/tbb.h"

#include "luxrays/core/randomgen.h"
#include "luxrays/utils/mc.h"

#include "slg/engines/caches/photongi/photongicache.h"
#include "slg/engines/caches/photongi/pgicoctree.h"
#include "slg/cameras/camera.h"
#include "slg/utils/scenevisibility.h"

using namespace std;
using namespace luxrays;
using namespace slg;

//------------------------------------------------------------------------------
// TraceVisibilityParticles
//------------------------------------------------------------------------------

namespace slg {

class PGICSceneVisibility : public SceneVisibility<PGICVisibilityParticle> {
public:
	PGICSceneVisibility(PhotonGICache &cache) :
		SceneVisibility(
			*cache.scene,
			cache.visibilityParticles,
			cache.params.photon.maxPathDepth,
			cache.params.visibility.maxSampleCount,
			cache.params.visibility.targetHitRate,
			cache.params.visibility.lookUpRadius,
			cache.params.visibility.lookUpNormalAngle,
			cache.params.photon.timeStart,
			cache.params.photon.timeEnd
		),
		pgic(cache)
	{}

	virtual ~PGICSceneVisibility() { }
	
protected:
	virtual IndexOctree<PGICVisibilityParticle> *AllocOctree() const {
		auto bbox = scene.GetDataSet().GetBBox();
		return new PGICOctree(
			visibilityParticles,
			bbox,
			lookUpRadius,
			lookUpNormalAngle
		);
	}

	virtual bool ProcessHitPoint(const BSDF &bsdf, const PathVolumeInfo &volInfo,
			vector<PGICVisibilityParticle> &visibilityParticles) const {
		if (pgic.IsPhotonGIEnabled(bsdf)) {
			// The total reflectance of a particle on a surface is computed by
			// ComputeVisibilityParticlesArea(), only for the particles kept
			// (most hits fall on a particle already there, and the generic
			// EvaluateTotal() of a glossy material was most of the time of
			// this pass: 5 of 6.7 s on LuxCore2.1Benchmark)
			const Spectrum bsdfEvalTotal = bsdf.IsVolume() ? bsdf.EvaluateTotal() : Spectrum();
			assert (bsdfEvalTotal.IsValid());

			// Check if I have to flip the normal
			const Normal surfaceNormal = (bsdf.hitPoint.intoObject ?
				1.f : -1.f) * bsdf.hitPoint.geometryN;

			visibilityParticles.push_back(PGICVisibilityParticle(bsdf.hitPoint.p,
					surfaceNormal, bsdfEvalTotal, bsdf.IsVolume()));
		}

		return true;
	}

	virtual bool ProcessVisibilityParticle(const PGICVisibilityParticle &vp,
			vector<PGICVisibilityParticle> &visibilityParticles,
			IndexOctree<PGICVisibilityParticle> *octree, const float maxDistance2) const {
		PGICOctree *particlesOctree = (PGICOctree *)octree;

		// Check if a cache entry is available for this point
		const u_int entryIndex = particlesOctree->GetNearestEntry(vp.p, vp.n, vp.isVolume);

		if (entryIndex == NULL_INDEX) {
			// Add as a new entry
			visibilityParticles.push_back(vp);
			particlesOctree->Add(visibilityParticles.size() - 1);

			return false;
		} else {
			PGICVisibilityParticle &entry = visibilityParticles[entryIndex];
			const float distance2 = DistanceSquared(vp.p, entry.p);

			if (distance2 > maxDistance2) {
				// Add as a new entry
				visibilityParticles.push_back(vp);
				particlesOctree->Add(visibilityParticles.size() - 1);
				
				return false;
			} else {
				// Update the statistics about the area covered by this entry
				entry.hitsAccumulatedDistance += sqrtf(distance2);
				entry.hitsCount += 1;
				
				return true;
			}
		}
	}
	
	PhotonGICache &pgic;
};

}

void PhotonGICache::TraceVisibilityParticles() {
	PGICSceneVisibility pgicVisibility(*this);
	
	pgicVisibility.Build();

	if (visibilityParticles.size() == 0) {
		// Something wrong, nothing in the scene is visible and/or cache enabled
		return;
	}

	// Build the KdTree
	SLG_LOG("PhotonGI building visibility particles KdTree");
	visibilityParticlesKdTree = new PGICKdTree(visibilityParticles);
}

//------------------------------------------------------------------------------
// ComputeVisibilityParticlesArea
//------------------------------------------------------------------------------

// A photon is stored in every particle within the lookup radius whose normal
// is within the lookup angle of the photon's (TracePhotonsThread), so the
// photons of a particle on a surface come from the part of the surfaces in its
// sphere that faces its way and stores photons: a disc of radius r on an open
// flat surface, less at an edge or a corner. The radiance estimate divides by
// that area. It is measured with probe rays cast down the particle's normal at
// points of the disc of its tangent plane: a probe counts when the first
// surface it hits in the sphere faces the particle's way (within the angle)
// and stores photons, as a photon arriving from that side. The probes start a
// little above the tangent plane (a facing surface rises by at most tan(angle)
// per unit of distance), so the surface under a table or a shelf counts and
// the lower face of a thin board does not.
//
// The area from the mean distance of the visibility hits (still used for the
// particles in a volume) measured the cell of the particle among its
// neighbours, smaller than the disc the photons are gathered from (the SDL
// Cornell box was 5.4% too bright); the plain disc is too large at the edges
// and on thin parts (the classroom's chairs 57% too dark).
//
// The same pass computes the total reflectance of each particle on a surface,
// from a BSDF at its point.

void PhotonGICache::ComputeVisibilityParticlesArea() {
	const double startTime = WallClockTime();

	const float radius = params.visibility.lookUpRadius;
	const float radius2 = params.visibility.lookUpRadius2;
	const float normalCosAngle = params.visibility.lookUpNormalCosAngle;
	const float startHeight = radius * tanf(Radians(Min(params.visibility.lookUpNormalAngle, 45.f)));
	const float discArea = M_PI * radius2;

	const Camera &camera = scene->GetCamera();
	const float time = (params.photon.timeStart <= params.photon.timeEnd) ?
		Lerp(.5f, params.photon.timeStart, params.photon.timeEnd) :
		camera.GenerateRayTime(.5f);

	const auto accel = scene->GetDataSet().GetAccelerator(ACCEL_EMBREE);
	const SceneObjectDefinitions &objDefs = scene->GetObjects();
	// The objects that stop every ray (and have no bevel, which moves the hit)
	vector<bool> opaqueObjects(objDefs.GetSize());
	for (u_int i = 0; i < objDefs.GetSize(); ++i) {
		const SceneObject &obj = objDefs.GetSceneObject(i);
		opaqueObjects[i] = !obj.GetMaterial().HasPassThroughTransparency() &&
				(obj.GetExtMesh().GetBevelRadius() == 0.f);
	}

	// 8x8 stratified probes
	const u_int gridSize = 8;
	const float invGridSize = 1.f / gridSize;

	visibilityParticlesArea.assign(visibilityParticles.size(), discArea);
	std::atomic<u_int> missedCount(0);

	tbb::parallel_for(tbb::blocked_range<size_t>(0, visibilityParticles.size(), 64),
		[&](const tbb::blocked_range<size_t> &range) {
			for (size_t index = range.begin(); index != range.end(); ++index) {
				PGICVisibilityParticle &vp = visibilityParticles[index];
				if (vp.isVolume)
					continue;

				const Vector n(vp.n);
				Vector t1, t2;
				CoordinateSystem(n, &t1, &t2);

				// The particle's own surface: its object and its total
				// reflectance
				u_int ownObjectIndex = NULL_INDEX;
				{
					// From a little above the particle (its point is only
					// as precise as the eye ray's distance)
					Ray ray(vp.p + (.05f * radius) * n, -n);
					ray.maxt = .1f * radius;
					ray.time = time;
					RayHit rayHit;
					if (accel->Intersect(&ray, &rayHit)) {
						ownObjectIndex = rayHit.meshIndex;

						BSDF bsdf;
						PathVolumeInfo volInfo;
						bsdf.Init(false, false, *scene, ray, rayHit, .5f, &volInfo);
						vp.bsdfEvaluateTotal = bsdf.EvaluateTotal();
					} else
						++missedCount;
				}

				TauswortheRandomGenerator rng((u_int)(index + 1));

				u_int probeCount = 0;
				u_int facingCount = 0;
				// A probe at (dx, dy) of the unit disc
				auto probe = [&](const float dx, const float dy) {
					// The probe's half chord in the sphere
					const float h = radius * sqrtf(Max(0.f, 1.f - (dx * dx + dy * dy)));
					const float z0 = Min(startHeight, h);

					Ray ray(vp.p + radius * (dx * t1 + dy * t2) + z0 * n, -n);
					ray.maxt = z0 + h;
					ray.time = time;
					const float passThroughEvent = rng.floatValue();

					// The fast path: the probe hits the particle's own object,
					// which stops every ray: its material stores photons as at
					// the particle (a texture that blacks out its albedo is not
					// seen)
					RayHit rayHit;
					if (!accel->Intersect(&ray, &rayHit)) {
						++probeCount;
						return;
					}
					if ((rayHit.meshIndex == ownObjectIndex) && opaqueObjects[rayHit.meshIndex]) {
						const ExtMesh &mesh = objDefs.GetSceneObject(rayHit.meshIndex).GetExtMesh();
						Transform local2World;
						mesh.GetLocal2World(time, local2World);
						const Normal geometryN = mesh.GetGeometryNormal(local2World, rayHit.triangleIndex);

						++probeCount;
						if (AbsDot(geometryN, n) > normalCosAngle)
							++facingCount;
						return;
					}

					// Any other object: as a photon (pass-through, volumes, the
					// material at the point)
					PathVolumeInfo volInfo;
					BSDF bsdf;
					Spectrum connectionThroughput;
					const bool hit = scene->Intersect(nullptr, LIGHT_RAY | GENERIC_RAY, &volInfo,
							passThroughEvent, &ray, &rayHit, &bsdf, &connectionThroughput);

					// A probe scattered by a volume tells nothing
					if (hit && bsdf.IsVolume())
						return;

					++probeCount;
					if (hit && (AbsDot(bsdf.hitPoint.geometryN, n) > normalCosAngle) &&
							IsPhotonDepositEnabled(bsdf))
						++facingCount;
				};

				// The stratified probes
				for (u_int y = 0; y < gridSize; ++y) {
					for (u_int x = 0; x < gridSize; ++x) {
						float dx, dy;
						ConcentricSampleDisk((x + rng.floatValue()) * invGridSize,
								(y + rng.floatValue()) * invGridSize, &dx, &dy);
						probe(dx, dy);
					}
				}

				// At least the particle's own point
				if (probeCount > 0)
					visibilityParticlesArea[index] = discArea * Max(facingCount, 1u) / probeCount;
			}
		});

	double areaSum = 0.0;
	u_int surfaceCount = 0;
	u_int discCount = 0;
	for (u_int index = 0; index < visibilityParticles.size(); ++index) {
		if (!visibilityParticles[index].isVolume) {
			const float area = visibilityParticlesArea[index];
			areaSum += area;
			++surfaceCount;
			if (area == discArea)
				++discCount;
		}
	}

	SLG_LOG("PhotonGI visibility particles area: mean " <<
			((surfaceCount > 0) ? (areaSum / (surfaceCount * discArea)) : 0.0) <<
			" of the lookup disc, whole disc " << discCount << "/" << surfaceCount <<
			", surface not found " << missedCount <<
			" (" << (WallClockTime() - startTime) << " secs)");
}

float PhotonGICache::GetVisibilityParticleArea(const u_int index) const {
	const PGICVisibilityParticle &vp = visibilityParticles[index];

	return (vp.isVolume || (index >= visibilityParticlesArea.size())) ?
		vp.GetHitsArea(params.indirect.lookUpRadius2) : visibilityParticlesArea[index];
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
