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
		if (pgic.IsVisibilityParticleEnabled(bsdf)) {
			// The total reflectance is used only by the particles in a volume
			// (the indirect cache holds their outgoing radiance, the irradiance
			// of the ones on a surface)
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
// The photons and the probes count only where the particle's viewpoint sees
// them (IsVisibleFromParticle()): the point up to the radius above the
// particle (below anything over it), which looks at each of 8x8 cells of the
// square of the lookup disc and keeps the height of the first surface it sees
// there; a point below that height (beyond a tolerance) is hidden. Without it the
// gather crossed occluders: the surfaces on the other side of a thin wall, or
// the part of the SDL classroom's window portal behind the wall, 2.2 cm under
// it, where the light trapped between the portal and the wall (the cache's
// reflected radiance at the visible portal was 3x the path traced one) was
// gathered by the particles of the visible part across the edge of the
// opening. The radiance filter keeps only the particles whose viewpoints see
// each other (FilterVisibilityParticlesRadiance()).
//
// The area from the mean distance of the visibility hits (still used for the
// particles in a volume) measured the cell of the particle among its
// neighbours, smaller than the disc the photons are gathered from (the SDL
// Cornell box was 5.4% too bright); the plain disc is too large at the edges
// and on thin parts (the classroom's chairs 57% too dark).
//
// The indirect photons are the ones of the surfaces with a diffuse lobe
// (IsIndirectDepositEnabled()).

// The cells of the lookup disc: an 8x8 grid over its square
static const u_int PGIC_VIEW_GRID_SIZE = 8;
static const u_int PGIC_VIEW_CELL_COUNT = PGIC_VIEW_GRID_SIZE * PGIC_VIEW_GRID_SIZE; // PGICViewMap::cellHeights
// A cell where the viewpoint sees nothing (or a volume): everything in it is
// visible
static const int8_t PGIC_VIEW_NO_HIT = -128;

// The cell of a point of the unit disc
static inline u_int ViewCell(const float dx, const float dy) {
	const u_int x = Min<u_int>(Floor2UInt(Max(0.f, (dx + 1.f) * (.5f * PGIC_VIEW_GRID_SIZE))), PGIC_VIEW_GRID_SIZE - 1);
	const u_int y = Min<u_int>(Floor2UInt(Max(0.f, (dy + 1.f) * (.5f * PGIC_VIEW_GRID_SIZE))), PGIC_VIEW_GRID_SIZE - 1);

	return x + y * PGIC_VIEW_GRID_SIZE;
}

void PhotonGICache::ComputeVisibilityParticlesArea() {
	const double startTime = WallClockTime();

	const float radius = params.visibility.lookUpRadius;
	const float radius2 = params.visibility.lookUpRadius2;
	const float normalCosAngle = params.visibility.lookUpNormalCosAngle;
	const float tanAngle = tanf(Radians(Min(params.visibility.lookUpNormalAngle, 45.f)));
	const float startHeight = radius * tanAngle;
	const float discArea = M_PI * radius2;
	// A facing surface seen through a cell (0.25 radius wide) rises or
	// falls by tan(angle) per unit of distance, more for a curved one; and
	// the hidden part must be well under what hides it (0.1 radius: the
	// SDL classroom 4.9% too bright, LuxCore2.1Benchmark 2.3%; 0.2: 3.0% and
	// 2.0%; 0.3: 3.4% and 1.7%, the classroom's portal is 0.33 radius under
	// the wall)
	visibilityHeightTolerance = .2f * radius;

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
	const u_int gridSize = PGIC_VIEW_GRID_SIZE;
	const float invGridSize = 1.f / gridSize;

	visibilityParticlesArea.assign(visibilityParticles.size(), discArea);
	PGICViewMap emptyViewMap;
	emptyViewMap.viewHeight = 0.f;
	for (u_int i = 0; i < PGIC_VIEW_CELL_COUNT; ++i) {
		emptyViewMap.cellHeights[i] = PGIC_VIEW_NO_HIT;
		emptyViewMap.cellDistances[i] = 0;
	}
	visibilityParticlesViewMap.assign(visibilityParticles.size(), emptyViewMap);
	std::atomic<u_int> missedCount(0);
	std::atomic<u_int> hiddenCount(0);

	tbb::parallel_for(tbb::blocked_range<size_t>(0, visibilityParticles.size(), 64),
		[&](const tbb::blocked_range<size_t> &range) {
			for (size_t index = range.begin(); index != range.end(); ++index) {
				PGICVisibilityParticle &vp = visibilityParticles[index];
				if (vp.isVolume)
					continue;

				const Vector n(vp.n);
				Vector t1, t2;
				CoordinateSystem(n, &t1, &t2);

				// The particle's own surface: its object
				u_int ownObjectIndex = NULL_INDEX;
				{
					// From a little above the particle (its point is only
					// as precise as the eye ray's distance)
					Ray ray(vp.p + (.05f * radius) * n, -n);
					ray.maxt = .1f * radius;
					ray.time = time;
					RayHit rayHit;
					if (accel->Intersect(&ray, &rayHit))
						ownObjectIndex = rayHit.meshIndex;
					else
						++missedCount;
				}

				TauswortheRandomGenerator rng((u_int)(index + 1));

				// The viewpoint: up to the radius above the particle, below
				// anything over it
				float viewHeight = radius;
				{
					Ray ray(vp.p + (.05f * radius) * n, n);
					ray.maxt = 2.f * radius;
					ray.time = time;
					RayHit rayHit;
					if (accel->Intersect(&ray, &rayHit))
						viewHeight = Min(radius, .5f * (rayHit.t + .05f * radius));
				}
				const Point viewPoint = vp.p + viewHeight * n;
				visibilityParticlesViewMap[index].viewHeight = viewHeight;

				// What it sees of the lookup disc: the height of the first
				// surface through each cell, down to the bottom of the sphere
				// (the first surface a photon would stop on: pass-through
				// materials as a photon, a volume tells nothing)
				// (the points of the sphere are within 2 radii of the viewpoint)
				int8_t *heightMap = visibilityParticlesViewMap[index].cellHeights;
				u_char *distanceMap = visibilityParticlesViewMap[index].cellDistances;
				for (u_int y = 0; y < gridSize; ++y) {
					for (u_int x = 0; x < gridSize; ++x) {
						float dx = 2.f * (x + .5f) * invGridSize - 1.f;
						float dy = 2.f * (y + .5f) * invGridSize - 1.f;
						// The cells across the edge of the disc look at its
						// part inside
						const float r2 = dx * dx + dy * dy;
						if (r2 > Sqr(.95f)) {
							const float scale = .95f / sqrtf(r2);
							dx *= scale;
							dy *= scale;
						}

						Vector d = (vp.p + radius * (dx * t1 + dy * t2)) - viewPoint;
						const float length = d.Length();
						d /= length;

						Ray ray(viewPoint, d);
						ray.maxt = Min(length * (viewHeight + radius) / viewHeight, 2.f * radius);
						ray.time = time;
						const float passThroughEvent = rng.floatValue();

						RayHit rayHit;
						if (!accel->Intersect(&ray, &rayHit))
							continue;

						Point hitPoint;
						if (opaqueObjects[rayHit.meshIndex])
							hitPoint = ray(rayHit.t);
						else {
							PathVolumeInfo volInfo;
							BSDF bsdf;
							Spectrum connectionThroughput;
							const bool hit = scene->Intersect(nullptr, LIGHT_RAY | GENERIC_RAY, &volInfo,
									passThroughEvent, &ray, &rayHit, &bsdf, &connectionThroughput);
							if (!hit || bsdf.IsVolume())
								continue;
							hitPoint = bsdf.hitPoint.p;
						}

						const float h = Dot(hitPoint - vp.p, n) / radius;
						heightMap[x + y * gridSize] = (int8_t)Clamp(Round2Int(h * 127.f), -127, 127);
						const float distance = Distance(hitPoint, viewPoint) / (2.f * radius);
						distanceMap[x + y * gridSize] = (u_char)Clamp(Round2Int(distance * 255.f), 0, 255);
					}
				}

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
						if (AbsDot(geometryN, n) > normalCosAngle) {
							if (IsVisibleFromParticle(index, ray(rayHit.t)))
								++facingCount;
							else
								++hiddenCount;
						}
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
							IsIndirectDepositEnabled(bsdf)) {
						if (IsVisibleFromParticle(index, bsdf.hitPoint.p))
							++facingCount;
						else
							++hiddenCount;
					}
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
			", hidden probes " << hiddenCount <<
			" (" << (WallClockTime() - startTime) << " secs)");
}

float PhotonGICache::GetVisibilityParticleArea(const u_int index) const {
	const PGICVisibilityParticle &vp = visibilityParticles[index];

	return (vp.isVolume || (index >= visibilityParticlesArea.size())) ?
		vp.GetHitsArea(params.indirect.lookUpRadius2) : visibilityParticlesArea[index];
}

Point PhotonGICache::GetParticleViewPoint(const u_int index) const {
	const PGICVisibilityParticle &vp = visibilityParticles[index];
	const float viewHeight = (index < visibilityParticlesViewMap.size()) ?
		visibilityParticlesViewMap[index].viewHeight : 0.f;

	return vp.p + viewHeight * Vector(vp.n);
}

bool PhotonGICache::IsVisibleFromParticle(const u_int index, const Point &p) const {
	if (index >= visibilityParticlesViewMap.size())
		return true;
	const PGICViewMap &viewMap = visibilityParticlesViewMap[index];
	const float viewHeight = viewMap.viewHeight;
	// A particle in a volume
	if (!(viewHeight > 0.f))
		return true;

	const PGICVisibilityParticle &vp = visibilityParticles[index];
	const float radius = params.visibility.lookUpRadius;

	const Vector n(vp.n);
	const Vector d = p - vp.p;
	const float h = Dot(d, n);
	// Above the viewpoint: on top of what is over the particle
	if (h >= .999f * viewHeight)
		return false;

	// The point of the lookup disc the viewpoint sees it through
	Vector t1, t2;
	CoordinateSystem(n, &t1, &t2);
	const float scale = viewHeight / ((viewHeight - h) * radius);
	const float dx = Dot(d, t1) * scale;
	const float dy = Dot(d, t2) * scale;
	// Out of the cone the viewpoint looked at
	if (dx * dx + dy * dy >= 1.f)
		return true;

	const u_int cell = ViewCell(dx, dy);
	const int8_t cellHeight = viewMap.cellHeights[cell];
	if (cellHeight == PGIC_VIEW_NO_HIT)
		return true;

	// The point is hidden if it is below the first surface seen through the
	// cell and farther from the viewpoint (a surface beyond it, like a wall
	// at the edge of the disc, hides nothing)
	return (h >= cellHeight * (radius / 127.f) - visibilityHeightTolerance) ||
			(Distance(p, vp.p + viewHeight * n) <= viewMap.cellDistances[cell] * (2.f * radius / 255.f));
}

bool PhotonGICache::AreParticlesVisible(const u_int index1, const u_int index2) const {
	if ((index1 >= visibilityParticlesViewMap.size()) || (index2 >= visibilityParticlesViewMap.size()) ||
			!(visibilityParticlesViewMap[index1].viewHeight > 0.f) || !(visibilityParticlesViewMap[index2].viewHeight > 0.f))
		return true;

	const Point p1 = GetParticleViewPoint(index1);
	const Point p2 = GetParticleViewPoint(index2);
	Vector d = p2 - p1;
	const float distance = d.Length();
	if (distance == 0.f)
		return true;
	d /= distance;

	Ray ray(p1, d);
	ray.maxt = distance;
	ray.time = (params.photon.timeStart <= params.photon.timeEnd) ?
		Lerp(.5f, params.photon.timeStart, params.photon.timeEnd) :
		scene->GetCamera().GenerateRayTime(.5f);
	RayHit rayHit;

	return !scene->GetDataSet().GetAccelerator(ACCEL_EMBREE)->Intersect(&ray, &rayHit);
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
