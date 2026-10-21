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

// The light tree strategy. The sampling and the pdf are compiled from the
// kernel source shared with the GPU (single-source, see
// luxrays/devices/cpudevice_oclemul.h).

#include <algorithm>
#include <cstring>
#include <cmath>
#include <memory>

#include "luxrays/utils/properties.h"
#include "slg/lights/strategies/lighttree.h"
#include "slg/scene/scene.h"
#include "slg/lights/distantlight.h"
#include "slg/lights/laserlight.h"
#include "slg/lights/pointlight.h"
#include "slg/lights/projectionlight.h"
#include "slg/lights/sharpdistantlight.h"
#include "slg/lights/spherelight.h"
#include "slg/lights/spotlight.h"
#include "slg/lights/sunlight.h"
#include "slg/lights/trianglelight.h"

namespace slg { namespace ocl { namespace {
#include "luxrays/devices/cpudevice_oclemul.h"
#include "slg/lights/sphericaltriangle_funcs.cl"
#include "slg/lights/strategies/lighttree_funcs.cl"

float3 ToFloat3(const luxrays::Point &p) {
	return MAKE_FLOAT3(p.x, p.y, p.z);
}

float3 ToFloat3(const luxrays::Normal &n) {
	return MAKE_FLOAT3(n.x, n.y, n.z);
}
} } }

using namespace std;
using namespace luxrays;
using namespace slg;

//------------------------------------------------------------------------------
// The bounds of the lights
//------------------------------------------------------------------------------

namespace slg { namespace {

struct LightBounds {
	LightBounds() : axis(0.f, 0.f, 1.f), cosThetaO(1.f), cosThetaE(1.f),
			phi(0.f), twoSided(false), isotropic(false), isTriangle(false) { }

	Point Centroid() const { return bbox.Center(); }

	BBox bbox;
	// The cone of the normals (theta_o) and of the emission around each
	// normal (theta_e, at most 90 degrees)
	Vector axis;
	float cosThetaO, cosThetaE;
	// The maximum radiant intensity (luminance)
	float phi;
	bool twoSided;
	// The light emits the same in every direction (point and sphere lights)
	bool isotropic;

	// A triangle light with a fixed position and a Lambertian emission: its
	// vertices, in the order of its emitting normal
	bool isTriangle;
	Point vertices[3];
};

float SafeACos(const float c) {
	return acosf(Clamp(c, -1.f, 1.f));
}

// The smallest cone holding two cones (pbrt-v4 Union() of DirectionCone)
void ConeUnion(const Vector &wa, const float cosA, const Vector &wb, const float cosB,
		Vector *w, float *cosTheta) {
	const float thetaA = SafeACos(cosA);
	const float thetaB = SafeACos(cosB);
	const float thetaD = SafeACos(Dot(wa, wb));

	if (Min(thetaD + thetaB, (float)M_PI) <= thetaA) {
		*w = wa;
		*cosTheta = cosA;
		return;
	}
	if (Min(thetaD + thetaA, (float)M_PI) <= thetaB) {
		*w = wb;
		*cosTheta = cosB;
		return;
	}

	const float thetaO = .5f * (thetaA + thetaD + thetaB);
	const Vector wr = Cross(wa, wb);
	if ((thetaO >= M_PI) || (wr.LengthSquared() == 0.f)) {
		// The whole sphere
		*w = wa;
		*cosTheta = -1.f;
		return;
	}

	// Rotate wa toward wb by thetaO - thetaA (wr is orthogonal to wa)
	const float thetaR = thetaO - thetaA;
	const Vector k = Normalize(wr);
	*w = Normalize(wa * cosf(thetaR) + Cross(k, wa) * sinf(thetaR));
	*cosTheta = cosf(thetaO);
}

LightBounds Union(const LightBounds &a, const LightBounds &b) {
	if (a.phi == 0.f)
		return b;
	if (b.phi == 0.f)
		return a;

	LightBounds result;
	result.bbox = luxrays::Union(a.bbox, b.bbox);
	ConeUnion(a.axis, a.cosThetaO, b.axis, b.cosThetaO, &result.axis, &result.cosThetaO);
	result.cosThetaE = Min(a.cosThetaE, b.cosThetaE);
	result.phi = a.phi + b.phi;
	result.twoSided = a.twoSided || b.twoSided;
	result.isotropic = a.isotropic && b.isotropic;
	result.isTriangle = false;

	return result;
}

// The surface area orientation heuristic of a split (pbrt-v4
// BVHLightSampler::EvaluateCost())
float SplitCost(const LightBounds &b, const BBox &bounds, const u_int dim) {
	const float thetaO = SafeACos(b.cosThetaO);
	const float thetaE = SafeACos(b.cosThetaE);
	const float thetaW = Min(thetaO + thetaE, (float)M_PI);
	const float sinThetaO = sqrtf(Max(0.f, 1.f - b.cosThetaO * b.cosThetaO));
	const float mOmega = 2.f * M_PI * (1.f - b.cosThetaO) +
			.5f * M_PI * (2.f * thetaW * sinThetaO - cosf(thetaO - 2.f * thetaW) -
			2.f * thetaO * sinThetaO + b.cosThetaO);

	// Favor the splits of the long axes
	const Vector diagonal = bounds.pMax - bounds.pMin;
	const float kr = Max(diagonal.x, Max(diagonal.y, diagonal.z)) / diagonal[dim];

	return b.phi * mOmega * kr * b.bbox.SurfaceArea();
}

typedef enum {
	LIGHT_NOT_SAMPLED, LIGHT_IN_TREE, LIGHT_DISTANT
} LightKind;

// The bounds of a light, or its importance if it is a distant light
LightKind GetLightBounds(SceneConstRef scene, LightSourceConstRef l,
		LightBounds *bounds, ocl::LightTreeDistantLight *distant) {
	const float importance = l.GetImportance();
	if (!(importance > 0.f))
		return LIGHT_NOT_SAMPLED;

	switch (l.GetType()) {
		case TYPE_TRIANGLE: {
			const TriangleLight &tl = static_cast<const TriangleLight &>(l);
			const Material &mat = *tl.lightMaterial;
			const ExtMesh &mesh = tl.sceneObject->GetExtMesh();
			const Triangle &tri = mesh.GetTriangles()[tl.triangleIndex];

			// The bounds over the motion of the mesh, sampled at the key
			// times and halfway between them
			vector<float> times;
			const ExtMotionTriangleMesh *motionMesh = dynamic_cast<const ExtMotionTriangleMesh *>(&mesh);
			if (motionMesh) {
				const vector<float> &keyTimes = motionMesh->GetMotionSystem().times;
				for (u_int i = 0; i < keyTimes.size(); ++i) {
					times.push_back(keyTimes[i]);
					if (i + 1 < keyTimes.size())
						times.push_back(.5f * (keyTimes[i] + keyTimes[i + 1]));
				}
			} else
				times.push_back(0.f);

			for (u_int i = 0; i < times.size(); ++i) {
				Transform local2World;
				mesh.GetLocal2World(times[i], local2World);

				for (u_int j = 0; j < 3; ++j)
					bounds->bbox = luxrays::Union(bounds->bbox, mesh.GetVertex(local2World, tri.v[j]));

				// The emitting side (TriangleLight::Illuminate())
				const Vector n(mesh.GetGeometryNormal(local2World, tl.triangleIndex));
				if (i == 0)
					bounds->axis = n;
				else
					ConeUnion(bounds->axis, bounds->cosThetaO, n, 1.f, &bounds->axis, &bounds->cosThetaO);
			}

			if (mat.GetEmissionFunc()) {
				// It can emit on both sides
				bounds->twoSided = true;
				bounds->cosThetaE = 0.f;
			} else
				bounds->cosThetaE = (mat.GetEmittedTheta() < 90.f) ? Max(0.f, mat.GetEmittedCosThetaMax()) : 0.f;

			// The intensity along the normal
			bounds->phi = mat.GetEmittedRadianceY(1.f / tl.GetMeshArea()) * tl.GetTriangleArea() * importance;

			if (!motionMesh && !mat.GetEmissionFunc() && (bounds->cosThetaE == 0.f) &&
					(tl.GetTriangleArea() > 0.f)) {
				Transform local2World;
				mesh.GetLocal2World(0.f, local2World);
				bounds->isTriangle = true;
				for (u_int j = 0; j < 3; ++j)
					bounds->vertices[j] = mesh.GetVertex(local2World, tri.v[j]);
				if (Dot(Cross(bounds->vertices[1] - bounds->vertices[0], bounds->vertices[2] - bounds->vertices[0]),
						bounds->axis) < 0.f)
					swap(bounds->vertices[1], bounds->vertices[2]);
			}
			break;
		}
		case TYPE_POINT:
		case TYPE_MAPPOINT:
		case TYPE_SPHERE:
		case TYPE_MAPSPHERE: {
			const PointLight &pl = static_cast<const PointLight &>(l);
			float absolutePos[3];
			pl.GetPreprocessedData(nullptr, absolutePos, nullptr);
			const Point pos(absolutePos[0], absolutePos[1], absolutePos[2]);

			const float radius = ((l.GetType() == TYPE_SPHERE) || (l.GetType() == TYPE_MAPSPHERE)) ?
				static_cast<const SphereLight &>(l).radius : 0.f;
			bounds->bbox = BBox(pos - Vector(radius, radius, radius), pos + Vector(radius, radius, radius));

			// Isotropic: every direction
			bounds->cosThetaO = -1.f;
			bounds->cosThetaE = 0.f;
			bounds->isotropic = true;
			bounds->phi = l.GetPower(scene) * (1.f / (4.f * M_PI)) * importance;
			break;
		}
		case TYPE_SPOT: {
			const SpotLight &sl = static_cast<const SpotLight &>(l);
			float emittedFactor[3], absolutePos[3], cosTotalWidth, cosFalloffStart;
			const Transform *alignedLight2World;
			sl.GetPreprocessedData(emittedFactor, absolutePos, &cosTotalWidth, &cosFalloffStart,
					&alignedLight2World);

			bounds->bbox = BBox(Point(absolutePos[0], absolutePos[1], absolutePos[2]));
			bounds->axis = Normalize((*alignedLight2World) * Vector(0.f, 0.f, 1.f));
			// As pbrt-v4: the full intensity inside the falloff start, the
			// falloff as the emission angle
			bounds->cosThetaO = cosFalloffStart;
			bounds->cosThetaE = cosf(Max(0.f, SafeACos(cosTotalWidth) - SafeACos(cosFalloffStart)));
			bounds->phi = RGBColor(emittedFactor).Y() * importance;
			break;
		}
		case TYPE_PROJECTION: {
			const ProjectionLight &pl = static_cast<const ProjectionLight &>(l);
			float emittedFactor[3], absolutePos[3], lightNormal[3];
			float screenX0, screenX1, screenY0, screenY1;
			const Transform *alignedLight2World, *lightProjection;
			pl.GetPreprocessedData(emittedFactor, absolutePos, lightNormal,
					&screenX0, &screenX1, &screenY0, &screenY1,
					&alignedLight2World, &lightProjection);

			bounds->bbox = BBox(Point(absolutePos[0], absolutePos[1], absolutePos[2]));
			bounds->axis = Normalize(Vector(lightNormal[0], lightNormal[1], lightNormal[2]));
			// The projection covers a cone up to the corners of its screen:
			// bound it with the hemisphere
			bounds->cosThetaO = 0.f;
			bounds->cosThetaE = 1.f;
			// The intensity without the image map
			bounds->phi = RGBColor(emittedFactor).Y() * importance;
			break;
		}
		case TYPE_LASER: {
			const LaserLight &ll = static_cast<const LaserLight &>(l);
			float emittedFactor[3], absoluteDir[3];
			ll.GetPreprocessedData(emittedFactor, nullptr, absoluteDir, nullptr, nullptr);

			distant->dir[0] = absoluteDir[0];
			distant->dir[1] = absoluteDir[1];
			distant->dir[2] = absoluteDir[2];
			distant->cosThetaMax = 1.f;
			distant->irradiance = RGBColor(emittedFactor).Y() * importance;
			return LIGHT_DISTANT;
		}
		case TYPE_SUN:
		case TYPE_DISTANT:
		case TYPE_SHARPDISTANT: {
			float dir[3];
			float cosThetaMax = 1.f;
			float sin2ThetaMax = 0.f;
			const float envRadius = InfiniteLightSource::GetEnvRadius(scene);
			const float power = l.GetPower(scene);
			float irradiance;
			if (l.GetType() == TYPE_SUN) {
				static_cast<const SunLight &>(l).GetPreprocessedData(dir, nullptr, nullptr,
						nullptr, nullptr, nullptr, &cosThetaMax, &sin2ThetaMax);
				// The radiance over the cone it is sampled in: GetPower() is
				// L * pi * r^2 * 2 * pi * sin^2(theta_max)
				const float radiance = power / (M_PI * envRadius * envRadius * 2.f * M_PI * Max(sin2ThetaMax, 1e-12f));
				irradiance = radiance * 2.f * M_PI * (1.f - cosThetaMax);
			} else if (l.GetType() == TYPE_DISTANT) {
				static_cast<const DistantLight &>(l).GetPreprocessedData(dir, nullptr, nullptr,
						nullptr, &cosThetaMax);
				// The radiance over the cone it is sampled in: GetPower() is
				// L * pi * r^2
				const float radiance = power / (M_PI * envRadius * envRadius);
				irradiance = radiance * 2.f * M_PI * (1.f - cosThetaMax);
			} else {
				static_cast<const SharpDistantLight &>(l).GetPreprocessedData(dir, nullptr, nullptr);
				// GetPower() is the irradiance times the disk of the scene
				irradiance = power / (M_PI * envRadius * envRadius);
			}

			distant->dir[0] = dir[0];
			distant->dir[1] = dir[1];
			distant->dir[2] = dir[2];
			distant->cosThetaMax = cosThetaMax;
			distant->irradiance = irradiance * importance;
			return LIGHT_DISTANT;
		}
		case TYPE_IL:
		case TYPE_IL_CONSTANT:
		case TYPE_IL_SKY2: {
			const float envRadius = InfiniteLightSource::GetEnvRadius(scene);

			distant->dir[0] = 0.f;
			distant->dir[1] = 0.f;
			distant->dir[2] = 1.f;
			distant->cosThetaMax = -1.f;
			// The irradiance pi * L of the average radiance L: GetPower() is
			// L * 4 * pi^2 * r^2 (L * 8 * pi^2 * r^2 for the sky)
			const float scale = (l.GetType() == TYPE_IL_SKY2) ? 8.f : 4.f;
			distant->irradiance = l.GetPower(scene) / (scale * M_PI * envRadius * envRadius) * importance;
			return LIGHT_DISTANT;
		}
		default:
			throw runtime_error("Unknown light source type in the light tree: " + ToString(l.GetType()));
	}

	return LIGHT_IN_TREE;
}

void ToOCL(const LightBounds &bounds, ocl::LightTreeNode *node) {
	node->bboxMin[0] = bounds.bbox.pMin.x;
	node->bboxMin[1] = bounds.bbox.pMin.y;
	node->bboxMin[2] = bounds.bbox.pMin.z;
	node->bboxMax[0] = bounds.bbox.pMax.x;
	node->bboxMax[1] = bounds.bbox.pMax.y;
	node->bboxMax[2] = bounds.bbox.pMax.z;
	node->radius = .5f * (bounds.bbox.pMax - bounds.bbox.pMin).Length();
	node->axis[0] = bounds.axis.x;
	node->axis[1] = bounds.axis.y;
	node->axis[2] = bounds.axis.z;
	// Normals within 0.06 degrees are parallel (the union of the cones of two
	// coplanar triangles is a little wider than zero in float, and the plane
	// cull of LightTree_NodeImportance() asks for exactly parallel normals)
	node->cosThetaO = (bounds.cosThetaO > .9999995f) ? 1.f : bounds.cosThetaO;
	node->cosThetaE = bounds.cosThetaE;
	node->phi = bounds.phi;
	node->flags = (bounds.twoSided ? LIGHTTREE_NODE_TWOSIDED : 0u) |
			(bounds.isotropic ? LIGHTTREE_NODE_ISOTROPIC : 0u);
}

typedef vector<pair<u_int, LightBounds> > BuildLights;

// Builds the subtree of the lights [start, end) and returns its root index
// (pbrt-v4 BVHLightSampler::buildBVH())
u_int BuildNode(BuildLights &lights, const u_int start, const u_int end,
		const u_int parentIndex, const u_int depth,
		vector<ocl::LightTreeNode> &nodes, vector<u_int> &lightIndex,
		vector<u_int> &nodeDepths) {
	const u_int nodeIndex = nodes.size();
	nodes.emplace_back();
	nodes[nodeIndex].parentIndex = parentIndex;
	nodeDepths.push_back(depth);

	if (end - start == 1) {
		const LightBounds &bounds = lights[start].second;
		ToOCL(bounds, &nodes[nodeIndex]);
		nodes[nodeIndex].flags |= LIGHTTREE_NODE_LEAF;
		if (bounds.isTriangle) {
			ocl::LightTreeNode &node = nodes[nodeIndex];
			for (u_int j = 0; j < 3; ++j) {
				node.bboxMin[j] = bounds.vertices[0][j];
				node.bboxMax[j] = bounds.vertices[1][j];
				node.axis[j] = bounds.vertices[2][j];
			}
			const Point centroid = (bounds.vertices[0] + bounds.vertices[1] + bounds.vertices[2]) * (1.f / 3.f);
			node.radius = sqrtf(Max(Max(
					DistanceSquared(bounds.vertices[0], centroid),
					DistanceSquared(bounds.vertices[1], centroid)),
					DistanceSquared(bounds.vertices[2], centroid)));
			node.flags |= LIGHTTREE_NODE_TRIANGLE;
		}
		nodes[nodeIndex].childOrLightIndex = lights[start].first;
		lightIndex[lights[start].first] = nodeIndex;

		return nodeIndex;
	}

	LightBounds nodeBounds;
	BBox centroidBounds;
	for (u_int i = start; i < end; ++i) {
		nodeBounds = Union(nodeBounds, lights[i].second);
		centroidBounds = luxrays::Union(centroidBounds, lights[i].second.Centroid());
	}

	// The split of least cost, among 12 buckets along each axis
	const u_int bucketCount = 12;
	float minCost = INFINITY;
	int minCostBucket = -1;
	int minCostDim = -1;
	// A degenerate tree (one light split off at a time) would recurse once
	// per light: split in the middle below some depth
	if (depth < 64) {
		for (u_int dim = 0; dim < 3; ++dim) {
			const float centroidMin = centroidBounds.pMin[dim];
			const float centroidMax = centroidBounds.pMax[dim];
			if (!(centroidMax > centroidMin))
				continue;

			LightBounds bucketBounds[bucketCount];
			for (u_int i = start; i < end; ++i) {
				const float offset = (lights[i].second.Centroid()[dim] - centroidMin) / (centroidMax - centroidMin);
				const u_int b = Min((u_int)(bucketCount * offset), bucketCount - 1);
				bucketBounds[b] = Union(bucketBounds[b], lights[i].second);
			}

			// The costs of the splits after each bucket but the last
			LightBounds below[bucketCount];
			below[0] = bucketBounds[0];
			for (u_int b = 1; b < bucketCount; ++b)
				below[b] = Union(below[b - 1], bucketBounds[b]);
			LightBounds above;
			for (int b = bucketCount - 2; b >= 0; --b) {
				above = Union(above, bucketBounds[b + 1]);
				if ((below[b].phi == 0.f) || (above.phi == 0.f))
					continue;

				const float cost = SplitCost(below[b], nodeBounds.bbox, dim) +
						SplitCost(above, nodeBounds.bbox, dim);
				if ((cost > 0.f) && (cost < minCost)) {
					minCost = cost;
					minCostBucket = b;
					minCostDim = dim;
				}
			}
		}
	}

	u_int mid;
	if (minCostDim == -1)
		mid = (start + end) / 2;
	else {
		const float centroidMin = centroidBounds.pMin[minCostDim];
		const float centroidMax = centroidBounds.pMax[minCostDim];
		auto midIter = std::partition(lights.begin() + start, lights.begin() + end,
				[=](const pair<u_int, LightBounds> &l) {
					const float offset = (l.second.Centroid()[minCostDim] - centroidMin) / (centroidMax - centroidMin);
					const int b = Min((u_int)(bucketCount * offset), bucketCount - 1);
					return b <= minCostBucket;
				});
		mid = midIter - lights.begin();
		if ((mid == start) || (mid == end))
			mid = (start + end) / 2;
	}

	// The first child is the next node
	BuildNode(lights, start, mid, nodeIndex, depth + 1, nodes, lightIndex, nodeDepths);
	const u_int child1 = BuildNode(lights, mid, end, nodeIndex, depth + 1, nodes, lightIndex, nodeDepths);

	ToOCL(nodeBounds, &nodes[nodeIndex]);
	nodes[nodeIndex].childOrLightIndex = child1;

	return nodeIndex;
}

} }

//------------------------------------------------------------------------------
// LightStrategyLightTree
//------------------------------------------------------------------------------

// The slot probabilities from the bounds of the last pick of the thread,
// kept for the AddSample() that follows it (the same point: no second walk)
namespace {
struct LearnPickStash {
	float boundsProb[LIGHTTREE_MAX_SLOTS];
	float px, py, pz;
	bool valid = false;
};
thread_local LearnPickStash learnPickStash;
}


LightStrategyLightTree::LightStrategyLightTree(const bool learnEnbl, const u_int learnDpth,
		const float learnMaxWght, const float learnCellSz, const float learnSmpls) : LightStrategy(TYPE_LIGHT_TREE),
		taskType(TASK_ILLUMINATE), learnEnable(learnEnbl), learnDepth(learnDpth),
		learnMaxWeight(learnMaxWght), learnCellSize(learnCellSz), learnSamples(learnSmpls), learnCellCount(0),
		learnLiveIndex(0), learnSampleCount(0), learnNextEpoch(0), learnRebuilding(false) {
	memset(&learn, 0, sizeof(learn));
}

LightStrategyLightTree::~LightStrategyLightTree() {
}

void LightStrategyLightTree::Preprocess(SceneConstRef scn, const LightStrategyTask type,
			const bool useRTMode) {
	taskType = type;

	distributionStrategy->Preprocess(scn, taskType, useRTMode);

	nodes.clear();
	distantLights.clear();
	lightIndex.clear();
	learnCellCount = 0;
	learnCells[0].clear();
	learnCells[1].clear();
	learnSums.reset();
	learnBoundsSums.reset();
	learnCounts.reset();

	if (UseTree())
		Build(scn);
}

void LightStrategyLightTree::Build(SceneConstRef scene) {
	const double startTime = WallClockTime();

	const u_int lightCount = scene.GetLightSources().GetSize();
	lightIndex.resize(lightCount, NULL_INDEX);

	BuildLights lights;
	for (u_int i = 0; i < lightCount; ++i) {
		auto& l = scene.GetLightSources().GetLightSource(i);
		// The emissive meshes over the 256 triangles of the AUTO direct light
		// sampling stay out, as with the other strategies: in the tree they
		// take the picks of the lights next to them (measured on the Blender
		// classroom, whose lamp meshes sit around its sphere lights)
		if (!l.IsDirectLightSamplingEnabled())
			continue;

		LightBounds bounds;
		ocl::LightTreeDistantLight distant;
		switch (GetLightBounds(scene, l, &bounds, &distant)) {
			case LIGHT_IN_TREE:
				if (bounds.phi > 0.f)
					lights.push_back(make_pair(i, bounds));
				break;
			case LIGHT_DISTANT:
				if (distant.irradiance > 0.f) {
					distant.lightIndex = i;
					lightIndex[i] = LIGHTTREE_DISTANT_LIGHT | distantLights.size();
					distantLights.push_back(distant);
				}
				break;
			default:
				break;
		}
	}

	vector<u_int> nodeDepths;
	if (lights.size() > 0) {
		nodes.reserve(2 * lights.size() - 1);
		nodeDepths.reserve(2 * lights.size() - 1);
		BuildNode(lights, 0, lights.size(), NULL_INDEX, 0, nodes, lightIndex, nodeDepths);
	}

	if (learnEnable)
		BuildLearning(scene, nodeDepths);

	SLG_LOG("Light tree: " << lights.size() << " lights in " << nodes.size() << " nodes, " <<
			distantLights.size() << " distant lights, " <<
			(UseLearning() ? ToString(learn.slotCount) + " learned slots in " + ToString(learnCellCount) + " cells, " : string("")) <<
			"built in " << (WallClockTime() - startTime) << " secs");
}

//------------------------------------------------------------------------------
// The learned slots
//------------------------------------------------------------------------------

void LightStrategyLightTree::BuildLearning(SceneConstRef scene, const vector<u_int> &nodeDepths) {
	memset(&learn, 0, sizeof(learn));
	learnCellCount = 0;

	// The cut: the nodes at the learning depth and the leaves above it, at a
	// smaller depth if they would not fit the slots with the distant lights
	vector<u_int> cut;
	for (int depth = learnDepth; depth >= 0; --depth) {
		cut.clear();
		for (u_int i = 0; i < nodes.size(); ++i) {
			if ((nodeDepths[i] == (u_int)depth) ||
					((nodeDepths[i] < (u_int)depth) && (nodes[i].flags & LIGHTTREE_NODE_LEAF)))
				cut.push_back(i);
		}
		if (cut.size() + distantLights.size() <= LIGHTTREE_MAX_SLOTS)
			break;
	}
	const u_int slotCount = cut.size() + distantLights.size();
	if ((slotCount > LIGHTTREE_MAX_SLOTS) || (slotCount < 2))
		return;

	learn.slotCount = slotCount;
	learn.distantCount = distantLights.size();
	for (u_int i = 0; i < cut.size(); ++i) {
		learn.cutNodes[i] = cut[i];
		nodes[cut[i]].flags |= (distantLights.size() + i + 1) << LIGHTTREE_NODE_SLOT_SHIFT;
	}

	const BSphere &bsphere = scene.GetSceneBSphere();
	learn.center[0] = bsphere.center.x;
	learn.center[1] = bsphere.center.y;
	learn.center[2] = bsphere.center.z;
	learn.invCellSize = 1.f / Max(learnCellSize * bsphere.rad, 1e-6f);
	// A cell stops learning at 4 times the samples its weight needs to reach
	// maxWeight (Min(maxWeight, N / (N + samples)) at N = samples * maxWeight /
	// (1 - maxWeight), 1024 samples with the defaults): the atomics cost up to
	// 14% per sample on the GPU (kitchen), and the ratios still improve after
	// the weight is saturated (stopping there loses 13% of the variance gain
	// on the Blender classroom, at 4 times 2%)
	learn.stopCount = 4.f * learnSamples * learnMaxWeight / (1.f - learnMaxWeight);

	learnCellCount = 1u << 17;
	learn.cellMask = learnCellCount - 1;
	for (u_int i = 0; i < 2; ++i)
		learnCells[i].assign((size_t)learnCellCount * (slotCount + 1), 0.f);
	learnLiveIndex = 0;
	learnSums.reset(new atomic<float>[(size_t)learnCellCount * slotCount]);
	learnBoundsSums.reset(new atomic<float>[(size_t)learnCellCount * slotCount]);
	for (size_t i = 0; i < (size_t)learnCellCount * slotCount; ++i) {
		learnSums[i] = 0.f;
		learnBoundsSums[i] = 0.f;
	}
	learnCounts.reset(new atomic<u_int>[learnCellCount]);
	for (size_t i = 0; i < learnCellCount; ++i)
		learnCounts[i] = 0;
	learnSampleCount = 0;
	learnNextEpoch = 1u << 16;
	learnRebuilding = false;
}

u_int LightStrategyLightTree::SlotOf(const u_int lightIdx) const {
	const u_int entry = lightIndex[lightIdx];
	if (entry == NULL_INDEX)
		return NULL_INDEX;
	if (entry & LIGHTTREE_DISTANT_LIGHT)
		return entry & ~LIGHTTREE_DISTANT_LIGHT;

	u_int nodeIndex = entry;
	for (;;) {
		const u_int slot = nodes[nodeIndex].flags >> LIGHTTREE_NODE_SLOT_SHIFT;
		if (slot)
			return slot - 1;
		nodeIndex = nodes[nodeIndex].parentIndex;
		if (nodeIndex == NULL_INDEX)
			return NULL_INDEX;
	}
}

void LightStrategyLightTree::AddSample(const Point &p, const Normal &n,
		const bool isVolume, const u_int lightIdx, const float pickPdf,
		const float contribution) const {
	if (!UseLearning())
		return;

	const ocl::float3 p3 = ocl::ToFloat3(p);
	const ocl::float3 n3 = ocl::ToFloat3(n);
	const u_int cell = ocl::LightTree_LearnCell(&learn, p3, n3, !isVolume);
	// A cell that has learned enough skips the atomics, as on the GPU
	if (learnCounts[cell].load(std::memory_order_relaxed) >= learn.stopCount) {
		learnPickStash.valid = false;
		return;
	}
	++learnCounts[cell];

	// The bounds probabilities of the slots at the point, the denominators
	// of the learned ratios: the ones of the pick, if it was made by this
	// thread at this point
	float boundsProbRecomputed[LIGHTTREE_MAX_SLOTS];
	const float *boundsProb;
	if (learnPickStash.valid && (learnPickStash.px == p.x) && (learnPickStash.py == p.y) && (learnPickStash.pz == p.z))
		boundsProb = learnPickStash.boundsProb;
	else {
		ocl::LightTree_SlotBoundsProbabilities(nodes.size() > 0 ? nodes.data() : nullptr,
				distantLights.data(), &learn, p3, n3, !isVolume, boundsProbRecomputed);
		boundsProb = boundsProbRecomputed;
	}
	learnPickStash.valid = false;
	for (u_int j = 0; j < learn.slotCount; ++j)
		if (boundsProb[j] > 0.f)
			learnBoundsSums[(size_t)cell * learn.slotCount + j].fetch_add(boundsProb[j]);

	if ((lightIdx != NULL_INDEX) && (contribution > 0.f) && (pickPdf > 0.f)) {
		const u_int slot = SlotOf(lightIdx);
		if (slot != NULL_INDEX)
			learnSums[(size_t)cell * learn.slotCount + slot].fetch_add(contribution / pickPdf);
	}

	// An epoch: rebuild the cells the sampling reads
	const unsigned long long count = ++learnSampleCount;
	if (count >= learnNextEpoch.load()) {
		bool expected = false;
		if (learnRebuilding.compare_exchange_strong(expected, true)) {
			if (count >= learnNextEpoch.load()) {
				RebuildLearnedCells();
				learnNextEpoch = 2 * count;
			}
			learnRebuilding = false;
		}
	}
}

void LightStrategyLightTree::RebuildLearnedCells() const {
	const u_int next = 1 - learnLiveIndex.load();
	BuildLearnedCells(learnSums.get(), learnBoundsSums.get(), learnCounts.get(),
			learnCellCount, learn.slotCount, learnMaxWeight, learnSamples,
			learnCells[next].data());
	learnLiveIndex = next;
}

LightSourcePtr LightStrategyLightTree::SampleLights(
	SceneConstRef scene,
	const float u,
	const Point &p, const Normal &n,
	const bool isVolume,
	float *pdf
	) const {
	if (!UseTree())
		return distributionStrategy->SampleLights(scene, u, p, n, isVolume, pdf);

	const bool learning = UseLearning();
	const u_int index = ocl::LightTree_SampleLights(
			nodes.size() > 0 ? nodes.data() : nullptr,
			distantLights.data(), distantLights.size(),
			learning ? &learn : nullptr, learning ? LiveLearnCells() : nullptr,
			ocl::ToFloat3(p), ocl::ToFloat3(n), isVolume,
			u, pdf, learning ? learnPickStash.boundsProb : nullptr);
	if (learning) {
		learnPickStash.px = p.x;
		learnPickStash.py = p.y;
		learnPickStash.pz = p.z;
		learnPickStash.valid = true;
	}

	if ((index == NULL_INDEX) || !(*pdf > 0.f))
		return nullptr;

	return LightSourcePtr(&scene.GetLightSources().GetLightSource(index));
}

float LightStrategyLightTree::SampleLightPdf(LightSourceConstRef light,
		const Point &p, const Normal &n, const bool isVolume) const {
	if (!UseTree())
		return distributionStrategy->SampleLightPdf(light, p, n, isVolume);

	const bool learning = UseLearning();
	return ocl::LightTree_SampleLightPdf(
			nodes.size() > 0 ? nodes.data() : nullptr,
			distantLights.data(), distantLights.size(),
			lightIndex.data(),
			learning ? &learn : nullptr, learning ? LiveLearnCells() : nullptr,
			ocl::ToFloat3(p), ocl::ToFloat3(n), isVolume,
			light.lightSceneIndex);
}

LightSourcePtr LightStrategyLightTree::SampleLights(SceneConstRef scene, const float u,
		float *pdf) const {
	return distributionStrategy->SampleLights(scene, u, pdf);
}

PropertiesUPtr LightStrategyLightTree::ToProperties() const {
	PropertiesUPtr props = std::make_unique<Properties>();

	*props <<
			Property("lightstrategy.type")(LightStrategyType2String(GetType())) <<
			Property("lightstrategy.lighttree.learning.enable")(learnEnable) <<
			Property("lightstrategy.lighttree.learning.depth")(learnDepth) <<
			Property("lightstrategy.lighttree.learning.maxweight")(learnMaxWeight) <<
			Property("lightstrategy.lighttree.learning.cellsize")(learnCellSize) <<
			Property("lightstrategy.lighttree.learning.samples")(learnSamples);

	return props;
}

// Static methods used by LightStrategyRegistry

PropertiesUPtr LightStrategyLightTree::ToProperties(const Properties &cfg) {
	PropertiesUPtr props = std::make_unique<Properties>();

	*props <<
			cfg.Get(GetDefaultProps()->Get("lightstrategy.type")) <<
			cfg.Get(GetDefaultProps()->Get("lightstrategy.lighttree.learning.enable")) <<
			cfg.Get(GetDefaultProps()->Get("lightstrategy.lighttree.learning.depth")) <<
			cfg.Get(GetDefaultProps()->Get("lightstrategy.lighttree.learning.maxweight")) <<
			cfg.Get(GetDefaultProps()->Get("lightstrategy.lighttree.learning.cellsize")) <<
			cfg.Get(GetDefaultProps()->Get("lightstrategy.lighttree.learning.samples"));

	return props;
}

LightStrategyUPtr LightStrategyLightTree::FromProperties(const Properties &cfg) {
	const bool learnEnable = cfg.Get(GetDefaultProps()->Get("lightstrategy.lighttree.learning.enable")).Get<bool>();
	const u_int learnDepth = Clamp(cfg.Get(GetDefaultProps()->Get("lightstrategy.lighttree.learning.depth")).Get<u_int>(), 0u, 4u);
	const float learnMaxWeight = Clamp(cfg.Get(GetDefaultProps()->Get("lightstrategy.lighttree.learning.maxweight")).Get<float>(), 0.f, .99f);
	const float learnCellSize = Max(cfg.Get(GetDefaultProps()->Get("lightstrategy.lighttree.learning.cellsize")).Get<float>(), 1e-4f);

	const float learnSamples = Max(cfg.Get(GetDefaultProps()->Get("lightstrategy.lighttree.learning.samples")).Get<float>(), 1.f);

	return std::make_unique<LightStrategyLightTree>(learnEnable, learnDepth, learnMaxWeight, learnCellSize, learnSamples);
}

PropertiesUPtr LightStrategyLightTree::GetDefaultProps() {
	auto props = std::make_unique<Properties>();
	*props <<
			LightStrategy::GetDefaultProps() <<
			Property("lightstrategy.type")(GetObjectTag()) <<
			// The learned slots (see LightTreeLearn in lighttree_types.cl):
			// the depth of the cut of the tree, the largest weight of the
			// learned probabilities against the ones from the bounds, and the
			// cell size as a fraction of the scene radius
			Property("lightstrategy.lighttree.learning.enable")(true) <<
			Property("lightstrategy.lighttree.learning.depth")(2u) <<
			Property("lightstrategy.lighttree.learning.maxweight")(.8f) <<
			Property("lightstrategy.lighttree.learning.cellsize")(.015f) <<
			// The samples a cell needs to weigh what it learned as much as
			// the bounds
			Property("lightstrategy.lighttree.learning.samples")(64.f);

	return props;
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
