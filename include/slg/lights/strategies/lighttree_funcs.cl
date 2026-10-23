#line 2 "lighttree_funcs.cl"

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

//------------------------------------------------------------------------------
// The light tree: the direct light sampling picks a light by its importance
// at the shading point, walking a BVH over the lights with their power,
// bounds and orientation cones (Conty Estevez and Kulla, "Importance Sampling
// of Many Lights with Adaptive Tree Splitting", 2018), in the form of Cycles'
// light tree (intern/cycles/kernel/light/tree.h): no splitting (one light per
// shading point), and at each node the probability of a child is the average
// of the probabilities computed with the largest and the smallest
// contribution its lights can make, which keeps it away from zero where the
// bounds are loose. The lights whose contribution doesn't depend on the
// distance (environment, sun, distant, laser) are a separate group, weighed
// against the root of the tree.
//
// The pdf of a light (for the MIS weight of a BSDF hit on it) walks up from
// its leaf: it is the probability of the sampling, so the pdf of a light the
// tree never picks at a point is zero there, for both strategies.
//
// It is the same code on the CPU (compiled through the OpenCL emulation, see
// src/slg/lights/strategies/lighttree.cpp) and on the GPU. It uses
// sphericaltriangle_funcs.cl.
//------------------------------------------------------------------------------

OPENCL_FORCE_INLINE float LightTree_SinFromCos(const float cosTheta) {
	return sqrt(fmax(0.f, 1.f - cosTheta * cosTheta));
}

// The importance of a triangle light leaf: its radiance times its solid angle
// from p times the bound of the incidence. Far from the triangle it is the
// importance of LightTree_NodeImportance(), close to it the two triangles of a
// quad differ. A quad leaf is the parallelogram of its 3 vertices.
OPENCL_FORCE_INLINE float LightTree_TriangleImportance(__global const LightTreeNode *node,
		const float3 p, const float3 n, const bool hasNormal) {
	const float3 v0 = MAKE_FLOAT3(node->bboxMin[0], node->bboxMin[1], node->bboxMin[2]);
	const float3 v1 = MAKE_FLOAT3(node->bboxMax[0], node->bboxMax[1], node->bboxMax[2]);
	const float3 v2 = MAKE_FLOAT3(node->axis[0], node->axis[1], node->axis[2]);

	const bool isQuad = (node->flags & LIGHTTREE_NODE_QUAD);

	const float3 crossN = cross(v1 - v0, v2 - v0);
	const float doubleArea = length(crossN);
	if (!(doubleArea > 0.f))
		return 0.f;
	const float3 triN = crossN / doubleArea;

	// The emitting side
	const float side = dot(triN, p - v0);
	if (!(node->flags & LIGHTTREE_NODE_TWOSIDED) && !(side > 0.f))
		return 0.f;

	const float3 toCentroid = (isQuad ? (.5f * (v1 + v2)) : ((v0 + v1 + v2) * (1.f / 3.f))) - p;
	const float centroidDistance2 = dot(toCentroid, toCentroid);
	if (!(centroidDistance2 > 0.f))
		return 0.f;
	const float centroidDistance = sqrt(centroidDistance2);
	const float3 wi = toCentroid / centroidDistance;

	// The solid angle: exact close to the triangle, the projected area far
	// from it (where the float precision of the exact one breaks down)
	const float area = isQuad ? doubleArea : (.5f * doubleArea);
	const float solidAngle = (centroidDistance2 < 64.f * area) ?
		(SphericalTriangle_SolidAngle(v0, v1, v2, p) +
			(isQuad ? SphericalTriangle_SolidAngle(v1, v1 + v2 - v0, v2, p) : 0.f)) :
		(area * fabs(dot(triN, wi)) / centroidDistance2);

	float cosIncidence = 1.f;
	if (hasNormal) {
		// theta_u: the half angle of the cone of directions from p that holds
		// the bounding sphere of the triangle around its centroid
		const float radius2 = node->radius * node->radius;
		const float cosThetaU = (centroidDistance2 > radius2) ?
				sqrt(1.f - radius2 / centroidDistance2) : -1.f;
		const float cosThetaI = fabs(dot(n, wi));
		cosIncidence = (cosThetaI >= cosThetaU) ? 1.f :
			(cosThetaI * cosThetaU + LightTree_SinFromCos(cosThetaI) * LightTree_SinFromCos(cosThetaU));
	}

	// phi is the radiance times the area
	return node->phi / area * solidAngle * cosIncidence;
}

// The largest and the smallest contribution the lights of a node can make to
// the point p. The normal n is used on both sides (the BSDF may transmit) and
// not at all in a volume.
OPENCL_FORCE_INLINE void LightTree_NodeImportance(__global const LightTreeNode *node,
		const float3 p, const float3 n, const bool hasNormal,
		float *maxImportance, float *minImportance) {
	if (node->flags & LIGHTTREE_NODE_TRIANGLE) {
		*maxImportance = LightTree_TriangleImportance(node, p, n, hasNormal);
		*minImportance = *maxImportance;
		return;
	}

	*maxImportance = 0.f;
	*minImportance = 0.f;

	const float3 bboxMin = MAKE_FLOAT3(node->bboxMin[0], node->bboxMin[1], node->bboxMin[2]);
	const float3 bboxMax = MAKE_FLOAT3(node->bboxMax[0], node->bboxMax[1], node->bboxMax[2]);

	const float3 toCentroid = .5f * (bboxMin + bboxMax) - p;
	const float centroidDistance = length(toCentroid);
	const float invCentroidDistance = 1.f / centroidDistance;
	// The direction from p to the centroid (any direction at the centroid)
	const float3 wi = (centroidDistance > 0.f) ? (toCentroid * invCentroidDistance) : MAKE_FLOAT3(0.f, 0.f, 1.f);

	// The radius of the bounding sphere of the box
	const float radius = node->radius;

	// The distance used for the largest contribution, clamped to half the
	// radius of the bounding sphere (it would diverge at the centroid), and the
	// distance of the farthest corner for the smallest one. Half the radius is
	// Cycles' clamp: with the whole radius, a point inside a large node (e.g.
	// the root, in a room) underrates the node 4x against the distant lights
	// (a sun behind blinds took 66% of the picks on the Blender classroom, 55%
	// with this clamp)
	const float minDistance = fmax(centroidDistance, .5f * radius);
	const float3 toFarthestCorner = fmax(p - bboxMin, bboxMax - p);
	const float minDistance2 = fmax(minDistance * minDistance, 1e-12f);
	const float maxDistance2 = fmax(dot(toFarthestCorner, toFarthestCorner), 1e-12f);

	const float3 axis = MAKE_FLOAT3(node->axis[0], node->axis[1], node->axis[2]);
	const float cosThetaO = node->cosThetaO;

	// The lights of a node whose normals are all parallel (a quad, a panel,
	// the two triangles of a window) can't light a point behind every point
	// of the box: the cone bound below only sees that past theta_u, which
	// leaves the points near the plane of the lights, and all the points
	// inside the bounding sphere, with an importance no child can confirm
	// (on the Blender classroom 17% of the walks ended this way)
	if ((cosThetaO >= 1.f) && !(node->flags & LIGHTTREE_NODE_TWOSIDED)) {
		const float3 halfExtent = .5f * (bboxMax - bboxMin);
		if (!(-dot(axis, toCentroid) + dot(fabs(axis), halfExtent) > 0.f))
			return;
	}

	// theta_u: the half angle of the cone of directions from p that holds the
	// node: of its bounding sphere when p is outside it (pbrt-v4), of its 8
	// corners when p is inside the sphere but outside the box (the directions
	// of a convex box are within the cone of its corners if that cone is
	// inside a hemisphere; otherwise, or inside the box, every direction)
	float cosThetaU, sinThetaU;
	if (centroidDistance > radius) {
		sinThetaU = radius * invCentroidDistance;
		cosThetaU = sqrt(fmax(0.f, 1.f - sinThetaU * sinThetaU));
	} else {
		const bool insideBox = (p.x >= bboxMin.x) && (p.x <= bboxMax.x) &&
				(p.y >= bboxMin.y) && (p.y <= bboxMax.y) &&
				(p.z >= bboxMin.z) && (p.z <= bboxMax.z);
		float minCos = -1.f;
		if (!insideBox) {
			minCos = 1.f;
			for (uint i = 0; i < 8; ++i) {
				const float3 corner = MAKE_FLOAT3(
						(i & 1) ? bboxMax.x : bboxMin.x,
						(i & 2) ? bboxMax.y : bboxMin.y,
						(i & 4) ? bboxMax.z : bboxMin.z) - p;
				minCos = fmin(minCos, dot(wi, corner) / sqrt(fmax(dot(corner, corner), 1e-24f)));
			}
		}
		cosThetaU = (minCos > 0.f) ? minCos : -1.f;
		sinThetaU = LightTree_SinFromCos(cosThetaU);
	}

	// theta: the angle between the axis of the cone and the direction from
	// the centroid to p
	float cosTheta = -dot(axis, wi);
	if (node->flags & LIGHTTREE_NODE_TWOSIDED)
		cosTheta = fabs(cosTheta);
	const float sinTheta = LightTree_SinFromCos(cosTheta);

	const float sinThetaO = LightTree_SinFromCos(cosThetaO);

	// The smallest angle an emitter normal can make with the direction to p:
	// max(0, theta - theta_u - theta_o)
	const float cosThetaMinusU = (cosTheta >= cosThetaU) ? 1.f :
		(cosTheta * cosThetaU + sinTheta * sinThetaU);
	float cosMinOutgoing;
	if (cosThetaMinusU >= cosThetaO)
		cosMinOutgoing = 1.f;
	else
		cosMinOutgoing = cosThetaMinusU * cosThetaO + LightTree_SinFromCos(cosThetaMinusU) * sinThetaO;

	// No light of the node emits toward p
	if (cosMinOutgoing < node->cosThetaE)
		return;

	// The incidence at p: max(0, theta_i - theta_u) for the largest
	// contribution, theta_i + theta_u for the smallest
	float cosMinIncidence = 1.f;
	float cosMaxIncidence = 1.f;
	if (hasNormal) {
		const float cosThetaI = fabs(dot(n, wi));
		const float sinThetaI = LightTree_SinFromCos(cosThetaI);

		cosMinIncidence = (cosThetaI >= cosThetaU) ? 1.f :
			(cosThetaI * cosThetaU + sinThetaI * sinThetaU);
		cosMaxIncidence = fmax(0.f, cosThetaI * cosThetaU - sinThetaI * sinThetaU);
	}

	*maxImportance = node->phi * cosMinOutgoing * cosMinIncidence / minDistance2;

	// The largest angle an emitter normal can make with the direction to p:
	// theta + theta_u + theta_o, zero contribution past theta_e (which is
	// never more than 90 degrees). The isotropic lights have no outgoing
	// cosine (their cone is the whole sphere, which would make it zero)
	if (node->flags & LIGHTTREE_NODE_ISOTROPIC)
		*minImportance = node->phi * cosMaxIncidence / maxDistance2;
	else if ((cosTheta >= 0.f) && (cosThetaU >= 0.f) && (cosThetaO >= 0.f)) {
		const float cosThetaPlusU = cosTheta * cosThetaU - sinTheta * sinThetaU;

		if (cosThetaPlusU > 0.f) {
			const float cosMaxOutgoing = cosThetaPlusU * cosThetaO - LightTree_SinFromCos(cosThetaPlusU) * sinThetaO;

			if (cosMaxOutgoing > node->cosThetaE)
				*minImportance = node->phi * cosMaxOutgoing * cosMaxIncidence / maxDistance2;
		}
	}
}

// The probability of the first of two children (or groups) from their
// largest and smallest importances, -1 if neither can contribute
OPENCL_FORCE_INLINE float LightTree_FirstChildProbability(
		const float maxImportance0, const float minImportance0,
		const float maxImportance1, const float minImportance1) {
	const float totalMax = maxImportance0 + maxImportance1;
	if (!(totalMax > 0.f))
		return -1.f;

	const float probMax = maxImportance0 / totalMax;

	const float totalMin = minImportance0 + minImportance1;
	const float probMin = (totalMin > 0.f) ? (minImportance0 / totalMin) :
		// Without a smallest importance, the largest one decides alone if it
		// is zero for one of the two
		(.5f * (((maxImportance0 > 0.f) ? 1.f : 0.f) + ((maxImportance1 > 0.f) ? 0.f : 1.f)));

	const float prob = .5f * (probMax + probMin);

	// A pick close to 1/2 is made exactly 1/2: the Sobol sampler is Owen
	// scrambled, so a cut of the pick dimension at 1/2 keeps the samples of
	// each child stratified in the dimensions of the light sample, which a
	// cut at 0.46 or 0.54 loses (on a nearly noiseless direct-light
	// estimator, e.g. the two triangles of a Cornell box light, that loss is
	// 20% of the variance, more than any importance gain). The cost of the
	// snap is at most 4% of the direct-light variance at the edge of the
	// window; 0.05 misses part of the picks and 0.15 costs importance.
	return (fabs(prob - .5f) <= .1f) ? .5f : prob;
}

// The importance of a light outside the tree
OPENCL_FORCE_INLINE float LightTree_DistantLightImportance(__global const LightTreeDistantLight *light,
		const float3 n, const bool hasNormal) {
	if (!hasNormal)
		return light->irradiance;

	// The incidence: max(0, theta_i - theta_max) on both sides of the surface
	const float3 dir = MAKE_FLOAT3(light->dir[0], light->dir[1], light->dir[2]);
	const float cosThetaI = fabs(dot(n, dir));
	const float cosThetaMax = light->cosThetaMax;

	const float cosIncidence = (cosThetaI >= cosThetaMax) ? 1.f :
		(cosThetaI * cosThetaMax + LightTree_SinFromCos(cosThetaI) * LightTree_SinFromCos(cosThetaMax));

	return light->irradiance * cosIncidence;
}

// The probability of picking the tree instead of a distant light, -1 if no
// light can contribute
OPENCL_FORCE_INLINE float LightTree_TreeProbability(
		__global const LightTreeNode* restrict lightTreeNodes,
		__global const LightTreeDistantLight* restrict lightTreeDistantLights,
		const uint lightTreeDistantLightCount,
		const float3 p, const float3 n, const bool hasNormal,
		float *distantImportance) {
	// Only the tree
	if (lightTreeDistantLightCount == 0) {
		*distantImportance = 0.f;
		return lightTreeNodes ? 1.f : -1.f;
	}

	float treeMaxImportance = 0.f;
	float treeMinImportance = 0.f;
	if (lightTreeNodes)
		LightTree_NodeImportance(&lightTreeNodes[0], p, n, hasNormal,
				&treeMaxImportance, &treeMinImportance);

	float importance = 0.f;
	for (uint i = 0; i < lightTreeDistantLightCount; ++i)
		importance += LightTree_DistantLightImportance(&lightTreeDistantLights[i], n, hasNormal);
	*distantImportance = importance;

	return LightTree_FirstChildProbability(treeMaxImportance, treeMinImportance,
			importance, importance);
}

// The probability of the first child of an interior node
OPENCL_FORCE_INLINE float LightTree_NodeFirstChildProbability(
		__global const LightTreeNode* restrict lightTreeNodes, const uint nodeIndex,
		const float3 p, const float3 n, const bool hasNormal) {
	float maxImportance0, minImportance0, maxImportance1, minImportance1;
	LightTree_NodeImportance(&lightTreeNodes[nodeIndex + 1], p, n, hasNormal,
			&maxImportance0, &minImportance0);
	LightTree_NodeImportance(&lightTreeNodes[lightTreeNodes[nodeIndex].childOrLightIndex], p, n, hasNormal,
			&maxImportance1, &minImportance1);

	return LightTree_FirstChildProbability(maxImportance0, minImportance0,
			maxImportance1, minImportance1);
}

//------------------------------------------------------------------------------
// The learned slots (see LightTreeLearn)
//------------------------------------------------------------------------------

// The cell of the point p with the normal n
OPENCL_FORCE_INLINE uint LightTree_LearnCell(__global const LightTreeLearn* restrict learn,
		const float3 p, const float3 n, const bool hasNormal) {
	const int ix = (int)floor((p.x - learn->center[0]) * learn->invCellSize);
	const int iy = (int)floor((p.y - learn->center[1]) * learn->invCellSize);
	const int iz = (int)floor((p.z - learn->center[2]) * learn->invCellSize);
	const uint octant = hasNormal ?
		(((n.x < 0.f) ? 1u : 0u) | ((n.y < 0.f) ? 2u : 0u) | ((n.z < 0.f) ? 4u : 0u)) : 8u;

	uint h = ((uint)ix * 73856093u) ^ ((uint)iy * 19349663u) ^ ((uint)iz * 83492791u) ^ (octant * 2654435761u);
	h ^= h >> 13;
	h *= 0x5bd1e995u;
	h ^= h >> 15;

	return h & learn->cellMask;
}

// The probabilities of the slots at the point p from the bounds, as
// LightTree_FirstChildProbability() computes them for two children, normalized
// (all 0 when no slot can contribute)
OPENCL_FORCE_INLINE void LightTree_SlotBoundsProbabilities(
		__global const LightTreeNode* restrict lightTreeNodes,
		__global const LightTreeDistantLight* restrict lightTreeDistantLights,
		__global const LightTreeLearn* restrict learn,
		const float3 p, const float3 n, const bool hasNormal,
		float *prob) {
	const uint slotCount = learn->slotCount;
	const uint distantCount = learn->distantCount;

	float minImportance[LIGHTTREE_MAX_SLOTS];
	float totalMax = 0.f;
	float totalMin = 0.f;
	for (uint j = 0; j < slotCount; ++j) {
		if (j < distantCount) {
			prob[j] = LightTree_DistantLightImportance(&lightTreeDistantLights[j], n, hasNormal);
			minImportance[j] = prob[j];
		} else
			LightTree_NodeImportance(&lightTreeNodes[learn->cutNodes[j - distantCount]], p, n, hasNormal,
					&prob[j], &minImportance[j]);
		totalMax += prob[j];
		totalMin += minImportance[j];
	}

	// The average of the probabilities by the largest and by the smallest
	// importance, the largest alone when no slot has a smallest one
	const float maxScale = (totalMax > 0.f) ? (((totalMin > 0.f) ? .5f : 1.f) / totalMax) : 0.f;
	const float minScale = (totalMin > 0.f) ? (.5f / totalMin) : 0.f;
	for (uint j = 0; j < slotCount; ++j)
		prob[j] = prob[j] * maxScale + minImportance[j] * minScale;
}

// The probabilities of the slots at the point p (not normalized: the sum is
// returned, 0 when no slot can be picked): the ones from the bounds, scaled
// by the ratio of the contribution to the bounds probability the cell has
// learned for each slot, and blended with the plain ones by the weight the
// cell has earned
OPENCL_FORCE_INLINE float LightTree_SlotProbabilities(
		__global const LightTreeNode* restrict lightTreeNodes,
		__global const LightTreeDistantLight* restrict lightTreeDistantLights,
		__global const LightTreeLearn* restrict learn,
		__global const float* restrict learnCells,
		const float3 p, const float3 n, const bool hasNormal,
		float *prob, float *boundsProbOut) {
	const uint slotCount = learn->slotCount;
	LightTree_SlotBoundsProbabilities(lightTreeNodes, lightTreeDistantLights, learn,
			p, n, hasNormal, prob);
	if (boundsProbOut) {
		for (uint j = 0; j < slotCount; ++j)
			boundsProbOut[j] = prob[j];
	}

	__global const float* restrict cell = &learnCells[LightTree_LearnCell(learn, p, n, hasNormal) * (slotCount + 1)];
	float learnedWeight = cell[0];

	float learned[LIGHTTREE_MAX_SLOTS];
	float learnedSum = 0.f;
	for (uint j = 0; j < slotCount; ++j) {
		learned[j] = prob[j] * cell[1 + j];
		learnedSum += learned[j];
	}
	if (!(learnedSum > 0.f))
		learnedWeight = 0.f;
	const float learnedScale = (learnedSum > 0.f) ? (learnedWeight / learnedSum) : 0.f;
	const float boundsWeight = 1.f - learnedWeight;

	float sum = 0.f;
	for (uint j = 0; j < slotCount; ++j) {
		prob[j] = boundsWeight * prob[j] + learnedScale * learned[j];
		sum += prob[j];
	}

	return sum;
}

// Picks a light for the direct light sampling of the point p
OPENCL_FORCE_INLINE uint LightTree_SampleLights(
		__global const LightTreeNode* restrict lightTreeNodes,
		__global const LightTreeDistantLight* restrict lightTreeDistantLights,
		const uint lightTreeDistantLightCount,
		__global const LightTreeLearn* restrict learn,
		__global const float* restrict learnCells,
		const float3 p, const float3 n, const bool isVolume,
		float u, float *pdf, float *learnBoundsProb) {
	*pdf = 0.f;
	const bool hasNormal = !isVolume;

	float pmf;
	uint nodeIndex;
	if (learn) {
		// A slot first (learnBoundsProb, if given, receives the slot
		// probabilities from the bounds, for the learning)
		float prob[LIGHTTREE_MAX_SLOTS];
		const float sum = LightTree_SlotProbabilities(lightTreeNodes, lightTreeDistantLights,
				learn, learnCells, p, n, hasNormal, prob, learnBoundsProb);
		if (!(sum > 0.f))
			return NULL_INDEX;

		const float target = u * sum;
		float below = 0.f;
		uint slot = NULL_INDEX;
		for (uint j = 0; j < learn->slotCount; ++j) {
			if (prob[j] > 0.f) {
				slot = j;
				if (target < below + prob[j])
					break;
				below += prob[j];
			}
		}
		if (slot == NULL_INDEX)
			return NULL_INDEX;
		// The last slot with a probability if the rounding of the sum left
		// the target above it
		below = fmin(below, target);

		pmf = prob[slot] / sum;
		u = fmin((target - below) / prob[slot], 0x1.fffffep-1f);

		if (slot < learn->distantCount) {
			*pdf = pmf;
			return lightTreeDistantLights[slot].lightIndex;
		}
		nodeIndex = learn->cutNodes[slot - learn->distantCount];
	} else {
		float distantImportance;
		const float treeProb = LightTree_TreeProbability(lightTreeNodes,
				lightTreeDistantLights, lightTreeDistantLightCount,
				p, n, hasNormal, &distantImportance);
		if (treeProb < 0.f)
			return NULL_INDEX;

		if (u >= treeProb) {
			// A distant light
			const float distantProb = 1.f - treeProb;
			const float target = fmin((u - treeProb) / distantProb, 1.f) * distantImportance;

			float sum = 0.f;
			uint lastIndex = NULL_INDEX;
			float lastImportance = 0.f;
			for (uint i = 0; i < lightTreeDistantLightCount; ++i) {
				const float importance = LightTree_DistantLightImportance(&lightTreeDistantLights[i], n, hasNormal);
				if (importance > 0.f) {
					lastIndex = i;
					lastImportance = importance;

					sum += importance;
					if (target < sum)
						break;
				}
			}

			// The last light with an importance if the float rounding of the
			// sum left the target above it
			if (lastIndex == NULL_INDEX)
				return NULL_INDEX;
			*pdf = distantProb * lastImportance / distantImportance;
			return lightTreeDistantLights[lastIndex].lightIndex;
		}

		// The tree
		pmf = treeProb;
		u = fmin(u / treeProb, 0x1.fffffep-1f);
		nodeIndex = 0;
	}

	for (;;) {
		__global const LightTreeNode *node = &lightTreeNodes[nodeIndex];
		if (node->flags & LIGHTTREE_NODE_LEAF) {
			*pdf = pmf;
			return node->childOrLightIndex;
		}

		const float prob0 = LightTree_NodeFirstChildProbability(lightTreeNodes, nodeIndex,
				p, n, hasNormal);
		if (prob0 < 0.f)
			return NULL_INDEX;

		if (u < prob0) {
			u = fmin(u / prob0, 0x1.fffffep-1f);
			pmf *= prob0;
			nodeIndex = nodeIndex + 1;
		} else {
			const float prob1 = 1.f - prob0;
			u = fmin((u - prob0) / prob1, 0x1.fffffep-1f);
			pmf *= prob1;
			nodeIndex = node->childOrLightIndex;
		}
	}
}

// The probability of LightTree_SampleLights() picking the light at the point p
OPENCL_FORCE_INLINE float LightTree_SampleLightPdf(
		__global const LightTreeNode* restrict lightTreeNodes,
		__global const LightTreeDistantLight* restrict lightTreeDistantLights,
		const uint lightTreeDistantLightCount,
		__global const uint* restrict lightTreeLightIndex,
		__global const LightTreeLearn* restrict learn,
		__global const float* restrict learnCells,
		const float3 p, const float3 n, const bool isVolume,
		const uint lightIndex) {
	const uint entry = lightTreeLightIndex[lightIndex];
	if (entry == NULL_INDEX)
		return 0.f;

	const bool hasNormal = !isVolume;

	if (learn) {
		float prob[LIGHTTREE_MAX_SLOTS];
		const float sum = LightTree_SlotProbabilities(lightTreeNodes, lightTreeDistantLights,
				learn, learnCells, p, n, hasNormal, prob, NULL);
		if (!(sum > 0.f))
			return 0.f;

		if (entry & LIGHTTREE_DISTANT_LIGHT)
			return prob[entry & ~LIGHTTREE_DISTANT_LIGHT] / sum;

		// Walk up from the leaf to its slot
		float pmf = 1.f;
		uint nodeIndex = entry;
		for (;;) {
			const uint slot = lightTreeNodes[nodeIndex].flags >> LIGHTTREE_NODE_SLOT_SHIFT;
			if (slot)
				return pmf * prob[slot - 1] / sum;

			const uint parentIndex = lightTreeNodes[nodeIndex].parentIndex;
			const float prob0 = LightTree_NodeFirstChildProbability(lightTreeNodes, parentIndex,
					p, n, hasNormal);
			if (prob0 < 0.f)
				return 0.f;

			pmf *= (nodeIndex == parentIndex + 1) ? prob0 : (1.f - prob0);
			nodeIndex = parentIndex;
		}
	}

	float distantImportance;
	const float treeProb = LightTree_TreeProbability(lightTreeNodes,
			lightTreeDistantLights, lightTreeDistantLightCount,
			p, n, hasNormal, &distantImportance);
	if (treeProb < 0.f)
		return 0.f;

	if (entry & LIGHTTREE_DISTANT_LIGHT) {
		if (!(distantImportance > 0.f))
			return 0.f;

		const float importance = LightTree_DistantLightImportance(
				&lightTreeDistantLights[entry & ~LIGHTTREE_DISTANT_LIGHT], n, hasNormal);

		return (1.f - treeProb) * importance / distantImportance;
	}

	// Walk up from the leaf
	float pmf = treeProb;
	uint nodeIndex = entry;
	for (;;) {
		const uint parentIndex = lightTreeNodes[nodeIndex].parentIndex;
		if (parentIndex == NULL_INDEX)
			return pmf;

		const float prob0 = LightTree_NodeFirstChildProbability(lightTreeNodes, parentIndex,
				p, n, hasNormal);
		if (prob0 < 0.f)
			return 0.f;

		pmf *= (nodeIndex == parentIndex + 1) ? prob0 : (1.f - prob0);
		nodeIndex = parentIndex;
	}
}

#if defined(SLG_OPENCL_KERNEL)
//------------------------------------------------------------------------------
// The learning updates on the GPU (LightStrategyLightTree::AddSample() does
// them on the CPU): a pick, with the slot probabilities from the bounds it
// was made with, and the contribution of a light that was visible
//------------------------------------------------------------------------------

OPENCL_FORCE_INLINE void LightTree_LearnAddPick(
		__global const LightTreeLearn* restrict learn,
		__global float* restrict learnBoundsSums,
		__global float* restrict learnCounts,
		const float3 p, const float3 n, const bool hasNormal,
		const float *boundsProb) {
	const uint cell = LightTree_LearnCell(learn, p, n, hasNormal);
	// A cell that has learned enough skips the atomics
	if (learnCounts[cell] >= learn->stopCount)
		return;
	AtomicAdd(&learnCounts[cell], 1.f);

	__global float* restrict sums = &learnBoundsSums[cell * learn->slotCount];
	for (uint j = 0; j < learn->slotCount; ++j) {
		if (boundsProb[j] > 0.f)
			AtomicAdd(&sums[j], boundsProb[j]);
	}
}

OPENCL_FORCE_INLINE void LightTree_LearnAddContribution(
		__global const LightTreeNode* restrict lightTreeNodes,
		__global const uint* restrict lightTreeLightIndex,
		__global const LightTreeLearn* restrict learn,
		__global float* restrict learnSums,
		__global const float* restrict learnCounts,
		const float3 p, const float3 n, const bool hasNormal,
		const uint lightIndex, const float pickPdf, const float contribution) {
	if (!(contribution > 0.f) || !(pickPdf > 0.f))
		return;
	const uint entry = lightTreeLightIndex[lightIndex];
	if (entry == NULL_INDEX)
		return;

	uint slot;
	if (entry & LIGHTTREE_DISTANT_LIGHT)
		slot = entry & ~LIGHTTREE_DISTANT_LIGHT;
	else {
		uint nodeIndex = entry;
		for (;;) {
			const uint s = lightTreeNodes[nodeIndex].flags >> LIGHTTREE_NODE_SLOT_SHIFT;
			if (s) {
				slot = s - 1;
				break;
			}
			nodeIndex = lightTreeNodes[nodeIndex].parentIndex;
			if (nodeIndex == NULL_INDEX)
				return;
		}
	}

	const uint cell = LightTree_LearnCell(learn, p, n, hasNormal);
	// As LightTree_LearnAddPick() (the picks still in flight when the cell
	// stops add their contribution or not: it only steers the probabilities)
	if (learnCounts[cell] >= learn->stopCount)
		return;
	AtomicAdd(&learnSums[cell * learn->slotCount + slot], contribution / pickPdf);
}
#endif
