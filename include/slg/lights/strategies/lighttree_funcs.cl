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
// quad differ.
OPENCL_FORCE_INLINE float LightTree_TriangleImportance(__global const LightTreeNode *node,
		const float3 p, const float3 n, const bool hasNormal) {
	const float3 v0 = MAKE_FLOAT3(node->bboxMin[0], node->bboxMin[1], node->bboxMin[2]);
	const float3 v1 = MAKE_FLOAT3(node->bboxMax[0], node->bboxMax[1], node->bboxMax[2]);
	const float3 v2 = MAKE_FLOAT3(node->axis[0], node->axis[1], node->axis[2]);

	const float3 crossN = cross(v1 - v0, v2 - v0);
	const float doubleArea = length(crossN);
	if (!(doubleArea > 0.f))
		return 0.f;
	const float3 triN = crossN / doubleArea;

	// The emitting side
	const float side = dot(triN, p - v0);
	if (!(node->flags & LIGHTTREE_NODE_TWOSIDED) && !(side > 0.f))
		return 0.f;

	const float3 toCentroid = (v0 + v1 + v2) * (1.f / 3.f) - p;
	const float centroidDistance2 = dot(toCentroid, toCentroid);
	if (!(centroidDistance2 > 0.f))
		return 0.f;
	const float centroidDistance = sqrt(centroidDistance2);
	const float3 wi = toCentroid / centroidDistance;

	// The solid angle: exact close to the triangle, the projected area far
	// from it (where the float precision of the exact one breaks down)
	const float area = .5f * doubleArea;
	const float solidAngle = (centroidDistance2 < 64.f * area) ?
		SphericalTriangle_SolidAngle(v0, v1, v2, p) :
		(area * fabs(dot(triN, wi)) / centroidDistance2);

	float cosIncidence = 1.f;
	if (hasNormal) {
		// theta_u: the half angle of the cone of directions from p that holds
		// the bounding sphere of the triangle around its centroid
		const float3 centroid = p + toCentroid;
		const float radius2 = fmax(fmax(
				dot(v0 - centroid, v0 - centroid),
				dot(v1 - centroid, v1 - centroid)),
				dot(v2 - centroid, v2 - centroid));
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
	// The direction from p to the centroid (any direction at the centroid)
	const float3 wi = (centroidDistance > 0.f) ? (toCentroid / centroidDistance) : MAKE_FLOAT3(0.f, 0.f, 1.f);

	// The radius of the bounding sphere of the box
	const float radius = .5f * length(bboxMax - bboxMin);

	// The distance used for the largest contribution, clamped to the bounding
	// sphere (it would diverge at the centroid), and the distance of the
	// farthest corner for the smallest one
	const float minDistance = fmax(centroidDistance, radius);
	const float maxDistance = length(fmax(p - bboxMin, bboxMax - p));
	const float minDistance2 = fmax(minDistance * minDistance, 1e-12f);
	const float maxDistance2 = fmax(maxDistance * maxDistance, 1e-12f);

	// theta_u: the half angle of the cone of directions from p that holds the
	// bounding sphere of the box (pbrt-v4; Cycles bounds the 8 corners, which is
	// tighter and costs 8 normalizations)
	const float cosThetaU = (centroidDistance > radius) ?
		sqrt(fmax(0.f, 1.f - (radius * radius) / (centroidDistance * centroidDistance))) : -1.f;
	const float sinThetaU = LightTree_SinFromCos(cosThetaU);

	// theta: the angle between the axis of the cone and the direction from
	// the centroid to p
	const float3 axis = MAKE_FLOAT3(node->axis[0], node->axis[1], node->axis[2]);
	float cosTheta = -dot(axis, wi);
	if (node->flags & LIGHTTREE_NODE_TWOSIDED)
		cosTheta = fabs(cosTheta);
	const float sinTheta = LightTree_SinFromCos(cosTheta);

	const float cosThetaO = node->cosThetaO;
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
	// never more than 90 degrees)
	if ((cosTheta >= 0.f) && (cosThetaU >= 0.f) && (cosThetaO >= 0.f)) {
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

// Picks a light for the direct light sampling of the point p
OPENCL_FORCE_INLINE uint LightTree_SampleLights(
		__global const LightTreeNode* restrict lightTreeNodes,
		__global const LightTreeDistantLight* restrict lightTreeDistantLights,
		const uint lightTreeDistantLightCount,
		const float3 p, const float3 n, const bool isVolume,
		float u, float *pdf) {
	*pdf = 0.f;
	const bool hasNormal = !isVolume;

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

		// The last light with an importance if the float rounding of the sum
		// left the target above it
		if (lastIndex == NULL_INDEX)
			return NULL_INDEX;
		*pdf = distantProb * lastImportance / distantImportance;
		return lightTreeDistantLights[lastIndex].lightIndex;
	}

	// The tree
	float pmf = treeProb;
	u = fmin(u / treeProb, 0x1.fffffep-1f);

	uint nodeIndex = 0;
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
		const float3 p, const float3 n, const bool isVolume,
		const uint lightIndex) {
	const uint entry = lightTreeLightIndex[lightIndex];
	if (entry == NULL_INDEX)
		return 0.f;

	const bool hasNormal = !isVolume;

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
