#line 2 "lighttree_types.cl"

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
// The light tree (see lighttree_funcs.cl)
//------------------------------------------------------------------------------

#define LIGHTTREE_NODE_LEAF 1u
// The lights emit on both sides of their normals
#define LIGHTTREE_NODE_TWOSIDED 2u
// A leaf of a triangle light: bboxMin, bboxMax and axis hold its vertices,
// in the order of its emitting normal
#define LIGHTTREE_NODE_TRIANGLE 4u

// A node of the tree. The nodes are stored depth first: the first child of an
// interior node is the next node.
typedef struct {
	// The bounding box of the lights below the node
	float bboxMin[3], bboxMax[3];
	// The radius of the bounding sphere of the box around its center (of a
	// triangle leaf: of its vertices around their centroid)
	float radius;
	// The bounding cone of their emission: the normals are within theta_o of
	// the axis and each light emits up to theta_e away from its normal
	float axis[3];
	float cosThetaO, cosThetaE;
	// The sum of the maximum radiant intensities of the lights (luminance)
	float phi;
	// An interior node: the index of the second child. A leaf: the light index.
	unsigned int childOrLightIndex;
	// NULL_INDEX for the root
	unsigned int parentIndex;
	unsigned int flags;
} LightTreeNode;

// A light outside the tree, whose importance doesn't depend on the distance:
// the environment, sun and distant lights, and the lasers
typedef struct {
	// The direction of the light and the cosine of the half angle of the cone
	// of directions it arrives from (-1 for an environment light: all of them)
	float dir[3];
	float cosThetaMax;
	// The irradiance at normal incidence (luminance)
	float irradiance;
	unsigned int lightIndex;
} LightTreeDistantLight;

// The entry of a light in the light index of the tree: its leaf node, the
// index of a distant light with this flag, or NULL_INDEX for a light the
// strategy never picks
#define LIGHTTREE_DISTANT_LIGHT 0x80000000u
