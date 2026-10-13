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

#if !defined(LUXRAYS_DISABLE_HIP)

#include "luxrays/accelerators/hiprtaccel.h"
#include "luxrays/utils/utils.h"
#include "luxrays/core/context.h"

using namespace std;

namespace luxrays {

// HiprtAccel Method Definitions

HiprtAccel::HiprtAccel(const Context & context) : ctx(context) {
	initialized = false;
}

HiprtAccel::~HiprtAccel() {
}

void HiprtAccel::Init(const deque<const Mesh * > &ms, const u_longlong totVert,
		const u_longlong totTri) {
	assert (!initialized);

	meshes = ms;
	totalVertexCount = totVert;
	totalTriangleCount = totTri;

	// Handle the empty DataSet case
	if (totalTriangleCount == 0) {
		LR_LOG(ctx, "Empty HIPRT accelerator");
		initialized = true;

		return;
	}

	initialized = true;
}

bool HiprtAccel::Intersect(const Ray *initialRay, RayHit *rayHit) const {
	throw runtime_error("Called HiprtAccel::Intersect()");
}

}

#endif
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
