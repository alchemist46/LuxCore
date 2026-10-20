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

#ifndef _SLG_PATHDEPTHINFO_H
#define	_SLG_PATHDEPTHINFO_H

#include <ostream>

#include "slg/slg.h"
#include "slg/bsdf/bsdf.h"

namespace slg {

// OpenCL data types
namespace ocl {
#include "slg/utils/pathdepthinfo_types.cl"
}

class PathDepthInfo {
public:
	PathDepthInfo();
	~PathDepthInfo() { }

	void IncDepths(const BSDFEvent event);
	// The kind of an event (or of the possible events of a vertex) for the
	// path depth: a glossy or specular reflection on the back side of a
	// transmissive surface (an internal reflection of a glass, the total
	// internal reflection) is part of the way through the object, so it
	// counts as a transmission
	static BSDFEvent GetDepthEvent(const BSDF &bsdf, const BSDFEvent event);
	bool IsLastPathVertex(const PathDepthInfo &maxPathDepth,
		const BSDFEvent event) const;
	// An eye path entering a subsurface material starts the random walk
	// while the subsurface budget lasts (the walk is the scattering of the
	// vertex, as a diffuse bounce is for the diffuse budget)
	bool CanStartSubsurfaceWalk(const PathDepthInfo &maxPathDepth) const;
	u_int GetRRDepth() const;

	// The budgets, as Arnold and Cycles:
	//  - transmissionDepth: a glossy or specular transmission (a refraction,
	//    a thin wall);
	//  - glossyDepth: a glossy or specular reflection (rough or sharp);
	//  - diffuseDepth: a diffuse event (the exit of a subsurface walk too);
	//  - subsurfaceDepth: the subsurface random walks.
	// rouletteDepth counts the diffuse events, the walks and the glossy
	// reflections (not the specular ones), for the Russian roulette.
	u_int depth, diffuseDepth, glossyDepth, transmissionDepth, subsurfaceDepth,
			rouletteDepth;
};

inline std::ostream &operator<<(std::ostream &os, const PathDepthInfo &pdi) {
	os << "PathDepthInfo[" << pdi.depth << ", " << pdi.diffuseDepth << ", " << pdi.glossyDepth << ", " <<
			pdi.transmissionDepth << ", " << pdi.subsurfaceDepth << ", " << pdi.rouletteDepth << "]";
	return os;
}

}

#endif	/* _SLG_PATHDEPTHINFO_H */
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
