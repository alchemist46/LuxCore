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

#include "slg/utils/pathdepthinfo.h"
#include "slg/film/sampleresult.h"

using namespace std;
using namespace luxrays;
using namespace slg;

//------------------------------------------------------------------------------
// PathDepthInfo
//------------------------------------------------------------------------------

PathDepthInfo::PathDepthInfo() {
	depth = 0;
	diffuseDepth = 0;
	glossyDepth = 0;
	specularDepth = 0;
	transmissionDepth = 0;
}

BSDFEvent PathDepthInfo::GetDepthEvent(const BSDF &bsdf, const BSDFEvent event) {
	if (!bsdf.IsVolume() && !bsdf.hitPoint.intoObject && (bsdf.GetEventTypes() & TRANSMIT) &&
			(event & (GLOSSY | SPECULAR)) && (event & REFLECT))
		return (event & ~REFLECT) | TRANSMIT;

	return event;
}

void PathDepthInfo::IncDepths(const BSDFEvent event) {
	++depth;
	if ((event & TRANSMIT) && (event & (GLOSSY | SPECULAR)))
		++transmissionDepth;
	else if (event & DIFFUSE)
		++diffuseDepth;
	else if (event & GLOSSY)
		++glossyDepth;
	else if (event & SPECULAR)
		++specularDepth;
}

// The possible events are a mask: a glossy or specular kind with TRANSMIT may
// transmit, with REFLECT reflect
bool PathDepthInfo::IsLastPathVertex(const PathDepthInfo &maxPathDepth, const BSDFEvent possibleEvents) const {
	return (depth + 1 >= maxPathDepth.depth) ||
			((possibleEvents & DIFFUSE) && (diffuseDepth + 1 >= maxPathDepth.diffuseDepth)) ||
			((possibleEvents & GLOSSY) && (possibleEvents & REFLECT) && (glossyDepth + 1 >= maxPathDepth.glossyDepth)) ||
			((possibleEvents & SPECULAR) && (possibleEvents & REFLECT) && (specularDepth + 1 >= maxPathDepth.specularDepth)) ||
			((possibleEvents & (GLOSSY | SPECULAR)) && (possibleEvents & TRANSMIT) &&
				(transmissionDepth + 1 >= maxPathDepth.transmissionDepth));
}

u_int PathDepthInfo::GetRRDepth() const {
	return diffuseDepth + glossyDepth;
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
