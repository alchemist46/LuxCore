#line 2 "pathinfo_funcs.cl"

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

//------------------------------------------------------------------------------
// PathDepthInfo
//------------------------------------------------------------------------------

OPENCL_FORCE_INLINE void PathDepthInfo_Init(__global PathDepthInfo *depthInfo) {
	depthInfo->depth = 0;
	depthInfo->diffuseDepth = 0;
	depthInfo->glossyDepth = 0;
	depthInfo->transmissionDepth = 0;
	depthInfo->subsurfaceDepth = 0;
	depthInfo->rouletteDepth = 0;
}

// See PathDepthInfo::IncDepths()
OPENCL_FORCE_INLINE void PathDepthInfo_IncDepths(__global PathDepthInfo *depthInfo, const BSDFEvent event) {
	++(depthInfo->depth);
	if (event & SUBSURFACE) {
		++(depthInfo->subsurfaceDepth);
		++(depthInfo->rouletteDepth);
	} else if ((event & TRANSMIT) && (event & (GLOSSY | SPECULAR)))
		++(depthInfo->transmissionDepth);
	else if (event & DIFFUSE) {
		++(depthInfo->diffuseDepth);
		++(depthInfo->rouletteDepth);
	} else if (event & (GLOSSY | SPECULAR)) {
		++(depthInfo->glossyDepth);
		if (event & GLOSSY)
			++(depthInfo->rouletteDepth);
	}
}

OPENCL_FORCE_INLINE bool PathDepthInfo_IsLastPathVertex(__global PathDepthInfo *depthInfo,
		__constant const PathDepthInfo* restrict maxDepthInfo, const BSDFEvent event) {
	return (depthInfo->depth + 1 >= maxDepthInfo->depth) ||
			((event & DIFFUSE) && (depthInfo->diffuseDepth + 1 >= maxDepthInfo->diffuseDepth)) ||
			((event & (GLOSSY | SPECULAR)) && (event & REFLECT) && (depthInfo->glossyDepth + 1 >= maxDepthInfo->glossyDepth)) ||
			((event & (GLOSSY | SPECULAR)) && (event & TRANSMIT) &&
				(depthInfo->transmissionDepth + 1 >= maxDepthInfo->transmissionDepth));
}

OPENCL_FORCE_INLINE bool PathDepthInfo_CanStartSubsurfaceWalk(__global PathDepthInfo *depthInfo,
		__constant const PathDepthInfo* restrict maxDepthInfo) {
	return (depthInfo->subsurfaceDepth + 1 < maxDepthInfo->subsurfaceDepth);
}

OPENCL_FORCE_INLINE bool PathDepthInfo_CheckComponentDepths(
		__constant const PathDepthInfo* restrict maxDepthInfo,
		const BSDFEvent component) {
	return ((maxDepthInfo->diffuseDepth > 0) && (component & DIFFUSE)) ||
			((maxDepthInfo->glossyDepth > 0) && (component & (GLOSSY | SPECULAR)) && (component & REFLECT)) ||
			((maxDepthInfo->transmissionDepth > 0) && (component & TRANSMIT) && (component & (GLOSSY | SPECULAR)));
}

OPENCL_FORCE_INLINE uint PathDepthInfo_GetRRDepth(__global PathDepthInfo *depthInfo) {
	return depthInfo->rouletteDepth;
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
