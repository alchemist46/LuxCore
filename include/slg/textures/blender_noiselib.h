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

// Dade - sources imported from Blender with authors permission
//
// The enums of the Blender textures. The noise functions are in
// texture_blender_noise_funcs.cl and texture_blender_noise_funcs2.cl.

#ifndef _SLG_BLENDER_NOISELIB_H
#define	_SLG_BLENDER_NOISELIB_H

namespace slg {
	
namespace blender {

typedef enum {
	TEX_SIN, TEX_SAW, TEX_TRI
} BlenderNoiseBase;

typedef enum {
	TEX_SOFT, TEX_SHARP, TEX_SHARPER
} BlenderMarbleType;

typedef enum {
	TEX_MULTIFRACTAL, TEX_RIDGED_MULTIFRACTAL, TEX_HYBRID_MULTIFRACTAL, TEX_FBM, TEX_HETERO_TERRAIN
} BlenderMusgraveType;

typedef enum {
	TEX_PLASTIC, TEX_WALL_IN, TEX_WALL_OUT
} BlenderStucciType;

typedef enum {
	BANDS, RINGS, BANDNOISE, RINGNOISE
} BlenderWoodType;

typedef enum {
	TEX_LIN, TEX_QUAD, TEX_EASE, TEX_DIAG, TEX_SPHERE, TEX_HALO, TEX_RAD
} ProgressionType;

typedef enum { 
	ACTUAL_DISTANCE, DISTANCE_SQUARED, MANHATTAN, CHEBYCHEV, MINKOWSKI_HALF, 
	MINKOWSKI_FOUR, MINKOWSKI
} DistanceMetric;

typedef enum {
    BLENDER_ORIGINAL, ORIGINAL_PERLIN, IMPROVED_PERLIN,
    VORONOI_F1, VORONOI_F2, VORONOI_F3, VORONOI_F4, VORONOI_F2_F1,
    VORONOI_CRACKLE, CELL_NOISE
} BlenderNoiseBasis;

} // namespace blender

} // namespace slg

#endif	/* _SLG_BLENDER_NOISELIB_H */
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
