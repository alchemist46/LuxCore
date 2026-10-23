#line 2 "hitpoint_types.cl"

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

// The values of HitPoint::subsurface:
//  HITPOINT_SSS_SURFACE: the subsurface material is a plain surface (light
//                        paths, inside a Mix, ...), with a Lambertian base
//  HITPOINT_SSS_ENTRY:   an eye path vertex: the base lobe enters the object and
//                        the path tracer runs the walk
//  HITPOINT_SSS_EXIT:    the exit point of a walk: a Lambertian transmission
//                        out of the object
#define HITPOINT_SSS_SURFACE 0
#define HITPOINT_SSS_ENTRY 1
#define HITPOINT_SSS_EXIT 2

typedef struct {
	// The incoming direction. It is the eyeDir when fromLight = false and
	// lightDir when fromLight = true
	Vector fixedDir;
	Point p;
	Normal geometryN;
	Normal interpolatedN;
	Normal shadeN;

	UV defaultUV;

	// Note: dpdu and dpdv are orthogonal to shading normal (i.e not geometry normal)
	Vector dpdu, dpdv;
	Normal dndu, dndv;

	// Mesh information
	unsigned int meshIndex;
	unsigned int triangleIndex;
	float triangleBariCoord1, triangleBariCoord2;

	// passThroughEvent can be stored here in a path state even before of
	// BSDF initialization (while tracing the next path vertex ray)
	float passThroughEvent;

	// The time of the ray: the local to world transformation of the mesh is
	// read from the mesh when needed (see HitPoint_GetLocalToWorld())
	float time;

	// Interior and exterior volume (this includes volume priority system
	// computation and scene default world volume)
	unsigned int interiorVolumeIndex, exteriorVolumeIndex;
	// Material code (i.e. glass, etc.) doesn't have access to materials list
	// so I use HitPoint to carry texture index information
	unsigned int interiorIorTexIndex, exteriorIorTexIndex;

	unsigned int objectID;

	int intoObject, throughShadowTransparency;
	// Path regularization (see PathTracer): the floor of the GGX alphas of
	// the materials at this vertex, 0 when the vertex is not regularized
	float regularization;
	// Random-walk subsurface scattering (see SubsurfaceMaterial): the role of
	// the vertex, HITPOINT_SSS_SURFACE unless the eye path tracer sets it
	int subsurface;
	// The dispersion (see HitPoint::wavelength): the wavelength (nm) of the
	// path or 0, and whether the path already has its colour
	float wavelength;
	int dispersionTinted;
	// The slots of the material bump cache filled at this vertex (see
	// Material_BumpSubMaterial()), cleared by BSDF_Init()
	unsigned int bumpCacheValidSlots;
} HitPoint;
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
