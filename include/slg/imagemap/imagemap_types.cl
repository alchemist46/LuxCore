#line 2 "imagemap_types.cl"

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

typedef enum {
	BYTE, HALF, FLOAT
} ImageMapStorageType;

typedef enum {
	WRAP_REPEAT,
	WRAP_BLACK,
	WRAP_WHITE,
	WRAP_CLAMP
} ImageWrapType;

typedef enum {
	FILTER_NEAREST,
	FILTER_LINEAR
} ImageFilterType;

// A 64bit unsigned integer, 64bit aligned (for OpenCL)
#if defined(SLG_OPENCL_KERNEL)
// CUDA unsigned long has a 32bit size so we need different definitions
// between CUDA and OpenCL
#if defined(LUXRAYS_CUDA_DEVICE)
typedef unsigned long long ImageMapULong;
#elif defined(LUXRAYS_OPENCL_DEVICE)
typedef unsigned long ImageMapULong;
#else
#error "Unsupported device in ImageMap struct"
#endif
#else
#if defined(LUXRAYS_ENABLE_OPENCL)
typedef cl_ulong ImageMapULong;
#else
// In this, case cl_ulong is not defined. The type, in this case, doesn't
// really matter because this structure is not used at all.
typedef unsigned long long ImageMapULong;
#endif
#endif

typedef struct {
	ImageMapStorageType storageType;
	ImageWrapType wrapType;
	ImageFilterType filterType;
	unsigned int channelCount, width, height;
	unsigned int pageIndex;
	ImageMapULong pixelsIndex;
	// The HIP texture objects of the image map (hipTextureObject_t), 0 when
	// there are none: the lookups then use the texture hardware instead of
	// the pixels at pixelsIndex, which may not be uploaded. textureObject has
	// the filter of the image map, texelTextureObject reads single texels.
	ImageMapULong textureObject, texelTextureObject;
} ImageMap;

//------------------------------------------------------------------------------
// Some macro trick in order to have more readable code
//------------------------------------------------------------------------------

#if defined(SLG_OPENCL_KERNEL)

#define IMAGEMAPS_PARAM_DECL , __global const ImageMap* restrict imageMapDescs, __global const float* restrict* restrict imageMapBuff
#define IMAGEMAPS_PARAM , imageMapDescs, imageMapBuff

#endif
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
