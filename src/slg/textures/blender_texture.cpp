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

#include <cmath>
#include <limits>

#include "slg/core/sdl.h"
#include "slg/bsdf/bsdf.h"
#include "slg/textures/blender_texture.h"

// The Blender textures are evaluated with the kernel sources of the GPU
// (see luxrays/devices/cpudevice_oclemul.h)
namespace slg { namespace ocl { namespace {
#include "luxrays/devices/cpudevice_oclemul.h"
#include "luxrays/core/utils_funcs.cl"
#include "slg/textures/texture_blender_noise_funcs.cl"
#include "slg/textures/texture_blender_noise_funcs2.cl"
#include "slg/textures/texture_blender_funcs.cl"
} } }

using namespace luxrays;
using namespace slg;
using namespace slg::blender;

// The enums of the textures are cast to the kernel ones
static_assert(int(TEX_RAD) == int(slg::ocl::TEX_RAD));
static_assert(int(TEX_SHARPER) == int(slg::ocl::TEX_SHARPER));
static_assert(int(TEX_HETERO_TERRAIN) == int(slg::ocl::TEX_HETERO_TERRAIN));
static_assert(int(TEX_WALL_OUT) == int(slg::ocl::TEX_WALL_OUT));
static_assert(int(RINGNOISE) == int(slg::ocl::RINGNOISE));
static_assert(int(TEX_TRI) == int(slg::ocl::TEX_TRI));
static_assert(int(MINKOWSKI) == int(slg::ocl::MINKOWSKI));
static_assert(int(CELL_NOISE) == int(slg::ocl::CELL_NOISE));

static inline slg::ocl::float3 ToFloat3(const Point &p) {
	return slg::ocl::make_float3(p.x, p.y, p.z);
}

//------------------------------------------------------------------------------
// Blender blend texture
//------------------------------------------------------------------------------

BlenderBlendTexture::BlenderBlendTexture(TextureMapping3DUPtr&& mp, const std::string ptype,
										 const bool direction, float bright, float contrast) :
		mapping(std::move(mp)), type(TEX_LIN), direction(direction), bright(bright), contrast(contrast) {

		if (ptype == "linear") {type = TEX_LIN;}
		else if (ptype == "quadratic") {type = TEX_QUAD;}
		else if (ptype == "easing") {type = TEX_EASE;}
		else if (ptype == "diagonal") {type = TEX_DIAG;}
		else if (ptype == "spherical") {type = TEX_SPHERE;}
		else if (ptype == "halo") {type = TEX_HALO;}
		else if (ptype == "radial") {type = TEX_RAD;};
}

float BlenderBlendTexture::GetFloatValue(const HitPoint &hitPoint) const {
	return ocl::BlenderBlendTexture_Evaluate(ToFloat3(mapping->Map(hitPoint)),
			static_cast<ocl::ProgressionType>(type), direction, contrast, bright);
}

Spectrum BlenderBlendTexture::GetSpectrumValue(const HitPoint &hitPoint) const {
	return Spectrum(GetFloatValue(hitPoint));
}

PropertiesUPtr BlenderBlendTexture::ToProperties(const ImageMapCache &imgMapCache, const bool useRealFileName) const {
	auto props = std::make_unique<Properties>();

	std::string progressiontype;
	switch(type) {
		default:
		case TEX_LIN:
			progressiontype = "linear";
			break;
		case TEX_QUAD:
			progressiontype = "quadratic";
			break;
		case TEX_EASE:
			progressiontype = "easing";
			break;
		case TEX_DIAG:
			progressiontype = "diagonal";
			break;
		case TEX_SPHERE:
			progressiontype = "spherical";
			break;
		case TEX_HALO:
			progressiontype = "halo";
			break;
		case TEX_RAD:
			progressiontype = "radial";
			break;
	}
	std::string directiontype = "horizontal";
	if(direction) directiontype = "vertical";

	const std::string name = GetName();

	props->Set(Property("scene.textures." + name + ".type")("blender_blend"));
	props->Set(Property("scene.textures." + name + ".progressiontype")(progressiontype));
	props->Set(Property("scene.textures." + name + ".direction")(directiontype));
	props->Set(Property("scene.textures." + name + ".bright")(bright));
	props->Set(Property("scene.textures." + name + ".contrast")(contrast));
	props->Set(mapping->ToProperties("scene.textures." + name + ".mapping"));

	return props;
}

//------------------------------------------------------------------------------
// Blender clouds texture
//------------------------------------------------------------------------------

BlenderCloudsTexture::BlenderCloudsTexture(TextureMapping3DUPtr&& mp, const std::string &pnoisebasis, const float noisesize, const int noisedepth,
		bool hard, float bright, float contrast) :
		mapping(std::move(mp)), noisebasis(BLENDER_ORIGINAL), noisedepth(noisedepth), noisesize(noisesize),
		hard(hard), bright(bright), contrast(contrast) {

	if(pnoisebasis == "blender_original") {
		noisebasis = BLENDER_ORIGINAL;
	} else if(pnoisebasis == "original_perlin") {
		noisebasis = ORIGINAL_PERLIN;
	} else if(pnoisebasis == "improved_perlin") {
		noisebasis = IMPROVED_PERLIN;
	} else if(pnoisebasis == "voronoi_f1") {
		noisebasis = VORONOI_F1;
	} else if(pnoisebasis == "voronoi_f2") {
		noisebasis = VORONOI_F2;
	} else if(pnoisebasis == "voronoi_f3") {
		noisebasis = VORONOI_F3;
	} else if(pnoisebasis == "voronoi_f4") {
		noisebasis = VORONOI_F4;
	} else if(pnoisebasis == "voronoi_f2_f1") {
		noisebasis = VORONOI_F2_F1;
	} else if(pnoisebasis == "voronoi_crackle") {
		noisebasis = VORONOI_CRACKLE;
	} else if(pnoisebasis == "cell_noise") {
		noisebasis = CELL_NOISE;
	};
}

float BlenderCloudsTexture::GetFloatValue(const HitPoint &hitPoint) const {
	return ocl::BlenderCloudsTexture_Evaluate(ToFloat3(mapping->Map(hitPoint)),
			static_cast<ocl::BlenderNoiseBasis>(noisebasis), noisesize, noisedepth,
			contrast, bright, hard);
}

Spectrum BlenderCloudsTexture::GetSpectrumValue(const HitPoint &hitPoint) const {
	return Spectrum(GetFloatValue(hitPoint));
}

PropertiesUPtr BlenderCloudsTexture::ToProperties(const ImageMapCache &imgMapCache, const bool useRealFileName) const {
	auto props = std::make_unique<Properties>();

	std::string noisetype = "soft_noise";
	if(hard) noisetype = "hard_noise";

	std::string nbas;
	switch(noisebasis) {
		default:
		case BLENDER_ORIGINAL:
			nbas = "blender_original";
			break;
		case ORIGINAL_PERLIN:
			nbas = "original_perlin";
			break;
		case IMPROVED_PERLIN:
			nbas = "improved_perlin";
			break;
		case VORONOI_F1:
			nbas = "voronoi_f1";
			break;
		case VORONOI_F2:
			nbas = "voronoi_f2";
			break;
		case VORONOI_F3:
			nbas = "voronoi_f3";
			break;
		case VORONOI_F4:
			nbas = "voronoi_f4";
			break;
		case VORONOI_F2_F1:
			nbas = "voronoi_f2_f1";
			break;
		case VORONOI_CRACKLE:
			nbas = "voronoi_crackle";
			break;
		case CELL_NOISE:
			nbas = "cell_noise";
			break;
	}

	const std::string name = GetName();

	props->Set(Property("scene.textures." + name + ".type")("blender_clouds"));
	props->Set(Property("scene.textures." + name + ".noisetype")(noisetype));
	props->Set(Property("scene.textures." + name + ".noisebasis")(nbas));
	props->Set(Property("scene.textures." + name + ".noisesize")(noisesize));
	props->Set(Property("scene.textures." + name + ".noisedepth")(noisedepth));
	props->Set(Property("scene.textures." + name + ".bright")(bright));
	props->Set(Property("scene.textures." + name + ".contrast")(contrast));
	props->Set(mapping->ToProperties("scene.textures." + name + ".mapping"));

	return props;
}

//------------------------------------------------------------------------------
// Blender distorted noise texture
//------------------------------------------------------------------------------

BlenderDistortedNoiseTexture::BlenderDistortedNoiseTexture(TextureMapping3DUPtr&& mp, const std::string &pnoisedistortion,
		const std::string &pnoisebasis, float distortion, float noisesize, float bright, float contrast) :
		mapping(std::move(mp)), noisedistortion(BLENDER_ORIGINAL), noisebasis(BLENDER_ORIGINAL), distortion(distortion), noisesize(noisesize),
		bright(bright), contrast(contrast) {

	if(pnoisedistortion == "blender_original") {
		noisedistortion = BLENDER_ORIGINAL;
	} else if(pnoisedistortion == "original_perlin") {
		noisedistortion = ORIGINAL_PERLIN;
	} else if(pnoisedistortion == "improved_perlin") {
		noisedistortion = IMPROVED_PERLIN;
	} else if(pnoisedistortion == "voronoi_f1") {
		noisedistortion = VORONOI_F1;
	} else if(pnoisedistortion == "voronoi_f2") {
		noisedistortion = VORONOI_F2;
	} else if(pnoisedistortion == "voronoi_f3") {
		noisedistortion = VORONOI_F3;
	} else if(pnoisedistortion == "voronoi_f4") {
		noisedistortion = VORONOI_F4;
	} else if(pnoisedistortion == "voronoi_f2_f1") {
		noisedistortion = VORONOI_F2_F1;
	} else if(pnoisedistortion == "voronoi_crackle") {
		noisedistortion = VORONOI_CRACKLE;
	} else if(pnoisedistortion == "cell_noise") {
		noisedistortion = CELL_NOISE;
	};

	if(pnoisebasis == "blender_original") {
		noisebasis = BLENDER_ORIGINAL;
	} else if(pnoisebasis == "original_perlin") {
		noisebasis = ORIGINAL_PERLIN;
	} else if(pnoisebasis == "improved_perlin") {
		noisebasis = IMPROVED_PERLIN;
	} else if(pnoisebasis == "voronoi_f1") {
		noisebasis = VORONOI_F1;
	} else if(pnoisebasis == "voronoi_f2") {
		noisebasis = VORONOI_F2;
	} else if(pnoisebasis == "voronoi_f3") {
		noisebasis = VORONOI_F3;
	} else if(pnoisebasis == "voronoi_f4") {
		noisebasis = VORONOI_F4;
	} else if(pnoisebasis == "voronoi_f2_f1") {
		noisebasis = VORONOI_F2_F1;
	} else if(pnoisebasis == "voronoi_crackle") {
		noisebasis = VORONOI_CRACKLE;
	} else if(pnoisebasis == "cell_noise") {
		noisebasis = CELL_NOISE;
	};

}

float BlenderDistortedNoiseTexture::GetFloatValue(const HitPoint &hitPoint) const {
	return ocl::BlenderDistortedNoiseTexture_Evaluate(ToFloat3(mapping->Map(hitPoint)),
			static_cast<ocl::BlenderNoiseBasis>(noisedistortion),
			static_cast<ocl::BlenderNoiseBasis>(noisebasis), distortion, noisesize,
			contrast, bright);
}

Spectrum BlenderDistortedNoiseTexture::GetSpectrumValue(const HitPoint &hitPoint) const {
	return Spectrum(GetFloatValue(hitPoint));
}

PropertiesUPtr BlenderDistortedNoiseTexture::ToProperties(const ImageMapCache &imgMapCache, const bool useRealFileName) const {
	auto props = std::make_unique<Properties>();

	const std::string name = GetName();

	props->Set(Property("scene.textures." + name + ".type")("blender_distortednoise"));
	props->Set(Property("scene.textures." + name + ".noisebasis")(noisebasis));
	props->Set(Property("scene.textures." + name + ".noise_distortion")(noisedistortion));
	props->Set(Property("scene.textures." + name + ".noisesize")(noisesize));
	props->Set(Property("scene.textures." + name + ".distortion")(distortion));
	props->Set(Property("scene.textures." + name + ".bright")(bright));
	props->Set(Property("scene.textures." + name + ".contrast")(contrast));
	props->Set(mapping->ToProperties("scene.textures." + name + ".mapping"));

	return props;
}

//------------------------------------------------------------------------------
// Blender magic texture
//------------------------------------------------------------------------------

BlenderMagicTexture::BlenderMagicTexture(TextureMapping3DUPtr&& mp, const int noisedepth,
										 const float turbulence, float bright, float contrast) :
		mapping(std::move(mp)), noisedepth(noisedepth), turbulence(turbulence), bright(bright), contrast(contrast) {

}

float BlenderMagicTexture::GetFloatValue(const HitPoint &hitPoint) const {
	return GetSpectrumValue(hitPoint).Y();
}

Spectrum BlenderMagicTexture::GetSpectrumValue(const HitPoint &hitPoint) const {
	const ocl::float3 rgb = ocl::BlenderMagicTexture_Evaluate(ToFloat3(mapping->Map(hitPoint)),
			noisedepth, turbulence, contrast, bright);

	return Spectrum(RGBColor(rgb.x, rgb.y, rgb.z));
}

float BlenderMagicTexture::Y() const {
	static float c[][3] = { { .58f, .58f, .6f }, { .58f, .58f, .6f }, { .58f, .58f, .6f },
		{ .5f, .5f, .5f }, { .6f, .59f, .58f }, { .58f, .58f, .6f },
		{ .58f, .58f, .6f }, {.2f, .2f, .33f }, { .58f, .58f, .6f }, };
	luxrays::Spectrum cs;
#define NC  sizeof(c) / sizeof(c[0])
	for (u_int i = 0; i < NC; ++i)
		cs += luxrays::Spectrum(c[i]);
	return cs.Y() / NC;
#undef NC
}

float BlenderMagicTexture::Filter() const {
	static float c[][3] = { { .58f, .58f, .6f }, { .58f, .58f, .6f }, { .58f, .58f, .6f },
		{ .5f, .5f, .5f }, { .6f, .59f, .58f }, { .58f, .58f, .6f },
		{ .58f, .58f, .6f }, {.2f, .2f, .33f }, { .58f, .58f, .6f }, };
	luxrays::Spectrum cs;
#define NC  sizeof(c) / sizeof(c[0])
	for (u_int i = 0; i < NC; ++i)
		cs += luxrays::Spectrum(c[i]);
	return cs.Filter() / NC;
#undef NC
}

PropertiesUPtr BlenderMagicTexture::ToProperties(const ImageMapCache &imgMapCache, const bool useRealFileName) const {
	auto props = std::make_unique<Properties>();
	const std::string name = GetName();

	props->Set(Property("scene.textures." + name + ".type")("blender_magic"));
	props->Set(Property("scene.textures." + name + ".noisedepth")(noisedepth));
	props->Set(Property("scene.textures." + name + ".turbulence")(turbulence));
	props->Set(Property("scene.textures." + name + ".bright")(bright));
	props->Set(Property("scene.textures." + name + ".contrast")(contrast));
	props->Set(mapping->ToProperties("scene.textures." + name + ".mapping"));

	return props;
}
//------------------------------------------------------------------------------
// Blender marble texture
//------------------------------------------------------------------------------

BlenderMarbleTexture::BlenderMarbleTexture(TextureMapping3DUPtr&& mp, const std::string &ptype, const std::string &pnoisebasis,
		const std::string &pnoise, float noisesize, float turb, int noisedepth, bool hard, float bright, float contrast) :
		mapping(std::move(mp)), type(TEX_SOFT), noisebasis(BLENDER_ORIGINAL), noisebasis2(TEX_SIN), noisesize(noisesize),
		turbulence(turb), noisedepth(noisedepth), hard(hard), bright(bright), contrast(contrast) {

	if(pnoisebasis == "blender_original") {
		noisebasis = BLENDER_ORIGINAL;
	} else if(pnoisebasis == "original_perlin") {
		noisebasis = ORIGINAL_PERLIN;
	} else if(pnoisebasis == "improved_perlin") {
		noisebasis = IMPROVED_PERLIN;
	} else if(pnoisebasis == "voronoi_f1") {
		noisebasis = VORONOI_F1;
	} else if(pnoisebasis == "voronoi_f2") {
		noisebasis = VORONOI_F2;
	} else if(pnoisebasis == "voronoi_f3") {
		noisebasis = VORONOI_F3;
	} else if(pnoisebasis == "voronoi_f4") {
		noisebasis = VORONOI_F4;
	} else if(pnoisebasis == "voronoi_f2_f1") {
		noisebasis = VORONOI_F2_F1;
	} else if(pnoisebasis == "voronoi_crackle") {
		noisebasis = VORONOI_CRACKLE;
	} else if(pnoisebasis == "cell_noise") {
		noisebasis = CELL_NOISE;
	};

	if(ptype == "soft") {
		type = TEX_SOFT;
	} else if(ptype == "sharp") {
		type = TEX_SHARP;
	} else if(ptype == "sharper") {
		type = TEX_SHARPER;
	};

	if(pnoise == "sin") {
		noisebasis2 = TEX_SIN;
	} else if(pnoise == "saw") {
		noisebasis2 = TEX_SAW;
	} else if(pnoise == "tri") {
		noisebasis2 = TEX_TRI;
	};
}

float BlenderMarbleTexture::GetFloatValue(const HitPoint &hitPoint) const {
	return ocl::BlenderMarbleTexture_Evaluate(ToFloat3(mapping->Map(hitPoint)),
			static_cast<ocl::BlenderMarbleType>(type),
			static_cast<ocl::BlenderNoiseBasis>(noisebasis),
			static_cast<ocl::BlenderNoiseBase>(noisebasis2), noisesize, turbulence,
			noisedepth, contrast, bright, hard);
}

Spectrum BlenderMarbleTexture::GetSpectrumValue(const HitPoint &hitPoint) const {
	return Spectrum(GetFloatValue(hitPoint));
}

PropertiesUPtr BlenderMarbleTexture::ToProperties(const ImageMapCache &imgMapCache, const bool useRealFileName) const {
	auto props = std::make_unique<Properties>();

	std::string nbas;
	switch(noisebasis) {
		default:
		case BLENDER_ORIGINAL:
			nbas = "blender_original";
			break;
		case ORIGINAL_PERLIN:
			nbas = "original_perlin";
			break;
		case IMPROVED_PERLIN:
			nbas = "improved_perlin";
			break;
		case VORONOI_F1:
			nbas = "voronoi_f1";
			break;
		case VORONOI_F2:
			nbas = "voronoi_f2";
			break;
		case VORONOI_F3:
			nbas = "voronoi_f3";
			break;
		case VORONOI_F4:
			nbas = "voronoi_f4";
			break;
		case VORONOI_F2_F1:
			nbas = "voronoi_f2_f1";
			break;
		case VORONOI_CRACKLE:
			nbas = "voronoi_crackle";
			break;
		case CELL_NOISE:
			nbas = "cell_noise";
			break;
	}

	std::string noise;
	switch(noisebasis2) {
		default:
		case TEX_SIN:
			noise = "sin";
			break;
		case TEX_SAW:
			noise = "saw";
			break;
		case TEX_TRI:
			noise = "tri";
			break;
	}

	std::string noisetype = "soft_noise";
	if(hard) noisetype = "hard_noise";

	const std::string name = GetName();

	props->Set(Property("scene.textures." + name + ".type")("blender_marble"));
	props->Set(Property("scene.textures." + name + ".noisebasis")(nbas));
	props->Set(Property("scene.textures." + name + ".noisebasis2")(noise));
	props->Set(Property("scene.textures." + name + ".noisesize")(noisesize));
	props->Set(Property("scene.textures." + name + ".noisetype")(noisetype));
	props->Set(Property("scene.textures." + name + ".turbulence")(turbulence));
	props->Set(Property("scene.textures." + name + ".bright")(bright));
	props->Set(Property("scene.textures." + name + ".contrast")(contrast));
	props->Set(mapping->ToProperties("scene.textures." + name + ".mapping"));

	return props;
}

//------------------------------------------------------------------------------
// Blender musgrave texture
//------------------------------------------------------------------------------

BlenderMusgraveTexture::BlenderMusgraveTexture(TextureMapping3DUPtr&& mp, const std::string &ptype, const std::string &pnoisebasis,
		const float dimension, const float intensity, const float lacunarity, const float offset, const float gain,
		const float octaves, float noisesize, float bright, float contrast) :
		mapping(std::move(mp)), type(TEX_MULTIFRACTAL), noisebasis(BLENDER_ORIGINAL), dimension(dimension), intensity(intensity),
		lacunarity(lacunarity), offset(offset), gain(gain), octaves(octaves),
		noisesize(noisesize), bright(bright), contrast(contrast) {

	if(ptype == "multifractal") {
		type = TEX_MULTIFRACTAL;
	} else if(ptype == "ridged_multifractal") {
		type = TEX_RIDGED_MULTIFRACTAL;
	} else if(ptype == "hybrid_multifractal") {
		type = TEX_HYBRID_MULTIFRACTAL;
	} else if(ptype == "fBM") {
		type = TEX_FBM;
	} else if(ptype == "hetero_terrain") {
		type = TEX_HETERO_TERRAIN;
	};

	if(pnoisebasis == "blender_original") {
		noisebasis = BLENDER_ORIGINAL;
	} else if(pnoisebasis == "original_perlin") {
		noisebasis = ORIGINAL_PERLIN;
	} else if(pnoisebasis == "improved_perlin") {
		noisebasis = IMPROVED_PERLIN;
	} else if(pnoisebasis == "voronoi_f1") {
		noisebasis = VORONOI_F1;
	} else if(pnoisebasis == "voronoi_f2") {
		noisebasis = VORONOI_F2;
	} else if(pnoisebasis == "voronoi_f3") {
		noisebasis = VORONOI_F3;
	} else if(pnoisebasis == "voronoi_f4") {
		noisebasis = VORONOI_F4;
	} else if(pnoisebasis == "voronoi_f2_f1") {
		noisebasis = VORONOI_F2_F1;
	} else if(pnoisebasis == "voronoi_crackle") {
		noisebasis = VORONOI_CRACKLE;
	} else if(pnoisebasis == "cell_noise") {
		noisebasis = CELL_NOISE;
	};
}

float BlenderMusgraveTexture::GetFloatValue(const HitPoint &hitPoint) const {
	return ocl::BlenderMusgraveTexture_Evaluate(ToFloat3(mapping->Map(hitPoint)),
			static_cast<ocl::BlenderMusgraveType>(type),
			static_cast<ocl::BlenderNoiseBasis>(noisebasis), dimension, intensity,
			lacunarity, offset, gain, octaves, noisesize, contrast, bright);
}

Spectrum BlenderMusgraveTexture::GetSpectrumValue(const HitPoint &hitPoint) const {
	return Spectrum(GetFloatValue(hitPoint));
}

PropertiesUPtr BlenderMusgraveTexture::ToProperties(const ImageMapCache &imgMapCache, const bool useRealFileName) const {
	auto props = std::make_unique<Properties>();

	std::string nbas;
	switch(noisebasis) {
		default:
		case BLENDER_ORIGINAL:
			nbas = "blender_original";
			break;
		case ORIGINAL_PERLIN:
			nbas = "original_perlin";
			break;
		case IMPROVED_PERLIN:
			nbas = "improved_perlin";
			break;
		case VORONOI_F1:
			nbas = "voronoi_f1";
			break;
		case VORONOI_F2:
			nbas = "voronoi_f2";
			break;
		case VORONOI_F3:
			nbas = "voronoi_f3";
			break;
		case VORONOI_F4:
			nbas = "voronoi_f4";
			break;
		case VORONOI_F2_F1:
			nbas = "voronoi_f2_f1";
			break;
		case VORONOI_CRACKLE:
			nbas = "voronoi_crackle";
			break;
		case CELL_NOISE:
			nbas = "cell_noise";
			break;
	}

	const std::string name = GetName();

	props->Set(Property("scene.textures." + name + ".type")("blender_musgrave"));
	props->Set(Property("scene.textures." + name + ".musgravetype")(type));
	props->Set(Property("scene.textures." + name + ".noisebasis")(nbas));
	props->Set(Property("scene.textures." + name + ".dimension")(dimension));
	props->Set(Property("scene.textures." + name + ".intensity")(intensity));
	props->Set(Property("scene.textures." + name + ".lacunarity")(lacunarity));
	props->Set(Property("scene.textures." + name + ".offset")(offset));
	props->Set(Property("scene.textures." + name + ".gain")(gain));
	props->Set(Property("scene.textures." + name + ".octaves")(octaves));
	props->Set(Property("scene.textures." + name + ".noisesize")(noisesize));
	props->Set(Property("scene.textures." + name + ".bright")(bright));
	props->Set(Property("scene.textures." + name + ".contrast")(contrast));
	props->Set(mapping->ToProperties("scene.textures." + name + ".mapping"));

	return props;
}

//------------------------------------------------------------------------------
// Blender noise texture
//------------------------------------------------------------------------------

BlenderNoiseTexture::BlenderNoiseTexture(int noisedepth, float bright, float contrast) :
		noisedepth(noisedepth), bright(bright), contrast(contrast) {
}

float BlenderNoiseTexture::GetFloatValue(const HitPoint &hitPoint) const {
	return ocl::BlenderNoiseTexture_Evaluate(hitPoint.passThroughEvent, noisedepth,
			bright, contrast);
}

Spectrum BlenderNoiseTexture::GetSpectrumValue(const HitPoint &hitPoint) const {
	return Spectrum(GetFloatValue(hitPoint));
}

PropertiesUPtr BlenderNoiseTexture::ToProperties(const ImageMapCache &imgMapCache, const bool useRealFileName) const {
	auto props = std::make_unique<Properties>();

	const std::string name = GetName();

	props->Set(Property("scene.textures." + name + ".type")("blender_noise"));
	props->Set(Property("scene.textures." + name + ".noisedepth")(noisedepth));
	props->Set(Property("scene.textures." + name + ".bright")(bright));
	props->Set(Property("scene.textures." + name + ".contrast")(contrast));
	return props;
}

//------------------------------------------------------------------------------
// Blender stucci texture
//------------------------------------------------------------------------------

BlenderStucciTexture::BlenderStucciTexture(TextureMapping3DUPtr&& mp, const std::string &ptype, const std::string &pnoisebasis,
		const float noises, float turb, bool hard, float bright, float contrast) :
		mapping(std::move(mp)), type(TEX_PLASTIC), noisebasis(BLENDER_ORIGINAL), noisesize(noises),
		turbulence(turb), hard(hard), bright(bright), contrast(contrast) {

	if(pnoisebasis == "blender_original") {
		noisebasis = BLENDER_ORIGINAL;
	} else if(pnoisebasis == "original_perlin") {
		noisebasis = ORIGINAL_PERLIN;
	} else if(pnoisebasis == "improved_perlin") {
		noisebasis = IMPROVED_PERLIN;
	} else if(pnoisebasis == "voronoi_f1") {
		noisebasis = VORONOI_F1;
	} else if(pnoisebasis == "voronoi_f2") {
		noisebasis = VORONOI_F2;
	} else if(pnoisebasis == "voronoi_f3") {
		noisebasis = VORONOI_F3;
	} else if(pnoisebasis == "voronoi_f4") {
		noisebasis = VORONOI_F4;
	} else if(pnoisebasis == "voronoi_f2_f1") {
		noisebasis = VORONOI_F2_F1;
	} else if(pnoisebasis == "voronoi_crackle") {
		noisebasis = VORONOI_CRACKLE;
	} else if(pnoisebasis == "cell_noise") {
		noisebasis = CELL_NOISE;
	};

	if(ptype == "plastic") {
		type = TEX_PLASTIC;
	} else if(ptype == "wall_in") {
		type = TEX_WALL_IN;
	} else if(ptype == "wall_out") {
		type = TEX_WALL_OUT;
	};
}

float BlenderStucciTexture::GetFloatValue(const HitPoint &hitPoint) const {
	return ocl::BlenderStucciTexture_Evaluate(ToFloat3(mapping->Map(hitPoint)),
			static_cast<ocl::BlenderStucciType>(type),
			static_cast<ocl::BlenderNoiseBasis>(noisebasis), noisesize, turbulence,
			contrast, bright, hard);
}

Spectrum BlenderStucciTexture::GetSpectrumValue(const HitPoint &hitPoint) const {
	return Spectrum(GetFloatValue(hitPoint));
}

PropertiesUPtr BlenderStucciTexture::ToProperties(const ImageMapCache &imgMapCache, const bool useRealFileName) const {
	auto props = std::make_unique<Properties>();

	std::string nbas;
	switch(noisebasis) {
		default:
		case BLENDER_ORIGINAL:
			nbas = "blender_original";
			break;
		case ORIGINAL_PERLIN:
			nbas = "original_perlin";
			break;
		case IMPROVED_PERLIN:
			nbas = "improved_perlin";
			break;
		case VORONOI_F1:
			nbas = "voronoi_f1";
			break;
		case VORONOI_F2:
			nbas = "voronoi_f2";
			break;
		case VORONOI_F3:
			nbas = "voronoi_f3";
			break;
		case VORONOI_F4:
			nbas = "voronoi_f4";
			break;
		case VORONOI_F2_F1:
			nbas = "voronoi_f2_f1";
			break;
		case VORONOI_CRACKLE:
			nbas = "voronoi_crackle";
			break;
		case CELL_NOISE:
			nbas = "cell_noise";
			break;
	}

	std::string stuccitype;
	switch(type) {
		default:
		case TEX_PLASTIC:
			stuccitype = "plastic";
			break;
		case TEX_WALL_IN:
			stuccitype = "wall_in";
			break;
		case TEX_WALL_OUT:
			stuccitype = "wall_out";
			break;
	}

	std::string noisetype = "soft_noise";
	if(hard) noisetype = "hard_noise";

	const std::string name = GetName();

	props->Set(Property("scene.textures." + name + ".type")("blender_stucci"));
	props->Set(Property("scene.textures." + name + ".stuccitype")(stuccitype));
	props->Set(Property("scene.textures." + name + ".noisebasis")(nbas));
	props->Set(Property("scene.textures." + name + ".noisesize")(noisesize));
	props->Set(Property("scene.textures." + name + ".noisetype")(noisetype));
	props->Set(Property("scene.textures." + name + ".turbulence")(turbulence));
	props->Set(Property("scene.textures." + name + ".bright")(bright));
	props->Set(Property("scene.textures." + name + ".contrast")(contrast));
	props->Set(mapping->ToProperties("scene.textures." + name + ".mapping"));

	return props;
}

//------------------------------------------------------------------------------
// Blender voronoi texture
//------------------------------------------------------------------------------

BlenderVoronoiTexture::BlenderVoronoiTexture(TextureMapping3DUPtr&& mp, const float intensity, const float exponent,
        const float fw1, const float fw2, const float fw3, const float fw4, const std::string distmetric, float noisesize,  float bright, float contrast) :
		mapping(std::move(mp)), intensity(intensity), feature_weight1(fw1), feature_weight2(fw2), feature_weight3(fw3), feature_weight4(fw4),
		distancemetric(ACTUAL_DISTANCE), exponent(exponent), noisesize(noisesize), bright(bright), contrast(contrast) {

	if(distmetric == "actual_distance") {
		distancemetric = ACTUAL_DISTANCE;
	} else if(distmetric == "distance_squared") {
		distancemetric = DISTANCE_SQUARED;
	} else if(distmetric == "manhattan") {
		distancemetric = MANHATTAN;
	} else if(distmetric == "chebychev") {
		distancemetric = CHEBYCHEV;
	} else if(distmetric == "minkowski_half") {
		distancemetric = MINKOWSKI_HALF;
	} else if(distmetric == "minkowski_four") {
		distancemetric = MINKOWSKI_FOUR;
	} else if(distmetric == "minkowski") {
		distancemetric = MINKOWSKI;
	};
}

float BlenderVoronoiTexture::GetFloatValue(const HitPoint &hitPoint) const {
	return ocl::BlenderVoronoiTexture_Evaluate(ToFloat3(mapping->Map(hitPoint)),
			static_cast<ocl::DistanceMetric>(distancemetric), feature_weight1,
			feature_weight2, feature_weight3, feature_weight4, noisesize, intensity,
			exponent, contrast, bright);
}

Spectrum BlenderVoronoiTexture::GetSpectrumValue(const HitPoint &hitPoint) const {
	return Spectrum(GetFloatValue(hitPoint));
}

PropertiesUPtr BlenderVoronoiTexture::ToProperties(const ImageMapCache &imgMapCache, const bool useRealFileName) const {
	auto props = std::make_unique<Properties>();

	const std::string name = GetName();

	std::string dm = "";
	switch(distancemetric) {
		case DISTANCE_SQUARED:
			dm = "distance_squared";
			break;
		case MANHATTAN:
			dm = "manhattan";
			break;
		case CHEBYCHEV:
			dm = "chebychev";
			break;
		case MINKOWSKI_HALF:
			dm = "minkowski_half";
			break;
		case MINKOWSKI_FOUR:
			dm = "minkowski_four";
			break;
		case MINKOWSKI:
			dm = "minkowski";
			break;
		case ACTUAL_DISTANCE:
		default:
			dm = "actual_distance";
			break;
	};

	props->Set(Property("scene.textures." + name + ".type")("blender_voronoi"));
	props->Set(Property("scene.textures." + name + ".distancemetric")(dm));
	props->Set(Property("scene.textures." + name + ".intensity")(intensity));
	props->Set(Property("scene.textures." + name + ".exponent")(exponent));
	props->Set(Property("scene.textures." + name + ".w1")(feature_weight1));
	props->Set(Property("scene.textures." + name + ".w2")(feature_weight2));
	props->Set(Property("scene.textures." + name + ".w3")(feature_weight3));
	props->Set(Property("scene.textures." + name + ".w4")(feature_weight4));
	props->Set(Property("scene.textures." + name + ".noisesize")(noisesize));
	props->Set(Property("scene.textures." + name + ".bright")(bright));
	props->Set(Property("scene.textures." + name + ".contrast")(contrast));
	props->Set(mapping->ToProperties("scene.textures." + name + ".mapping"));

	return props;
}

//------------------------------------------------------------------------------
// Blender wood texture
//------------------------------------------------------------------------------

BlenderWoodTexture::BlenderWoodTexture(TextureMapping3DUPtr&& mp, const std::string &ptype, const std::string &pnoise,
		const std::string &pnoisebasis, const float noises, float turb, bool hard, float bright, float contrast) :
		mapping(std::move(mp)), type(BANDS), noisebasis(BLENDER_ORIGINAL), noisebasis2(TEX_SIN), noisesize(noises),
		turbulence(turb), hard(hard), bright(bright), contrast(contrast) {

	if(pnoisebasis == "blender_original") {
		noisebasis = BLENDER_ORIGINAL;
	} else if(pnoisebasis == "original_perlin") {
		noisebasis = ORIGINAL_PERLIN;
	} else if(pnoisebasis == "improved_perlin") {
		noisebasis = IMPROVED_PERLIN;
	} else if(pnoisebasis == "voronoi_f1") {
		noisebasis = VORONOI_F1;
	} else if(pnoisebasis == "voronoi_f2") {
		noisebasis = VORONOI_F2;
	} else if(pnoisebasis == "voronoi_f3") {
		noisebasis = VORONOI_F3;
	} else if(pnoisebasis == "voronoi_f4") {
		noisebasis = VORONOI_F4;
	} else if(pnoisebasis == "voronoi_f2_f1") {
		noisebasis = VORONOI_F2_F1;
	} else if(pnoisebasis == "voronoi_crackle") {
		noisebasis = VORONOI_CRACKLE;
	} else if(pnoisebasis == "cell_noise") {
		noisebasis = CELL_NOISE;
	};

	if(ptype == "bands") {
		type = BANDS;
	} else if(ptype == "rings") {
		type = RINGS;
	} else if(ptype == "bandnoise") {
		type = BANDNOISE;
	} else if(ptype == "ringnoise") {
		type = RINGNOISE;
	};

	if(pnoise == "sin") {
		noisebasis2 = TEX_SIN;
	} else if(pnoise == "saw") {
		noisebasis2 = TEX_SAW;
	} else if(pnoise == "tri") {
		noisebasis2 = TEX_TRI;
	};
}

float BlenderWoodTexture::GetFloatValue(const HitPoint &hitPoint) const {
	return ocl::BlenderWoodTexture_Evaluate(ToFloat3(mapping->Map(hitPoint)),
			static_cast<ocl::BlenderWoodType>(type),
			static_cast<ocl::BlenderNoiseBase>(noisebasis2),
			static_cast<ocl::BlenderNoiseBasis>(noisebasis), noisesize, turbulence,
			contrast, bright, hard);
}

Spectrum BlenderWoodTexture::GetSpectrumValue(const HitPoint &hitPoint) const {
	return Spectrum(GetFloatValue(hitPoint));
}

PropertiesUPtr BlenderWoodTexture::ToProperties(const ImageMapCache &imgMapCache, const bool useRealFileName) const {
	auto props = std::make_unique<Properties>();

	std::string noise;
	switch(noisebasis2) {
		default:
		case TEX_SIN:
			noise = "sin";
			break;
		case TEX_SAW:
			noise = "saw";
			break;
		case TEX_TRI:
			noise = "tri";
			break;
	}
	std::string woodtype;
	switch(type) {
		default:
		case BANDS:
			woodtype = "bands";
			break;
		case RINGS:
			woodtype = "rings";
			break;
		case BANDNOISE:
			woodtype = "bandnoise";
			break;
		case RINGNOISE:
			woodtype = "ringnoise";
			break;
	}

	std::string nbas;
	switch(noisebasis) {
		default:
		case BLENDER_ORIGINAL:
			nbas = "blender_original";
			break;
		case ORIGINAL_PERLIN:
			nbas = "original_perlin";
			break;
		case IMPROVED_PERLIN:
			nbas = "improved_perlin";
			break;
		case VORONOI_F1:
			nbas = "voronoi_f1";
			break;
		case VORONOI_F2:
			nbas = "voronoi_f2";
			break;
		case VORONOI_F3:
			nbas = "voronoi_f3";
			break;
		case VORONOI_F4:
			nbas = "voronoi_f4";
			break;
		case VORONOI_F2_F1:
			nbas = "voronoi_f2_f1";
			break;
		case VORONOI_CRACKLE:
			nbas = "voronoi_crackle";
			break;
		case CELL_NOISE:
			nbas = "cell_noise";
			break;
	}
	std::string noisetype = "soft_noise";
	if(hard) noisetype = "hard_noise";

	const std::string name = GetName();

	props->Set(Property("scene.textures." + name + ".type")("blender_wood"));
	props->Set(Property("scene.textures." + name + ".woodtype")(woodtype));
	props->Set(Property("scene.textures." + name + ".noisebasis")(nbas));
	props->Set(Property("scene.textures." + name + ".noisebasis2")(noise));
	props->Set(Property("scene.textures." + name + ".noisesize")(noisesize));
	props->Set(Property("scene.textures." + name + ".noisetype")(noisetype));
	props->Set(Property("scene.textures." + name + ".turbulence")(turbulence));
	props->Set(Property("scene.textures." + name + ".bright")(bright));
	props->Set(Property("scene.textures." + name + ".contrast")(contrast));
	props->Set(mapping->ToProperties("scene.textures." + name + ".mapping"));

	return props;
}

// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
