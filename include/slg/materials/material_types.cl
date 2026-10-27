#line 2 "material_types.cl"

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
// Material evaluation op
//------------------------------------------------------------------------------

typedef enum {
	EVAL_ALBEDO,
	EVAL_GET_INTERIOR_VOLUME,
	EVAL_GET_EXTERIOR_VOLUME,
	EVAL_GET_EMITTED_RADIANCE,
	EVAL_GET_PASS_TROUGH_TRANSPARENCY,
	EVAL_EVALUATE,
	EVAL_SAMPLE,
	EVAL_CONDITIONAL_GOTO,
	EVAL_UNCONDITIONAL_GOTO,
	// For the very special case of Mix material
	EVAL_GET_VOLUME_MIX_SETUP1,
	EVAL_GET_VOLUME_MIX_SETUP2,
	EVAL_GET_EMITTED_RADIANCE_MIX_SETUP1,
	EVAL_GET_EMITTED_RADIANCE_MIX_SETUP2,
	EVAL_GET_PASS_TROUGH_TRANSPARENCY_MIX_SETUP1,
	EVAL_GET_PASS_TROUGH_TRANSPARENCY_MIX_SETUP2,
	EVAL_EVALUATE_MIX_SETUP1,
	EVAL_EVALUATE_MIX_SETUP2,
	EVAL_SAMPLE_MIX_SETUP1,
	EVAL_SAMPLE_MIX_SETUP2,
	// For the very special case of GlossyCoating material
	EVAL_EVALUATE_GLOSSYCOATING_SETUP,
	EVAL_SAMPLE_GLOSSYCOATING_SETUP,
	EVAL_SAMPLE_GLOSSYCOATING_CLOSE_SAMPLE_BASE,
	EVAL_SAMPLE_GLOSSYCOATING_CLOSE_EVALUATE_BASE,
	// For the very special case of TwoSided material
	EVAL_TWOSIDED_SETUP
} MaterialEvalOpType;

// The bump cache keeps, for the vertex being shaded, the shading normal of
// each bumped sub-material of a Mix or GlossyCoating tree, so its bump is
// evaluated once per vertex instead of once per Evaluate and Sample (see
// Material_BumpSubMaterial()). A slot per edge of the tree, at most 32 (the
// bits of the valid mask); the edges past them are not cached.
#define MATERIAL_BUMP_CACHE_MAX_SLOTS 32
#define MATERIAL_BUMP_CACHE_NO_SLOT 0xffffu
#define MATERIAL_BUMP_CACHE_SLOTS(slotA, slotB) ((slotA) | ((slotB) << 16))
#define MATERIAL_BUMP_CACHE_SLOT_A(bumpSlots) ((bumpSlots) & 0xffffu)
#define MATERIAL_BUMP_CACHE_SLOT_B(bumpSlots) ((bumpSlots) >> 16)

// The smallest roughness (GGX alpha) of the microfacet lobes: a smooth surface
// is a very sharp lobe, not a delta. Below about 1e-7 the float D and pdf of
// a near-mirror direction overflow in the kernels (-ffast-math), the same
// floor as OPENPBR_MIN_ALPHA.
#define MICROFACET_MIN_ROUGHNESS 1e-4f

typedef struct {
	unsigned int matIndex;
	MaterialEvalOpType evalType;
	union {
		// EVAL_CONDITIONAL_GOTO and EVAL_UNCONDITIONAL_GOTO
		unsigned int opsCount;
		// The other ops: the bump cache slots of the sub-materials bumped by
		// the op (MATERIAL_BUMP_CACHE_SLOTS(): material A or the base, and
		// material B), set for the Mix and GlossyCoating setup ops
		unsigned int bumpSlots;
	} opData;
} MaterialEvalOp;

//------------------------------------------------------------------------------
// Materials
//------------------------------------------------------------------------------

typedef enum {
	MATTE, MIRROR, GLASS, ARCHGLASS, MIX, NULLMAT, MATTETRANSLUCENT,
	GLOSSY2, METAL2, ROUGHGLASS, VELVET, CLOTH, CARPAINT, ROUGHMATTE,
	ROUGHMATTETRANSLUCENT, GLOSSYTRANSLUCENT, GLOSSYCOATING, DISNEY,
	TWOSIDED, SUBSURFACEMAT, OPENPBR,
			
	// Volumes
	HOMOGENEOUS_VOL, CLEAR_VOL, HETEROGENEOUS_VOL
} MaterialType;

typedef struct {
    unsigned int kdTexIndex;
} MatteParam;

typedef struct {
    unsigned int kdTexIndex;
    unsigned int sigmaTexIndex;
} RoughMatteParam;

typedef struct {
    unsigned int krTexIndex;
} MirrorParam;

typedef struct {
    unsigned int krTexIndex;
	unsigned int ktTexIndex;
	unsigned int exteriorIorTexIndex, interiorIorTexIndex;
	unsigned int cauchyBTex;
	unsigned int filmThicknessTexIndex;
	unsigned int filmIorTexIndex;
} GlassParam;

typedef struct {
    unsigned int krTexIndex;
	unsigned int expTexIndex;
} MetalParam;

typedef struct {
    unsigned int krTexIndex;
	unsigned int ktTexIndex;
	unsigned int exteriorIorTexIndex, interiorIorTexIndex;
	unsigned int filmThicknessTexIndex;
	unsigned int filmIorTexIndex;
} ArchGlassParam;

typedef struct {
	unsigned int matAIndex, matBIndex;
	unsigned int mixFactorTexIndex;
	// The weight of material B at a vertex (MixMaterial::GetVertexWeight2()):
	// mixFactorTexIndex itself or, with vertex weights, a texture of the
	// compiled scene (CompiledScene::CompileMixVertexFactors())
	unsigned int vertexFactorTexIndex;
} MixParam;

typedef struct {
    unsigned int krTexIndex;
	unsigned int ktTexIndex;
} MatteTranslucentParam;

typedef struct {
    unsigned int krTexIndex;
	unsigned int ktTexIndex;
	unsigned int sigmaTexIndex;
} RoughMatteTranslucentParam;

typedef struct {
	unsigned int kdTexIndex;
	unsigned int ksTexIndex;
	unsigned int nuTexIndex;
	unsigned int nvTexIndex;
	unsigned int kaTexIndex;
	unsigned int depthTexIndex;
	unsigned int indexTexIndex;
	int multibounce;
	int doublesided;
} Glossy2Param;

typedef struct {
    unsigned int fresnelTexIndex;
    unsigned int nTexIndex;
	unsigned int kTexIndex;
	unsigned int nuTexIndex;
	unsigned int nvTexIndex;
} Metal2Param;

typedef struct {
    unsigned int krTexIndex;
	unsigned int ktTexIndex;
	unsigned int exteriorIorTexIndex, interiorIorTexIndex;
	unsigned int nuTexIndex;
	unsigned int nvTexIndex;
	unsigned int filmThicknessTexIndex;
	unsigned int filmIorTexIndex;
} RoughGlassParam;

typedef struct {
	unsigned int kdTexIndex;
	unsigned int roughnessTexIndex;
} VelvetParam;

typedef struct {
	unsigned int colorTexIndex;
	unsigned int radiusTexIndex;
	unsigned int scaleTexIndex;
	unsigned int anisotropyTexIndex;
	unsigned int iorTexIndex;
	unsigned int roughnessTexIndex;
} SubsurfaceParam;

// The inputs of the openpbr material (see openpbr.cpp), the textures of
// OpenPBRParam::texIndex[]
typedef enum {
	OPENPBR_BASE_WEIGHT, OPENPBR_BASE_COLOR, OPENPBR_BASE_DIFFUSEROUGHNESS,
	OPENPBR_BASE_METALNESS,
	OPENPBR_SPECULAR_WEIGHT, OPENPBR_SPECULAR_COLOR, OPENPBR_SPECULAR_ROUGHNESS,
	OPENPBR_SPECULAR_IOR, OPENPBR_SPECULAR_ANISOTROPY,
	OPENPBR_TRANSMISSION_WEIGHT, OPENPBR_TRANSMISSION_COLOR, OPENPBR_TRANSMISSION_DEPTH,
	OPENPBR_TRANSMISSION_SCATTER, OPENPBR_TRANSMISSION_SCATTERANISOTROPY,
	OPENPBR_TRANSMISSION_DISPERSIONSCALE, OPENPBR_TRANSMISSION_DISPERSIONABBENUMBER,
	OPENPBR_SUBSURFACE_WEIGHT, OPENPBR_SUBSURFACE_COLOR, OPENPBR_SUBSURFACE_RADIUS,
	OPENPBR_SUBSURFACE_RADIUSSCALE, OPENPBR_SUBSURFACE_ANISOTROPY,
	OPENPBR_FUZZ_WEIGHT, OPENPBR_FUZZ_COLOR, OPENPBR_FUZZ_ROUGHNESS,
	OPENPBR_COAT_WEIGHT, OPENPBR_COAT_COLOR, OPENPBR_COAT_ROUGHNESS,
	OPENPBR_COAT_ANISOTROPY, OPENPBR_COAT_IOR, OPENPBR_COAT_DARKENING,
	OPENPBR_THINFILM_WEIGHT, OPENPBR_THINFILM_THICKNESS, OPENPBR_THINFILM_IOR,
	OPENPBR_TANGENT_ROTATION,

	OPENPBR_INPUT_COUNT
} OpenPBRInput;

typedef struct {
	// NULL_INDEX when the input is not set: its value is the default of the
	// specification (OpenPBR_Defaults[])
	unsigned int texIndex[OPENPBR_INPUT_COUNT];
	int thinWalled;
	// The subsurface part is the random walk on eye paths
	// (OpenPBRMaterial::GetSubsurfaceWalkMaterial())
	int subsurfaceWalk;
} OpenPBRParam;

typedef enum {
	DENIM, SILKSHANTUNG, SILKCHARMEUSE, COTTONTWILL, WOOLGABARDINE, POLYESTER
} ClothPreset;

typedef enum {
	WARP, WEFT
} YarnType;

// Data structure describing the properties of a single yarn
typedef struct {
	// Fiber twist angle
	float psi;
	// Maximum inclination angle
	float umax;
	// Spine curvature
	float kappa;
	// Width of segment rectangle
	float width;
	// Length of segment rectangle
	float length;
	/*! u coordinate of the yarn segment center,
	 * assumes that the tile covers 0 <= u, v <= 1.
	 * (0, 0) is lower left corner of the weave pattern
	 */
	float centerU;
	// v coordinate of the yarn segment center
	float centerV;

	// Weft/Warp flag
	YarnType yarn_type;
} Yarn;

typedef struct  {
	// Size of the weave pattern
	unsigned int tileWidth, tileHeight;
	
	// Uniform scattering parameter
	float alpha;
	// Forward scattering parameter
	float beta;
	// Filament smoothing
	float ss;
	// Highlight width
	float hWidth;
	// Combined area taken up by the warp & weft
	float warpArea, weftArea;

	// Noise-related parameters
	float fineness;

	float dWarpUmaxOverDWarp;
	float dWarpUmaxOverDWeft;
	float dWeftUmaxOverDWarp;
	float dWeftUmaxOverDWeft;
	float period;
} WeaveConfig;

typedef struct {
    ClothPreset Preset;
	unsigned int Weft_KdIndex;
	unsigned int Weft_KsIndex;
	unsigned int Warp_KdIndex;
	unsigned int Warp_KsIndex;
	float Repeat_U;
	float Repeat_V;
	float specularNormalization;
} ClothParam;

typedef struct {
	unsigned int KdTexIndex;
	unsigned int Ks1TexIndex;
	unsigned int Ks2TexIndex;
	unsigned int Ks3TexIndex;
	unsigned int M1TexIndex;
	unsigned int M2TexIndex;
	unsigned int M3TexIndex;
	unsigned int R1TexIndex;
	unsigned int R2TexIndex;
	unsigned int R3TexIndex;
	unsigned int KaTexIndex;
	unsigned int depthTexIndex;
} CarPaintParam;

typedef struct {
	unsigned int kdTexIndex;
	unsigned int ktTexIndex;
	unsigned int ksTexIndex;
	unsigned int ksbfTexIndex;
	unsigned int nuTexIndex;
	unsigned int nubfTexIndex;
	unsigned int nvTexIndex;
	unsigned int nvbfTexIndex;
	unsigned int kaTexIndex;
	unsigned int kabfTexIndex;
	unsigned int depthTexIndex;
	unsigned int depthbfTexIndex;
	unsigned int indexTexIndex;
	unsigned int indexbfTexIndex;
	int multibounce;
	int multibouncebf;
} GlossyTranslucentParam;

typedef struct {
	unsigned int matBaseIndex;
	unsigned int ksTexIndex;
	unsigned int nuTexIndex;
	unsigned int nvTexIndex;
	unsigned int kaTexIndex;
	unsigned int depthTexIndex;
	unsigned int indexTexIndex;
	int multibounce;
	// The darkening by the internal reflection of the coating: when it is not
	// the constant 0, the eval. ops of the base push its albedo (see
	// CompiledScene::CompileMaterialOps())
	unsigned int darkeningTexIndex;
	int darkening;
} GlossyCoatingParam;

typedef struct {
	unsigned int baseColorTexIndex;
	unsigned int subsurfaceTexIndex;
	unsigned int roughnessTexIndex;
	unsigned int metallicTexIndex;
	unsigned int specularTexIndex;
	unsigned int specularTintTexIndex;
	unsigned int clearcoatTexIndex;
	unsigned int clearcoatGlossTexIndex;
	unsigned int anisotropicTexIndex;
	unsigned int sheenTexIndex;
	unsigned int sheenTintTexIndex;
	unsigned int filmAmountTexIndex;
	unsigned int filmThicknessTexIndex;
	unsigned int filmIorTexIndex;
} DisneyParam;

typedef struct {
	unsigned int frontMatIndex;
	unsigned int backMatIndex;
} TwoSidedParam;

typedef struct {
	unsigned int sigmaATexIndex;
} ClearVolumeParam;

typedef struct {
	unsigned int sigmaATexIndex;
	unsigned int sigmaSTexIndex;
	unsigned int gTexIndex;
	int multiScattering;
} HomogenousVolumeParam;

typedef struct {
	unsigned int sigmaATexIndex;
	unsigned int sigmaSTexIndex;
	unsigned int gTexIndex;
	float stepSize;
	unsigned int maxStepsCount;
	int multiScattering;
} HeterogenousVolumeParam;

typedef struct {
	unsigned int iorTexIndex;
	// This is a different kind of emission texture from the one in
	// Material class (i.e. is not sampled by direct light).
	unsigned int volumeEmissionTexIndex;
	unsigned int volumeLightID;
	int priority;

	union {
		ClearVolumeParam clear;
		HomogenousVolumeParam homogenous;
		HeterogenousVolumeParam heterogenous;
	};
} VolumeParam;

typedef struct {
	MaterialType type;
	unsigned int matID, lightID;
	float bumpSampleDistance;
	Spectrum emittedFactor;
	float emittedCosThetaMax;
	int usePrimitiveArea;
	unsigned int frontTranspTexIndex, backTranspTexIndex;
	Spectrum passThroughShadowTransparency;
	unsigned int emitTexIndex, bumpTexIndex;
	// Type of indirect paths where a light source is visible with a direct hit. It is
	// an OR of DIFFUSE, GLOSSY and SPECULAR.
	BSDFEvent visibility;
	unsigned int interiorVolumeIndex, exteriorVolumeIndex;
	float glossiness, avgPassThroughTransparency;
	int isShadowCatcher, isShadowCatcherOnlyInfiniteLights, isPhotonGIEnabled,
			isHoldout;

	// The result of calling Material::GetEventTypes()
	BSDFEvent eventTypes;
	// The result of calling Material::IsDelta()
	int isDelta; 
	// Material::IsRegularizedDelta() and Material::GetRegularizedEventTypes():
	// the same at a regularized vertex (HitPoint::regularization > 0)
	int isRegularizedDelta;
	BSDFEvent regularizedEventTypes;

	// Material::GetRefractiveMaterial(): the IOR textures of the glass or
	// rough glass reached through the composite materials, for the Russian
	// roulette on the path throughput (BSDF_GetRefractionEtaScale())
	int isRefractive;
	unsigned int refractionExteriorIorTexIndex, refractionInteriorIorTexIndex;

	// Material::GetSubsurfaceWalkMaterial(): the subsurface or OpenPBR
	// material whose random walk an eye path entering this one starts (itself,
	// or the base of a glossy coating), NULL_INDEX when there is none
	unsigned int subsurfaceWalkMatIndex;
	// Material::HasDiffuseBesideSubsurfaceWalk()
	int hasDiffuseBesideSubsurfaceWalk;

	// Material eval. ops start index and length 
	unsigned int evalAlbedoOpStartIndex, evalAlbedoOpLength;
	unsigned int evalGetInteriorVolumeOpStartIndex, evalGetInteriorVolumeOpLength;
	unsigned int evalGetExteriorVolumeOpStartIndex, evalGetExteriorVolumeOpLength;
	unsigned int evalGetEmittedRadianceOpStartIndex, evalGetEmittedRadianceOpLength;
	unsigned int evalGetPassThroughTransparencyOpStartIndex, evalGetPassThroughTransparencyOpLength;
	unsigned int evalEvaluateOpStartIndex, evalEvaluateOpLength;
	unsigned int evalSampleOpStartIndex, evalSampleOpLength;
	// Shortcuts of the interpreter (CompiledScene::CompileMaterialOps()): the
	// volumes are interiorVolumeIndex and exteriorVolumeIndex, the pass
	// through transparency is black, the emitted radiance is the one of
	// DefaultMaterial_GetEmittedRadiance() and, with constEmittedRadiance,
	// it doesn't read the hit point; nullPassThrough: a null material with
	// no transparency, which lets every ray through (BSDF_Init() builds only
	// what Scene_Intersect() reads)
	int constVolumes, noPassThroughTransparency, defaultEmittedRadiance, constEmittedRadiance,
			nullPassThrough;

	union {
		MatteParam matte;
		RoughMatteParam roughmatte;
		MirrorParam mirror;
		GlassParam glass;
		MetalParam metal;
		ArchGlassParam archglass;
		MixParam mix;
		// NULLMAT has no parameters
		MatteTranslucentParam matteTranslucent;
		RoughMatteTranslucentParam roughmatteTranslucent;
		Glossy2Param glossy2;
		Metal2Param metal2;
		RoughGlassParam roughglass;
		VelvetParam velvet;
        ClothParam cloth;
		CarPaintParam carpaint;
		GlossyTranslucentParam glossytranslucent;
		GlossyCoatingParam glossycoating;
		DisneyParam disney;
		TwoSidedParam twosided;
		SubsurfaceParam subsurface;
		OpenPBRParam openpbr;
		VolumeParam volume;
	};
} Material;

//------------------------------------------------------------------------------
// Some macro trick in order to have more readable code
//------------------------------------------------------------------------------

#if defined(SLG_OPENCL_KERNEL)

#define MATERIALS_PARAM_DECL \
	, __global const Material* restrict mats \
	, __global const MaterialEvalOp* restrict matEvalOps \
	, __global float *matEvalStacks \
	, const uint maxMaterialEvalStackSize \
	TEXTURES_PARAM_DECL
#define MATERIALS_PARAM \
	, mats \
	, matEvalOps \
	, matEvalStacks \
	, maxMaterialEvalStackSize \
	TEXTURES_PARAM

#define MATERIAL_EVALUATE_RETURN_BLACK { \
		const float3 result = BLACK; \
		EvalStack_PushFloat3(result); \
		const BSDFEvent event = NONE; \
		EvalStack_PushBSDFEvent(event); \
		const float directPdfW = 0.f; \
		EvalStack_PushFloat(directPdfW); \
		return; \
	}

#define MATERIAL_SAMPLE_RETURN_BLACK { \
		const float3 result = BLACK; \
		EvalStack_PushFloat3(result); \
		const float3 sampledDir = ZERO; \
		EvalStack_PushFloat3(sampledDir); \
		const BSDFEvent event = NONE; \
		EvalStack_PushBSDFEvent(event); \
		const float pdfW = 0.f; \
		EvalStack_PushFloat(pdfW); \
		return; \
	}

#endif
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
