#line 2 "materialdefs_funcs_openpbr_bsdf.cl"

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
// The OpenPBR Surface BSDF (OpenPBR Surface specification v1.1, Portsmouth,
// Kutz, Hill et al. 2025). Pure functions of the parameters at the hit point
// (OpenPBRBSDF) and of local directions, compiled for the GPU
// (materialdefs_funcs_openpbr.cl fills OpenPBRBSDF from the textures) and for
// the CPU (openpbr.cpp, through luxrays/devices/cpudevice_oclemul.h), so the
// two devices run the same code.
//
// The lobes, from the top:
//
//  base       mix(dielectric, metal, base.metalness)
//    metal    GGX with the F82-tint Fresnel of F0 = base.weight base.color
//             and the edge tint specular.color, times specular.weight
//    dielectric
//      specular  GGX with the exact Fresnel of specular.ior, F0 modulated by
//                specular.weight, tinted by specular.color
//      diffuse   EON of the albedo base.weight base.color and the roughness
//                base.diffuseroughness, under the specular scaled by
//                1 - E_spec(mu_eye) (the albedo scaling of the
//                specification: exact for any base)
//
// Every GGX lobe is energy compensated (microfacet_energy_tables.cl), the
// specular alphas are alpha = specular.roughness^2 split by
// specular.anisotropy as the specification does, in the tangent frame rotated
// by tangent.rotation (in turns). Both faces of the surface are the same: the
// back face is the mirror image of the front one.
//------------------------------------------------------------------------------

// The defaults of the specification, 3 values per input in OpenPBRInput
// order (the float inputs repeat their value)
__constant float OpenPBR_Defaults[OPENPBR_INPUT_COUNT * 3] = {
	1.f, 1.f, 1.f,		// base.weight
	.8f, .8f, .8f,		// base.color
	0.f, 0.f, 0.f,		// base.diffuseroughness
	0.f, 0.f, 0.f,		// base.metalness
	1.f, 1.f, 1.f,		// specular.weight
	1.f, 1.f, 1.f,		// specular.color
	.3f, .3f, .3f,		// specular.roughness
	1.5f, 1.5f, 1.5f,	// specular.ior
	0.f, 0.f, 0.f,		// specular.anisotropy
	0.f, 0.f, 0.f,		// transmission.weight
	1.f, 1.f, 1.f,		// transmission.color
	0.f, 0.f, 0.f,		// transmission.depth
	0.f, 0.f, 0.f,		// transmission.scatter
	0.f, 0.f, 0.f,		// transmission.scatteranisotropy
	0.f, 0.f, 0.f,		// subsurface.weight
	.8f, .8f, .8f,		// subsurface.color
	1.f, 1.f, 1.f,		// subsurface.radius
	1.f, .5f, .25f,		// subsurface.radiusscale
	0.f, 0.f, 0.f,		// subsurface.anisotropy
	0.f, 0.f, 0.f,		// fuzz.weight
	1.f, 1.f, 1.f,		// fuzz.color
	.5f, .5f, .5f,		// fuzz.roughness
	0.f, 0.f, 0.f,		// coat.weight
	1.f, 1.f, 1.f,		// coat.color
	0.f, 0.f, 0.f,		// coat.roughness
	0.f, 0.f, 0.f,		// coat.anisotropy
	1.6f, 1.6f, 1.6f,	// coat.ior
	1.f, 1.f, 1.f,		// coat.darkening
	0.f, 0.f, 0.f,		// thinfilm.weight
	.5f, .5f, .5f,		// thinfilm.thickness
	1.4f, 1.4f, 1.4f,	// thinfilm.ior
	0.f, 0.f, 0.f		// tangent.rotation
};

// The smallest GGX alpha (a smooth surface is a very sharp lobe, not a delta)
#define OPENPBR_MIN_ALPHA 1e-4f

typedef struct {
	// The inputs at the hit point (OpenPBR_Init() clamps them)
	float baseWeight, baseDiffuseRoughness, baseMetalness;
	float3 baseColor;
	float specularWeight, specularRoughness, specularIor, specularAnisotropy;
	float3 specularColor;
	float tangentRotation;
	// The path regularization of the hit point (HitPoint::regularization)
	float regularization;

	// Derived by OpenPBR_Init()
	// base.weight base.color: the diffuse albedo and the F0 of the metal
	float3 albedo;
	// The GGX alphas of the base specular lobes in the rotated tangent frame
	float ax, ay;
	float cosRot, sinRot;
	// The relative IOR of the dielectric Fresnel (specular.weight applied) and
	// its hemispherical average reflectance
	float eta, etaFavg;
	// The F82-tint Fresnel of the metal: F(mu) = F0 + (1 - F0) (1 - mu)^5 -
	// b mu (1 - mu)^6, and its hemispherical average
	float3 metalB, metalFavg;

#if defined(LUXRAYS_CPU_DEVICE)
	// The luminance weights of the working colour space (compile-time
	// constants on the GPU, see Spectrum_Y())
	float3 lumWeights;
#endif
} OpenPBRBSDF;

// The luminance of a working space colour, for the lobe probabilities (the
// same choices in every working colour space)
#if defined(LUXRAYS_CPU_DEVICE)
#define OpenPBR_Y(b, c) fmax(0.f, dot((b)->lumWeights, (c)))
#else
#define OpenPBR_Y(b, c) fmax(0.f, Spectrum_Y(c))
#endif

// The hemispherical average of the dielectric Fresnel reflectance for the
// relative IOR eta >= 1 (the fit of Kulla & Conty 2017), for the energy
// compensation only
OPENCL_FORCE_INLINE float OpenPBR_DielectricFavg(const float eta) {
	return (eta - 1.f) / (4.08567f + 1.00071f * eta);
}

OPENCL_FORCE_INLINE void OpenPBR_Init(OpenPBRBSDF *b) {
	b->baseWeight = clamp(b->baseWeight, 0.f, 1.f);
	b->baseColor = Spectrum_Clamp(b->baseColor);
	b->baseDiffuseRoughness = clamp(b->baseDiffuseRoughness, 0.f, 1.f);
	b->baseMetalness = clamp(b->baseMetalness, 0.f, 1.f);
	b->specularWeight = fmax(b->specularWeight, 0.f);
	b->specularColor = Spectrum_Clamp(b->specularColor);
	b->specularRoughness = clamp(b->specularRoughness, 0.f, 1.f);
	b->specularIor = fmax(b->specularIor, 1e-3f);
	b->specularAnisotropy = clamp(b->specularAnisotropy, 0.f, 1.f);

	b->albedo = b->baseWeight * b->baseColor;

	// The specification's alphas: alpha = r^2, alpha_t = alpha
	// sqrt(2 / (1 + (1 - a)^2)), alpha_b = (1 - a) alpha_t
	const float alpha = b->specularRoughness * b->specularRoughness;
	const float oneMinusA = 1.f - b->specularAnisotropy;
	const float alphaT = alpha * sqrt(2.f / (1.f + oneMinusA * oneMinusA));
	const float alphaB = oneMinusA * alphaT;
	b->ax = fmax(OPENPBR_MIN_ALPHA, Microfacet_Regularize(alphaT, b->regularization));
	b->ay = fmax(OPENPBR_MIN_ALPHA, Microfacet_Regularize(alphaB, b->regularization));

	const float angle = 2.f * M_PI_F * b->tangentRotation;
	b->cosRot = cos(angle);
	b->sinRot = sin(angle);

	// specular.weight multiplies the F0 of the IOR, mapped back to an IOR
	// (the specification's modulation). The opaque base reflects the same
	// from both sides: an IOR below 1 is inverted.
	const float etaIn = (b->specularIor < 1.f) ? (1.f / b->specularIor) : b->specularIor;
	const float r0 = (etaIn - 1.f) / (etaIn + 1.f);
	const float sqrtF0 = sqrt(clamp(b->specularWeight * r0 * r0, 0.f, .99f));
	b->eta = (1.f + sqrtF0) / (1.f - sqrtF0);
	b->etaFavg = OpenPBR_DielectricFavg(b->eta);

	// The F82-tint metal (Kutz et al. 2021, the specification's "Metal"): b
	// makes F(1/7) the Schlick value times specular.color. mu (1 - mu)^6 at
	// mu = 1/7 is 0.0566527, its cosine-weighted average 1/126.
	const float3 F0 = b->albedo;
	const float3 schlickBar = F0 + (WHITE - F0) * .462664f;
	b->metalB = schlickBar * (WHITE - b->specularColor) * (1.f / .0566527f);
	b->metalFavg = Spectrum_Clamp(b->specularWeight * (F0 + (WHITE - F0) * (1.f / 21.f) - b->metalB * (1.f / 126.f)));
}

// The metal Fresnel at the cosine mu between a direction and the microfacet
// normal, times specular.weight
OPENCL_FORCE_INLINE float3 OpenPBR_MetalFresnel(const OpenPBRBSDF *b, const float mu) {
	const float m = 1.f - clamp(mu, 0.f, 1.f);
	const float m2 = m * m;
	const float m5 = m2 * m2 * m;
	const float3 F0 = b->albedo;
	const float3 F = F0 + (WHITE - F0) * m5 - b->metalB * (mu * m5 * m);

	return Spectrum_Clamp(b->specularWeight * F);
}

// The albedo tables are isotropic: an anisotropic lobe is looked up with the
// alpha projected on the azimuth of the direction w (the roughness its Smith
// masking sees) toward grazing, the geometric mean of the alphas at normal
// incidence (where the azimuth has no meaning)
OPENCL_FORCE_INLINE float OpenPBR_ViewAlpha(const OpenPBRBSDF *b, const float3 w) {
	const float geo2 = b->ax * b->ay;
	const float sin2 = w.x * w.x + w.y * w.y;
	if ((b->ax == b->ay) || !(sin2 > 1e-8f))
		return sqrt(geo2);

	const float proj2 = (w.x * w.x * b->ax * b->ax + w.y * w.y * b->ay * b->ay) / sin2;
	return sqrt(geo2 + fmin(sin2, 1.f) * (proj2 - geo2));
}

// The energy compensation scale of the dielectric specular lobe seen from the
// eye direction with the cosine mu and the view alpha av, and the albedo of the
// compensated lobe (specular.color applied)
OPENCL_FORCE_INLINE float OpenPBR_DielectricMSScale(const OpenPBRBSDF *b, const float av, const float mu) {
	return GGXEnergy_ReflectScale(GGXEnergy_E(av, av, mu, false),
			GGXEnergy_Eavg(av, av, false), b->etaFavg);
}

OPENCL_FORCE_INLINE float3 OpenPBR_DielectricE(const OpenPBRBSDF *b, const float av, const float mu, const float msScale) {
	return b->specularColor * (GGXEnergy_DielectricE(av, av, mu, b->eta) * msScale);
}

// The probabilities of the glossy and the diffuse lobes from the direction w
// (front face, rotated frame)
OPENCL_FORCE_INLINE void OpenPBR_LobeWeights(const OpenPBRBSDF *b, const float3 w,
		float *wGlossy, float *wDiffuse) {
	const float m = b->baseMetalness;
	const float mu = w.z;

	float glossy = 0.f;
	float diffuse = 0.f;
	if (m > 0.f)
		glossy += m * OpenPBR_Y(b, OpenPBR_MetalFresnel(b, mu));
	if (m < 1.f) {
		const float av = OpenPBR_ViewAlpha(b, w);
		const float Ed = OpenPBR_Y(b, OpenPBR_DielectricE(b, av, mu, OpenPBR_DielectricMSScale(b, av, mu)));
		glossy += (1.f - m) * Ed;
		diffuse += (1.f - m) * fmax(0.f, 1.f - Ed) * OpenPBR_Y(b, b->albedo);
	}

	const float sum = glossy + diffuse;
	if (sum > 0.f) {
		*wGlossy = glossy / sum;
		*wDiffuse = diffuse / sum;
	} else {
		*wGlossy = 0.f;
		*wDiffuse = 0.f;
	}
}

// f |cos light| for two directions above the surface (front face, rotated
// frame)
OPENCL_FORCE_NOT_INLINE float3 OpenPBR_EvaluateFront(const OpenPBRBSDF *b,
		const float3 lightDir, const float3 eyeDir) {
	const float muL = lightDir.z;
	const float muE = eyeDir.z;
	if ((muL < DEFAULT_COS_EPSILON_STATIC) || (muE < DEFAULT_COS_EPSILON_STATIC))
		return BLACK;

	const float m = b->baseMetalness;

	// The microfacet lobe with F = 1, shared by the metal and the dielectric
	const float3 wh = normalize(lightDir + eyeDir);
	const float cosEH = fmax(0.f, dot(eyeDir, wh));
	const float spec = Microfacet_D(b->ax, b->ay, wh) * Microfacet_G(b->ax, b->ay, lightDir, eyeDir) / (4.f * muE);

	const float av = OpenPBR_ViewAlpha(b, eyeDir);

	float3 result = BLACK;
	if (m > 0.f) {
		const float E = GGXEnergy_E(av, av, muE, false);
		const float Eavg = GGXEnergy_Eavg(av, av, false);
		const float3 ms = MAKE_FLOAT3(GGXEnergy_ReflectScale(E, Eavg, b->metalFavg.x),
				GGXEnergy_ReflectScale(E, Eavg, b->metalFavg.y),
				GGXEnergy_ReflectScale(E, Eavg, b->metalFavg.z));

		result += (m * spec) * OpenPBR_MetalFresnel(b, cosEH) * ms;
	}
	if (m < 1.f) {
		const float msScale = OpenPBR_DielectricMSScale(b, av, muE);
		const float3 specular = (spec * FresnelCauchy_Evaluate(b->eta, cosEH) * msScale) * b->specularColor;
		const float3 diffuse = EON_Evaluate(b->albedo, b->baseDiffuseRoughness, lightDir, eyeDir) * muL;
		const float3 Espec = OpenPBR_DielectricE(b, av, muE, msScale);

		result += (1.f - m) * (specular + (WHITE - Espec) * diffuse);
	}

	return result;
}

// The pdf of sampling sampledDir from fixedDir (front face, rotated frame)
OPENCL_FORCE_INLINE float OpenPBR_PdfFront(const OpenPBRBSDF *b,
		const float wGlossy, const float wDiffuse,
		const float3 fixedDir, const float3 sampledDir) {
	if (sampledDir.z <= 0.f)
		return 0.f;

	float pdf = wDiffuse * sampledDir.z * M_1_PI_F;
	if (wGlossy > 0.f)
		pdf += wGlossy * Microfacet_PdfReflect(b->ax, b->ay, fixedDir, sampledDir);

	return pdf;
}

// The shading frame to the frame of the lobes (rotated by tangent.rotation
// about the normal, the back face mirrored by flip) and back
OPENCL_FORCE_INLINE float3 OpenPBR_ToLobeFrame(const OpenPBRBSDF *b, const float3 w, const float flip) {
	return MAKE_FLOAT3(b->cosRot * w.x + b->sinRot * w.y, b->cosRot * w.y - b->sinRot * w.x, flip * w.z);
}

OPENCL_FORCE_INLINE float3 OpenPBR_FromLobeFrame(const OpenPBRBSDF *b, const float3 w, const float flip) {
	return MAKE_FLOAT3(b->cosRot * w.x - b->sinRot * w.y, b->sinRot * w.x + b->cosRot * w.y, flip * w.z);
}

//------------------------------------------------------------------------------
// The material interface (local directions in the shading frame)
//------------------------------------------------------------------------------

OPENCL_FORCE_INLINE float3 OpenPBR_Albedo(const OpenPBRBSDF *b) {
	return b->albedo;
}

// f |cos light| (as Material::Evaluate()). directPdfW is the pdf of sampling
// the light direction from the eye one on eye paths (fromLight = false) and
// the opposite on light paths, reversePdfW the other way round.
OPENCL_FORCE_INLINE float3 OpenPBR_Evaluate(const OpenPBRBSDF *b,
		const float3 localLightDir, const float3 localEyeDir, const bool fromLight,
		BSDFEvent *event, float *directPdfW, float *reversePdfW) {
	if (localLightDir.z * localEyeDir.z <= 0.f)
		return BLACK;

	const float flip = (localEyeDir.z < 0.f) ? -1.f : 1.f;
	const float3 lightDir = OpenPBR_ToLobeFrame(b, localLightDir, flip);
	const float3 eyeDir = OpenPBR_ToLobeFrame(b, localEyeDir, flip);
	const float3 fixedDir = fromLight ? lightDir : eyeDir;
	const float3 sampledDir = fromLight ? eyeDir : lightDir;

	const float3 result = OpenPBR_EvaluateFront(b, lightDir, eyeDir);
	if (Spectrum_IsBlack(result))
		return BLACK;

	float wGlossy, wDiffuse;
	OpenPBR_LobeWeights(b, fixedDir, &wGlossy, &wDiffuse);
	if (directPdfW)
		*directPdfW = OpenPBR_PdfFront(b, wGlossy, wDiffuse, fixedDir, sampledDir);
	if (reversePdfW) {
		float rGlossy, rDiffuse;
		OpenPBR_LobeWeights(b, sampledDir, &rGlossy, &rDiffuse);
		*reversePdfW = OpenPBR_PdfFront(b, rGlossy, rDiffuse, sampledDir, fixedDir);
	}
	*event = ((wGlossy >= wDiffuse) ? GLOSSY : DIFFUSE) | REFLECT;

	return result;
}

// f |cos light| / pdf (as Material::Sample()), BLACK when there is no sample
OPENCL_FORCE_INLINE float3 OpenPBR_Sample(const OpenPBRBSDF *b,
		const float3 localFixedDir, const float u0, const float u1, const float passThroughEvent,
		const bool fromLight, float3 *localSampledDir, float *pdfW, BSDFEvent *event) {
	if (fabs(localFixedDir.z) < DEFAULT_COS_EPSILON_STATIC)
		return BLACK;

	const float flip = (localFixedDir.z < 0.f) ? -1.f : 1.f;
	const float3 fixedDir = OpenPBR_ToLobeFrame(b, localFixedDir, flip);

	float wGlossy, wDiffuse;
	OpenPBR_LobeWeights(b, fixedDir, &wGlossy, &wDiffuse);
	if (wGlossy + wDiffuse <= 0.f)
		return BLACK;

	float3 sampledDir;
	if (passThroughEvent < wGlossy) {
		if (Microfacet_SampleReflect(b->ax, b->ay, fixedDir, u0, u1, &sampledDir) <= 0.f)
			return BLACK;
		*event = GLOSSY | REFLECT;
	} else {
		float pdf;
		sampledDir = CosineSampleHemisphereWithPdf(u0, u1, &pdf);
		*event = DIFFUSE | REFLECT;
	}
	if (sampledDir.z < DEFAULT_COS_EPSILON_STATIC)
		return BLACK;

	// The value and the pdf of the mixture
	*pdfW = OpenPBR_PdfFront(b, wGlossy, wDiffuse, fixedDir, sampledDir);
	if (!(*pdfW > 0.f))
		return BLACK;
	const float3 f = fromLight ?
		OpenPBR_EvaluateFront(b, fixedDir, sampledDir) :
		OpenPBR_EvaluateFront(b, sampledDir, fixedDir);
	if (Spectrum_IsBlack(f))
		return BLACK;

	*localSampledDir = OpenPBR_FromLobeFrame(b, sampledDir, flip);

	return f / *pdfW;
}

// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
