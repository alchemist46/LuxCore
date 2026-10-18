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
// The layers, from the top. Each one scales what is beneath by
// 1 - E_layer(mu_eye), the albedo scaling of the specification (exact for any
// base, and the engines evaluate f with the eye and light roles fixed on
// every path, so the non-reciprocal form is consistent):
//
//  fuzz       fuzz.weight x the microfiber sheen of Zeltner, Burley & Chiang
//             2022 (the LTC table of the velvet material) of roughness
//             fuzz.roughness, tinted by fuzz.color
//  coat       coat.weight x GGX with the exact Fresnel of coat.ior (alpha =
//             coat.roughness^2 split by coat.anisotropy). Beneath it, as the
//             specification does for the coated part (all blended by
//             coat.weight): the transmission tinted by coat.color^(1 / cos_t)
//             (cos_t of the eye direction refracted into the coat, as Cycles),
//             the darkening (1 - K) / (1 - E_base K) of the internal
//             reflection K of the coat (times coat.darkening), the base IOR
//             relative to the coat and the base roughness raised to
//             (r^4 + 2 r_coat^4)^(1/4)
//  base       mix(dielectric, metal, base.metalness)
//    metal    GGX with the F82-tint Fresnel of F0 = base.weight base.color
//             and the edge tint specular.color, times specular.weight
//    dielectric
//      specular  GGX with the exact Fresnel of specular.ior, F0 modulated by
//                specular.weight, tinted by specular.color
//      diffuse   EON of the albedo base.weight base.color and the roughness
//                base.diffuseroughness
//  The thin film (thinfilm.weight, thickness in micrometres, ior) tints the
//  base specular and the metal with CalcFilmColor().
//
// Every GGX lobe is energy compensated (microfacet_energy_tables.cl), the
// specular alphas are alpha = specular.roughness^2 split by
// specular.anisotropy as the specification does, in the tangent frame rotated
// by tangent.rotation (in turns), shared by the coat. Both faces of the
// surface are the same: the back face is the mirror image of the front one.
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
	float coatWeight, coatRoughness, coatAnisotropy, coatIor, coatDarkening;
	float3 coatColor;
	float fuzzWeight, fuzzRoughness;
	float3 fuzzColor;
	float thinFilmWeight, thinFilmThickness, thinFilmIor;
	float tangentRotation;
	// The path regularization of the hit point (HitPoint::regularization)
	float regularization;

	// Derived by OpenPBR_Init()
	// base.weight base.color: the diffuse albedo and the F0 of the metal
	float3 albedo;
	// The GGX alphas of the base specular lobes and of the coat, in the
	// rotated tangent frame
	float ax, ay, coatAx, coatAy;
	float cosRot, sinRot;
	// The relative IOR of the base dielectric Fresnel (specular.weight and the
	// coat applied) and of the coat, with their hemispherical average
	// reflectances
	float eta, etaFavg, coatEta, coatEtaFavg;
	// The hemispherical average reflectance of the coat seen from inside
	float coatK;
	// The F82-tint Fresnel of the metal: F(mu) = F0 + (1 - F0) (1 - mu)^5 -
	// b mu (1 - mu)^6, and its hemispherical average
	float3 metalB, metalFavg;

#if defined(LUXRAYS_CPU_DEVICE)
	// The luminance weights of the working colour space and the rows of the
	// Rec.709 to working space matrix (compile-time constants on the GPU, see
	// color_funcs.cl)
	float3 lumWeights;
	float3 fromRec709[3];
#endif
} OpenPBRBSDF;

// The luminance of a working space colour, for the lobe probabilities (the
// same choices in every working colour space)
#if defined(LUXRAYS_CPU_DEVICE)
#define OpenPBR_Y(b, c) fmax(0.f, dot((b)->lumWeights, (c)))
#else
#define OpenPBR_Y(b, c) fmax(0.f, Spectrum_Y(c))
#endif

// The thin film interference colour of the eye direction in the working
// colour space
OPENCL_FORCE_INLINE float3 OpenPBR_FilmColor(const OpenPBRBSDF *b, const float3 eyeDir) {
	const float3 c = CalcFilmColor(eyeDir, 1000.f * b->thinFilmThickness, b->thinFilmIor);
#if defined(LUXRAYS_CPU_DEVICE)
	// Rec.709 on the CPU: WorkingColorSpace_FromRec709() is the identity there
	return MAKE_FLOAT3(dot(b->fromRec709[0], c), dot(b->fromRec709[1], c), dot(b->fromRec709[2], c));
#else
	return c;
#endif
}

// The hemispherical average of the dielectric Fresnel reflectance for the
// relative IOR eta >= 1 (the fit of Kulla & Conty 2017)
OPENCL_FORCE_INLINE float OpenPBR_DielectricFavg(const float eta) {
	return (eta - 1.f) / (4.08567f + 1.00071f * eta);
}

// The specification's alphas: alpha = r^2, alpha_t = alpha
// sqrt(2 / (1 + (1 - a)^2)), alpha_b = (1 - a) alpha_t, then regularized
OPENCL_FORCE_INLINE void OpenPBR_Alphas(const float roughness, const float anisotropy,
		const float regularization, float *ax, float *ay) {
	const float alpha = roughness * roughness;
	const float oneMinusA = 1.f - anisotropy;
	const float alphaT = alpha * sqrt(2.f / (1.f + oneMinusA * oneMinusA));
	const float alphaB = oneMinusA * alphaT;
	*ax = fmax(OPENPBR_MIN_ALPHA, Microfacet_Regularize(alphaT, regularization));
	*ay = fmax(OPENPBR_MIN_ALPHA, Microfacet_Regularize(alphaB, regularization));
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
	b->coatWeight = clamp(b->coatWeight, 0.f, 1.f);
	b->coatColor = Spectrum_Clamp(b->coatColor);
	b->coatRoughness = clamp(b->coatRoughness, 0.f, 1.f);
	b->coatAnisotropy = clamp(b->coatAnisotropy, 0.f, 1.f);
	b->coatIor = fmax(b->coatIor, 1.f);
	b->coatDarkening = clamp(b->coatDarkening, 0.f, 1.f);
	b->fuzzWeight = clamp(b->fuzzWeight, 0.f, 1.f);
	b->fuzzColor = Spectrum_Clamp(b->fuzzColor);
	b->fuzzRoughness = clamp(b->fuzzRoughness, 0.f, 1.f);
	b->thinFilmWeight = clamp(b->thinFilmWeight, 0.f, 1.f);
	b->thinFilmThickness = fmax(b->thinFilmThickness, 0.f);
	b->thinFilmIor = fmax(b->thinFilmIor, 1e-3f);

	b->albedo = b->baseWeight * b->baseColor;

	// The coat roughens the base (the specification: r^4 + 2 r_coat^4, the
	// GGX alpha^2 of the two crossings added), where it is present
	const float r = b->specularRoughness;
	const float r2 = r * r;
	const float rc2 = b->coatRoughness * b->coatRoughness;
	const float rCoated = fmin(1.f, sqrt(sqrt(r2 * r2 + 2.f * rc2 * rc2)));
	OpenPBR_Alphas(r + b->coatWeight * (rCoated - r), b->specularAnisotropy,
			b->regularization, &b->ax, &b->ay);
	OpenPBR_Alphas(b->coatRoughness, b->coatAnisotropy, b->regularization,
			&b->coatAx, &b->coatAy);

	const float angle = 2.f * M_PI_F * b->tangentRotation;
	b->cosRot = cos(angle);
	b->sinRot = sin(angle);

	// The base IOR is relative to the coat where coated. specular.weight
	// multiplies its F0, mapped back to an IOR (the specification's
	// modulation). The opaque base reflects the same from both sides: an IOR
	// below 1 is inverted.
	const float ior = b->specularIor * (1.f + b->coatWeight * (1.f / b->coatIor - 1.f));
	const float etaIn = (ior < 1.f) ? (1.f / ior) : ior;
	const float r0 = (etaIn - 1.f) / (etaIn + 1.f);
	const float sqrtF0 = sqrt(clamp(b->specularWeight * r0 * r0, 0.f, .99f));
	b->eta = (1.f + sqrtF0) / (1.f - sqrtF0);
	b->etaFavg = OpenPBR_DielectricFavg(b->eta);

	b->coatEta = b->coatIor;
	b->coatEtaFavg = OpenPBR_DielectricFavg(b->coatEta);
	// What the coat transmits from inside, averaged over the hemisphere, is
	// (1 - Favg) / eta^2 of the outside transmission
	b->coatK = 1.f - (1.f - b->coatEtaFavg) / (b->coatEta * b->coatEta);

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
OPENCL_FORCE_INLINE float OpenPBR_ViewAlpha(const float ax, const float ay, const float3 w) {
	const float geo2 = ax * ay;
	const float sin2 = w.x * w.x + w.y * w.y;
	if ((ax == ay) || !(sin2 > 1e-8f))
		return sqrt(geo2);

	const float proj2 = (w.x * w.x * ax * ax + w.y * w.y * ay * ay) / sin2;
	return sqrt(geo2 + fmin(sin2, 1.f) * (proj2 - geo2));
}

// The energy compensation scale of a reflection lobe of the view alpha av,
// seen from the eye direction with the cosine mu, with the average Fresnel
// reflectance Favg
OPENCL_FORCE_INLINE float OpenPBR_MSScale(const float av, const float mu, const float Favg) {
	return GGXEnergy_ReflectScale(GGXEnergy_E(av, av, mu, false),
			GGXEnergy_Eavg(av, av, false), Favg);
}

// The albedo of the coat seen from the direction w, times coat.weight
OPENCL_FORCE_INLINE float OpenPBR_CoatE(const OpenPBRBSDF *b, const float3 w) {
	if (!(b->coatWeight > 0.f))
		return 0.f;

	const float av = OpenPBR_ViewAlpha(b->coatAx, b->coatAy, w);
	return b->coatWeight * GGXEnergy_DielectricE(av, av, w.z, b->coatEta) *
			OpenPBR_MSScale(av, w.z, b->coatEtaFavg);
}

// The albedo of the fuzz seen from the direction with the cosine mu, times
// fuzz.weight, and the LTC of that direction
OPENCL_FORCE_INLINE float OpenPBR_FuzzE(const OpenPBRBSDF *b, const float mu,
		float *ltcA, float *ltcB) {
	*ltcA = 0.f;
	*ltcB = 0.f;
	if (!(b->fuzzWeight > 0.f))
		return 0.f;

	float albedo;
	SheenLTC_Lookup(mu, b->fuzzRoughness, ltcA, ltcB, &albedo);
	if ((fabs(*ltcA) < 1e-5f) || (albedo < 1e-5f))
		return 0.f;

	return b->fuzzWeight * albedo;
}

// The factor of what is beneath the coat seen from the eye direction, blended
// by coat.weight: the transmission coat.color^(1 / cos_t) and the darkening
// (1 - K) / (1 - E_base K) of the base albedo E_base
OPENCL_FORCE_INLINE float3 OpenPBR_CoatTint(const OpenPBRBSDF *b, const float3 eyeDir, const float3 baseE) {
	const float mu = eyeDir.z;
	const float sin2T = (1.f - mu * mu) / (b->coatEta * b->coatEta);
	const float invCosT = 1.f / fmax(sqrt(fmax(0.f, 1.f - sin2T)), 1e-4f);
	const float3 T = MAKE_FLOAT3(pow(b->coatColor.x, invCosT), pow(b->coatColor.y, invCosT),
			pow(b->coatColor.z, invCosT));

	const float K = b->coatK;
	const float3 delta = (1.f - K) / (WHITE - K * Spectrum_Clamp(baseE));
	const float3 darkening = WHITE + b->coatDarkening * (delta - WHITE);

	return WHITE + b->coatWeight * (T * darkening - WHITE);
}

// f |cos light| of the base for two directions above the surface (front
// face, rotated frame, wh their half vector), and the albedo of the base seen
// from the eye direction (for the coat darkening)
OPENCL_FORCE_INLINE float3 OpenPBR_EvaluateBase(const OpenPBRBSDF *b,
		const float3 lightDir, const float3 eyeDir, const float3 wh, float3 *baseE) {
	const float muL = lightDir.z;
	const float muE = eyeDir.z;
	const float m = b->baseMetalness;

	// The microfacet lobe with F = 1, shared by the metal and the dielectric
	const float cosEH = fmax(0.f, dot(eyeDir, wh));
	const float spec = Microfacet_D(b->ax, b->ay, wh) * Microfacet_G(b->ax, b->ay, lightDir, eyeDir) / (4.f * muE);
	const float av = OpenPBR_ViewAlpha(b->ax, b->ay, eyeDir);
	const float E = GGXEnergy_E(av, av, muE, false);
	const float Eavg = GGXEnergy_Eavg(av, av, false);

	float3 film = WHITE;
	if ((b->thinFilmWeight > 0.f) && (b->thinFilmThickness > 0.f))
		film = WHITE + b->thinFilmWeight * (OpenPBR_FilmColor(b, eyeDir) - WHITE);

	float3 result = BLACK;
	*baseE = BLACK;
	if (m > 0.f) {
		const float3 ms = MAKE_FLOAT3(GGXEnergy_ReflectScale(E, Eavg, b->metalFavg.x),
				GGXEnergy_ReflectScale(E, Eavg, b->metalFavg.y),
				GGXEnergy_ReflectScale(E, Eavg, b->metalFavg.z));

		result += (m * spec) * OpenPBR_MetalFresnel(b, cosEH) * ms * film;
		*baseE += m * OpenPBR_MetalFresnel(b, muE);
	}
	if (m < 1.f) {
		const float msScale = GGXEnergy_ReflectScale(E, Eavg, b->etaFavg);
		const float3 specular = (spec * FresnelCauchy_Evaluate(b->eta, cosEH) * msScale) * b->specularColor * film;
		const float3 diffuse = EON_Evaluate(b->albedo, b->baseDiffuseRoughness, lightDir, eyeDir) * muL;
		const float3 Espec = b->specularColor * (GGXEnergy_DielectricE(av, av, muE, b->eta) * msScale);

		result += (1.f - m) * (specular + (WHITE - Espec) * diffuse);
		*baseE += (1.f - m) * (Espec + (WHITE - Espec) * b->albedo);
	}

	return result;
}

// f |cos light| for two directions above the surface (front face, rotated
// frame)
OPENCL_FORCE_NOT_INLINE float3 OpenPBR_EvaluateFront(const OpenPBRBSDF *b,
		const float3 lightDir, const float3 eyeDir) {
	const float muL = lightDir.z;
	const float muE = eyeDir.z;
	if ((muL < DEFAULT_COS_EPSILON_STATIC) || (muE < DEFAULT_COS_EPSILON_STATIC))
		return BLACK;

	const float3 wh = normalize(lightDir + eyeDir);
	float3 baseE;
	float3 result = OpenPBR_EvaluateBase(b, lightDir, eyeDir, wh, &baseE);

	if (b->coatWeight > 0.f) {
		const float cosEH = fmax(0.f, dot(eyeDir, wh));
		const float av = OpenPBR_ViewAlpha(b->coatAx, b->coatAy, eyeDir);
		const float msScale = OpenPBR_MSScale(av, muE, b->coatEtaFavg);
		const float coat = b->coatWeight * msScale *
				Microfacet_D(b->coatAx, b->coatAy, wh) * Microfacet_G(b->coatAx, b->coatAy, lightDir, eyeDir) / (4.f * muE) *
				FresnelCauchy_Evaluate(b->coatEta, cosEH);
		const float Ec = b->coatWeight * GGXEnergy_DielectricE(av, av, muE, b->coatEta) * msScale;

		result = TO_FLOAT3(coat) + (1.f - Ec) * OpenPBR_CoatTint(b, eyeDir, baseE) * result;
	}

	if (b->fuzzWeight > 0.f) {
		float ltcA, ltcB;
		const float Ef = OpenPBR_FuzzE(b, muE, &ltcA, &ltcB);
		if (Ef > 0.f)
			result = (Ef * Sheen_LTC(ltcA, ltcB, eyeDir, lightDir)) * b->fuzzColor + (1.f - Ef) * result;
	}

	return result;
}

// The probabilities of the lobes from the fixed direction, and the LTC of the
// fixed direction (the fuzz of eye paths)
typedef struct {
	float wFuzz, wCoat, wGlossy, wDiffuse;
	float ltcA, ltcB;
} OpenPBRLobes;

OPENCL_FORCE_INLINE void OpenPBR_LobeWeights(const OpenPBRBSDF *b, const float3 w,
		OpenPBRLobes *lobes) {
	const float mu = w.z;
	const float Ef = OpenPBR_FuzzE(b, mu, &lobes->ltcA, &lobes->ltcB);
	const float Ec = OpenPBR_CoatE(b, w);

	const float m = b->baseMetalness;
	float glossy = 0.f;
	float diffuse = 0.f;
	if (m > 0.f)
		glossy += m * OpenPBR_Y(b, OpenPBR_MetalFresnel(b, mu));
	if (m < 1.f) {
		const float av = OpenPBR_ViewAlpha(b->ax, b->ay, w);
		const float Ed = OpenPBR_Y(b, b->specularColor *
				(GGXEnergy_DielectricE(av, av, mu, b->eta) * OpenPBR_MSScale(av, mu, b->etaFavg)));
		glossy += (1.f - m) * Ed;
		diffuse += (1.f - m) * fmax(0.f, 1.f - Ed) * OpenPBR_Y(b, b->albedo);
	}

	const float rest = (1.f - Ef) * (1.f - Ec);
	lobes->wFuzz = Ef * OpenPBR_Y(b, b->fuzzColor);
	lobes->wCoat = (1.f - Ef) * Ec;
	lobes->wGlossy = rest * glossy;
	lobes->wDiffuse = rest * diffuse;

	const float sum = lobes->wFuzz + lobes->wCoat + lobes->wGlossy + lobes->wDiffuse;
	if (sum > 0.f) {
		const float invSum = 1.f / sum;
		lobes->wFuzz *= invSum;
		lobes->wCoat *= invSum;
		lobes->wGlossy *= invSum;
		lobes->wDiffuse *= invSum;
	}
}

// The pdf of sampling sampledDir from fixedDir (front face, rotated frame).
// The fuzz is sampled from its LTC on eye paths and from the cosine on light
// paths (the LTC is oriented by the eye direction).
OPENCL_FORCE_INLINE float OpenPBR_PdfFront(const OpenPBRBSDF *b, const OpenPBRLobes *lobes,
		const bool fromLight, const float3 fixedDir, const float3 sampledDir) {
	if (sampledDir.z <= 0.f)
		return 0.f;

	float pdf = lobes->wDiffuse * sampledDir.z * M_1_PI_F;
	if (lobes->wGlossy > 0.f)
		pdf += lobes->wGlossy * Microfacet_PdfReflect(b->ax, b->ay, fixedDir, sampledDir);
	if (lobes->wCoat > 0.f)
		pdf += lobes->wCoat * Microfacet_PdfReflect(b->coatAx, b->coatAy, fixedDir, sampledDir);
	if (lobes->wFuzz > 0.f)
		pdf += lobes->wFuzz * (fromLight ? (sampledDir.z * M_1_PI_F) :
			Sheen_LTC(lobes->ltcA, lobes->ltcB, fixedDir, sampledDir));

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
// the opposite on light paths, reversePdfW the pdf of the other path type
// sampling the fixed direction.
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

	OpenPBRLobes lobes;
	OpenPBR_LobeWeights(b, fixedDir, &lobes);
	if (directPdfW)
		*directPdfW = OpenPBR_PdfFront(b, &lobes, fromLight, fixedDir, sampledDir);
	if (reversePdfW) {
		OpenPBRLobes reverseLobes;
		OpenPBR_LobeWeights(b, sampledDir, &reverseLobes);
		*reversePdfW = OpenPBR_PdfFront(b, &reverseLobes, !fromLight, sampledDir, fixedDir);
	}
	*event = ((lobes.wGlossy + lobes.wCoat >= lobes.wDiffuse + lobes.wFuzz) ? GLOSSY : DIFFUSE) | REFLECT;

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

	OpenPBRLobes lobes;
	OpenPBR_LobeWeights(b, fixedDir, &lobes);
	if (lobes.wFuzz + lobes.wCoat + lobes.wGlossy + lobes.wDiffuse <= 0.f)
		return BLACK;

	float3 sampledDir;
	if (passThroughEvent < lobes.wFuzz) {
		if (fromLight) {
			float pdf;
			sampledDir = CosineSampleHemisphereWithPdf(u0, u1, &pdf);
		} else
			sampledDir = Sheen_SampleLTC(lobes.ltcA, lobes.ltcB, fixedDir, u0, u1);
		*event = DIFFUSE | REFLECT;
	} else if (passThroughEvent < lobes.wFuzz + lobes.wCoat) {
		if (Microfacet_SampleReflect(b->coatAx, b->coatAy, fixedDir, u0, u1, &sampledDir) <= 0.f)
			return BLACK;
		*event = GLOSSY | REFLECT;
	} else if (passThroughEvent < lobes.wFuzz + lobes.wCoat + lobes.wGlossy) {
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
	*pdfW = OpenPBR_PdfFront(b, &lobes, fromLight, fixedDir, sampledDir);
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
