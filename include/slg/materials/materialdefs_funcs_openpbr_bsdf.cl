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
//      mix(opaque, translucent, transmission.weight)
//      opaque    the specular over mix(diffuse, subsurface, subsurface.weight)
//        specular  GGX with the exact Fresnel of specular.ior, F0 modulated by
//                  specular.weight, tinted by specular.color
//        diffuse   EON of the albedo base.weight base.color and the roughness
//                  base.diffuseroughness
//        subsurface  on eye paths (HITPOINT_SSS_ENTRY) a cosine-weighted
//                  entry into the object for the random walk of
//                  subsurface_funcs.cl (subsurface.color the albedo of the
//                  medium, subsurface.radius x subsurface.radiusscale its mean
//                  free path), a Lambertian base of subsurface.color elsewhere
//                  (as the subsurface material); thin-walled, a Lambertian
//                  half reflected and half transmitted
//      translucent  the rough dielectric interface of the same GGX lobe: the
//                specular reflection and the refraction tinted by
//                transmission.color (white when transmission.depth > 0: the
//                interior volume does the colour); thin-walled, the
//                transmission is the reflection lobe mirrored through the
//                surface, without refraction. The dispersion
//                (transmission.dispersionscale, dispersionabbenumber) moves
//                the IOR of the interface to the wavelength of the vertex,
//                which tints the refraction
//  The thin film (thinfilm.weight, thickness in micrometres, ior) tints the
//  base specular and the metal with CalcFilmColor().
//
// Every GGX lobe is energy compensated (microfacet_energy_tables.cl), the
// specular alphas are alpha = specular.roughness^2 split by
// specular.anisotropy as the specification does, in the tangent frame rotated
// by tangent.rotation (in turns), shared by the coat. A smooth lobe (roughness
// up to 0.01) is a delta, as the glass material and Cycles' sharp closures:
// Sample() gives it a SPECULAR event (the specular path depth, no MIS),
// Evaluate() and the pdf leave it out. The back face is the mirror image of
// the front one, except for the IOR of the translucent interface, inverted
// inside a (not thin-walled) object.
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
	0.f, 0.f, 0.f,		// transmission.dispersionscale
	20.f, 20.f, 20.f,	// transmission.dispersionabbenumber
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
	// transmissionColor is the tint of the surface transmission: white when
	// transmission.depth > 0 (the interior volume does the colour)
	float transmissionWeight;
	float3 transmissionColor;
	float transmissionDispersionScale, transmissionDispersionAbbe;
	float subsurfaceWeight;
	float3 subsurfaceColor;
	float coatWeight, coatRoughness, coatAnisotropy, coatIor, coatDarkening;
	float3 coatColor;
	float fuzzWeight, fuzzRoughness;
	float3 fuzzColor;
	float thinFilmWeight, thinFilmThickness, thinFilmIor;
	float tangentRotation;
	// The IOR of the exterior volume (1 without one)
	float exteriorIor;
	int thinWalled;
	// The role of the vertex for the subsurface random walk
	// (HitPoint::subsurface, HITPOINT_SSS_*)
	int subsurfaceMode;
	// The path regularization of the hit point (HitPoint::regularization)
	float regularization;
	// A random number of the hit point (HitPoint::passThroughEvent): the
	// wavelength of the dispersion
	float wavelengthU;

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
	// The relative IOR of the translucent interface, the interior over the
	// exterior (specular.weight applied, >= 1 when thin-walled, at the
	// wavelength of the dispersion), and the luminances of its reflection and
	// transmission tints
	float etaT, transYR, transYT;
	// The colour of the wavelength of the dispersion (white without)
	float3 dispersionColor;
	// Whether the base specular lobes (metal, dielectric, translucent) and the
	// coat are smooth enough to be delta lobes, and whether every lobe of the
	// hit point is a delta (OpenPBR_SampleDelta())
	int specDelta, coatDelta, allDelta;
	// Whether the opaque dielectric has a specular reflection (specular.weight
	// and specular.color not 0): without, its albedo tables are skipped
	int dielectricSpec;
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

// The relative IOR of the translucent interface for the IOR iorT relative to
// the exterior: specular.weight modulates its F0 as the base's, kept away
// from 1 (where the refraction is a delta), inverted below 1 unless
// thin-walled
OPENCL_FORCE_INLINE float OpenPBR_TranslucentEta(const OpenPBRBSDF *b, const float iorT) {
	const float etaTIn = (iorT < 1.f) ? (1.f / iorT) : iorT;
	const float rT = (etaTIn - 1.f) / (etaTIn + 1.f);
	const float sqrtF0T = sqrt(clamp(b->specularWeight * rT * rT, 0.f, .99f));
	const float etaT = fmax((1.f + sqrtF0T) / (1.f - sqrtF0T), 1.001f);

	return ((iorT < 1.f) && !b->thinWalled) ? (1.f / etaT) : etaT;
}

// The colour of a wavelength (nm) in the working colour space, averaging to
// white over 380-780 nm (the glass material's)
OPENCL_FORCE_INLINE float3 OpenPBR_WaveLengthColor(const OpenPBRBSDF *b, const float waveLength) {
	float3 c;
	if (waveLength < 440.f)
		c = MAKE_FLOAT3((440.f - waveLength) / (440.f - 380.f), 0.f, 1.f);
	else if (waveLength < 490.f)
		c = MAKE_FLOAT3(0.f, (waveLength - 440.f) / (490.f - 440.f), 1.f);
	else if (waveLength < 510.f)
		c = MAKE_FLOAT3(0.f, 1.f, (510.f - waveLength) / (510.f - 490.f));
	else if (waveLength < 580.f)
		c = MAKE_FLOAT3((waveLength - 510.f) / (580.f - 510.f), 1.f, 0.f);
	else if (waveLength < 645.f)
		c = MAKE_FLOAT3(1.f, (645.f - waveLength) / (645.f - 580.f), 0.f);
	else
		c = MAKE_FLOAT3(1.f, 0.f, 0.f);

	// The intensity falls off near the limits
	float factor = 1.f;
	if (waveLength < 420.f)
		factor = .3f + .7f * (waveLength - 380.f) / (420.f - 380.f);
	else if (waveLength >= 700.f)
		factor = .3f + .7f * (780.f - waveLength) / (780.f - 700.f);
	c = c * factor * MAKE_FLOAT3(1.f / .5652729f, 1.f / .36875f, 1.f / .265375f);

#if defined(LUXRAYS_CPU_DEVICE)
	return MAKE_FLOAT3(dot(b->fromRec709[0], c), dot(b->fromRec709[1], c), dot(b->fromRec709[2], c));
#else
	return WorkingColorSpace_FromRec709(c);
#endif
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
	b->transmissionWeight = clamp(b->transmissionWeight, 0.f, 1.f);
	b->transmissionColor = Spectrum_Clamp(b->transmissionColor);
	b->transmissionDispersionScale = clamp(b->transmissionDispersionScale, 0.f, 1.f);
	b->subsurfaceWeight = clamp(b->subsurfaceWeight, 0.f, 1.f);
	b->subsurfaceColor = Spectrum_Clamp(b->subsurfaceColor);
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
	b->exteriorIor = fmax(b->exteriorIor, 1e-3f);

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
	// A smooth lobe (roughness up to 0.01, not regularized) is a delta, as the
	// glass material and Cycles' sharp closures: sampled only, a SPECULAR
	// event (the specular path depth), left out of Evaluate() and of the pdf
	b->specDelta = (b->ax <= OPENPBR_MIN_ALPHA) && (b->ay <= OPENPBR_MIN_ALPHA);
	b->coatDelta = (b->coatAx <= OPENPBR_MIN_ALPHA) && (b->coatAy <= OPENPBR_MIN_ALPHA);

	const float angle = 2.f * M_PI_F * b->tangentRotation;
	b->cosRot = cos(angle);
	b->sinRot = sin(angle);

	// The base IOR is relative to the exterior, or to the coat where coated.
	// specular.weight multiplies its F0, mapped back to an IOR (the
	// specification's modulation). The opaque base reflects the same from
	// both sides: an IOR below 1 is inverted.
	const float invExteriorIor = 1.f / b->exteriorIor;
	const float ior = b->specularIor * (invExteriorIor + b->coatWeight * (1.f / b->coatIor - invExteriorIor));
	const float etaIn = (ior < 1.f) ? (1.f / ior) : ior;
	const float r0 = (etaIn - 1.f) / (etaIn + 1.f);
	const float sqrtF0 = sqrt(clamp(b->specularWeight * r0 * r0, 0.f, .99f));
	b->eta = (1.f + sqrtF0) / (1.f - sqrtF0);
	b->etaFavg = OpenPBR_DielectricFavg(b->eta);
	b->dielectricSpec = (sqrtF0 > 0.f) && (OpenPBR_Y(b, b->specularColor) > 0.f);

	// The translucent interface refracts with the IOR relative to the exterior
	// (the coat, a parallel layer, doesn't change the refracted direction)
	b->etaT = OpenPBR_TranslucentEta(b, b->specularIor * invExteriorIor);
	b->transYR = OpenPBR_Y(b, b->specularColor);
	b->transYT = OpenPBR_Y(b, b->transmissionColor);

	// The dispersion: the IOR at the wavelength lambda of this vertex (uniform
	// in 380-780 nm), n_d + (n_d - 1) / V (523640 nm^2 / lambda^2 - 1.5168),
	// Cauchy's law through the d, F and C lines (the specification's form),
	// with the Abbe number V = transmission.dispersionabbenumber /
	// transmission.dispersionscale, for the whole interface (a thin wall
	// doesn't refract), so that each wavelength conserves the energy. The
	// refraction takes the colour of the wavelength, the reflection stays
	// white. The wavelength is one per vertex, not one per path (as the glass
	// material), so this is an approximation: tinting the reflection too
	// keeps every channel exact but multiplies independent colours along the
	// internal reflections (the furnace of a rough sphere: relative std 0.07
	// -> 0.50); white, a rough glass reads 1.5% low near the total internal
	// reflection, where the reflected fraction depends on the wavelength.
	b->dispersionColor = WHITE;
	// (an Abbe number of 0 or less is no dispersion: no data)
	if ((b->transmissionWeight > 0.f) && !b->thinWalled && (b->transmissionDispersionScale > 0.f) &&
			(b->transmissionDispersionAbbe > 0.f)) {
		// The hit point's random number, decorrelated from its other uses
		// (the lobe selection)
		const float u = b->wavelengthU * 4093.f;
		const float lambda = 380.f + 400.f * (u - floor(u));
		const float abbe = b->transmissionDispersionAbbe / b->transmissionDispersionScale;
		const float iorL = b->specularIor + (b->specularIor - 1.f) / abbe * (523640.f / (lambda * lambda) - 1.5168f);
		b->etaT = OpenPBR_TranslucentEta(b, fmax(iorL, 1e-3f) * invExteriorIor);
		b->dispersionColor = OpenPBR_WaveLengthColor(b, lambda);
	}

	// Only delta lobes: a smooth metal or glass (no opaque dielectric, whose
	// diffuse isn't a delta), under a smooth coat or none, without fuzz
	b->allDelta = b->specDelta && !(b->fuzzWeight > 0.f) && (!(b->coatWeight > 0.f) || b->coatDelta) &&
			((b->baseMetalness >= 1.f) || (b->transmissionWeight >= 1.f)) &&
			(b->subsurfaceMode != HITPOINT_SSS_EXIT);

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

// The probability of the reflection of the translucent interface at a
// microfacet with the Fresnel reflectance F (the transmission otherwise)
OPENCL_FORCE_INLINE float OpenPBR_TranslucentReflectProb(const OpenPBRBSDF *b, const float F) {
	const float r = F * b->transYR;
	const float t = (1.f - F) * b->transYT;

	return (r + t > 0.f) ? (r / (r + t)) : 1.f;
}

// f |cos light| of the translucent interface for the eye direction above the
// surface (front face, rotated frame), eta the relative IOR of the light side
// over the eye side: the rough dielectric reflection tinted by
// specular.color and the thin film, and the transmission tinted by
// transmission.color, a refraction (Walter et al. 2007, times the colour of
// the dispersion) or, thin-walled, the reflection lobe mirrored through the
// surface. Both are energy compensated by the albedo of the whole interface
// seen from the eye (as Cycles), so the white interface conserves the
// energy.
OPENCL_FORCE_INLINE float3 OpenPBR_EvaluateTranslucent(const OpenPBRBSDF *b,
		const float3 lightDir, const float3 eyeDir, const float eta, const float3 film) {
	const float muE = eyeDir.z;
	const float av = OpenPBR_ViewAlpha(b->ax, b->ay, eyeDir);
	const float E = b->thinWalled ? GGXEnergy_E(av, av, muE, false) : GGXEnergy_GlassE(av, av, muE, eta);
	const float msScale = 1.f / fmax(E, 1e-3f);

	if ((lightDir.z > 0.f) || b->thinWalled) {
		const bool transmit = (lightDir.z < 0.f);
		const float3 l = transmit ? MAKE_FLOAT3(lightDir.x, lightDir.y, -lightDir.z) : lightDir;
		const float3 wh = normalize(l + eyeDir);
		const float cosEH = dot(eyeDir, wh);
		if (!(cosEH > 0.f))
			return BLACK;

		const float F = FresnelCauchy_Evaluate(eta, cosEH);
		const float spec = Microfacet_D(b->ax, b->ay, wh) * Microfacet_G(b->ax, b->ay, l, eyeDir) /
				(4.f * muE) * msScale;

		return transmit ? ((spec * (1.f - F)) * b->transmissionColor) : ((spec * F) * b->specularColor * film);
	}

	// The refraction: the generalized half vector of eta = eta_light / eta_eye
	float3 wh = eta * lightDir + eyeDir;
	if (wh.z < 0.f)
		wh = -wh;
	const float lengthSquared = dot(wh, wh);
	if (!(lengthSquared > 0.f))
		return BLACK;
	wh /= sqrt(lengthSquared);

	// Only the microfacets that both directions see (Sample() never generates
	// the others)
	const float cosEH = dot(eyeDir, wh);
	const float cosLH = dot(lightDir, wh);
	if (!(cosEH > 0.f) || !(cosLH < 0.f))
		return BLACK;

	const float F = FresnelCauchy_Evaluate(eta, cosEH);

	return ((-cosLH) * cosEH * Microfacet_D(b->ax, b->ay, wh) * Microfacet_G(b->ax, b->ay, lightDir, eyeDir) /
			(muE * lengthSquared) * (1.f - F) * msScale) * b->transmissionColor * b->dispersionColor;
}

// The pdf of sampling sampledDir from fixedDir (above the surface) with the
// translucent lobe, eta the relative IOR of the other side over the fixed
// side: a visible normal, then the reflection or the transmission with the
// probability of OpenPBR_TranslucentReflectProb()
OPENCL_FORCE_INLINE float OpenPBR_PdfTranslucent(const OpenPBRBSDF *b, const float eta,
		const float3 fixedDir, const float3 sampledDir) {
	if ((sampledDir.z > 0.f) || b->thinWalled) {
		const bool transmit = (sampledDir.z < 0.f);
		const float3 s = transmit ? MAKE_FLOAT3(sampledDir.x, sampledDir.y, -sampledDir.z) : sampledDir;
		const float3 wh = normalize(fixedDir + s);
		const float cosFH = dot(fixedDir, wh);
		if (!(cosFH > 0.f))
			return 0.f;

		const float pR = OpenPBR_TranslucentReflectProb(b, FresnelCauchy_Evaluate(eta, cosFH));

		return (transmit ? (1.f - pR) : pR) * Microfacet_PdfH(b->ax, b->ay, fixedDir, wh) / (4.f * cosFH);
	}

	float3 wh = fixedDir + eta * sampledDir;
	if (wh.z < 0.f)
		wh = -wh;
	const float lengthSquared = dot(wh, wh);
	if (!(lengthSquared > 0.f))
		return 0.f;
	wh /= sqrt(lengthSquared);

	const float cosFH = dot(fixedDir, wh);
	const float cosSH = dot(sampledDir, wh);
	if (!(cosFH > 0.f) || !(cosSH < 0.f))
		return 0.f;

	const float pR = OpenPBR_TranslucentReflectProb(b, FresnelCauchy_Evaluate(eta, cosFH));

	return (1.f - pR) * Microfacet_PdfH(b->ax, b->ay, fixedDir, wh) * eta * eta * (-cosSH) / lengthSquared;
}

// The albedo of the base seen from the eye direction with the cosine muE (for
// the coat darkening), with Espec the albedo of the dielectric specular
// reflection and av, E, Eavg the albedo terms of its lobe
OPENCL_FORCE_INLINE float3 OpenPBR_BaseAlbedo(const OpenPBRBSDF *b, const float muE,
		const float av, const float msScale, float3 *Espec) {
	const float m = b->baseMetalness;

	float3 baseE = BLACK;
	if (m > 0.f)
		baseE += m * OpenPBR_MetalFresnel(b, muE);
	*Espec = b->dielectricSpec ? (b->specularColor * (GGXEnergy_DielectricE(av, av, muE, b->eta) * msScale)) : BLACK;
	if (m < 1.f) {
		const float opaque = 1.f - b->transmissionWeight;
		if (opaque > 0.f) {
			const float3 diffuseAlbedo = b->albedo + b->subsurfaceWeight * (b->subsurfaceColor - b->albedo);
			baseE += ((1.f - m) * opaque) * (*Espec + (WHITE - *Espec) * diffuseAlbedo);
		}
		if (b->transmissionWeight > 0.f)
			baseE += ((1.f - m) * b->transmissionWeight) * *Espec;
	}

	return baseE;
}

// f |cos light| of the base for the eye direction above the surface (front
// face, rotated frame), etaEye the relative IOR of the translucent interface
// seen from the eye, sssEntry when the subsurface part is the random walk
// (sampled only), delta for the delta lobes only (the others otherwise), and
// the albedo of the base seen from the eye direction (for the coat darkening)
OPENCL_FORCE_INLINE float3 OpenPBR_EvaluateBase(const OpenPBRBSDF *b,
		const float3 lightDir, const float3 eyeDir, const float etaEye,
		const bool sssEntry, const bool delta, float3 *baseE) {
	const float muL = lightDir.z;
	const float muE = eyeDir.z;
	const float m = b->baseMetalness;
	const bool reflect = (muL > 0.f);

	// The albedo terms of the specular lobe, for the metal and the opaque
	// dielectric's specular (the translucent interface has its own)
	const bool specTerms = (m > 0.f) || b->dielectricSpec;
	float av = 0.f, E = 1.f, Eavg = 1.f, msScale = 1.f;
	if (specTerms) {
		av = OpenPBR_ViewAlpha(b->ax, b->ay, eyeDir);
		E = GGXEnergy_E(av, av, muE, false);
		Eavg = GGXEnergy_Eavg(av, av, false);
		msScale = GGXEnergy_ReflectScale(E, Eavg, b->etaFavg);
	}
	float3 Espec;
	*baseE = OpenPBR_BaseAlbedo(b, muE, av, msScale, &Espec);

	float3 film = WHITE;
	if ((b->thinFilmWeight > 0.f) && (b->thinFilmThickness > 0.f))
		film = WHITE + b->thinFilmWeight * (OpenPBR_FilmColor(b, eyeDir) - WHITE);

	const bool specLobes = (b->specDelta != 0) == delta;
	float3 result = BLACK;
	if (reflect) {
		// The microfacet lobe with F = 1, shared by the metal and the dielectric
		const float3 wh = normalize(lightDir + eyeDir);
		const float cosEH = fmax(0.f, dot(eyeDir, wh));
		const float spec = (specLobes && specTerms) ?
			(Microfacet_D(b->ax, b->ay, wh) * Microfacet_G(b->ax, b->ay, lightDir, eyeDir) / (4.f * muE)) : 0.f;

		if ((m > 0.f) && specLobes) {
			const float3 ms = MAKE_FLOAT3(GGXEnergy_ReflectScale(E, Eavg, b->metalFavg.x),
					GGXEnergy_ReflectScale(E, Eavg, b->metalFavg.y),
					GGXEnergy_ReflectScale(E, Eavg, b->metalFavg.z));

			result += (m * spec) * OpenPBR_MetalFresnel(b, cosEH) * ms * film;
		}
		const float opaque = 1.f - b->transmissionWeight;
		if ((m < 1.f) && (opaque > 0.f)) {
			const float3 specular = (b->dielectricSpec && (spec > 0.f)) ?
				((spec * FresnelCauchy_Evaluate(b->eta, cosEH) * msScale) * b->specularColor * film) : BLACK;
			float3 diffuse = BLACK;
			if (!delta) {
				// EON is Lambertian at roughness 0 (E = 1, no multiple scattering)
				if (b->subsurfaceWeight < 1.f)
					diffuse = (1.f - b->subsurfaceWeight) * muL * ((b->baseDiffuseRoughness > 0.f) ?
						EON_Evaluate(b->albedo, b->baseDiffuseRoughness, lightDir, eyeDir) : (M_1_PI_F * b->albedo));
				// The subsurface is a Lambertian base where it isn't the random
				// walk, half reflected and half transmitted when thin-walled
				if ((b->subsurfaceWeight > 0.f) && !sssEntry)
					diffuse += (b->subsurfaceWeight * (b->thinWalled ? .5f : 1.f) * M_1_PI_F * muL) * b->subsurfaceColor;
			}

			result += ((1.f - m) * opaque) * (specular + (WHITE - Espec) * diffuse);
		}
	} else if (!delta && b->thinWalled && (b->subsurfaceWeight > 0.f) && (b->transmissionWeight < 1.f)) {
		result += ((1.f - m) * (1.f - b->transmissionWeight) * b->subsurfaceWeight * .5f * M_1_PI_F * (-muL)) *
				(WHITE - Espec) * b->subsurfaceColor;
	}

	if ((m < 1.f) && (b->transmissionWeight > 0.f) && specLobes)
		result += ((1.f - m) * b->transmissionWeight) * OpenPBR_EvaluateTranslucent(b, lightDir, eyeDir, etaEye, film);

	return result;
}

// f |cos light| for the eye direction above the surface (front face, rotated
// frame), see OpenPBR_EvaluateBase() for etaEye, sssEntry and delta
OPENCL_FORCE_NOT_INLINE float3 OpenPBR_EvaluateFront(const OpenPBRBSDF *b,
		const float3 lightDir, const float3 eyeDir, const float etaEye,
		const bool sssEntry, const bool delta) {
	const float muE = eyeDir.z;
	if ((fabs(lightDir.z) < DEFAULT_COS_EPSILON_STATIC) || (muE < DEFAULT_COS_EPSILON_STATIC))
		return BLACK;
	const bool reflect = (lightDir.z > 0.f);

	float3 baseE;
	float3 result = OpenPBR_EvaluateBase(b, lightDir, eyeDir, etaEye, sssEntry, delta, &baseE);

	if (b->coatWeight > 0.f) {
		const float av = OpenPBR_ViewAlpha(b->coatAx, b->coatAy, eyeDir);
		const float msScale = OpenPBR_MSScale(av, muE, b->coatEtaFavg);
		const float Ec = b->coatWeight * GGXEnergy_DielectricE(av, av, muE, b->coatEta) * msScale;

		float coat = 0.f;
		if (reflect && ((b->coatDelta != 0) == delta)) {
			const float3 wh = normalize(lightDir + eyeDir);
			const float cosEH = fmax(0.f, dot(eyeDir, wh));
			coat = b->coatWeight * msScale *
					Microfacet_D(b->coatAx, b->coatAy, wh) * Microfacet_G(b->coatAx, b->coatAy, lightDir, eyeDir) / (4.f * muE) *
					FresnelCauchy_Evaluate(b->coatEta, cosEH);
		}

		result = TO_FLOAT3(coat) + (1.f - Ec) * OpenPBR_CoatTint(b, eyeDir, baseE) * result;
	}

	if (b->fuzzWeight > 0.f) {
		float ltcA, ltcB;
		const float Ef = OpenPBR_FuzzE(b, muE, &ltcA, &ltcB);
		if (Ef > 0.f)
			result = ((reflect && !delta) ? (Ef * Sheen_LTC(ltcA, ltcB, eyeDir, lightDir)) * b->fuzzColor : BLACK) +
					(1.f - Ef) * result;
	}

	return result;
}

// The probabilities of the lobes from the fixed direction, the LTC of the
// fixed direction (the fuzz of eye paths) and the relative IOR of the
// translucent interface seen from it
typedef struct {
	float wFuzz, wCoat, wGlossy, wDiffuse, wTranslucent, wDiffuseT, wSubsurface;
	float ltcA, ltcB;
	float eta;
} OpenPBRLobes;

// w is the fixed direction above the surface (front face, rotated frame),
// outside when it is on the front side of the shading normal
OPENCL_FORCE_INLINE void OpenPBR_LobeWeights(const OpenPBRBSDF *b, const float3 w,
		const bool outside, const bool fromLight, OpenPBRLobes *lobes) {
	const float mu = w.z;
	const float Ef = OpenPBR_FuzzE(b, mu, &lobes->ltcA, &lobes->ltcB);
	const float Ec = OpenPBR_CoatE(b, w);
	lobes->eta = (b->thinWalled || outside) ? b->etaT : (1.f / b->etaT);

	const float m = b->baseMetalness;
	float glossy = 0.f;
	float diffuse = 0.f;
	float translucent = 0.f;
	float diffuseT = 0.f;
	float subsurface = 0.f;
	if (m > 0.f)
		glossy += m * OpenPBR_Y(b, OpenPBR_MetalFresnel(b, mu));
	if (m < 1.f) {
		const float opaque = (1.f - m) * (1.f - b->transmissionWeight);
		if (opaque > 0.f) {
			float Ed = 0.f;
			if (b->dielectricSpec) {
				const float av = OpenPBR_ViewAlpha(b->ax, b->ay, w);
				Ed = OpenPBR_Y(b, b->specularColor *
						(GGXEnergy_DielectricE(av, av, mu, b->eta) * OpenPBR_MSScale(av, mu, b->etaFavg)));
			}
			glossy += opaque * Ed;
			const float under = opaque * fmax(0.f, 1.f - Ed);
			diffuse += under * (1.f - b->subsurfaceWeight) * OpenPBR_Y(b, b->albedo);
			if (b->subsurfaceWeight > 0.f) {
				const float sss = under * b->subsurfaceWeight * OpenPBR_Y(b, b->subsurfaceColor);
				if (b->thinWalled) {
					diffuse += .5f * sss;
					diffuseT += .5f * sss;
				} else if ((b->subsurfaceMode == HITPOINT_SSS_ENTRY) && outside && !fromLight)
					subsurface += sss;
				else
					diffuse += sss;
			}
		}
		if (b->transmissionWeight > 0.f) {
			const float F = FresnelCauchy_Evaluate(lobes->eta, mu);
			translucent += (1.f - m) * b->transmissionWeight * (F * b->transYR + (1.f - F) * b->transYT);
		}
	}

	const float rest = (1.f - Ef) * (1.f - Ec);
	lobes->wFuzz = Ef * OpenPBR_Y(b, b->fuzzColor);
	lobes->wCoat = (1.f - Ef) * Ec;
	lobes->wGlossy = rest * glossy;
	lobes->wDiffuse = rest * diffuse;
	lobes->wTranslucent = rest * translucent;
	lobes->wDiffuseT = rest * diffuseT;
	lobes->wSubsurface = rest * subsurface;

	const float sum = lobes->wFuzz + lobes->wCoat + lobes->wGlossy + lobes->wDiffuse +
			lobes->wTranslucent + lobes->wDiffuseT + lobes->wSubsurface;
	if (sum > 0.f) {
		const float invSum = 1.f / sum;
		lobes->wFuzz *= invSum;
		lobes->wCoat *= invSum;
		lobes->wGlossy *= invSum;
		lobes->wDiffuse *= invSum;
		lobes->wTranslucent *= invSum;
		lobes->wDiffuseT *= invSum;
		lobes->wSubsurface *= invSum;
	}
}

// The pdf of sampling sampledDir from fixedDir (fixedDir above the surface,
// front face, rotated frame), of the delta lobes when delta (Sample() of a
// delta lobe), of the others otherwise. The fuzz is sampled from its LTC on
// eye paths and from the cosine on light paths (the LTC is oriented by the
// eye direction). The subsurface entry has no pdf: its sample is the random
// walk, never a direction to evaluate.
OPENCL_FORCE_INLINE float OpenPBR_PdfFront(const OpenPBRBSDF *b, const OpenPBRLobes *lobes,
		const bool fromLight, const float3 fixedDir, const float3 sampledDir, const bool delta) {
	const bool specLobes = (b->specDelta != 0) == delta;
	float pdf = 0.f;
	if (sampledDir.z > 0.f) {
		if (!delta) {
			pdf = lobes->wDiffuse * sampledDir.z * M_1_PI_F;
			if (lobes->wFuzz > 0.f)
				pdf += lobes->wFuzz * (fromLight ? (sampledDir.z * M_1_PI_F) :
					Sheen_LTC(lobes->ltcA, lobes->ltcB, fixedDir, sampledDir));
		}
		if ((lobes->wGlossy > 0.f) && specLobes)
			pdf += lobes->wGlossy * Microfacet_PdfReflect(b->ax, b->ay, fixedDir, sampledDir);
		if ((lobes->wCoat > 0.f) && ((b->coatDelta != 0) == delta))
			pdf += lobes->wCoat * Microfacet_PdfReflect(b->coatAx, b->coatAy, fixedDir, sampledDir);
	} else if (sampledDir.z < 0.f) {
		if (!delta)
			pdf = lobes->wDiffuseT * (-sampledDir.z) * M_1_PI_F;
	} else
		return 0.f;

	if ((lobes->wTranslucent > 0.f) && specLobes)
		pdf += lobes->wTranslucent * OpenPBR_PdfTranslucent(b, lobes->eta, fixedDir, sampledDir);

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

// Whether the material can transmit at this hit point
OPENCL_FORCE_INLINE bool OpenPBR_CanTransmit(const OpenPBRBSDF *b) {
	return (b->baseMetalness < 1.f) && ((b->transmissionWeight > 0.f) ||
			(b->thinWalled && (b->subsurfaceWeight > 0.f) && (b->transmissionWeight < 1.f)));
}

// f |cos light| for the local directions in the shading frame (of the delta
// lobes when delta, of the others otherwise): the eye direction decides the
// face. Light paths get the adjoint of the refraction, the radiance BTDF
// times (eta_light / eta_eye)^2.
OPENCL_FORCE_INLINE float3 OpenPBR_EvaluateLocal(const OpenPBRBSDF *b,
		const float3 localLightDir, const float3 localEyeDir, const bool fromLight, const bool delta) {
	const bool eyeOutside = (localEyeDir.z > 0.f);
	const float flip = eyeOutside ? 1.f : -1.f;
	const float3 lightDir = OpenPBR_ToLobeFrame(b, localLightDir, flip);
	const float3 eyeDir = OpenPBR_ToLobeFrame(b, localEyeDir, flip);
	const float etaEye = (b->thinWalled || eyeOutside) ? b->etaT : (1.f / b->etaT);
	const bool sssEntry = (b->subsurfaceMode == HITPOINT_SSS_ENTRY) && eyeOutside && !b->thinWalled;

	const float3 result = OpenPBR_EvaluateFront(b, lightDir, eyeDir, etaEye, sssEntry, delta);

	return (fromLight && (lightDir.z < 0.f) && !b->thinWalled) ? (result * (etaEye * etaEye)) : result;
}

// What the coat leaves of the base of a delta-only hit point seen from the eye
// direction with the cosine muE: 1 - the coat albedo, times the tint and the
// darkening of OpenPBR_CoatTint() (the albedos are the Fresnel terms: the
// limit of the energy tables at the smallest alpha)
OPENCL_FORCE_INLINE float3 OpenPBR_DeltaBelowCoat(const OpenPBRBSDF *b, const float muE) {
	if (!(b->coatWeight > 0.f))
		return WHITE;

	const float m = b->baseMetalness;
	float3 baseE = BLACK;
	if (m > 0.f)
		baseE += m * OpenPBR_MetalFresnel(b, muE);
	if (m < 1.f)
		baseE += ((1.f - m) * FresnelCauchy_Evaluate(b->eta, muE)) * b->specularColor;
	const float Ec = b->coatWeight * FresnelCauchy_Evaluate(b->coatEta, muE);

	return (1.f - Ec) * OpenPBR_CoatTint(b, MAKE_FLOAT3(0.f, 0.f, muE), baseE);
}

// Sample() of a hit point with only delta lobes (OpenPBRBSDF::allDelta): the
// mirror reflection or the refraction (straight through when thin-walled),
// chosen by the luminance of the energy they carry (without the radiance
// scaling of the refraction, as the general path: a closed white glass then
// weighs exactly 1 along a path), in closed form. These are the
// limits of the general path's f / pdf (bsdftest compares the albedos):
// without the microfacet sampling, the albedo tables and the mixture.
OPENCL_FORCE_INLINE float3 OpenPBR_SampleDelta(const OpenPBRBSDF *b,
		const float3 localFixedDir, const float passThroughEvent, const bool fromLight,
		float3 *localSampledDir, float *pdfW, BSDFEvent *event) {
	const bool outside = (localFixedDir.z > 0.f);
	const float mu = fabs(localFixedDir.z);
	const float m = b->baseMetalness;
	// The relative IOR of the other side over the fixed one
	const float eta = (b->thinWalled || outside) ? b->etaT : (1.f / b->etaT);

	float3 film = WHITE;
	if ((b->thinFilmWeight > 0.f) && (b->thinFilmThickness > 0.f))
		film = WHITE + b->thinFilmWeight * (OpenPBR_FilmColor(b, MAKE_FLOAT3(0.f, 0.f, mu)) - WHITE);

	// The reflection: the coat, then what it leaves of the metal and of the
	// translucent interface's reflection
	const float FT = (m < 1.f) ? FresnelCauchy_Evaluate(eta, mu) : 1.f;
	float3 baseR = BLACK;
	if (m > 0.f)
		baseR += m * OpenPBR_MetalFresnel(b, mu);
	if (m < 1.f)
		baseR += ((1.f - m) * FT) * b->specularColor;
	float3 reflect = OpenPBR_DeltaBelowCoat(b, mu) * baseR * film;
	if (b->coatWeight > 0.f)
		reflect += TO_FLOAT3(b->coatWeight * FresnelCauchy_Evaluate(b->coatEta, mu));

	// The transmission (none at the total internal reflection, F = 1)
	float3 transmit = BLACK;
	float3 transDir = BLACK;
	float scale = 1.f;
	if ((m < 1.f) && (FT < 1.f)) {
		float muE = mu;
		if (b->thinWalled)
			transDir = MAKE_FLOAT3(-localFixedDir.x, -localFixedDir.y, -localFixedDir.z);
		else {
			const float invEta = 1.f / eta;
			const float cosT = sqrt(fmax(0.f, 1.f - invEta * invEta * (1.f - mu * mu)));
			transDir = MAKE_FLOAT3(-invEta * localFixedDir.x, -invEta * localFixedDir.y,
					outside ? -cosT : cosT);
			// The radiance form on eye paths, (eta_eye / eta_light)^2, and on
			// light paths the same times |cos fixed / cos sampled| (the
			// Jacobian of the refraction)
			scale = invEta * invEta;
			if (fromLight) {
				muE = cosT;
				scale *= mu / fmax(cosT, 1e-6f);
			}
		}
		// Without the scale for now
		transmit = ((1.f - m) * (1.f - FT)) * OpenPBR_DeltaBelowCoat(b, muE) *
				b->transmissionColor * b->dispersionColor;
	}

	const float yR = OpenPBR_Y(b, reflect);
	const float yT = OpenPBR_Y(b, transmit);
	if (!(yR + yT > 0.f))
		return BLACK;
	const float pR = yR / (yR + yT);

	if (passThroughEvent < pR) {
		*localSampledDir = MAKE_FLOAT3(-localFixedDir.x, -localFixedDir.y, localFixedDir.z);
		*event = SPECULAR | REFLECT;
		*pdfW = pR;

		return reflect / pR;
	}

	*localSampledDir = transDir;
	if (fabs(transDir.z) < DEFAULT_COS_EPSILON_STATIC)
		return BLACK;
	*event = SPECULAR | TRANSMIT;
	*pdfW = 1.f - pR;

	return transmit * (scale / (1.f - pR));
}

//------------------------------------------------------------------------------
// The material interface (local directions in the shading frame)
//------------------------------------------------------------------------------

OPENCL_FORCE_INLINE float3 OpenPBR_Albedo(const OpenPBRBSDF *b) {
	const float3 opaque = b->albedo + b->subsurfaceWeight * (b->subsurfaceColor - b->albedo);
	const float3 dielectric = opaque + b->transmissionWeight * (b->transmissionColor - opaque);

	return dielectric + b->baseMetalness * (b->albedo - dielectric);
}

// f |cos light| (as Material::Evaluate()). directPdfW is the pdf of sampling
// the light direction from the eye one on eye paths (fromLight = false) and
// the opposite on light paths, reversePdfW the pdf of the other path type
// sampling the fixed direction.
OPENCL_FORCE_INLINE float3 OpenPBR_Evaluate(const OpenPBRBSDF *b,
		const float3 localLightDir, const float3 localEyeDir, const bool fromLight,
		BSDFEvent *event, float *directPdfW, float *reversePdfW) {
	const float3 localFixedDir = fromLight ? localLightDir : localEyeDir;
	const float3 localSampledDir = fromLight ? localEyeDir : localLightDir;

	if (b->subsurfaceMode == HITPOINT_SSS_EXIT) {
		// The exit point of a random walk: a Lambertian transmission out of
		// the object (as the subsurface material)
		if (localLightDir.z * localEyeDir.z >= 0.f)
			return BLACK;

		if (directPdfW)
			*directPdfW = fabs(localSampledDir.z) * M_1_PI_F;
		if (reversePdfW)
			*reversePdfW = fabs(localFixedDir.z) * M_1_PI_F;
		*event = DIFFUSE | TRANSMIT;

		return TO_FLOAT3(fabs(localLightDir.z) * M_1_PI_F);
	}

	// Only delta lobes: nothing to evaluate
	if (b->allDelta)
		return BLACK;

	const bool reflect = (localLightDir.z * localEyeDir.z > 0.f);
	if (!reflect && ((localLightDir.z * localEyeDir.z == 0.f) || !OpenPBR_CanTransmit(b)))
		return BLACK;

	const float3 result = OpenPBR_EvaluateLocal(b, localLightDir, localEyeDir, fromLight, false);
	if (Spectrum_IsBlack(result))
		return BLACK;

	const bool fixedOutside = (localFixedDir.z > 0.f);
	const float fixedFlip = fixedOutside ? 1.f : -1.f;
	const float3 fixedDir = OpenPBR_ToLobeFrame(b, localFixedDir, fixedFlip);
	OpenPBRLobes lobes;
	OpenPBR_LobeWeights(b, fixedDir, fixedOutside, fromLight, &lobes);
	if (directPdfW)
		*directPdfW = OpenPBR_PdfFront(b, &lobes, fromLight, fixedDir,
				OpenPBR_ToLobeFrame(b, localSampledDir, fixedFlip), false);
	if (reversePdfW) {
		const bool sampledOutside = (localSampledDir.z > 0.f);
		const float sampledFlip = sampledOutside ? 1.f : -1.f;
		const float3 sampledDir = OpenPBR_ToLobeFrame(b, localSampledDir, sampledFlip);
		OpenPBRLobes reverseLobes;
		OpenPBR_LobeWeights(b, sampledDir, sampledOutside, !fromLight, &reverseLobes);
		*reversePdfW = OpenPBR_PdfFront(b, &reverseLobes, !fromLight, sampledDir,
				OpenPBR_ToLobeFrame(b, localFixedDir, sampledFlip), false);
	}
	// The kind of the non-delta lobes
	const float wSpec = b->specDelta ? 0.f : (lobes.wGlossy + lobes.wTranslucent);
	const float wCoat = b->coatDelta ? 0.f : lobes.wCoat;
	if (reflect)
		*event = ((wSpec + wCoat >= lobes.wDiffuse + lobes.wFuzz) ? GLOSSY : DIFFUSE) | REFLECT;
	else
		*event = ((!b->specDelta && (lobes.wTranslucent >= lobes.wDiffuseT)) ? GLOSSY : DIFFUSE) | TRANSMIT;

	return result;
}

// f |cos light| / pdf (as Material::Sample()), BLACK when there is no sample.
// The subsurface entry (event SUBSURFACE) returns the weight of the walk
// instead: the part of the light that enters the surface over the
// probability of the lobe.
OPENCL_FORCE_INLINE float3 OpenPBR_Sample(const OpenPBRBSDF *b,
		const float3 localFixedDir, const float u0, const float u1, const float passThroughEvent,
		const bool fromLight, float3 *localSampledDir, float *pdfW, BSDFEvent *event) {
	if (fabs(localFixedDir.z) < DEFAULT_COS_EPSILON_STATIC)
		return BLACK;

	if (b->subsurfaceMode == HITPOINT_SSS_EXIT) {
		// The exit point of a random walk: a Lambertian transmission
		*localSampledDir = CosineSampleHemisphereWithPdf(u0, u1, pdfW);
		if (localFixedDir.z > 0.f)
			localSampledDir->z = -localSampledDir->z;
		if (fabs(localSampledDir->z) < DEFAULT_COS_EPSILON_STATIC)
			return BLACK;

		*event = DIFFUSE | TRANSMIT;
		return TO_FLOAT3(fromLight ? fabs(localFixedDir.z / localSampledDir->z) : 1.f);
	}

	if (b->allDelta)
		return OpenPBR_SampleDelta(b, localFixedDir, passThroughEvent, fromLight, localSampledDir, pdfW, event);

	const bool outside = (localFixedDir.z > 0.f);
	const float flip = outside ? 1.f : -1.f;
	const float3 fixedDir = OpenPBR_ToLobeFrame(b, localFixedDir, flip);

	OpenPBRLobes lobes;
	OpenPBR_LobeWeights(b, fixedDir, outside, fromLight, &lobes);
	if (lobes.wFuzz + lobes.wCoat + lobes.wGlossy + lobes.wDiffuse +
			lobes.wTranslucent + lobes.wDiffuseT + lobes.wSubsurface <= 0.f)
		return BLACK;

	float3 sampledDir;
	// Whether the sampled lobe is a delta
	bool delta = false;
	const float cFuzz = lobes.wFuzz;
	const float cCoat = cFuzz + lobes.wCoat;
	const float cGlossy = cCoat + lobes.wGlossy;
	const float cTranslucent = cGlossy + lobes.wTranslucent;
	const float cSubsurface = cTranslucent + lobes.wSubsurface;
	const float cDiffuseT = cSubsurface + lobes.wDiffuseT;
	if (passThroughEvent < cFuzz) {
		if (fromLight) {
			float pdf;
			sampledDir = CosineSampleHemisphereWithPdf(u0, u1, &pdf);
		} else
			sampledDir = Sheen_SampleLTC(lobes.ltcA, lobes.ltcB, fixedDir, u0, u1);
		*event = DIFFUSE | REFLECT;
	} else if (passThroughEvent < cCoat) {
		if (Microfacet_SampleReflect(b->coatAx, b->coatAy, fixedDir, u0, u1, &sampledDir) <= 0.f)
			return BLACK;
		delta = b->coatDelta;
		*event = (delta ? SPECULAR : GLOSSY) | REFLECT;
	} else if (passThroughEvent < cGlossy) {
		if (Microfacet_SampleReflect(b->ax, b->ay, fixedDir, u0, u1, &sampledDir) <= 0.f)
			return BLACK;
		delta = b->specDelta;
		*event = (delta ? SPECULAR : GLOSSY) | REFLECT;
	} else if (passThroughEvent < cTranslucent) {
		// A visible normal, then the reflection or the transmission chosen
		// with the rest of the lobe selection sample
		float3 wh;
		float d, pdfH;
		Microfacet_SampleH(b->ax, b->ay, fixedDir, u0, u1, &wh, &d, &pdfH);
		const float cosFH = dot(fixedDir, wh);
		if (!(cosFH > 0.f))
			return BLACK;

		const float pR = OpenPBR_TranslucentReflectProb(b, FresnelCauchy_Evaluate(lobes.eta, cosFH));
		const float u = (passThroughEvent - cGlossy) / lobes.wTranslucent;
		delta = b->specDelta;
		const BSDFEvent kind = delta ? SPECULAR : GLOSSY;
		if ((u < pR) || b->thinWalled) {
			sampledDir = 2.f * cosFH * wh - fixedDir;
			if (sampledDir.z <= 0.f)
				return BLACK;
			if (u < pR)
				*event = kind | REFLECT;
			else {
				// The thin wall: the reflection mirrored through the surface
				sampledDir.z = -sampledDir.z;
				*event = kind | TRANSMIT;
			}
		} else {
			// The refraction, with the IOR of the fixed side over the other
			const float invEta = 1.f / lobes.eta;
			const float sin2T = invEta * invEta * fmax(0.f, 1.f - cosFH * cosFH);
			if (sin2T >= 1.f)
				return BLACK;
			const float cosT = sqrt(1.f - sin2T);
			sampledDir = (invEta * cosFH - cosT) * wh - invEta * fixedDir;
			if (sampledDir.z >= 0.f)
				return BLACK;
			*event = kind | TRANSMIT;
		}
	} else if (passThroughEvent < cSubsurface) {
		// Into the object, for the random walk (eye paths)
		float pdf;
		sampledDir = CosineSampleHemisphereWithPdf(u0, u1, &pdf);
		if (sampledDir.z < DEFAULT_COS_EPSILON_STATIC)
			return BLACK;
		sampledDir.z = -sampledDir.z;

		// The part of the light that enters: what the layers above leave of
		// the subsurface fraction of the base, under its specular reflection
		const float muE = fixedDir.z;
		const float av = OpenPBR_ViewAlpha(b->ax, b->ay, fixedDir);
		float3 Espec;
		const float3 baseE = OpenPBR_BaseAlbedo(b, muE, av, OpenPBR_MSScale(av, muE, b->etaFavg), &Espec);
		float3 weight = ((1.f - b->baseMetalness) * (1.f - b->transmissionWeight) * b->subsurfaceWeight) *
				(WHITE - Espec);
		if (b->coatWeight > 0.f)
			weight *= (1.f - OpenPBR_CoatE(b, fixedDir)) * OpenPBR_CoatTint(b, fixedDir, baseE);
		if (b->fuzzWeight > 0.f) {
			float ltcA, ltcB;
			weight *= 1.f - OpenPBR_FuzzE(b, muE, &ltcA, &ltcB);
		}

		*localSampledDir = OpenPBR_FromLobeFrame(b, sampledDir, flip);
		*pdfW = lobes.wSubsurface * pdf;
		*event = DIFFUSE | TRANSMIT | SUBSURFACE;

		return weight / lobes.wSubsurface;
	} else if (passThroughEvent < cDiffuseT) {
		float pdf;
		sampledDir = CosineSampleHemisphereWithPdf(u0, u1, &pdf);
		sampledDir.z = -sampledDir.z;
		*event = DIFFUSE | TRANSMIT;
	} else {
		float pdf;
		sampledDir = CosineSampleHemisphereWithPdf(u0, u1, &pdf);
		*event = DIFFUSE | REFLECT;
	}
	if (fabs(sampledDir.z) < DEFAULT_COS_EPSILON_STATIC)
		return BLACK;

	// The value and the pdf of the mixture: of the delta lobes for a delta
	// sample (their lobes are very sharp GGX ones, so f / pdf is the weight
	// of the delta), of the others otherwise
	const float pdf = OpenPBR_PdfFront(b, &lobes, fromLight, fixedDir, sampledDir, delta);
	if (!(pdf > 0.f))
		return BLACK;

	*localSampledDir = OpenPBR_FromLobeFrame(b, sampledDir, flip);
	const float3 f = fromLight ?
		OpenPBR_EvaluateLocal(b, localFixedDir, *localSampledDir, true, delta) :
		OpenPBR_EvaluateLocal(b, *localSampledDir, localFixedDir, false, delta);
	if (Spectrum_IsBlack(f))
		return BLACK;

	// A delta sample's pdf is the probability of its lobes (as the glass
	// material: the engines don't use it for MIS)
	*pdfW = delta ? ((b->coatDelta ? lobes.wCoat : 0.f) +
			(b->specDelta ? (lobes.wGlossy + lobes.wTranslucent) : 0.f)) : pdf;

	return f / pdf;
}

// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
