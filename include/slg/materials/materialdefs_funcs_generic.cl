#line 2 "materialdefs_funcs_generic.cl"

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
//------------------------------------------------------------------------------
// Main material functions
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

// The functions below read the material and texture buffers: GPU only (the
// rest of this file is also compiled on the CPU, see openpbr.cpp)
#if !defined(LUXRAYS_CPU_DEVICE)

//------------------------------------------------------------------------------
// Material_GetEventTypes
//------------------------------------------------------------------------------

OPENCL_FORCE_INLINE BSDFEvent Material_GetEventTypes(const uint matIndex
		MATERIALS_PARAM_DECL) {
	__global const Material *material = &mats[matIndex];

	return material->eventTypes;
}

//------------------------------------------------------------------------------
// Material_IsDelta
//------------------------------------------------------------------------------

OPENCL_FORCE_INLINE bool Material_IsDelta(const uint matIndex
		MATERIALS_PARAM_DECL) {
	__global const Material *material = &mats[matIndex];

	return material->isDelta;
}

//------------------------------------------------------------------------------
// Material_GetEmittedCosThetaMax
//------------------------------------------------------------------------------

OPENCL_FORCE_INLINE float Material_GetEmittedCosThetaMax(const uint matIndex
		MATERIALS_PARAM_DECL) {
	__global const Material *material = &mats[matIndex];

	return material->emittedCosThetaMax;
}

//------------------------------------------------------------------------------
// Material_Bump
//------------------------------------------------------------------------------

OPENCL_FORCE_INLINE void Material_Bump(const uint matIndex, __global HitPoint *hitPoint
	MATERIALS_PARAM_DECL) {
	__global const Material *material = &mats[matIndex];

	uint bumpTexIndex;
	
	// A special case here for TWOSIDED material
	if (material->type == TWOSIDED) {
		bumpTexIndex = hitPoint->intoObject ? mats[material->twosided.frontMatIndex].bumpTexIndex : 
			mats[material->twosided.backMatIndex].bumpTexIndex;
	} else
		bumpTexIndex = material->bumpTexIndex;

	if (bumpTexIndex != NULL_INDEX) {
		const float3 shadeN = Texture_Bump(bumpTexIndex, hitPoint, material->bumpSampleDistance
			TEXTURES_PARAM);

		// Update dpdu and dpdv so they are still orthogonal to shadeN
		float3 dpdu = VLOAD3F(&hitPoint->dpdu.x);
		float3 dpdv = VLOAD3F(&hitPoint->dpdv.x);
		dpdu = cross(shadeN, cross(dpdu, shadeN));
		dpdv = cross(shadeN, cross(dpdv, shadeN));
		// Update HitPoint structure
		VSTORE3F(shadeN, &hitPoint->shadeN.x);
		VSTORE3F(dpdu, &hitPoint->dpdu.x);
		VSTORE3F(dpdv, &hitPoint->dpdv.x);
	}
}

//------------------------------------------------------------------------------
// Material_GetGlossiness
//------------------------------------------------------------------------------

OPENCL_FORCE_INLINE float Material_GetGlossiness(const uint matIndex
		MATERIALS_PARAM_DECL) {
	return mats[matIndex].glossiness;
}

//------------------------------------------------------------------------------
// Material_IsPhotonGIEnabled
//------------------------------------------------------------------------------

OPENCL_FORCE_INLINE float Material_IsPhotonGIEnabled(const uint matIndex
		MATERIALS_PARAM_DECL) {
	return mats[matIndex].isPhotonGIEnabled;
}

//------------------------------------------------------------------------------
// Material_IsHoldout
//------------------------------------------------------------------------------

OPENCL_FORCE_INLINE float Material_IsHoldout(const uint matIndex
		MATERIALS_PARAM_DECL) {
	return mats[matIndex].isHoldout;
}

//------------------------------------------------------------------------------
// Material_HasSubsurfaceWalk: an eye path entering the material starts the
// subsurface random walk (Material::GetSubsurfaceWalkMaterial())
//------------------------------------------------------------------------------

OPENCL_FORCE_INLINE bool Material_HasSubsurfaceWalk(__global const Material* restrict material) {
	return (material->subsurfaceWalkMatIndex != NULL_INDEX);
}

#endif

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
// Generic material related functions
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

OPENCL_FORCE_INLINE float3 FresnelSchlick_Evaluate(const float3 normalIncidence, const float cosi) {
	return normalIncidence + (WHITE - normalIncidence) *
		pow(1.f - cosi, 5.f);
}

OPENCL_FORCE_INLINE float3 CoatingAbsorption(const float cosi, const float coso,
		const float3 alpha, const float depth) {
	if (depth > 0.f) {
		// 1/cosi+1/coso=(cosi+coso)/(cosi*coso)
		const float depthFactor = depth * (cosi + coso) / (cosi * coso);
		return Spectrum_Exp(alpha * -depthFactor);
	} else
		return WHITE;
}

//------------------------------------------------------------------------------
// GGX distribution with the Smith masking (alpha = (alphaU, alphaV))
//------------------------------------------------------------------------------

OPENCL_FORCE_INLINE float GGXDistribution_D(const float alphaU, const float alphaV, const float3 wh) {
	const float cosTheta2 = wh.z * wh.z;
	if (!(cosTheta2 > 0.f))
		return 0.f;

	const float ax = wh.x / alphaU;
	const float ay = wh.y / alphaV;
	const float e = ax * ax + ay * ay + cosTheta2;

	return 1.f / (M_PI_F * alphaU * alphaV * e * e);
}

OPENCL_FORCE_INLINE float GGXDistribution_Lambda(const float alphaU, const float alphaV, const float3 w) {
	const float cosTheta2 = w.z * w.z;
	if (!(cosTheta2 > 0.f))
		return INFINITY;

	const float ax = alphaU * w.x;
	const float ay = alphaV * w.y;

	return (-1.f + sqrt(1.f + (ax * ax + ay * ay) / cosTheta2)) * .5f;
}

OPENCL_FORCE_INLINE float GGXDistribution_G1(const float alphaU, const float alphaV, const float3 w) {
	return 1.f / (1.f + GGXDistribution_Lambda(alphaU, alphaV, w));
}

OPENCL_FORCE_INLINE float GGXDistribution_G(const float alphaU, const float alphaV, const float3 wo, const float3 wi) {
	// Height-correlated Smith masking-shadowing
	return 1.f / (1.f + GGXDistribution_Lambda(alphaU, alphaV, wo) + GGXDistribution_Lambda(alphaU, alphaV, wi));
}

// The pdf of the visible normals of wo (wo.z > 0)
OPENCL_FORCE_INLINE float GGXDistribution_PdfVisibleH(const float alphaU, const float alphaV, const float3 wo, const float3 wh) {
	const float cosWH = dot(wo, wh);
	if (!(cosWH > 0.f))
		return 0.f;

	return GGXDistribution_G1(alphaU, alphaV, wo) * GGXDistribution_D(alphaU, alphaV, wh) * cosWH / wo.z;
}

// Dupuy & Benyoub 2023, spherical caps (wo.z > 0, wh.z >= 0)
OPENCL_FORCE_INLINE float3 GGXDistribution_SampleVisibleH(const float alphaU, const float alphaV, const float3 wo,
		const float u0, const float u1) {
	// Warp to the hemisphere configuration
	const float3 woStd = normalize(MAKE_FLOAT3(wo.x * alphaU, wo.y * alphaV, wo.z));

	// Sample a spherical cap in (-woStd.z, 1]
	const float phi = 2.f * M_PI_F * u0;
	const float z = (1.f - u1) * (1.f + woStd.z) - woStd.z;
	const float sinTheta = sqrt(clamp(1.f - z * z, 0.f, 1.f));
	const float3 h = MAKE_FLOAT3(sinTheta * cos(phi) + woStd.x, sinTheta * sin(phi) + woStd.y, z + woStd.z);

	// Warp back to the ellipsoid configuration
	return normalize(MAKE_FLOAT3(h.x * alphaU, h.y * alphaV, fmax(0.f, h.z)));
}

// Eto & Tokuyoshi 2023, bounded VNDF sampling of the reflected direction (wo.z > 0)
OPENCL_FORCE_INLINE float GGXDistribution_BoundedK(const float alphaU, const float alphaV, const float3 wo) {
	const float a = clamp(fmin(alphaU, alphaV), 0.f, 1.f);
	const float s = 1.f + sqrt(wo.x * wo.x + wo.y * wo.y);
	const float a2 = a * a;
	const float s2 = s * s;

	return (1.f - a2) * s2 / (s2 + a2 * wo.z * wo.z);
}

OPENCL_FORCE_INLINE float3 GGXDistribution_SampleBoundedReflect(const float alphaU, const float alphaV, const float3 wo,
		const float u0, const float u1) {
	const float3 woStd = normalize(MAKE_FLOAT3(wo.x * alphaU, wo.y * alphaV, wo.z));

	const float phi = 2.f * M_PI_F * u0;
	const float b = GGXDistribution_BoundedK(alphaU, alphaV, wo) * woStd.z;
	const float z = (1.f - u1) * (1.f + b) - b;
	const float sinTheta = sqrt(clamp(1.f - z * z, 0.f, 1.f));
	const float3 h = MAKE_FLOAT3(sinTheta * cos(phi) + woStd.x, sinTheta * sin(phi) + woStd.y, z + woStd.z);

	const float3 wh = normalize(MAKE_FLOAT3(h.x * alphaU, h.y * alphaV, fmax(0.f, h.z)));
	return 2.f * dot(wh, wo) * wh - wo;
}

OPENCL_FORCE_INLINE float GGXDistribution_PdfBoundedReflect(const float alphaU, const float alphaV, const float3 wo, const float3 wi) {
	if (!(wi.z > 0.f))
		return 0.f;

	const float3 wh = normalize(wo + wi);
	const float ax = alphaU * wo.x;
	const float ay = alphaV * wo.y;
	const float t = sqrt(ax * ax + ay * ay + wo.z * wo.z);
	const float k = GGXDistribution_BoundedK(alphaU, alphaV, wo);

	return GGXDistribution_D(alphaU, alphaV, wh) / (2.f * (k * wo.z + t));
}

//------------------------------------------------------------------------------
// Microfacet lobe shared by Metal2, RoughGlass, CarPaint and the coating of
// Glossy2, GlossyCoating and GlossyTranslucent (GGX, see material.h)
//------------------------------------------------------------------------------

// Path regularization (Kaplanyan & Dachsbacher 2013, pbrt-v4's rule): after
// the first non-specular vertex of an eye path the GGX alphas become
// max(alpha, clamp(2 alpha, f, 3 f)), f = HitPoint::regularization (0 = off),
// so a delta lobe gets the alpha f
OPENCL_FORCE_INLINE float Microfacet_Regularize(const float alpha, const float regularization) {
	return (regularization > 0.f) ?
		fmax(alpha, clamp(2.f * alpha, regularization, 3.f * regularization)) :
		alpha;
}

// The Disney material: its GGX alpha is roughness^2 and the clearcoat alpha
// is Lerp(clearcoatGloss, 0.1, 0.001), regularized through the parameters
OPENCL_FORCE_INLINE float Disney_RegularizeRoughness(const float roughness, const float regularization) {
	return (regularization > 0.f) ?
		sqrt(Microfacet_Regularize(roughness * roughness, regularization)) : roughness;
}
OPENCL_FORCE_INLINE float Disney_RegularizeClearcoatGloss(const float clearcoatGloss, const float regularization) {
	if (!(regularization > 0.f))
		return clearcoatGloss;
	const float alpha = Microfacet_Regularize(Lerp(clearcoatGloss, .1f, .001f), regularization);
	return clamp((alpha - .1f) / (.001f - .1f), 0.f, 1.f);
}

OPENCL_FORCE_INLINE float Microfacet_D(const float u, const float v,
		const float3 wh) {
	return GGXDistribution_D(u, v, wh);
}

OPENCL_FORCE_INLINE float Microfacet_G(const float u, const float v,
		const float3 wo, const float3 wi) {
	return GGXDistribution_G(u, v, wo, wi);
}

// Samples a half vector in the upper hemisphere (wo can be on either side).
// d is D(wh) and pdfH the pdf of wh over the solid angle.
OPENCL_FORCE_INLINE void Microfacet_SampleH(const float u, const float v,
		const float3 wo, const float u0, const float u1,
		float3 *wh, float *d, float *pdfH) {
	const float3 woUp = (wo.z < 0.f) ? -wo : wo;
	*wh = GGXDistribution_SampleVisibleH(u, v, woUp, u0, u1);
	*d = GGXDistribution_D(u, v, *wh);
	*pdfH = GGXDistribution_PdfVisibleH(u, v, woUp, *wh);
}

OPENCL_FORCE_INLINE float Microfacet_PdfH(const float u, const float v,
		const float3 wo, const float3 wh) {
	const float3 woUp = (wo.z < 0.f) ? -wo : wo;
	return GGXDistribution_PdfVisibleH(u, v, woUp, wh);
}

// Reflection lobe: samples wi on the same side as wo and returns its pdf over
// the solid angle (0 when the sample is not usable)
OPENCL_FORCE_INLINE float Microfacet_SampleReflect(const float u, const float v,
		const float3 wo, const float u0, const float u1, float3 *wi) {
	const bool flip = (wo.z < 0.f);
	const float3 woUp = flip ? -wo : wo;
	const float3 wiUp = GGXDistribution_SampleBoundedReflect(u, v, woUp, u0, u1);
	*wi = flip ? -wiUp : wiUp;
	return GGXDistribution_PdfBoundedReflect(u, v, woUp, wiUp);
}

OPENCL_FORCE_INLINE float Microfacet_PdfReflect(const float u, const float v,
		const float3 wo, const float3 wi) {
	const bool flip = (wo.z < 0.f);
	return GGXDistribution_PdfBoundedReflect(u, v, flip ? -wo : wo, flip ? -wi : wi);
}

//------------------------------------------------------------------------------
// Multiple-scattering energy compensation of the GGX lobes (always on since
// 2026-10-18), see microfacet_energy_tables.cl
//------------------------------------------------------------------------------

// The hemispherical average of the Schlick Fresnel term: F0 + (1 - F0) / 21
OPENCL_FORCE_INLINE float3 SchlickFresnel_Average(const float3 F0) {
	return F0 + (WHITE - F0) * (1.f / 21.f);
}

OPENCL_FORCE_INLINE float3 GGXEnergy_ReflectScale3(const float u, const float v, const float mu,
		const float3 Fss, const bool separable) {
	const float E = GGXEnergy_E(u, v, mu, separable);
	const float Eavg = GGXEnergy_Eavg(u, v, separable);

	return MAKE_FLOAT3(GGXEnergy_ReflectScale(E, Eavg, Fss.x),
			GGXEnergy_ReflectScale(E, Eavg, Fss.y),
			GGXEnergy_ReflectScale(E, Eavg, Fss.z));
}

// The energy compensation scale of a Microfacet_* reflection lobe seen with
// the cosine mu
OPENCL_FORCE_INLINE float3 Microfacet_MSScale(const float u, const float v, const float mu, const float3 Fss) {
	return GGXEnergy_ReflectScale3(u, v, mu, Fss, false);
}

//------------------------------------------------------------------------------
// Schlick BSDF (the coating lobe)
//------------------------------------------------------------------------------

// The layering of the coatings over their base (glossy2, glossycoating,
// glossytranslucent; see glossy2.cpp): the coating albedo per channel, its
// average, the base scale (1 - Ec(mu_light)) (1 - Ec(mu_eye)) / (1 - Ec_avg)
// and the probability of sampling the coating
OPENCL_FORCE_INLINE float3 GGXEnergy_CoatE3(const float u, const float v, const float mu, const float3 F0) {
	return MAKE_FLOAT3(GGXEnergy_CoatE(u, v, mu, F0.x), GGXEnergy_CoatE(u, v, mu, F0.y),
			GGXEnergy_CoatE(u, v, mu, F0.z));
}

OPENCL_FORCE_INLINE float3 GGXEnergy_CoatEavg3(const float u, const float v, const float3 F0) {
	return MAKE_FLOAT3(GGXEnergy_CoatEavg(u, v, F0.x), GGXEnergy_CoatEavg(u, v, F0.y),
			GGXEnergy_CoatEavg(u, v, F0.z));
}

OPENCL_FORCE_INLINE float3 Coating_LayerScale(const float3 EcLight, const float3 EcEye, const float3 EcAvg) {
	return (WHITE - EcLight) * (WHITE - EcEye) /
			MAKE_FLOAT3(fmax(1.f - EcAvg.x, 1e-4f), fmax(1.f - EcAvg.y, 1e-4f), fmax(1.f - EcAvg.z, 1e-4f));
}

OPENCL_FORCE_INLINE float Coating_Probability(const float3 Ec, const float3 baseAlbedo) {
	const float coat = Spectrum_Filter(Ec);
	const float base = Spectrum_Filter((WHITE - Ec) * baseAlbedo);

	return ((coat + base) > 0.f) ? coat / (coat + base) : 1.f;
}

OPENCL_FORCE_INLINE float SchlickBSDF_CoatingWeight(const float3 ks, const float3 fixedDir) {
	// Approximate H by using reflection direction for wi
	const float u = fabs(fixedDir.z);
	const float3 S = FresnelSchlick_Evaluate(ks, u);

	// Ensures coating is never sampled less than half the time
	return .5f * (1.f + Spectrum_Filter(S));
}

OPENCL_FORCE_INLINE float3 SchlickBSDF_CoatingF(const float3 ks,
		const float u, const float v,
		const float3 fixedDir, const float3 sampledDir) {
	const float coso = fabs(fixedDir.z);

	const float3 wh = normalize(fixedDir + sampledDir);
	const float cosi_Schlick = clamp(fabs(dot(sampledDir, wh)), 0.f, 1.f);
	const float3 S = FresnelSchlick_Evaluate(ks, cosi_Schlick);

	const float G = Microfacet_G(u, v, fixedDir, sampledDir);
	const float factor = Microfacet_D(u, v, wh) * G / (4.f * coso);

	// Multiple-scattering energy compensation (fixedDir is the eye direction)
	return factor * S * Microfacet_MSScale(u, v, coso, SchlickFresnel_Average(ks));
}

OPENCL_FORCE_INLINE float3 SchlickBSDF_CoatingSampleF(const float3 ks,
		const float u, const float v,
		const float3 fixedDir, float3 *sampledDir,
		float u0, float u1, float *pdf) {
	*pdf = Microfacet_SampleReflect(u, v, fixedDir, u0, u1, sampledDir);

	if ((fabs((*sampledDir).z) < DEFAULT_COS_EPSILON_STATIC) || (fixedDir.z * (*sampledDir).z < 0.f))
		return BLACK;

	if (*pdf <= 0.f)
		return BLACK;

	const float coso = fabs(fixedDir.z);

	const float3 wh = normalize(fixedDir + *sampledDir);
	const float cosWH = fabs(dot(fixedDir, wh));

	float3 S = FresnelSchlick_Evaluate(ks, cosWH);

	const float d = Microfacet_D(u, v, wh);
	const float G = Microfacet_G(u, v, fixedDir, *sampledDir);

	S *= (d / *pdf) * G / (4.f * coso);

	// Multiple-scattering energy compensation (fixedDir is the eye direction)
	return S * Microfacet_MSScale(u, v, coso, SchlickFresnel_Average(ks));
}

OPENCL_FORCE_INLINE float SchlickBSDF_CoatingPdf(const float u, const float v,
		const float3 fixedDir, const float3 sampledDir) {
	return Microfacet_PdfReflect(u, v, fixedDir, sampledDir);
}

OPENCL_FORCE_INLINE float FrDiel2(const float cosi, const float cost, const float eta) {
	float Rparl = eta * cosi;
	Rparl = (cost - Rparl) / (cost + Rparl);
	float Rperp = eta * cost;
	Rperp = (cosi - Rperp) / (cosi + Rperp);

	return (Rparl * Rparl + Rperp * Rperp) * .5f;
}

OPENCL_FORCE_INLINE float3 FrFull(const float cosi, const float3 cost, const float3 eta, const float3 k) {
	const float3 tmp = (eta * eta + k * k) * (cosi * cosi) + (cost * cost);
	const float3 Rparl2 = (tmp - (2.f * cosi * cost) * eta) /
		(tmp + (2.f * cosi * cost) * eta);
	const float3 tmp_f = (eta * eta + k * k) * (cost * cost) + (cosi * cosi);
	const float3 Rperp2 = (tmp_f - (2.f * cosi * cost) * eta) /
		(tmp_f + (2.f * cosi * cost) * eta);
	return (Rparl2 + Rperp2) * .5f;
}

OPENCL_FORCE_INLINE float3 FresnelGeneral_Evaluate(const float3 eta, const float3 k, const float cosi) {
	float3 sint2 = TO_FLOAT3(fmax(0.f, 1.f - cosi * cosi));
	if (cosi > 0.f)
		sint2 /= eta * eta;
	else
		sint2 *= eta * eta;
	sint2 = Spectrum_Clamp(sint2);

	const float3 cost2 = 1.f - sint2;
	if (cosi > 0.f) {
		const float3 a = 2.f * k * k * sint2;
		return FrFull(cosi, Spectrum_Sqrt((cost2 + Spectrum_Sqrt(cost2 * cost2 + a * a)) / 2.f), eta, k);
	} else {
		const float3 a = 2.f * k * k * sint2;
		const float3 d2 = eta * eta + k * k;
		return FrFull(-cosi, Spectrum_Sqrt((cost2 + Spectrum_Sqrt(cost2 * cost2 + a * a)) / 2.f), eta / d2, -k / d2);
	}
}

OPENCL_FORCE_INLINE float FresnelCauchy_Evaluate(const float eta, const float cosi) {
	// Compute indices of refraction for dielectric
	const bool entering = (cosi > 0.f);

	// Compute _sint_ using Snell's law
	const float eta2 = eta * eta;
	const float sint2 = (entering ? 1.f / eta2 : eta2) *
		fmax(0.f, 1.f - cosi * cosi);
	// Handle total internal reflection
	if (sint2 >= 1.f)
		return 1.f;
	else
		return FrDiel2(fabs(cosi), sqrt(fmax(0.f, 1.f - sint2)),
			entering ? eta : 1.f / eta);
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4

//------------------------------------------------------------------------------
// EON, the energy-preserving Oren-Nayar lobe (Portsmouth, Kutz & Hill 2025):
// Fujii's single-scattering lobe with its exact albedo and a Lambertian-like
// multiple-scattering term. The roughness r is sigma clamped to [0, 1]. Same
// code as EON_E()/EON_Evaluate() in material.cpp.
//------------------------------------------------------------------------------

#define EON_CONST1 (.5f - 2.f / (3.f * M_PI_F))
#define EON_CONST2 (2.f / 3.f - 28.f / (15.f * M_PI_F))

OPENCL_FORCE_INLINE float EON_E(const float muIn, const float r) {
	// A direction rounded to z slightly above 1 made acos() NaN (times Si = 0)
	const float mu = clamp(muIn, 1e-4f, 1.f);
	const float AF = 1.f / (1.f + EON_CONST1 * r);
	const float Si = sqrt(fmax(0.f, 1.f - mu * mu));
	const float G = Si * (acos(mu) - Si * mu) +
			(2.f / 3.f) * ((Si / mu) * (1.f - Si * Si * Si) - Si);

	return AF * (1.f + (r * M_1_PI_F) * G);
}

// f_ss + f_ms without the cosine term, for two directions in the same hemisphere
OPENCL_FORCE_INLINE float3 EON_Evaluate(const float3 rho, const float r, const float3 wi, const float3 wo) {
	const float mui = fmax(fabs(wi.z), 1e-4f);
	const float muo = fmax(fabs(wo.z), 1e-4f);
	const float AF = 1.f / (1.f + EON_CONST1 * r);

	// dot(wi, wo) - mui * muo
	const float s = wi.x * wo.x + wi.y * wo.y;
	const float stinv = (s > 0.f) ? s / fmax(mui, muo) : s;
	const float3 fss = rho * (M_1_PI_F * AF * (1.f + r * stinv));

	const float Eo = EON_E(muo, r);
	const float Ei = EON_E(mui, r);
	const float Eavg = AF * (1.f + EON_CONST2 * r);
	// Only the denominators are guarded (see material.cpp)
	const float eps = 1e-6f;
	const float3 denom = WHITE - rho * (1.f - Eavg);
	const float3 rhoMS = (rho * rho) * Eavg / MAKE_FLOAT3(fmax(denom.x, eps), fmax(denom.y, eps), fmax(denom.z, eps));
	const float3 fms = rhoMS * (M_1_PI_F * fmax(0.f, 1.f - Eo) * fmax(0.f, 1.f - Ei) / fmax(eps, 1.f - Eavg));

	return fss + fms;
}

