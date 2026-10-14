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
//------------------------------------------------------------------------------
// Generic material related functions
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

OPENCL_FORCE_INLINE float SchlickDistribution_SchlickZ(const float roughness, float cosNH) {
	if (roughness > 0.f) {
		const float cosNH2 = cosNH * cosNH;
		// expanded for increased numerical stability
		const float d = cosNH2 * roughness + (1.f - cosNH2);
		// use double division to avoid overflow in d*d product
		return (roughness / d) / d;
	}
	return 0.f;
}

OPENCL_FORCE_INLINE float SchlickDistribution_SchlickA(const float3 H, const float anisotropy) {
	const float h = sqrt(H.x * H.x + H.y * H.y);
	if (h > 0.f) {
		const float w = (anisotropy > 0.f ? H.x : H.y) / h;
		const float p = 1.f - fabs(anisotropy);
		return sqrt(p / (p * p + w * w * (1.f - p * p)));
	}

	return 1.f;
}

OPENCL_FORCE_INLINE float SchlickDistribution_D(const float roughness, const float3 wh, const float anisotropy) {
	const float cosTheta = fabs(wh.z);
	return SchlickDistribution_SchlickZ(roughness, cosTheta) * SchlickDistribution_SchlickA(wh, anisotropy) * M_1_PI_F;
}

OPENCL_FORCE_INLINE float SchlickDistribution_SchlickG(const float roughness, const float costheta) {
	return costheta / (costheta * (1.f - roughness) + roughness);
}

OPENCL_FORCE_INLINE float SchlickDistribution_G(const float roughness, const float3 fixedDir, const float3 sampledDir) {
	return SchlickDistribution_SchlickG(roughness, fabs(fixedDir.z)) *
			SchlickDistribution_SchlickG(roughness, fabs(sampledDir.z));
}

OPENCL_FORCE_INLINE float GetPhi(const float a, const float b) {
	return M_PI_F * .5f * sqrt(a * b / (1.f - a * (1.f - b)));
}

OPENCL_FORCE_INLINE void SchlickDistribution_SampleH(const float roughness, const float anisotropy,
		const float u0, const float u1, float3 *wh, float *d, float *pdf) {
	float u1x4 = u1 * 4.f;
	const float cos2Theta = u0 / (roughness * (1.f - u0) + u0);
	const float cosTheta = sqrt(cos2Theta);
	const float sinTheta = sqrt(1.f - cos2Theta);
	const float p = 1.f - fabs(anisotropy);
	float phi;
	if (u1x4 < 1.f) {
		phi = GetPhi(u1x4 * u1x4, p * p);
	} else if (u1x4 < 2.f) {
		u1x4 = 2.f - u1x4;
		phi = M_PI_F - GetPhi(u1x4 * u1x4, p * p);
	} else if (u1x4 < 3.f) {
		u1x4 -= 2.f;
		phi = M_PI_F + GetPhi(u1x4 * u1x4, p * p);
	} else {
		u1x4 = 4.f - u1x4;
		phi = M_PI_F * 2.f - GetPhi(u1x4 * u1x4, p * p);
	}

	if (anisotropy > 0.f)
		phi += M_PI_F * .5f;

	*wh = MAKE_FLOAT3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
	*d = SchlickDistribution_SchlickZ(roughness, cosTheta) * SchlickDistribution_SchlickA(*wh, anisotropy) * M_1_PI_F;
	*pdf = *d;
}

OPENCL_FORCE_INLINE float SchlickDistribution_Pdf(const float roughness, const float3 wh,
		const float anisotropy) {
	return SchlickDistribution_D(roughness, wh, anisotropy);
}

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
// Glossy2, GlossyCoating and GlossyTranslucent (see material.h for the
// description of the two distributions)
//------------------------------------------------------------------------------

OPENCL_FORCE_INLINE void SchlickDistribution_Params(const float u, const float v,
		float *roughness, float *anisotropy) {
	const float u2 = u * u;
	const float v2 = v * v;
	*anisotropy = (u2 < v2) ? (1.f - u2 / v2) : u2 > 0.f ? (v2 / u2 - 1.f) : 0.f;
	*roughness = u * v;
}

OPENCL_FORCE_INLINE float Microfacet_D(const MicrofacetDistributionType dist, const float u, const float v,
		const float3 wh) {
	if (dist == GGX_DISTRIBUTION)
		return GGXDistribution_D(u, v, wh);
	else {
		float roughness, anisotropy;
		SchlickDistribution_Params(u, v, &roughness, &anisotropy);
		return SchlickDistribution_D(roughness, wh, anisotropy);
	}
}

OPENCL_FORCE_INLINE float Microfacet_G(const MicrofacetDistributionType dist, const float u, const float v,
		const float3 wo, const float3 wi) {
	if (dist == GGX_DISTRIBUTION)
		return GGXDistribution_G(u, v, wo, wi);
	else {
		float roughness, anisotropy;
		SchlickDistribution_Params(u, v, &roughness, &anisotropy);
		return SchlickDistribution_G(roughness, wo, wi);
	}
}

// Samples a half vector in the upper hemisphere (wo can be on either side).
// d is D(wh) and pdfH the pdf of wh over the solid angle.
OPENCL_FORCE_INLINE void Microfacet_SampleH(const MicrofacetDistributionType dist, const float u, const float v,
		const float3 wo, const float u0, const float u1,
		float3 *wh, float *d, float *pdfH) {
	if (dist == GGX_DISTRIBUTION) {
		const float3 woUp = (wo.z < 0.f) ? -wo : wo;
		*wh = GGXDistribution_SampleVisibleH(u, v, woUp, u0, u1);
		*d = GGXDistribution_D(u, v, *wh);
		*pdfH = GGXDistribution_PdfVisibleH(u, v, woUp, *wh);
	} else {
		float roughness, anisotropy;
		SchlickDistribution_Params(u, v, &roughness, &anisotropy);
		SchlickDistribution_SampleH(roughness, anisotropy, u0, u1, wh, d, pdfH);
	}
}

OPENCL_FORCE_INLINE float Microfacet_PdfH(const MicrofacetDistributionType dist, const float u, const float v,
		const float3 wo, const float3 wh) {
	if (dist == GGX_DISTRIBUTION) {
		const float3 woUp = (wo.z < 0.f) ? -wo : wo;
		return GGXDistribution_PdfVisibleH(u, v, woUp, wh);
	} else {
		float roughness, anisotropy;
		SchlickDistribution_Params(u, v, &roughness, &anisotropy);
		return SchlickDistribution_Pdf(roughness, wh, anisotropy);
	}
}

// Reflection lobe: samples wi on the same side as wo and returns its pdf over
// the solid angle (0 when the sample is not usable)
OPENCL_FORCE_INLINE float Microfacet_SampleReflect(const MicrofacetDistributionType dist, const float u, const float v,
		const float3 wo, const float u0, const float u1, float3 *wi) {
	if (dist == GGX_DISTRIBUTION) {
		const bool flip = (wo.z < 0.f);
		const float3 woUp = flip ? -wo : wo;
		const float3 wiUp = GGXDistribution_SampleBoundedReflect(u, v, woUp, u0, u1);
		*wi = flip ? -wiUp : wiUp;
		return GGXDistribution_PdfBoundedReflect(u, v, woUp, wiUp);
	} else {
		float roughness, anisotropy;
		SchlickDistribution_Params(u, v, &roughness, &anisotropy);
		float3 wh;
		float d, pdfH;
		SchlickDistribution_SampleH(roughness, anisotropy, u0, u1, &wh, &d, &pdfH);
		const float cosWH = dot(wo, wh);
		*wi = 2.f * cosWH * wh - wo;
		return pdfH / (4.f * fabs(cosWH));
	}
}

OPENCL_FORCE_INLINE float Microfacet_PdfReflect(const MicrofacetDistributionType dist, const float u, const float v,
		const float3 wo, const float3 wi) {
	if (dist == GGX_DISTRIBUTION) {
		const bool flip = (wo.z < 0.f);
		return GGXDistribution_PdfBoundedReflect(u, v, flip ? -wo : wo, flip ? -wi : wi);
	} else {
		float roughness, anisotropy;
		SchlickDistribution_Params(u, v, &roughness, &anisotropy);
		const float3 wh = normalize(wo + wi);
		return SchlickDistribution_Pdf(roughness, wh, anisotropy) / (4.f * fabs(dot(wo, wh)));
	}
}

//------------------------------------------------------------------------------
// Multiple-scattering energy compensation of the GGX lobes (property
// "multiscattering", GGX_DISTRIBUTION only), see microfacet_energy_tables.cl
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

// The scale of a Microfacet_* reflection lobe seen with the cosine mu: 1 unless
// multiscattering is enabled and the distribution is GGX
OPENCL_FORCE_INLINE float3 Microfacet_MSScale(const MicrofacetDistributionType dist, const int multiscattering,
		const float u, const float v, const float mu, const float3 Fss) {
	if (multiscattering && (dist == GGX_DISTRIBUTION))
		return GGXEnergy_ReflectScale3(u, v, mu, Fss, false);
	else
		return WHITE;
}

//------------------------------------------------------------------------------
// Schlick BSDF (the coating lobe)
//------------------------------------------------------------------------------

OPENCL_FORCE_INLINE float SchlickBSDF_CoatingWeight(const float3 ks, const float3 fixedDir) {
	// Approximate H by using reflection direction for wi
	const float u = fabs(fixedDir.z);
	const float3 S = FresnelSchlick_Evaluate(ks, u);

	// Ensures coating is never sampled less than half the time
	return .5f * (1.f + Spectrum_Filter(S));
}

OPENCL_FORCE_INLINE float3 SchlickBSDF_CoatingF(const float3 ks,
		const MicrofacetDistributionType dist, const float u, const float v,
		const int multibounce, const int multiscattering, const float3 fixedDir,
		const float3 sampledDir) {
	const float coso = fabs(fixedDir.z);
	const float cosi = fabs(sampledDir.z);

	const float3 wh = normalize(fixedDir + sampledDir);
	const float cosi_Schlick = clamp(fabs(dot(sampledDir, wh)), 0.f, 1.f);
	const float3 S = FresnelSchlick_Evaluate(ks, cosi_Schlick);

	const float G = Microfacet_G(dist, u, v, fixedDir, sampledDir);

	// Multibounce - alternative with interreflection in the coating creases
	float factor = Microfacet_D(dist, u, v, wh) * G;
	//if (!fromLight)
		factor = factor / (4.f * coso) +
				(multibounce ? cosi * clamp((1.f - G) / (4.f * coso * cosi), 0.f, 1.f) : 0.f);
	//else
	//	factor = factor / (4.f * cosi) + 
	//			(multibounce ? coso * Clamp((1.f - G) / (4.f * cosi * coso), 0.f, 1.f) : 0.f);

	// Multiple-scattering energy compensation (fixedDir is the eye direction)
	return factor * S * Microfacet_MSScale(dist, multiscattering, u, v, coso, SchlickFresnel_Average(ks));
}

OPENCL_FORCE_INLINE float3 SchlickBSDF_CoatingSampleF(const float3 ks,
		const MicrofacetDistributionType dist, const float u, const float v,
		const int multibounce, const int multiscattering, const float3 fixedDir, float3 *sampledDir,
		float u0, float u1, float *pdf) {
	*pdf = Microfacet_SampleReflect(dist, u, v, fixedDir, u0, u1, sampledDir);

	if ((fabs((*sampledDir).z) < DEFAULT_COS_EPSILON_STATIC) || (fixedDir.z * (*sampledDir).z < 0.f))
		return BLACK;

	if (*pdf <= 0.f)
		return BLACK;

	const float coso = fabs(fixedDir.z);
	const float cosi = fabs((*sampledDir).z);

	const float3 wh = normalize(fixedDir + *sampledDir);
	const float cosWH = fabs(dot(fixedDir, wh));

	float3 S = FresnelSchlick_Evaluate(ks, cosWH);

	const float d = Microfacet_D(dist, u, v, wh);
	const float G = Microfacet_G(dist, u, v, fixedDir, *sampledDir);

	//CoatingF(sw, *wi, wo, f_);
	S *= (d / *pdf) * G / (4.f * coso) + 
			(multibounce ? cosi * clamp((1.f - G) / (4.f * coso * cosi), 0.f, 1.f) / *pdf : 0.f);

	// Multiple-scattering energy compensation (fixedDir is the eye direction)
	return S * Microfacet_MSScale(dist, multiscattering, u, v, coso, SchlickFresnel_Average(ks));
}

OPENCL_FORCE_INLINE float SchlickBSDF_CoatingPdf(const MicrofacetDistributionType dist, const float u, const float v,
		const float3 fixedDir, const float3 sampledDir) {
	return Microfacet_PdfReflect(dist, u, v, fixedDir, sampledDir);
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
