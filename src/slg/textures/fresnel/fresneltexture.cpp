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

#include "slg/textures/fresnel/fresneltexture.h"

using namespace std;
using namespace luxrays;
using namespace slg;

//------------------------------------------------------------------------------
// Fresnel texture
//------------------------------------------------------------------------------

float FresnelTexture::ApproxN(const float Fr) {
	const float sqrtReflectance = sqrtf(Clamp(Fr, 0.f, .999f));

	return (1.f + sqrtReflectance) /
		(1.f - sqrtReflectance);
}

Spectrum FresnelTexture::ApproxN(const Spectrum &Fr) {
	const Spectrum sqrtReflectance = Fr.Clamp(0.f, .999f).Sqrt();

	return (Spectrum(1.f) + sqrtReflectance) /
		(Spectrum(1.f) - sqrtReflectance);
}

float FresnelTexture::ApproxK(const float Fr) {
	const float reflectance = Clamp(Fr, 0.f, .999f);

	return 2.f * sqrtf(reflectance /
		(1.f - reflectance));
}

Spectrum FresnelTexture::ApproxK(const Spectrum &Fr) {
	const Spectrum reflectance = Fr.Clamp(0.f, .999f);

	return 2.f * Sqrt(reflectance /
		(Spectrum(1.f) - reflectance));
}

float FresnelTexture::FrDiel2(const float cosi, const float cost,
		const float eta) {
	float Rparl = eta * cosi;
	Rparl = (cost - Rparl) / (cost + Rparl);
	float Rperp = eta * cost;
	Rperp = (cosi - Rperp) / (cosi + Rperp);

	return (Rparl * Rparl + Rperp * Rperp) * .5f;
}

//Spectrum FrDiel(const float cosi, const float cost,
//		const Spectrum &etai, const Spectrum &etat) {
//	return FrDiel2(cosi, Spectrum(cost), etat / etai);
//}
//
//static Spectrum FrCond(const float cosi, const Spectrum &eta,
//		const Spectrum &k) {
//	const Spectrum tmp = (eta * eta + k*k) * (cosi * cosi) + 1;
//	const Spectrum Rparl2 = 
//		(tmp - (2.f * eta * cosi)) /
//		(tmp + (2.f * eta * cosi));
//	const Spectrum tmp_f = eta * eta + k*k + (cosi * cosi);
//	const Spectrum Rperp2 =
//		(tmp_f - (2.f * eta * cosi)) /
//		(tmp_f + (2.f * eta * cosi));
//	return (Rparl2 + Rperp2) * .5f;
//}

// The unpolarized reflectance of the interface to a medium of complex IOR
// eta + i k (relative to the incident one) at the cosine cosi > 0, exact (as
// pbrt-v3's FrConductor(); FresnelGeneral_Evaluate() in
// materialdefs_funcs_generic.cl is the same code)
float FresnelTexture::FrConductor(const float cosi, const float eta, const float k) {
	const float cos2 = cosi * cosi;
	const float sin2 = 1.f - cos2;
	const float t0 = eta * eta - k * k - sin2;
	const float a2PlusB2 = sqrtf(Max(0.f, t0 * t0 + 4.f * eta * eta * k * k));
	const float t1 = a2PlusB2 + cos2;
	const float a = sqrtf(Max(0.f, .5f * (a2PlusB2 + t0)));
	const float t2 = 2.f * cosi * a;
	const float rs = (t1 - t2) / Max(t1 + t2, 1e-20f);
	const float t3 = cos2 * a2PlusB2 + sin2 * sin2;
	const float t4 = t2 * sin2;
	const float rp = rs * (t3 - t4) / Max(t3 + t4, 1e-20f);

	return .5f * (rp + rs);
}

Spectrum FresnelTexture::SchlickEvaluate(const Spectrum &normalIncidence, const float cosi) {
	return normalIncidence + (Spectrum(1.f) - normalIncidence) *
		powf(1.f - cosi, 5.f);
}

// From the other side (cosi < 0) the relative IOR is 1 / (eta + i k)
Spectrum FresnelTexture::GeneralEvaluate(const Spectrum &eta, const Spectrum &k, const float cosi) {
	const float c = Min(fabsf(cosi), 1.f);

	Spectrum result;
	for (u_int i = 0; i < COLOR_SAMPLES; ++i) {
		if (cosi > 0.f)
			result.c[i] = FrConductor(c, eta.c[i], k.c[i]);
		else {
			const float d2 = eta.c[i] * eta.c[i] + k.c[i] * k.c[i];
			result.c[i] = FrConductor(c, eta.c[i] / d2, k.c[i] / d2);
		}
	}

	return result;
}

float FresnelTexture::CauchyEvaluate(const float eta, const float cosi) {
	// Compute indices of refraction for dielectric
	const bool entering = (cosi > 0.f);

	// Compute _sint_ using Snell's law
	const float eta2 = eta * eta;
	const float sint2 = (entering ? 1.f / eta2 : eta2) *
		Max(0.f, 1.f - cosi * cosi);
	// Handle total internal reflection
	if (sint2 >= 1.f)
		return 1.f;
	else
		return FrDiel2(fabsf(cosi), sqrtf(Max(0.f, 1.f - sint2)),
			entering ? eta : (1.f / eta));
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
