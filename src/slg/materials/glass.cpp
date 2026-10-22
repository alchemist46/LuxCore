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
#include "slg/materials/glass.h"
#include "slg/materials/roughglass.h"
#include "slg/materials/thinfilmcoating.h"

using namespace std;
using namespace luxrays;
using namespace slg;

//------------------------------------------------------------------------------
// Glass material
//------------------------------------------------------------------------------

GlassMaterial::GlassMaterial(TextureConstPtr frontTransp, TextureConstPtr backTransp,
		TextureConstPtr emitted, TextureConstPtr bump,
		TextureConstPtr refl, TextureConstPtr trans,
		TextureConstPtr exteriorIorFact, TextureConstPtr interiorIorFact,
		TextureConstPtr B, TextureConstPtr filmThickness, TextureConstPtr filmIor) :
			Material(frontTransp, backTransp, emitted, bump),
			Kr(refl), Kt(trans), exteriorIor(exteriorIorFact), interiorIor(interiorIorFact),
			cauchyB(B), filmThickness(filmThickness), filmIor(filmIor) {
}

// Path regularization (HitPoint::regularization > 0): the glass is a rough
// glass with the alpha floor, without dispersion
#define GLASS_REGULARIZED_PARAMS \
	const Spectrum kr = Kr->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f); \
	const Spectrum kt = Kt->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f); \
	const float nc = ExtractExteriorIors(hitPoint, exteriorIor); \
	const float nt = ExtractInteriorIors(hitPoint, interiorIor); \
	const float alpha = Microfacet_Regularize(0.f, hitPoint.regularization); \
	const float localFilmThickness = filmThickness ? filmThickness->GetFloatValue(hitPoint) : 0.f; \
	const float localFilmIor = ((localFilmThickness > 0.f) && filmIor) ? filmIor->GetFloatValue(hitPoint) : 1.f;

Spectrum GlassMaterial::Evaluate(const HitPoint &hitPoint,
	const Vector &localLightDir, const Vector &localEyeDir, BSDFEvent *event,
	float *directPdfW, float *reversePdfW) const {
	if (hitPoint.regularization > 0.f) {
		GLASS_REGULARIZED_PARAMS
		return RoughGlassMaterial::EvaluateRough(hitPoint, localLightDir, localEyeDir, event, directPdfW, reversePdfW,
				kt, kr, nc, nt, alpha, alpha, false, localFilmThickness, localFilmIor, workingColorSpace);
	}

	return Spectrum();
}

static Spectrum WaveLength2RGB(const float waveLength) {
	float r, g, b;
	if ((waveLength >= 380.f) && (waveLength < 440.f)) {
		r = -(waveLength - 440.f) / (440 - 380.f);
		g = 0.f;
		b = 1.f;
	} else if ((waveLength >= 440.f) && (waveLength < 490.f)) {
		r = 0.f;
		g = (waveLength - 440.f) / (490.f - 440.f);
		b = 1.f;
	} else if ((waveLength >= 490.f) && (waveLength < 510.f)) {
		r = 0.f;
		g = 1.f;
		b = -(waveLength - 510.f) / (510.f - 490.f);
	} else if ((waveLength >= 510.f) && (waveLength < 580.f)) {
		r = (waveLength - 510.f) / (580.f - 510.f);
		g = 1.f;
		b = 0.f;
	} else if ((waveLength >= 580.f) && (waveLength < 645.f)) {
		r = 1.f;
		g = -(waveLength - 645.f) / (645 - 580.f);
		b = 0.f;
	} else if ((waveLength >= 645.f) && (waveLength < 780.f)) {
		r = 1.f;
		g = 0.f;
		b = 0.f;
	} else
		return Spectrum();

	// The intensity fall off near the upper and lower limits
	float factor;
	if ((waveLength >= 380.f) && (waveLength < 420.f))
		factor = .3f + .7f * (waveLength - 380.f) / (420.f - 380.f);
	else if ((waveLength >= 420) && (waveLength < 700))
		factor = 1.f;
	else
		factor = .3f + .7f * (780.f - waveLength) / (780.f - 700.f);

	const Spectrum result = Spectrum(r, g, b) * factor;

	/*
	Spectrum white;
	for (u_int i = 380; i < 780; ++i)
		white += WaveLength2RGB(i);
	white *= 1.f / 400.f;
	cout << std::setprecision(std::numeric_limits<float>::digits10 + 1) << white.c[0] << ", " << white.c[1] << ", " << white.c[2] << "\n";
	 
	 Result: 0.5652729, 0.36875, 0.265375
	 */

	// To normalize the output
	const Spectrum normFactor(1.f / .5652729f, 1.f / .36875f, 1.f / .265375f);
	
	return result * normFactor;
}

Spectrum GlassMaterial::WaveLengthColor(const float waveLength, const WorkingColorSpace &workingColorSpace) {
	return workingColorSpace.FromRec709(WaveLength2RGB(waveLength));
}

static float WaveLength2IOR(const float waveLength, const float IOR, const float B) {
	// Cauchy's equation for relationship between the refractive index and wavelength
	// note: Cauchy's lambda is expressed in micrometers while waveLength is in nanometers

	// This is the formula suggested by Neo here, with a changed naming convention from B->A and C-> B:
	// https://github.com/LuxCoreRender/BlendLuxCore/commit/d3fed046ab62e18226e410b42a16ca1bccefb530#commitcomment-26617643
	
	// Compute Cauchy-A assuming the user input IOR at 587.56 nm 
	// (Fraunhofer d-line, Helium, used in one definition of the Abbe number)
	//const float A = IOR - B / Sqr(587.56f / 1000.f);

	// Use the user input IOR directly as Cauchy-A. Equivalent to the B used by old LuxRender.
	const float A = IOR;

	// Cauchy's equation
	const float cauchyEq = A + B / Sqr(waveLength / 1000.f);

	return cauchyEq;
}

Spectrum GlassMaterial::EvalSpecularReflection(const HitPoint &hitPoint,
		const Vector &localFixedDir, const Spectrum &kr,
		const float nc, const float nt,
		Vector *localSampledDir, 
		const float localFilmThickness, const float localFilmIor,
		const WorkingColorSpace &workingColorSpace) {
	if (kr.Black())
		return Spectrum();

	const float cosTheta = CosTheta(localFixedDir);
	*localSampledDir = Vector(-localFixedDir.x, -localFixedDir.y, localFixedDir.z);

	const float ntc = nt / nc;
	const Spectrum result = kr * FresnelTexture::CauchyEvaluate(ntc, cosTheta);

	if (localFilmThickness > 0.f) {
		// The film is on the exterior face of the interface
		const bool entering = (cosTheta > 0.f);
		const Spectrum filmColor = CalcFilmColor(localFixedDir, localFilmThickness, localFilmIor,
				entering ? nc : nt, entering ? nt : nc, workingColorSpace);
		return result * filmColor;
	}
	return result;
}

Spectrum GlassMaterial::EvalSpecularTransmission(const HitPoint &hitPoint,
		const Vector &localFixedDir, const Spectrum &kt, const float nc, const float nt,
		Vector *localSampledDir, const float localFilmThickness, const float localFilmIor,
		const WorkingColorSpace &workingColorSpace) {
	if (kt.Black())
		return Spectrum();

	// Compute transmitted ray direction
	const float ntc = nt / nc;
	const float cosTheta = CosTheta(localFixedDir);
	const bool entering = (cosTheta > 0.f);
	const float eta = entering ? (nc / nt) : ntc;
	const float eta2 = eta * eta;
	const float sini2 = SinTheta2(localFixedDir);
	const float sint2 = eta2 * sini2;

	// Handle total internal reflection for transmission
	if (sint2 >= 1.f)
		return Spectrum();

	const float cost = sqrtf(Max(0.f, 1.f - sint2)) * (entering ? -1.f : 1.f);
	*localSampledDir = Vector(-eta * localFixedDir.x, -eta * localFixedDir.y, cost);

	float ce;
	if (!hitPoint.fromLight)
		ce = (1.f - FresnelTexture::CauchyEvaluate(ntc, cost)) * eta2;
	else {
		const float absCosSampledDir = fabsf(CosTheta(*localSampledDir));
		ce = (1.f - FresnelTexture::CauchyEvaluate(ntc, cosTheta)) * fabsf(CosTheta(localFixedDir) / absCosSampledDir);
	}

	if (localFilmThickness > 0.f) {
		// A lossless film transmits what it does not reflect (the same from
		// both sides)
		const Spectrum filmColor = CalcFilmColor(localFixedDir, localFilmThickness, localFilmIor,
				entering ? nc : nt, entering ? nt : nc, workingColorSpace);
		return kt * ce * FilmTransmissionScale(FresnelTexture::CauchyEvaluate(ntc, cosTheta), filmColor);
	}

	return kt * ce;
}

bool GlassMaterial::GetRefractionIors(const HitPoint &hitPoint, float *nc, float *nt) const {
	*nc = ExtractExteriorIors(hitPoint, exteriorIor);
	*nt = ExtractInteriorIors(hitPoint, interiorIor);

	return true;
}

Spectrum GlassMaterial::Sample(const HitPoint &hitPoint,
		const Vector &localFixedDir, Vector *localSampledDir,
		const float u0, const float u1, const float passThroughEvent,
		float *pdfW, BSDFEvent *event) const {
	if (hitPoint.regularization > 0.f) {
		GLASS_REGULARIZED_PARAMS
		return RoughGlassMaterial::SampleRough(hitPoint, localFixedDir, localSampledDir, u0, u1, passThroughEvent, pdfW, event,
				kt, kr, nc, nt, alpha, alpha, false, localFilmThickness, localFilmIor, workingColorSpace);
	}

	const Spectrum kr = Kr->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
	const Spectrum kt = Kt->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);

	const float nc = ExtractExteriorIors(hitPoint, exteriorIor);
	const float nt = ExtractInteriorIors(hitPoint, interiorIor);

	const float cauchyBValue = cauchyB ? cauchyB->GetFloatValue(hitPoint) : 0.f;

	const float localFilmThickness = filmThickness ? filmThickness->GetFloatValue(hitPoint) : 0.f;
	const float localFilmIor = (localFilmThickness > 0.f && filmIor) ? filmIor->GetFloatValue(hitPoint) : 1.f;

	// The dispersion: the IOR at the wavelength of the path (or of this
	// vertex without one) for the reflection and the refraction alike, so
	// that they conserve the energy at that wavelength, and the colour of
	// the wavelength on both, once per path (see BSDFEvent DISPERSED)
	const bool dispersed = (cauchyBValue > 0.f);
	float lnt = nt;
	Spectrum dispersionColor(1.f);
	if (dispersed) {
		const float waveLength = (hitPoint.wavelength > 0.f) ? hitPoint.wavelength : Lerp(u0, 380.f, 780.f);
		lnt = WaveLength2IOR(waveLength, nt, cauchyBValue);
		if (!hitPoint.dispersionTinted)
			dispersionColor = WaveLengthColor(waveLength, workingColorSpace);
	}

	Vector transLocalSampledDir;
	const Spectrum trans = EvalSpecularTransmission(hitPoint, localFixedDir,
			kt * dispersionColor, nc, lnt, &transLocalSampledDir, localFilmThickness, localFilmIor,
			workingColorSpace);

	Vector reflLocalSampledDir;
	const Spectrum refl = EvalSpecularReflection(hitPoint, localFixedDir,
			kr * dispersionColor, nc, lnt, &reflLocalSampledDir, localFilmThickness, localFilmIor, workingColorSpace);

	// Decide to transmit or reflect
	float threshold;
	if (!refl.Black()) {
		if (!trans.Black()) {
			// Importance sampling
			const float reflFilter = refl.Filter();
			const float transFilter = trans.Filter();
			threshold = transFilter / (reflFilter + transFilter);

			// A place an upper and lower limit to not under sample
			// reflection or transmission
			threshold = Clamp(threshold, .25f, .75f);
		} else
			threshold = 0.f;
	} else {
		if (!trans.Black())
			threshold = 1.f;
		else
			return Spectrum();
	}

	Spectrum result;
	if (passThroughEvent < threshold) {
		// Transmit

		*localSampledDir = transLocalSampledDir;

		*event = SPECULAR | TRANSMIT | (dispersed ? DISPERSED : NONE);
		*pdfW = threshold;
	
		result = trans;
	} else {
		// Reflect

		*localSampledDir = reflLocalSampledDir;

		*event = SPECULAR | REFLECT | (dispersed ? DISPERSED : NONE);
		*pdfW = 1.f - threshold;

		result = refl;
	}
	
	return result / *pdfW;
}

void GlassMaterial::Pdf(const HitPoint &hitPoint,
		const luxrays::Vector &localLightDir, const luxrays::Vector &localEyeDir,
	float *directPdfW, float *reversePdfW) const {
	if (hitPoint.regularization > 0.f) {
		GLASS_REGULARIZED_PARAMS
		RoughGlassMaterial::PdfRough(hitPoint, localLightDir, localEyeDir, directPdfW, reversePdfW,
				kt, kr, nc, nt, alpha, alpha);
		return;
	}

	if (directPdfW)
		*directPdfW = 0.f;
	if (reversePdfW)
		*reversePdfW = 0.f;
}

void GlassMaterial::AddReferencedTextures(std::unordered_set<const Texture *>  &referencedTexs) const {
	Material::AddReferencedTextures(referencedTexs);

	Kr->AddReferencedTextures(referencedTexs);
	Kt->AddReferencedTextures(referencedTexs);
	if (exteriorIor)
		exteriorIor->AddReferencedTextures(referencedTexs);
	if (interiorIor)
		interiorIor->AddReferencedTextures(referencedTexs);
	if (filmThickness)
		filmThickness->AddReferencedTextures(referencedTexs);
	if (filmIor)
		filmIor->AddReferencedTextures(referencedTexs);
}

void GlassMaterial::UpdateTextureReferences(TextureConstRef oldTex, TextureRef newTex) {
	Material::UpdateTextureReferences(oldTex, newTex);

	if (Kr == &oldTex)
		Kr = &newTex;
	if (Kt == &oldTex)
		Kt = &newTex;
	if (exteriorIor == &oldTex)
		exteriorIor = &newTex;
	if (interiorIor == &oldTex)
		interiorIor = &newTex;
	if (filmThickness == &oldTex)
		filmThickness = &newTex;
	if (filmIor == &oldTex)
		filmIor = &newTex;
}

PropertiesUPtr GlassMaterial::ToProperties(const ImageMapCache &imgMapCache, const bool useRealFileName) const  {
	auto props = std::make_unique<Properties>();

	const string name = GetName();
	props->Set(Property("scene.materials." + name + ".type")("glass"));
	props->Set(Property("scene.materials." + name + ".kr")(Kr->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".kt")(Kt->GetSDLValue()));
	if (exteriorIor)
		props->Set(Property("scene.materials." + name + ".exteriorior")(exteriorIor->GetSDLValue()));
	if (interiorIor)
		props->Set(Property("scene.materials." + name + ".interiorior")(interiorIor->GetSDLValue()));
	if (cauchyB)
		props->Set(Property("scene.materials." + name + ".cauchyb")(cauchyB->GetSDLValue()));
	if (filmThickness)
		props->Set(Property("scene.materials." + name + ".filmthickness")(filmThickness->GetSDLValue()));
	if (filmIor)
		props->Set(Property("scene.materials." + name + ".filmior")(filmIor->GetSDLValue()));
	props->Set(Material::ToProperties(imgMapCache, useRealFileName));

	return props;
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
