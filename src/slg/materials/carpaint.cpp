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
#include "slg/materials/carpaint.h"

using namespace std;
using namespace luxrays;
using namespace slg;

//------------------------------------------------------------------------------
// CarPaint material
//
// LuxRender carpaint material porting.
//------------------------------------------------------------------------------

CarPaintMaterial::CarPaintMaterial(
	TextureConstPtr frontTransp, TextureConstPtr backTransp,
	TextureConstPtr emitted, TextureConstPtr bump,
	TextureConstPtr kd, TextureConstPtr ks1, TextureConstPtr ks2, TextureConstPtr ks3, TextureConstPtr m1, TextureConstPtr m2, TextureConstPtr m3,
	TextureConstPtr r1, TextureConstPtr r2, TextureConstPtr r3, TextureConstPtr ka, TextureConstPtr d) :
	Material(frontTransp, backTransp, emitted, bump), Kd(kd), Ks1(ks1), Ks2(ks2), Ks3(ks3), M1(m1), M2(m2), M3(m3),
	R1(r1), R2(r2), R3(r3),	Ka(ka), depth(d) {
	ComputeGlossiness(M1, M2, M3);
}

Spectrum CarPaintMaterial::Albedo(const HitPoint &hitPoint) const {
	return Kd->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
}

namespace {

// The glossy lobes of the car paint as a table of the active ones (ks > 0 and
// m > 0), in the order of the properties
struct CarPaintLobes {
	Spectrum ks[3];
	float m[3], r[3];
	u_int count;
};

CarPaintLobes GetCarPaintLobes(const CarPaintMaterial &mat, const HitPoint &hitPoint) {
	CarPaintLobes lobes;
	lobes.count = 0;

	const TextureConstPtr ksTex[3] = { mat.Ks1, mat.Ks2, mat.Ks3 };
	const TextureConstPtr mTex[3] = { mat.M1, mat.M2, mat.M3 };
	const TextureConstPtr rTex[3] = { mat.R1, mat.R2, mat.R3 };
	for (u_int i = 0; i < 3; ++i) {
		const Spectrum ks = ksTex[i]->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
		const float m = Microfacet_Regularize(mTex[i]->GetFloatValue(hitPoint), hitPoint.regularization);
		if ((ks.Filter() > 0.f) && (m > 0.f)) {
			lobes.ks[lobes.count] = ks;
			lobes.m[lobes.count] = m;
			lobes.r[lobes.count] = rTex[i]->GetFloatValue(hitPoint);
			++lobes.count;
		}
	}

	return lobes;
}

}

// The diffuse layer plus the glossy lobes, all of them evaluated (their sum is
// the BSDF) and sampled with the same probability each (the mixture pdf).
//
// The glossy lobes exist only on the front face: they reflect about the
// shading normal, so Sample() can not generate them on the back face, and
// Evaluate() must not count them there either.
Spectrum CarPaintMaterial::Evaluate(const HitPoint &hitPoint,
	const Vector &localLightDir, const Vector &localEyeDir, BSDFEvent *event,
	float *directPdfW, float *reversePdfW) const {
	const Vector &localFixedDir = hitPoint.fromLight ? localLightDir : localEyeDir;
	const Vector &localSampledDir = hitPoint.fromLight ? localEyeDir : localLightDir;

	*event = GLOSSY | REFLECT;

	// Absorption
	const float cosi = fabsf(localLightDir.z);
	const float coso = fabsf(localEyeDir.z);
	const Spectrum alpha = Ka->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
	const float d = depth->GetFloatValue(hitPoint);
	const Spectrum absorption = CoatingAbsorption(cosi, coso, alpha, d);

	// Diffuse layer
	Spectrum result = absorption * Kd->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f) * INV_PI * cosi;
	float pdf = fabsf(localSampledDir.z) * INV_PI;
	float pdfR = fabsf(localFixedDir.z) * INV_PI;
	u_int n = 1;

	// Glossy layers
	if ((localLightDir.z > 0.f) && (localEyeDir.z > 0.f)) {
		const Vector H = Normalize(localLightDir + localEyeDir);
		const CarPaintLobes lobes = GetCarPaintLobes(*this, hitPoint);
		for (u_int i = 0; i < lobes.count; ++i) {
			const float m = lobes.m[i];
			const float r = lobes.r[i];
			result += (Microfacet_D(m, m, H) * Microfacet_G(m, m, localLightDir, localEyeDir) / (4.f * coso)) *
					(lobes.ks[i] * FresnelTexture::SchlickEvaluate(r, AbsDot(localEyeDir, H))) *
					Microfacet_MSScale(IsMultiscattering(), m, m, coso, Spectrum(SchlickFresnel_Average(r)));
			pdf += Microfacet_PdfReflect(m, m, localFixedDir, localSampledDir);
			pdfR += Microfacet_PdfReflect(m, m, localSampledDir, localFixedDir);
		}
		n += lobes.count;
	}

	if (directPdfW)
		*directPdfW = pdf / n;
	if (reversePdfW)
		*reversePdfW = pdfR / n;

	return result;
}

Spectrum CarPaintMaterial::Sample(const HitPoint &hitPoint,
	const Vector &localFixedDir, Vector *localSampledDir,
	const float u0, const float u1, const float passThroughEvent,
	float *pdfW, BSDFEvent *event) const {
	if (fabsf(localFixedDir.z) < DEFAULT_COS_EPSILON_STATIC)
		return Spectrum();

	// One layer, each with the same probability (as the pdf of Evaluate())
	CarPaintLobes lobes;
	if (localFixedDir.z > 0.f)
		lobes = GetCarPaintLobes(*this, hitPoint);
	else
		lobes.count = 0;
	const u_int n = 1 + lobes.count;
	const u_int layer = Min(static_cast<u_int>(passThroughEvent * n), n - 1);

	if (layer == 0) {
		// Diffuse layer
		float pdf;
		*localSampledDir = Sgn(localFixedDir.z) * CosineSampleHemisphere(u0, u1, &pdf);
	} else {
		// Glossy layer
		const float m = lobes.m[layer - 1];
		const float pdf = Microfacet_SampleReflect(m, m, localFixedDir, u0, u1, localSampledDir);
		if ((pdf <= 0.f) || (localSampledDir->z < DEFAULT_COS_EPSILON_STATIC))
			return Spectrum();
	}
	if (fabsf(localSampledDir->z) < DEFAULT_COS_EPSILON_STATIC)
		return Spectrum();

	// The value and the pdf of the whole mixture
	const Vector &localLightDir = hitPoint.fromLight ? localFixedDir : *localSampledDir;
	const Vector &localEyeDir = hitPoint.fromLight ? *localSampledDir : localFixedDir;
	const Spectrum f = Evaluate(hitPoint, localLightDir, localEyeDir, event, pdfW, nullptr);
	if (f.Black() || (*pdfW <= 0.f))
		return Spectrum();

	return f / *pdfW;
}

void CarPaintMaterial::Pdf(const HitPoint &hitPoint,
	const Vector &localLightDir, const Vector &localEyeDir,
	float *directPdfW, float *reversePdfW) const {
	BSDFEvent event;
	Evaluate(hitPoint, localLightDir, localEyeDir, &event, directPdfW, reversePdfW);
}

void CarPaintMaterial::AddReferencedTextures(std::unordered_set<const Texture *>  &referencedTexs) const {
	Material::AddReferencedTextures(referencedTexs);

	Kd->AddReferencedTextures(referencedTexs);
	Ks1->AddReferencedTextures(referencedTexs);
	Ks2->AddReferencedTextures(referencedTexs);
	Ks3->AddReferencedTextures(referencedTexs);
	M1->AddReferencedTextures(referencedTexs);
	M2->AddReferencedTextures(referencedTexs);
	M3->AddReferencedTextures(referencedTexs);
	R1->AddReferencedTextures(referencedTexs);
	R2->AddReferencedTextures(referencedTexs);
	R3->AddReferencedTextures(referencedTexs);
	Ka->AddReferencedTextures(referencedTexs);
	depth->AddReferencedTextures(referencedTexs);
}

void CarPaintMaterial::UpdateTextureReferences(
	TextureConstRef oldTex,
	TextureRef newTex) {
	Material::UpdateTextureReferences(oldTex, newTex);

	bool updateGlossiness = false;
	if (Kd == &oldTex)
		Kd = &newTex;
	if (Ks1 == &oldTex)
		Ks1 = &newTex;
	if (Ks2 == &oldTex)
		Ks2 = &newTex;
	if (Ks3 == &oldTex)
		Ks3 = &newTex;
	if (M1 == &oldTex) {
		M1 = &newTex;
		updateGlossiness = true;
	}
	if (M2 == &oldTex) {
		M2 = &newTex;
		updateGlossiness = true;
	}
	if (M3 == &oldTex) {
		M3 = &newTex;
		updateGlossiness = true;
	}
	if (R1 == &oldTex)
		R1 = &newTex;
	if (R2 == &oldTex)
		R2 = &newTex;
	if (R3 == &oldTex)
		R3 = &newTex;
	if (Ka == &oldTex)
		Ka = &newTex;
	if (depth == &oldTex)
		depth = &newTex;
	
	if (updateGlossiness)
		ComputeGlossiness(M1, M2, M3);
}

PropertiesUPtr CarPaintMaterial::ToProperties(const ImageMapCache &imgMapCache, const bool useRealFileName) const  {
	auto props = std::make_unique<Properties>();

	const string name = GetName();
	props->Set(Property("scene.materials." + name + ".type")("carpaint"));
	props->Set(Property("scene.materials." + name + ".kd")(Kd->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".ks1")(Ks1->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".ks2")(Ks2->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".ks3")(Ks3->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".m1")(M1->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".m2")(M2->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".m3")(M3->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".r1")(R1->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".r2")(R2->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".r3")(R3->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".ka")(Ka->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".d")(depth->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".multiscattering")(IsMultiscattering()));
	props->Set(Material::ToProperties(imgMapCache, useRealFileName));

	return props;
}

const struct CarPaintMaterial::CarPaintData CarPaintMaterial::data[8] = {
  {"ford f8",
   {0.0012f, 0.0015f, 0.0018f},
   {0.0049f, 0.0076f, 0.0120f},
   {0.0100f, 0.0130f, 0.0180f},
   {0.0070f, 0.0065f, 0.0077f},
    0.1500f, 0.0870f, 0.9000f,
    0.3200f, 0.1100f, 0.0130f},
  {"polaris silber",
   {0.0550f, 0.0630f, 0.0710f},
   {0.0650f, 0.0820f, 0.0880f},
   {0.1100f, 0.1100f, 0.1300f},
   {0.0080f, 0.0130f, 0.0150f},
    1.0000f, 0.9200f, 0.9000f,
    0.3800f, 0.1700f, 0.0130f},
  {"opel titan",
   {0.0110f, 0.0130f, 0.0150f},
   {0.0570f, 0.0660f, 0.0780f},
   {0.1100f, 0.1200f, 0.1300f},
   {0.0095f, 0.0140f, 0.0160f},
    0.8500f, 0.8600f, 0.9000f,
    0.3800f, 0.1700f, 0.0140f},
  {"bmw339",
   {0.0120f, 0.0150f, 0.0160f},
   {0.0620f, 0.0760f, 0.0800f},
   {0.1100f, 0.1200f, 0.1200f},
   {0.0083f, 0.0150f, 0.0160f},
    0.9200f, 0.8700f, 0.9000f,
    0.3900f, 0.1700f, 0.0130f},
  {"2k acrylack",
   {0.4200f, 0.3200f, 0.1000f},
   {0.0000f, 0.0000f, 0.0000f},
   {0.0280f, 0.0260f, 0.0060f},
   {0.0170f, 0.0075f, 0.0041f},
    1.0000f, 0.9000f, 0.1700f,
    0.8800f, 0.8000f, 0.0150f},
  {"white",
   {0.6100f, 0.6300f, 0.5500f},
   {2.6e-06f, 0.00031f, 3.1e-08f},
   {0.0130f, 0.0110f, 0.0083f},
   {0.0490f, 0.0420f, 0.0370f},
    0.0490f, 0.4500f, 0.1700f,
    1.0000f, 0.1500f, 0.0150f},
  {"blue",
   {0.0079f, 0.0230f, 0.1000f},
   {0.0011f, 0.0015f, 0.0019f},
   {0.0250f, 0.0300f, 0.0430f},
   {0.0590f, 0.0740f, 0.0820f},
    1.0000f, 0.0940f, 0.1700f,
    0.1500f, 0.0430f, 0.0200f},
  {"blue matte",
   {0.0099f, 0.0360f, 0.1200f},
   {0.0032f, 0.0045f, 0.0059f},
   {0.1800f, 0.2300f, 0.2800f},
   {0.0400f, 0.0490f, 0.0510f},
    1.0000f, 0.0460f, 0.1700f,
    0.1600f, 0.0750f, 0.0340f}
};
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
