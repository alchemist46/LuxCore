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

#include <cstddef>

#include "slg/volumes/homogenous.h"
#include "luxrays/usings.h"
#include "slg/bsdf/bsdf.h"
#include "slg/textures/texture.h"

using namespace std;
using namespace luxrays;
using namespace slg;

//------------------------------------------------------------------------------
// HomogeneousVolume
//------------------------------------------------------------------------------

HomogeneousVolume::HomogeneousVolume(
	TextureConstRef iorTex,
	TextureConstPtr emiTex,
	TextureConstRef a, TextureConstRef s, TextureConstRef g,
	const bool multiScat
) :
	Volume(iorTex, emiTex),
	schlickScatter(*this, g),
	multiScattering(multiScat),
	sigmaA(a),
	sigmaS(s)
{}

float HomogeneousVolume::Scatter(const float u,
		const bool scatterAllowed, const float segmentLength,
		const Spectrum &sigmaA, const Spectrum &sigmaS, const Spectrum &emission,
		const Spectrum &throughput,
		Spectrum &segmentTransmittance, Spectrum &segmentEmission,
		float *uNoScatter) {
	// This code must work also with segmentLength = INFINITY

	bool scatter = false;
	segmentTransmittance = Spectrum(1.f);
	segmentEmission = Spectrum(0.f);

	//--------------------------------------------------------------------------
	// Check if there is a scattering event
	//--------------------------------------------------------------------------

	float scatterDistance = segmentLength;
	if (scatterAllowed && !sigmaS.Black()) {
		// Spectral MIS (Kutz et al. 2017, as Cycles): the distance is sampled
		// from the sigma_s of one colour channel, picked in proportion to the
		// throughput times the single scattering albedo, and weighted by the
		// mix of the three channels' pdfs. The distance takes u itself (the
		// sampler's stratification, see Scene::Intersect()), the channel a
		// hash of it (a sequence of its own, not the pass-through one)
		float channelPdfs[3];
		float channelPdfsSum = 0.f;
		for (u_int i = 0; i < 3; ++i) {
			channelPdfs[i] = (sigmaS.c[i] > 0.f) ?
				(throughput.c[i] * sigmaS.c[i] / (sigmaA.c[i] + sigmaS.c[i])) : 0.f;
			channelPdfsSum += channelPdfs[i];
		}
		if (!(channelPdfsSum > 0.f) || isinf(channelPdfsSum)) {
			// No throughput: the channels that scatter, uniformly
			channelPdfsSum = 0.f;
			for (u_int i = 0; i < 3; ++i) {
				channelPdfs[i] = (sigmaS.c[i] > 0.f) ? 1.f : 0.f;
				channelPdfsSum += channelPdfs[i];
			}
		}

		for (u_int i = 0; i < 3; ++i)
			channelPdfs[i] /= channelPdfsSum;

		union {
			float f;
			u_int i;
		} uBits;
		uBits.f = u;
		const float uChannel = TauswortheRandomGenerator(uBits.i ^ 0x9e3779b9u).floatValue();

		u_int channel = 0;
		float cdf = 0.f;
		for (u_int i = 0; i < 3; ++i) {
			if (channelPdfs[i] > 0.f) {
				channel = i;
				if (uChannel < cdf + channelPdfs[i])
					break;
				cdf += channelPdfs[i];
			}
		}

		// Determine scattering distance
		const float proposedScatterDistance = -logf(1.f - u) / sigmaS.c[channel];

		scatter = (proposedScatterDistance < segmentLength);
		scatterDistance = scatter ? proposedScatterDistance : segmentLength;

		// Without a scatter event, 1 - u <= exp(-sigma_s * segmentLength):
		// u remapped to [0, 1) (see Volume::Scatter())
		if (!scatter && uNoScatter) {
			const float noScatterProb = expf(-segmentLength * sigmaS.c[channel]);
			if (noScatterProb > 0.f)
				*uNoScatter = Clamp(1.f - (1.f - u) / noScatterProb, 0.f, .99999994f);
		}

		// Note: scatterDistance can not be infinity because otherwise there would
		// have been a scatter event before.
		float pdf = 0.f;
		for (u_int i = 0; i < 3; ++i) {
			if (channelPdfs[i] > 0.f)
				pdf += channelPdfs[i] * expf(-scatterDistance * sigmaS.c[i]) * (scatter ? sigmaS.c[i] : 1.f);
		}
		segmentTransmittance /= pdf;
	}

	//--------------------------------------------------------------------------
	// Volume transmittance
	//--------------------------------------------------------------------------
	
	const Spectrum sigmaT = sigmaA + sigmaS;
	if (!sigmaT.Black()) {
		if (isinf(scatterDistance)) {
			// This avoid NaN in case scatterDistance is inf
			segmentTransmittance = Spectrum(0.f);
		} else {
			const Spectrum tau = scatterDistance * sigmaT;
			segmentTransmittance *= Exp(-tau) * (scatter ? sigmaT : Spectrum(1.f));
		}
	}

	//--------------------------------------------------------------------------
	// Volume emission
	//--------------------------------------------------------------------------

	segmentEmission += segmentTransmittance * scatterDistance * emission;

	return scatter ? scatterDistance : -1.f;
}

Spectrum HomogeneousVolume::SigmaA(const HitPoint &hitPoint) const {
	return GetSigmaA().GetSpectrumValue(hitPoint).Clamp();
}

Spectrum HomogeneousVolume::SigmaS(const HitPoint &hitPoint) const {
	return GetSigmaS().GetSpectrumValue(hitPoint).Clamp();
}

float HomogeneousVolume::Scatter(const Ray &ray, const float u,
		const bool scatteredStart, const bool transmittanceOnly,
		const Spectrum &pathThroughput, Spectrum *connectionThroughput,
		Spectrum *connectionEmission, float *surfacePassThroughEvent) const {
	const float segmentLength = ray.maxt - ray.mint;

	// Check if I have to support multi-scattering
	const bool scatterAllowed = !transmittanceOnly && (!scatteredStart || multiScattering);

	// Point where to evaluate the volume
	HitPoint hitPoint;
	hitPoint.Init();
	hitPoint.fixedDir = ray.d;
	hitPoint.p = ray.o;
	hitPoint.geometryN = hitPoint.interpolatedN = hitPoint.shadeN = Normal(-ray.d);
	hitPoint.passThroughEvent = u;

	const Spectrum sigmaA = SigmaA(hitPoint);
	const Spectrum sigmaS = SigmaS(hitPoint);
	const Spectrum emission = Emission(hitPoint);

	Spectrum segmentTransmittance, segmentEmission;
	const float scatterDistance = HomogeneousVolume::Scatter(u, scatterAllowed,
			segmentLength, sigmaA, sigmaS, emission, pathThroughput * (*connectionThroughput),
			segmentTransmittance, segmentEmission, surfacePassThroughEvent);

	// I need to update first connectionEmission and than connectionThroughput
	*connectionEmission += *connectionThroughput * emission;
	*connectionThroughput *= segmentTransmittance;

	return (scatterDistance == -1.f) ? -1.f : (ray.mint + scatterDistance);
}

Spectrum HomogeneousVolume::Albedo(const HitPoint &hitPoint) const {
	return schlickScatter.Albedo(hitPoint);
}

Spectrum HomogeneousVolume::Evaluate(const HitPoint &hitPoint,
		const Vector &localLightDir, const Vector &localEyeDir, BSDFEvent *event,
		float *directPdfW, float *reversePdfW) const {
	return schlickScatter.Evaluate(hitPoint, localLightDir, localEyeDir, event, directPdfW, reversePdfW);
}

Spectrum HomogeneousVolume::Sample(const HitPoint &hitPoint,
		const Vector &localFixedDir, Vector *localSampledDir,
		const float u0, const float u1, const float passThroughEvent,
		float *pdfW, BSDFEvent *event) const {
	return schlickScatter.Sample(hitPoint, localFixedDir, localSampledDir,
			u0, u1, passThroughEvent, pdfW, event);
}

void HomogeneousVolume::Pdf(const HitPoint &hitPoint,
		const Vector &localLightDir, const Vector &localEyeDir,
		float *directPdfW, float *reversePdfW) const {
	schlickScatter.Pdf(hitPoint, localLightDir, localEyeDir, directPdfW, reversePdfW);
}

void HomogeneousVolume::AddReferencedTextures(std::unordered_set<const Texture *>  &referencedTexs) const {
	Volume::AddReferencedTextures(referencedTexs);

	GetSigmaA().AddReferencedTextures(referencedTexs);
	GetSigmaS().AddReferencedTextures(referencedTexs);
	schlickScatter.GetG().AddReferencedTextures(referencedTexs);
}

void HomogeneousVolume::UpdateTextureReferences(
	TextureConstRef oldTex, TextureRef newTex
) {
	Volume::UpdateTextureReferences(oldTex, newTex);

	updtex(sigmaA, oldTex, newTex);
	updtex(sigmaS, oldTex, newTex);
	if (&schlickScatter.GetG() == &oldTex)
		schlickScatter.SetG(newTex);
}

PropertiesUPtr HomogeneousVolume::ToProperties() const {
	PropertiesUPtr props = std::make_unique<Properties>();

	const string name = GetName();
	props->Set(Property("scene.volumes." + name + ".type")("homogeneous"));
	props->Set(Property("scene.volumes." + name + ".absorption")(GetSigmaA().GetSDLValue()));
	props->Set(Property("scene.volumes." + name + ".scattering")(GetSigmaS().GetSDLValue()));
	props->Set(Property("scene.volumes." + name + ".asymmetry")(schlickScatter.GetG().GetSDLValue()));
	props->Set(Property("scene.volumes." + name + ".multiscattering")(multiScattering));
	props->Set(Volume::ToProperties());

	return props;
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
