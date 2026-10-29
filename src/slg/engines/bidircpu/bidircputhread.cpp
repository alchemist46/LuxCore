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

// NOTE: this is code is heavily based on Tomas Davidovic's SmallVCM
// (http://www.davidovic.cz and http://www.smallvcm.com)

#include <thread>
#include <cassert>
#include <boost/format.hpp>

#include "luxrays/core/device.h"
#include "luxrays/core/intersectiondevice.h"
#include "luxrays/usings.h"
#include "luxrays/utils/thread.h"

#include "slg/engines/bidircpu/bidircpu.h"
#include "slg/cameras/camera.h"
#include "slg/lights/light.h"
#include "slg/materials/glass.h"

using namespace std;
using namespace luxrays;
using namespace slg;
using namespace std::literals::chrono_literals;

//------------------------------------------------------------------------------
// BiDirCPU RenderThread
//------------------------------------------------------------------------------

const Film::FilmChannels BiDirCPURenderThread::eyeSampleResultsChannels({
	Film::RADIANCE_PER_PIXEL_NORMALIZED, Film::ALPHA, Film::DEPTH,
	Film::POSITION, Film::GEOMETRY_NORMAL, Film::SHADING_NORMAL, Film::MATERIAL_ID,
	Film::UV, Film::OBJECT_ID, Film::SAMPLECOUNT, Film::CONVERGENCE,
	Film::MATERIAL_ID_COLOR, Film::ALBEDO, Film::AVG_SHADING_NORMAL, Film::NOISE
});

const Film::FilmChannels BiDirCPURenderThread::lightSampleResultsChannels({
	Film::RADIANCE_PER_SCREEN_NORMALIZED
}); 

BiDirCPURenderThread::BiDirCPURenderThread(BiDirCPURenderEngine *engine,
		const u_int index, IntersectionDeviceRef device) :
		CPUNoTileRenderThread(engine, index, device) {
}

void BiDirCPURenderThread::AOVWarmUp(
	std::stop_token stop_token,
	const luxrays::RandomGeneratorUPtr & rndGen
) {
	if (threadIndex == 0)
		SLG_LOG("[BiDirCPURenderThread::" << threadIndex << "] AOV warmup started");

	const double start = WallClockTime();
	double lastProgressPrint = start;

	BiDirCPURenderEngine *engine = (BiDirCPURenderEngine *)renderEngine;
	auto& scene = engine->renderConfig.GetScene();
	auto& camera = scene.GetCamera();

	SobolSampler sampler(rndGen, engine->GetFilm(), engine->GetSampleSplatter(), true, 0.f, 0.f,
		16, 16, 1, 1,
		engine->aovWarmupSamplerSharedData
	);

	// Request the samples
	const u_int sampleBootSize = 5;
	const u_int sampleStepSize = 3;
	const u_int sampleSize = 
		sampleBootSize + // To generate eye ray
		engine->maxEyePathDepth * sampleStepSize; // For each path vertex
	sampler.RequestSamples(ONLY_AOV_SAMPLE, sampleSize);

	// Initialize SampleResult 
	vector<SampleResult> sampleResults(1);
	SampleResult &sampleResult = sampleResults[0];
	const Film::FilmChannels sampleResultsChannels({
		Film::ALBEDO, Film::AVG_SHADING_NORMAL
	});

	sampleResult.Init(&sampleResultsChannels, engine->GetFilm().GetRadianceGroupCount());

	// Initialize the max. path depth
	PathDepthInfo maxPathDepthInfo;
	maxPathDepthInfo.depth = engine->maxEyePathDepth;
	maxPathDepthInfo.diffuseDepth = engine->maxEyePathDepth;
	maxPathDepthInfo.glossyDepth = engine->maxEyePathDepth;
	maxPathDepthInfo.subsurfaceDepth = engine->maxEyePathDepth;
	maxPathDepthInfo.transmissionDepth = engine->maxEyePathDepth;

	while (sampler.GetPassCount() < engine->aovWarmupSPP) {
		if (stop_token.stop_requested())
			return;

		sampleResult.filmX = sampler.GetSample(0);
		sampleResult.filmY = sampler.GetSample(1);

		const float timeSample = sampler.GetSample(4);
		const float time = scene.GetCamera().GenerateRayTime(timeSample);

		Ray eyeRay;
		PathVolumeInfo volInfo;
		camera.GenerateRay(time,
				sampleResult.filmX, sampleResult.filmY, &eyeRay,
				&volInfo, sampler.GetSample(2), sampler.GetSample(3));

		sampleResult.albedo = Spectrum(); // Just in case albedoToDo is never true
		sampleResult.shadingNormal = Normal();

		Spectrum pathThroughput(1.f);
		u_int depth = 0;
		BSDF bsdf;
		while (depth < engine->maxEyePathDepth) {
			sampleResult.firstPathVertex = (depth == 0);

			const u_int sampleOffset = sampleBootSize + depth * sampleStepSize;

			// NOTE: I account for volume emission only with path tracing (i.e. here and
			// not in any other place)
			RayHit eyeRayHit;
			Spectrum connectionThroughput;
			const bool hit = scene.Intersect(IntersectionDevicePtr(&device),
					EYE_RAY | (sampleResult.firstPathVertex ? CAMERA_RAY : INDIRECT_RAY),
					&volInfo, sampler.GetSample(sampleOffset),
					&eyeRay, &eyeRayHit, &bsdf,
					&connectionThroughput, &pathThroughput, &sampleResult);
			pathThroughput *= connectionThroughput;

			if (!hit) {
				// Nothing was hit
				break;
			}

			//------------------------------------------------------------------
			// Something was hit
			//------------------------------------------------------------------

			if (bsdf.IsAlbedoEndPoint(engine->albedoSpecularSetting,
					engine->albedoSpecularGlossinessThreshold)) {
				sampleResult.albedo = pathThroughput * bsdf.Albedo();
				sampleResult.shadingNormal = bsdf.hitPoint.shadeN;
				break;
			}

			// Check if I reached the max. depth
			if (depth == engine->maxEyePathDepth)
				break;

			//------------------------------------------------------------------
			// Build the next vertex path ray
			//------------------------------------------------------------------

			Vector sampledDir;
			float cosSampledDir, lastPdfW;
			BSDFEvent lastBSDFEvent;
			const Spectrum bsdfSample = bsdf.Sample(&sampledDir,
						sampler.GetSample(sampleOffset + 1),
						sampler.GetSample(sampleOffset + 2),
						&lastPdfW, &cosSampledDir, &lastBSDFEvent);

			assert (!bsdfSample.IsNaN() && !bsdfSample.IsInf());
			if (bsdfSample.Black())
				break;
			assert (!isnan(lastPdfW) && !isinf(lastPdfW));

			pathThroughput *= bsdfSample;
			assert (!pathThroughput.IsNaN() && !pathThroughput.IsInf());

			// Update volume information
			volInfo.Update(lastBSDFEvent, bsdf);

			eyeRay.Update(bsdf.GetRayOrigin(sampledDir), sampledDir);
			// A ray from a point in a volume starts at the point, see BSDF::GetRayOrigin()
			if (bsdf.IsVolume())
				eyeRay.mint = 0.f;
		}

		sampler.NextSample(sampleResults);

		if (threadIndex == 0) {
			const double end = WallClockTime();
			const double delta = end - lastProgressPrint;

			if (delta > 2.0) {
				const u_int *filmSubRegion = engine->GetFilm().GetSubRegion();
				const u_int subRegionWidth = filmSubRegion[1] - filmSubRegion[0] + 1;
				const u_int subRegionHeight = filmSubRegion[3] - filmSubRegion[2] + 1;

				const double samplesSec = sampler.GetPassCount() * (double)(subRegionWidth * subRegionHeight) / (1000000.0 * (end -start));

				SLG_LOG("[BiDirCPURenderThread::" << threadIndex << "] AOV warmup progress: " <<
						sampler.GetPassCount() << "/" << engine->aovWarmupSPP << " pass"
						" (" << boost::str(boost::format("%3.2fM") % samplesSec) << " samples/sec)");
				
				lastProgressPrint = WallClockTime();
			}
		}
#ifdef WIN32
		// Work around Windows bad scheduling
        std::this_thread::yield();
#endif
	}
	
	if (threadIndex == 0) {
		const double end = WallClockTime();
		
		const u_int *filmSubRegion = engine->GetFilm().GetSubRegion();
		const u_int subRegionWidth = filmSubRegion[1] - filmSubRegion[0] + 1;
		const u_int subRegionHeight = filmSubRegion[3] - filmSubRegion[2] + 1;
		
		const double samplesSec = engine->aovWarmupSPP * (double)(subRegionWidth * subRegionHeight) / (1000000.0 * (end -start));
		
		SLG_LOG("[BiDirCPURenderThread::" << threadIndex << "] AOV warmup done: " <<
				boost::str(boost::format("%3.2fM") % samplesSec) << " samples/sec");
	}
}

SampleResult &BiDirCPURenderThread::AddResult(vector<SampleResult> &sampleResults, const bool fromLight) const {
	BiDirCPURenderEngine *engine = (BiDirCPURenderEngine *)renderEngine;

	const u_int size = sampleResults.size();
	sampleResults.resize(size + 1);

	SampleResult &sampleResult = sampleResults[size];

	sampleResult.Init(
			fromLight ? &lightSampleResultsChannels : &eyeSampleResultsChannels,
			engine->GetFilm().GetRadianceGroupCount());

	return sampleResult;
}

Spectrum BiDirCPURenderThread::DispersionOnce(const PathVertexVM &eyeVertex, const BSDFEvent eyeEvent,
		const PathVertexVM &lightVertex, const BSDFEvent lightEvent, const Spectrum &radiance) const {
	// Each side has the colour of the wavelength once, from the refractions
	// of its path or from its end of the connection (eyeEvent, lightEvent)
	const bool eyeTinted = eyeVertex.dispersionTinted || (eyeEvent & DISPERSED);
	const bool lightTinted = lightVertex.dispersionTinted || (lightEvent & DISPERSED);
	if (!eyeTinted || !lightTinted)
		return radiance;

	// Both: the path needs it once (a channel the colour doesn't have is
	// already 0)
	const Spectrum c = GlassMaterial::WaveLengthColor(eyeVertex.wavelength,
			((BiDirCPURenderEngine *)renderEngine)->renderConfig.GetScene().GetWorkingColorSpace());
	Spectrum result;
	for (u_int i = 0; i < COLOR_SAMPLES; ++i)
		result.c[i] = (c.c[i] > 0.f) ? (radiance.c[i] / c.c[i]) : 0.f;

	return result;
}

void BiDirCPURenderThread::ConnectVertices(const float time,
		const PathVertexVM &eyeVertex, const PathVertexVM &lightVertex,
		SampleResult &eyeSampleResult, const float u0) const {
	BiDirCPURenderEngine *engine = (BiDirCPURenderEngine *)renderEngine;
	auto& scene = engine->renderConfig.GetScene();

	Vector p2pDir(lightVertex.bsdf.hitPoint.p - eyeVertex.bsdf.hitPoint.p);
	const float p2pDistance2 = p2pDir.LengthSquared();
	const float p2pDistance = sqrtf(p2pDistance2);
	p2pDir /= p2pDistance;

	// Check eye vertex BSDF
	float eyeBsdfPdfW, eyeBsdfRevPdfW;
	BSDFEvent eyeEvent;
	const Spectrum eyeBsdfEval = eyeVertex.bsdf.Evaluate(p2pDir, &eyeEvent, &eyeBsdfPdfW, &eyeBsdfRevPdfW);

	if (!eyeBsdfEval.Black()) {
		// Check light vertex BSDF
		float lightBsdfPdfW, lightBsdfRevPdfW;
		BSDFEvent lightEvent;
		const Spectrum lightBsdfEval = lightVertex.bsdf.Evaluate(-p2pDir, &lightEvent, &lightBsdfPdfW, &lightBsdfRevPdfW);

		if (!lightBsdfEval.Black()) {
			// Check if the 2 surfaces can see each other (a volume vertex has
			// no cosine: its shadeN is the fixed direction)
			const float cosThetaAtCamera = eyeVertex.bsdf.IsVolume() ? 1.f : Dot(eyeVertex.bsdf.hitPoint.shadeN, p2pDir);
			const float cosThetaAtLight = lightVertex.bsdf.IsVolume() ? 1.f : Dot(lightVertex.bsdf.hitPoint.shadeN, -p2pDir);
			// Was:
			//  const float geometryTerm = cosThetaAtCamera * cosThetaAtLight / eyeDistance2;
			//
			// but now BSDF::Evaluate() follows LuxRender habit to return the
			// result multiplied by cosThetaAtLight
			const float geometryTerm = 1.f / p2pDistance2;

			// Trace ray between the two vertices
			const Point shadowRayOrig = eyeVertex.bsdf.GetRayOrigin(p2pDir);
			const Vector shadowRayOrigP2P(lightVertex.bsdf.hitPoint.p - shadowRayOrig);
			const float shadowRayDistanceSquared = shadowRayOrigP2P.LengthSquared();
			const float shadowRayDistance = sqrtf(shadowRayDistanceSquared);
			const Vector shadowRayDir = shadowRayOrigP2P / shadowRayDistance;

			Ray p2pRay(shadowRayOrig, shadowRayDir,
					0.f,
					shadowRayDistance,
					time);
			p2pRay.UpdateMinMaxWithEpsilon();
			// A ray from a point in a volume starts at the point, see BSDF::GetRayOrigin()
			if (eyeVertex.bsdf.IsVolume())
				p2pRay.mint = 0.f;
			if (lightVertex.bsdf.IsVolume())
				p2pRay.maxt = shadowRayDistance;

			RayHit p2pRayHit;
			BSDF bsdfConn;
			Spectrum connectionThroughput;
			PathVolumeInfo volInfo = eyeVertex.volInfo; // I need to use a copy here
			// The ray starts in the volume of the eye vertex's side it leaves,
			// as in DirectLightSampling(). The volume of the light vertex's
			// side (the old rule) is wrong when the segment crosses another
			// object: with that object's interior as the current volume, the
			// volume priority system passed the ray through its surface
			// (PathVolumeInfo::ContinueToTrace()) instead of stopping it
			if (!eyeVertex.bsdf.IsVolume() && (eyeEvent & TRANSMIT))
				volInfo.Update(eyeEvent, eyeVertex.bsdf);
			if (!scene.Intersect(
					luxrays::make_observer(device),
					LIGHT_RAY | INDIRECT_RAY | SHADOW_RAY,
					&volInfo, u0, &p2pRay, &p2pRayHit, &bsdfConn,
					&connectionThroughput)) {
				// Nothing was hit, the light path vertex is visible

				// The MIS weights leave the Russian roulette out: the probability of the
				// other direction (and the depth of the other subpath) is not known here,
				// and weights that sum to 1 need no true pdfs (Bounce() leaves it out of
				// dVC and dVM too)
				// Convert pdfs to area pdfs
				const float eyeBsdfPdfA = PdfWtoA(eyeBsdfPdfW, p2pDistance, cosThetaAtLight);
				const float lightBsdfPdfA = PdfWtoA(lightBsdfPdfW,  p2pDistance, cosThetaAtCamera);

				// MIS weights
				float eyeDVCM, eyeDVC;
				EyeMISQuantities(eyeVertex, lightVertex.depth, &eyeDVCM, &eyeDVC);
				const float lightWeight = MIS(eyeBsdfPdfA) *
					(misVmWeightFactor + lightVertex.dVCM + lightVertex.dVC * MIS(lightBsdfRevPdfW));
				const float eyeWeight = MIS(lightBsdfPdfA) *
					(misVmWeightFactor + eyeDVCM + eyeDVC * MIS(eyeBsdfRevPdfW));

				const float misWeight = 1.f / (lightWeight + 1.f + eyeWeight);

				eyeSampleResult.radiance[lightVertex.lightID] += DispersionOnce(eyeVertex, eyeEvent, lightVertex, lightEvent,
						(misWeight * geometryTerm) * eyeVertex.throughput * eyeBsdfEval *
						connectionThroughput * lightBsdfEval * lightVertex.throughput);
			}
		}
	}
}

void BiDirCPURenderThread::ConnectToEye(const float time,
		const PathVertexVM &lightVertex, const float u0,
		const Point &lensPoint, vector<SampleResult> &sampleResults) const {
	// I don't connect camera invisible objects with the eye
	if (lightVertex.bsdf.IsCameraInvisible())
		return;

	BiDirCPURenderEngine *engine = (BiDirCPURenderEngine *)renderEngine;
	auto& scene = engine->renderConfig.GetScene();

	Vector eyeDir;
	float eyeDistance = 0;
    if (scene.GetCamera().GetType() == Camera::ORTHOGRAPHIC){
		// Orthographic camera need to be handled separately,
		// lensPoint can not be pre-calculated in this case
		Point p = lightVertex.bsdf.hitPoint.p;
		eyeDir = scene.GetCamera().GetDir();
		// calculate distance from vertex to camera plane
		const float D = -eyeDir.x*lensPoint.x - eyeDir.y*lensPoint.y - eyeDir.z*lensPoint.z;
		eyeDistance = eyeDir.x*p.x + eyeDir.y*p.y + eyeDir.z*p.z + D;
		eyeDistance = fabsf(eyeDistance);
	} else {
		eyeDir = Vector(lightVertex.bsdf.hitPoint.p - lensPoint);
		eyeDistance = eyeDir.Length();
		eyeDir /= eyeDistance;
	}

	float bsdfPdfW, bsdfRevPdfW;
	BSDFEvent event;
	const Spectrum bsdfEval = lightVertex.bsdf.Evaluate(-eyeDir, &event, &bsdfPdfW, &bsdfRevPdfW);

	if (!bsdfEval.Black()) {
		float filmX, filmY;
		bool sampleSuccess;
		Ray eyeRay;
		
		if (scene.GetCamera().GetType() == Camera::ORTHOGRAPHIC){
			// Along the camera direction, as the camera's rays: the vertex is
			// seen from the film point it projects to, not from lensPoint
			eyeRay = Ray(lightVertex.bsdf.hitPoint.p, eyeDir,
				0.f,
				eyeDistance,
				time);
			sampleSuccess = scene.GetCamera().ProjectToImage(&eyeRay, &filmX, &filmY);
		} else {
			eyeRay = Ray(lensPoint, eyeDir,
				0.f,
				eyeDistance,
				time);
			sampleSuccess = scene.GetCamera().GetSamplePosition(&eyeRay, &filmX, &filmY);
		}
		if (sampleSuccess) {
			// I have to flip the direction of the traced ray because
			// the information inside PathVolumeInfo are about the path from
			// the light toward the camera (i.e. ray.o would be in the wrong
			// place).
			Ray traceRay(lightVertex.bsdf.GetRayOrigin(-eyeRay.d), -eyeRay.d,
					eyeDistance - eyeRay.maxt,
					eyeDistance - eyeRay.mint,
					time);
			RayHit traceRayHit;

			BSDF bsdfConn;
			Spectrum connectionThroughput;
			PathVolumeInfo volInfo = lightVertex.volInfo; // I need to use a copy here
			// The camera on the other side of the surface is seen through its
			// interior volume, or from inside it
			if (!lightVertex.bsdf.IsVolume() && (event & TRANSMIT))
				volInfo.Update(event, lightVertex.bsdf);
			if (!scene.Intersect(
					luxrays::make_observer(device),
					LIGHT_RAY | CAMERA_RAY,
					&volInfo, u0, &traceRay, &traceRayHit, &bsdfConn,
					&connectionThroughput)) {
				// Nothing was hit, the light path vertex is visible

				// No Russian roulette in the MIS weights, see ConnectVertices()

				// A volume vertex has no cosine (its shadeN is the fixed direction)
				const float cosToCamera = lightVertex.bsdf.IsVolume() ? 1.f : Dot(lightVertex.bsdf.hitPoint.shadeN, -eyeDir);
				float cameraPdfW, fluxToRadianceFactor;
				scene.GetCamera().GetPDF(eyeRay, eyeDistance, filmX, filmY, &cameraPdfW, &fluxToRadianceFactor);
				// The orthographic camera's pdf is per area of its film plane,
				// which its parallel rays project with the cosine alone
				const float cameraPdfA = (scene.GetCamera().GetType() == Camera::ORTHOGRAPHIC) ?
						(cameraPdfW * fabsf(cosToCamera)) :
						PdfWtoA(cameraPdfW, eyeDistance, cosToCamera);
				// Was:
				//  const float fluxToRadianceFactor = cameraPdfA;
				//
				// but now BSDF::Evaluate() follows LuxRender habit to return the
				// result multiplied by cosThetaToLight, and the phase function of
				// a volume has no cosine to add (as PathTracer::ConnectToEye())

				const float weightLight = MIS(cameraPdfA) *
					(misVmWeightFactor + lightVertex.dVCM + lightVertex.dVC * MIS(bsdfRevPdfW));
				const float misWeight = 1.f / (weightLight + 1.f);

				const Spectrum radiance = (misWeight * fluxToRadianceFactor) *
					connectionThroughput * lightVertex.throughput * bsdfEval;

				SampleResult &sampleResult = AddResult(sampleResults, true);
				sampleResult.filmX = filmX;
				sampleResult.filmY = filmY;
				
				// Add radiance from the light source
				sampleResult.radiance[lightVertex.lightID] = radiance;
			}
		}
	}
}

void BiDirCPURenderThread::DirectLightSampling(const float time,
		const float u0, const float u1, const float u2,
		const float u3, const float u4,
		const PathVertexVM &eyeVertex,
		SampleResult &eyeSampleResult) const {
	BiDirCPURenderEngine *engine = (BiDirCPURenderEngine *)renderEngine;
	auto& scene = engine->renderConfig.GetScene();
	
	if (!eyeVertex.bsdf.IsDelta()) {
		// Pick a light source to sample
		const Normal landingNormal = eyeVertex.bsdf.hitPoint.intoObject ? eyeVertex.bsdf.hitPoint.geometryN : -eyeVertex.bsdf.hitPoint.geometryN;
		float lightPickPdf;
		auto light = scene.GetLightSources().GetEmitLightStrategy().SampleLights(
			scene, u0,
			eyeVertex.bsdf.hitPoint.p,
			landingNormal,
			eyeVertex.bsdf.IsVolume(),
			&lightPickPdf
		);

		if (light) {
			Ray shadowRay;
			float directPdfW, emissionPdfW, cosThetaAtLight;
			const Spectrum lightRadiance = light->Illuminate(scene, eyeVertex.bsdf,
					time, u1, u2, u3, shadowRay, directPdfW, &emissionPdfW,
					&cosThetaAtLight);

			if (!lightRadiance.Black()) {
				BSDFEvent event;
				float bsdfPdfW, bsdfRevPdfW;
				const Spectrum bsdfEval = eyeVertex.bsdf.Evaluate(shadowRay.d, &event, &bsdfPdfW, &bsdfRevPdfW);

				if (!bsdfEval.Black()) {
					RayHit shadowRayHit;
					BSDF shadowBsdf;
					Spectrum connectionThroughput;
					PathVolumeInfo volInfo = eyeVertex.volInfo; // I need to use a copy here
					// A light on the other side of the surface is seen through
					// its interior volume, or from inside it
					if (!eyeVertex.bsdf.IsVolume() && (event & TRANSMIT))
						volInfo.Update(event, eyeVertex.bsdf);
					// Check if the light source is visible
					if (!scene.Intersect(
						luxrays::make_observer(device),
						EYE_RAY | SHADOW_RAY, &volInfo, u4,
						&shadowRay, &shadowRayHit, &shadowBsdf, &connectionThroughput)) {
						// I'm ignoring volume emission because it is not sampled in
						// direct light step.

						// If the light source is not intersectable, it can not be
						// sampled with BSDF
						bsdfPdfW *= (light->IsEnvironmental() || light->IsIntersectable()) ? 1.f : 0.f;

						// No Russian roulette in the MIS weights, see ConnectVertices()

						const float cosThetaToLight = eyeVertex.bsdf.IsVolume() ? 1.f : AbsDot(shadowRay.d, eyeVertex.bsdf.hitPoint.shadeN);
						const float directLightSamplingPdfW = directPdfW * lightPickPdf;
						// An environmental light emits from a disk normal to its
						// direction (emissionPdfW is per area of that disk), so no
						// cosine at the light, as TraceLightPath(): with the one at
						// its sphere the light paths got too much weight, more so
						// in a small scene (an orthographic camera close to the
						// object read 0.05% dark)
						const float usedCosLight = light->IsEnvironmental() ? 1.f : cosThetaAtLight;

						// emissionPdfA / directPdfA = emissionPdfW / directPdfW
						float eyeDVCM, eyeDVC;
						EyeMISQuantities(eyeVertex, 0, &eyeDVCM, &eyeDVC);
						const float weightLight = MIS(bsdfPdfW / directLightSamplingPdfW);
						const float weightCamera = MIS(emissionPdfW * cosThetaToLight / (directPdfW * usedCosLight)) *
								(misVmWeightFactor + eyeDVCM + eyeDVC * MIS(bsdfRevPdfW));
						// Disable MIS if we have gone trough a shadow transparent object
						const float misWeight = shadowBsdf.hitPoint.throughShadowTransparency ?
							1.f : (1.f / (weightLight + 1.f + weightCamera));

						// Was:
						//  const float factor = cosThetaToLight / directLightSamplingPdfW;
						//
						// but now BSDF::Evaluate() follows LuxRender habit to return the
						// result multiplied by cosThetaToLight
						const float factor = 1.f / directLightSamplingPdfW;

						eyeSampleResult.radiance[light->GetID()] += (misWeight * factor) *
								eyeVertex.throughput * connectionThroughput * lightRadiance * bsdfEval;

						assert (eyeSampleResult.IsValid());
					}
				}
			}
		}
	}
}

void BiDirCPURenderThread::DirectHitLight(
	LightSourceConstRef light,
	const Spectrum &lightRadiance,
	const float directPdfA,
	const float emissionPdfW,
	const PathVertexVM &eyeVertex,
	Spectrum *radiance
) const {
	if (lightRadiance.Black())
		return;

	if (eyeVertex.depth == 1) {
		*radiance += eyeVertex.throughput * lightRadiance;
		return;
	}

	BiDirCPURenderEngine *engine = (BiDirCPURenderEngine *)renderEngine;
	auto& scene = engine->renderConfig.GetScene();

	const float lightPickPdf = scene.GetLightSources().GetEmitLightStrategy().SampleLightPdf(
		light,
		eyeVertex.bsdf.hitPoint.p,
		eyeVertex.bsdf.hitPoint.geometryN,
		eyeVertex.bsdf.IsVolume()
	);

	// MIS weight
	float eyeDVCM, eyeDVC;
	EyeMISQuantities(eyeVertex, -1, &eyeDVCM, &eyeDVC);
	const float weightCamera = MIS(directPdfA * lightPickPdf) * eyeDVCM +
		MIS(emissionPdfW * lightPickPdf) * eyeDVC;
	const float misWeight = 1.f / (weightCamera + 1.f);

	*radiance += misWeight * eyeVertex.throughput * lightRadiance;

	assert (radiance->IsValid());
}

void BiDirCPURenderThread::DirectHitLight(
	const bool finiteLightSource,
	const PathVertexVM &eyeVertex,
	SampleResult &eyeSampleResult
) const {
	float directPdfA, emissionPdfW;
	if (finiteLightSource) {
		const Spectrum lightRadiance = eyeVertex.bsdf.GetEmittedRadiance(
			&directPdfA, &emissionPdfW
		);

		DirectHitLight(
			*eyeVertex.bsdf.GetLightSource(),
			lightRadiance,
			directPdfA,
			emissionPdfW,
			eyeVertex, &eyeSampleResult.radiance[eyeVertex.bsdf.GetLightID()]
		);
	} else {
		BiDirCPURenderEngine *engine = (BiDirCPURenderEngine *)renderEngine;
		auto& scene = engine->renderConfig.GetScene();

		for(EnvLightSource& el: scene.GetLightSources().GetEnvLightSources()) {
			const Spectrum lightRadiance = el.GetRadiance(scene,
					(eyeVertex.depth == 1) ? nullptr : &eyeVertex.bsdf,
					eyeVertex.bsdf.hitPoint.fixedDir, &directPdfA, &emissionPdfW);

			DirectHitLight(el, lightRadiance, directPdfA, emissionPdfW,
					eyeVertex, &eyeSampleResult.radiance[el.GetID()]);
		}
	}
}

bool BiDirCPURenderThread::TraceLightPath(const float time, const float wavelength,
		const SamplerUPtr& sampler, CameraConstRef camera,
		vector<PathVertexVM> &lightPathVertices,
		vector<SampleResult> &sampleResults) const {
	BiDirCPURenderEngine *engine = (BiDirCPURenderEngine *)renderEngine;
	auto& scene = engine->renderConfig.GetScene();

	// Select one light source
	// BiDir can use only a single strategy, emit in this case
	float lightPickPdf;
	auto light = scene.GetLightSources().GetEmitLightStrategy().
			SampleLights(scene, sampler->GetSample(2), &lightPickPdf);
	if (!light)
		return false;

	// Initialize the light path
	PathVertexVM lightVertex;
	lightVertex.lightID = light->GetID();
	lightVertex.wavelength = wavelength;
	lightVertex.dispersionTinted = false;
	
	float lightEmitPdfW, lightDirectPdfW, cosThetaAtLight;
	Ray lightRay;
	lightVertex.throughput = light->Emit(scene,
			time, sampler->GetSample(5), sampler->GetSample(6),
			sampler->GetSample(7), sampler->GetSample(8), sampler->GetSample(9),
			lightRay, lightEmitPdfW,
			&lightDirectPdfW, &cosThetaAtLight);
	if (!lightVertex.throughput.Black()) {
		lightVertex.volInfo.AddVolume(light->volume);

		lightEmitPdfW *= lightPickPdf;
		lightDirectPdfW *= lightPickPdf;

		lightVertex.throughput /= lightEmitPdfW;
		assert (!lightVertex.throughput.IsNaN() && !lightVertex.throughput.IsInf());

		// I don't store the light vertex 0 because direct lighting will take
		// care of these kind of paths
		lightVertex.dVCM = MIS(lightDirectPdfW / lightEmitPdfW);
		// If the light source is not intersectable, it can not be
		// sampled with BSDF
		if (light->IsEnvironmental() || light->IsIntersectable()) {
			const float usedCosLight = light->IsEnvironmental() ? 1.f : cosThetaAtLight;
			lightVertex.dVC = MIS(usedCosLight / lightEmitPdfW);
		} else
			lightVertex.dVC = 0.f;
		lightVertex.dVM = lightVertex.dVC * misVcWeightFactor;

		lightVertex.depth = 1;
		while (lightVertex.depth <= engine->maxLightPathDepth) {
			const u_int sampleOffset = sampleBootSize + (lightVertex.depth - 1) * sampleLightStepSize;

			RayHit nextEventRayHit;
			Spectrum connectionThroughput;
			const bool hit = scene.Intersect(
					luxrays::make_observer(device),
					LIGHT_RAY | INDIRECT_RAY,
					&lightVertex.volInfo, sampler->GetSample(sampleOffset),
					&lightRay, &nextEventRayHit, &lightVertex.bsdf,
					&connectionThroughput);

			if (hit) {
				// Something was hit
				
				// Check if it is something with a not black shadow transparency
				// and stop if it has. Direct light sampling will take care of
				// this kind of paths.
				if (!lightVertex.bsdf.GetPassThroughShadowTransparency().Black() & !lightVertex.bsdf.GetPassThroughShadowTransparencyOverride())
					break;

				// Update the new light vertex
				lightVertex.SetDispersion();
				lightVertex.throughput *= connectionThroughput;

				// The direct light sampling at the first vertex picks the
				// direction of an environmental light with its visibility map
				// cache or its MIS distribution, not with the distribution
				// Emit() samples and returns the pdf of
				if ((lightVertex.depth == 1) && light->IsEnvironmental()) {
					// Left at -1 when the direction has no lat-long mapping
					float hitDirectPdfW = -1.f;
					static_cast<const EnvLightSource &>(*light).GetRadiance(scene,
							&lightVertex.bsdf, lightRay.d, &hitDirectPdfW);
					if (hitDirectPdfW >= 0.f)
						lightVertex.dVCM = MIS(hitDirectPdfW * lightPickPdf / lightEmitPdfW);
				}

				// Infinite lights use MIS based on solid angle instead of area
				if((lightVertex.depth > 1) || !light->IsEnvironmental())
					lightVertex.dVCM *= MIS(nextEventRayHit.t * nextEventRayHit.t);
				const float factor = 1.f / MIS(AbsDot(lightVertex.bsdf.hitPoint.shadeN, lightRay.d));
				lightVertex.dVCM *= factor;
				lightVertex.dVC *= factor;
				lightVertex.dVM *= factor;

				// Store the vertex only if it isn't specular
				if (!lightVertex.bsdf.IsDelta()) {
					lightPathVertices.push_back(lightVertex);

					//----------------------------------------------------------
					// Try to connect the light path vertex with the eye
					//----------------------------------------------------------

					// Sample a point on the camera lens
					Point lensPoint;
					camera.SampleLens(time, sampler->GetSample(3), sampler->GetSample(4), &lensPoint);

					ConnectToEye(time, lightVertex, sampler->GetSample(sampleOffset + 1),
							lensPoint, sampleResults);
				}

				if (lightVertex.depth >= engine->maxLightPathDepth)
					break;

				//--------------------------------------------------------------
				// Build the next vertex path ray
				//--------------------------------------------------------------

				if (!Bounce(time, sampler, sampleOffset + 2, &lightVertex, &lightRay, false))
					break;
			} else {
				// Ray lost in space...
				break;
			}
		}
	}
	
	return true;
}

void BiDirCPURenderThread::EyeMISQuantities(const PathVertexVM &eyeVertex, const int lightDepth,
		float *dVCM, float *dVC) const {
	BiDirCPURenderEngine *engine = (BiDirCPURenderEngine *)renderEngine;

	// The strategy that moves the connection j steps toward the camera has a
	// light path vertex of depth lightDepth + j (j = depth: the light tracing),
	// so the ones beyond light.maxdepth are left out
	const int maxJ = (int)engine->maxLightPathDepth - lightDepth;
	if (!engine->lightDepthCutsStrategies || (maxJ >= (int)eyeVertex.depth)) {
		*dVCM = eyeVertex.dVCM;
		*dVC = eyeVertex.dVC;
		return;
	}

	assert (eyeVertex.dVCTerms.size() + 1 == eyeVertex.depth);
	*dVCM = (maxJ >= 1) ? eyeVertex.dVCM : 0.f;
	// dVCTerms[size - (j - 1)] is the term j
	const int size = (int)eyeVertex.dVCTerms.size();
	float sum = 0.f;
	for (int j = 2; j <= maxJ; ++j)
		sum += eyeVertex.dVCTerms[size - (j - 1)];
	*dVC = sum;
}

bool BiDirCPURenderThread::Bounce(const float time, const SamplerUPtr& sampler,
		const u_int sampleOffset, PathVertexVM *pathVertex, Ray *nextEventRay,
		const bool eyePath) const {
	BiDirCPURenderEngine *engine = (BiDirCPURenderEngine *)renderEngine;

	Vector sampledDir;
	BSDFEvent &event = pathVertex->bsdfEvent;
	float bsdfPdfW, cosSampledDir;
	const Spectrum bsdfSample = pathVertex->bsdf.Sample(&sampledDir,
			sampler->GetSample(sampleOffset),
			sampler->GetSample(sampleOffset + 1),
			&bsdfPdfW, &cosSampledDir, &event);
	if (bsdfSample.Black())
		return false;

	// The path has the colour of its wavelength after a dispersive vertex
	pathVertex->dispersionTinted = pathVertex->dispersionTinted || (event & DISPERSED);

	float bsdfRevPdfW;
	if (event & SPECULAR)
		bsdfRevPdfW = bsdfPdfW;
	else
		pathVertex->bsdf.Pdf(sampledDir, NULL, &bsdfRevPdfW);

	if (pathVertex->depth >= engine->rrDepth) {
		// Russian Roulette
		const float prob = RenderEngine::RussianRouletteProb(bsdfSample, engine->rrImportanceCap);
		if (prob < sampler->GetSample(sampleOffset + 2))
			return false;

		pathVertex->throughput /= prob;
	}

	// Was:
	//  pathVertex->throughput *= bsdfSample * (cosSampledDir / bsdfPdfW);
	//
	// but now BSDF::Sample() follows LuxRender habit to return the
	// result multiplied by cosSampledDir / bsdfPdfW
	pathVertex->throughput *= bsdfSample;
	assert (!pathVertex->throughput.IsNaN() && !pathVertex->throughput.IsInf());

	// The terms of dVC (see PathVertexVM::dVCTerms), as dVC below: each one a
	// step further from the connection, and a new term j = 2 from dVCM (none
	// at a specular vertex, which can't be connected)
	if (eyePath && engine->lightDepthCutsStrategies) {
		const float stepFactor = (event & SPECULAR) ?
			MIS(cosSampledDir) : (MIS(cosSampledDir / bsdfPdfW) * MIS(bsdfRevPdfW));
		for (float &term : pathVertex->dVCTerms)
			term *= stepFactor;
		pathVertex->dVCTerms.push_back((event & SPECULAR) ?
			0.f : (MIS(cosSampledDir / bsdfPdfW) * (pathVertex->dVCM + misVmWeightFactor)));
	}

	// New MIS weights
	if (event & SPECULAR) {
		pathVertex->dVCM = 0.f;
		// Was:
		//  const float factor = MIS(cosSampledDir / bsdfPdfW) * MIS(bsdfRevPdfW);
		//
		// but bsdfPdfW = bsdfRevPdfW for specular material.
		assert (bsdfPdfW == bsdfRevPdfW);
		const float factor = MIS(cosSampledDir);
		pathVertex->dVC *= factor;
		pathVertex->dVM *= factor;
	} else {
		pathVertex->dVC = MIS(cosSampledDir / bsdfPdfW) * (pathVertex->dVC *
				MIS(bsdfRevPdfW) + pathVertex->dVCM + misVmWeightFactor);
		pathVertex->dVM = MIS(cosSampledDir / bsdfPdfW) * (pathVertex->dVM *
				MIS(bsdfRevPdfW) + pathVertex->dVCM * misVcWeightFactor + 1.f);
		pathVertex->dVCM = MIS(1.f / bsdfPdfW);
	}

	// Update volume information
	pathVertex->volInfo.Update(event, pathVertex->bsdf);

	*nextEventRay = Ray(pathVertex->bsdf.GetRayOrigin(sampledDir), sampledDir);
	nextEventRay->UpdateMinMaxWithEpsilon();
	// A ray from a point in a volume starts at the point, see BSDF::GetRayOrigin()
	if (pathVertex->bsdf.IsVolume())
		nextEventRay->mint = 0.f;
	nextEventRay->time = time;

	++pathVertex->depth;

	return true;
}

void BiDirCPURenderThread::RenderFunc(std::stop_token stop_token) {
#ifndef NDEBUG
	SLG_LOG("[BiDirCPURenderThread::" << threadIndex << "] Rendering thread started");
#endif

	//--------------------------------------------------------------------------
	// Initialization
	//--------------------------------------------------------------------------

	// This is really used only by Windows for 64+ threads support
	SetThreadGroupAffinity(threadIndex);

	BiDirCPURenderEngine *engine = (BiDirCPURenderEngine *)renderEngine;
	// (engine->seedBase + 1) seed is used for sharedRndGen

	auto rndGen = std::make_unique<RandomGenerator>(engine->seedBase + 1 + threadIndex);
	auto& scene = engine->renderConfig.GetScene();
	auto& camera = scene.GetCamera();
	PhotonGICache *photonGICache = engine->photonGICache;

	// Albedo and Normal AOV warm up
	if (engine->aovWarmupSPP > 0)
		AOVWarmUp(stop_token, rndGen);

	// Setup the sampler
	auto sampler = engine->renderConfig.AllocSampler(
		rndGen,
		engine->GetFilm(),
		engine->GetSampleSplatter(),
		engine->samplerSharedData,
		Properties()
	);
	const u_int sampleSize =
		sampleBootSize + // To generate the initial light vertex and trace eye ray
		engine->maxLightPathDepth * sampleLightStepSize + // For each light vertex
		engine->maxEyePathDepth * sampleEyeStepSize; // For each eye vertex
	sampler->SetThreadIndex(threadIndex);
	sampler->RequestSamples(PIXEL_NORMALIZED_AND_SCREEN_NORMALIZED, sampleSize);

	VarianceClamping varianceClamping(engine->sqrtVarianceClampMaxValue);

	// Disable vertex merging
	misVmWeightFactor = 0.f;
	misVcWeightFactor = 0.f;

	vector<SampleResult> sampleResults;
	vector<PathVertexVM> lightPathVertices;

	for(u_int steps = 0; !stop_token.stop_requested(); ++steps) {
		// Check if we are in pause mode
		if (engine->pauseMode) {
			// Check every 100ms if I have to continue the rendering
			while (!stop_token.stop_requested() && engine->pauseMode)
				std::this_thread::sleep_for(100ms);

			if (stop_token.stop_requested())
				break;
		}

		sampleResults.clear();
		lightPathVertices.clear();

		const float timeSample = sampler->GetSample(12);
		const float time = scene.GetCamera().GenerateRayTime(timeSample);
		// One wavelength of the dispersion for the eye and the light path
		const float wavelength = PathInfo::Wavelength(sampler->GetSample(13));

		/*
		// Sample a point on the camera lens
		Point lensPoint;
		if (!camera.SampleLens(time, sampler->GetSample(3), sampler->GetSample(4),
				&lensPoint)) {
			assert (SampleResult::IsAllValid(sampleResults));

			sampler->NextSample(sampleResults);
			continue;
		}
		*/

		//----------------------------------------------------------------------
		// Trace light path
		//----------------------------------------------------------------------

		const bool validLightPath = TraceLightPath(time, wavelength, sampler, camera, lightPathVertices, sampleResults);
		assert (SampleResult::IsAllValid(sampleResults));

		if (validLightPath) {
			//------------------------------------------------------------------
			// Trace eye path
			//------------------------------------------------------------------

			PathVertexVM eyeVertex;
			SampleResult &eyeSampleResult = AddResult(sampleResults, false);

			eyeSampleResult.filmX = sampler->GetSample(0);
			eyeSampleResult.filmY = sampler->GetSample(1);
			Ray eyeRay;
			camera.GenerateRay(time,
					eyeSampleResult.filmX, eyeSampleResult.filmY, &eyeRay,
					&eyeVertex.volInfo, sampler->GetSample(10), sampler->GetSample(11));

			// Required by MIS weights update
			eyeVertex.bsdf.hitPoint.fixedDir = -eyeRay.d;
			eyeVertex.throughput = Spectrum(1.f);
			eyeVertex.wavelength = wavelength;
			eyeVertex.dispersionTinted = false;
			float cameraPdfW;
			scene.GetCamera().GetPDF(eyeRay, 0.f, eyeSampleResult.filmX, eyeSampleResult.filmY, &cameraPdfW, nullptr);
			eyeVertex.dVCM = MIS(1.f / cameraPdfW);
			eyeVertex.dVC = 0.f;
			eyeVertex.dVM = 0.f;

			eyeVertex.depth = 1;
			bool albedoToDo = true;
			eyeSampleResult.albedo = Spectrum(); // Just in case albedoToDo is never true
			eyeSampleResult.shadingNormal = Normal();
			bool photonGICausticCacheUsed = false;
			bool isTransmittedEyePath = true;
			// The BSDF Scene::Intersect() initializes at each hit, pass-through
			// surfaces included: eyeVertex.bsdf stays the one of the last
			// scattering vertex until something is hit, for the pdf of an
			// environment light on a miss
			BSDF hitBsdf;
			while (eyeVertex.depth <= engine->maxEyePathDepth) {
				eyeSampleResult.firstPathVertex = (eyeVertex.depth == 1);
				eyeSampleResult.lastPathVertex = (eyeVertex.depth == engine->maxEyePathDepth);

				const u_int sampleOffset = sampleBootSize + engine->maxLightPathDepth * sampleLightStepSize +
					(eyeVertex.depth - 1) * sampleEyeStepSize;

				// NOTE: I account for volume emission only with path tracing (i.e. here and
				// not in any other place)
				RayHit eyeRayHit;
				Spectrum connectionThroughput;
				const bool hit = scene.Intersect(
						luxrays::make_observer(device),
						EYE_RAY | (eyeSampleResult.firstPathVertex ? CAMERA_RAY : INDIRECT_RAY),
						&eyeVertex.volInfo, sampler->GetSample(sampleOffset),
						&eyeRay, &eyeRayHit, &hitBsdf,
						&connectionThroughput, &eyeVertex.throughput, &eyeSampleResult);

				if (!hit) {
					// Nothing was hit, look for infinitelight

					// This is a trick, you can not have a BSDF of something that has
					// not been hit. DirectHitInfiniteLight must be aware of this.
					eyeVertex.bsdf.hitPoint.fixedDir = -eyeRay.d;
					eyeVertex.throughput *= connectionThroughput;

					DirectHitLight(false, eyeVertex, eyeSampleResult);

					if (eyeSampleResult.firstPathVertex) {
						eyeSampleResult.alpha = 0.f;
						eyeSampleResult.depth = numeric_limits<float>::infinity();
						eyeSampleResult.position = Point(
								numeric_limits<float>::infinity(),
								numeric_limits<float>::infinity(),
								numeric_limits<float>::infinity());
						eyeSampleResult.geometryNormal = Normal();
						eyeSampleResult.shadingNormal = Normal();
						eyeSampleResult.materialID = 0;
						eyeSampleResult.objectID = 0;
						eyeSampleResult.uv = UV(numeric_limits<float>::infinity(),
								numeric_limits<float>::infinity());
					} else if (isTransmittedEyePath) {
						// I set to 0.0 also the alpha all purely transmitted paths hitting nothing
						eyeSampleResult.alpha = 0.f;
					}
					break;
				}
				eyeVertex.throughput *= connectionThroughput;

				// Something was hit
				eyeVertex.bsdf = hitBsdf;

				eyeVertex.SetDispersion();

				if (albedoToDo && eyeVertex.bsdf.IsAlbedoEndPoint(engine->albedoSpecularSetting,
						engine->albedoSpecularGlossinessThreshold)) {
					eyeSampleResult.albedo = eyeVertex.throughput * eyeVertex.bsdf.Albedo();
					eyeSampleResult.shadingNormal = eyeVertex.bsdf.hitPoint.shadeN;
					albedoToDo = false;
				}

				float t_MIS;
				if (eyeSampleResult.firstPathVertex) {
					eyeSampleResult.alpha = 1.f;
					eyeSampleResult.depth = eyeRayHit.t;
					eyeSampleResult.position = eyeVertex.bsdf.hitPoint.p;
					eyeSampleResult.geometryNormal = eyeVertex.bsdf.hitPoint.geometryN;
					eyeSampleResult.materialID = eyeVertex.bsdf.GetMaterialID();
					eyeSampleResult.objectID = eyeVertex.bsdf.GetObjectID();
					eyeSampleResult.uv = eyeVertex.bsdf.hitPoint.GetUV(0);
					// for the camera ray, we need to add the clipping distance
					// because eyeRayHit.t is measured from clipping start.
					// Otherwise, the brightness near the front clipping plane may be distorted.
					t_MIS = eyeRayHit.t + camera.clipHither;
				} 
				else{
					t_MIS = eyeRayHit.t;
				}

				// Update MIS constants. The rays of an orthographic camera are
				// parallel: the area pdf of its first vertex has no 1 / t^2
				// (as ConnectToEye())
				const float factor = 1.f / MIS(AbsDot(eyeVertex.bsdf.hitPoint.shadeN, eyeVertex.bsdf.hitPoint.fixedDir));
				const bool orthographicFirstVertex = eyeSampleResult.firstPathVertex &&
						(camera.GetType() == Camera::ORTHOGRAPHIC);
				eyeVertex.dVCM *= (orthographicFirstVertex ? 1.f : MIS(t_MIS * t_MIS)) * factor;
				eyeVertex.dVC *= factor;
				eyeVertex.dVM *= factor;
				for (float &term : eyeVertex.dVCTerms)
					term *= factor;

				// Check if it is a light source
				if (eyeVertex.bsdf.IsLightSource() &&
					// Avoid to render caustic path if PhotonGI caustic cache
					// has been used (for SDS paths)
					!photonGICausticCacheUsed){
					DirectHitLight(true, eyeVertex, eyeSampleResult);
				}

				// A path of path.maxdepth segments ends here: the light
				// sample, the connections and the caustic cache would make it
				// longer
				if (eyeVertex.depth >= engine->maxEyePathDepth)
					break;

				// Note: pass-through check is done inside Scene::Intersect()
		
				//--------------------------------------------------------------
				// Check if I can use the photon cache
				//--------------------------------------------------------------

				if (photonGICache) {
					const bool isPhotonGIEnabled = photonGICache->IsPhotonGIEnabled(eyeVertex.bsdf);

					// Check if the cache is enabled for this material
					if (isPhotonGIEnabled) {
						// TODO: add support for AOVs (possible ?)

						if (photonGICache->IsCausticEnabled() && (eyeVertex.depth > 1)) {
							const SpectrumGroup causticRadiance = photonGICache->ConnectWithCausticPaths(eyeVertex.bsdf);

							if (!causticRadiance.Black())
								eyeSampleResult.radiance.AddWeighted(eyeVertex.throughput, causticRadiance);
							
							photonGICausticCacheUsed = true;
						}
					}
				}

				//--------------------------------------------------------------
				// Direct light sampling
				//--------------------------------------------------------------

				DirectLightSampling(time,
						sampler->GetSample(sampleOffset + 1),
						sampler->GetSample(sampleOffset + 2),
						sampler->GetSample(sampleOffset + 3),
						sampler->GetSample(sampleOffset + 4),
						sampler->GetSample(sampleOffset + 5),
						eyeVertex, eyeSampleResult);

				assert (eyeSampleResult.IsValid());

				//--------------------------------------------------------------
				// Connect vertex path ray with all light path vertices
				//--------------------------------------------------------------

				if (!eyeVertex.bsdf.IsDelta()) {
					// The light path vertices are in increasing depth: the
					// first one that makes the path longer than path.maxdepth
					// ends the loop
					for (vector<PathVertexVM>::const_iterator lightPathVertex = lightPathVertices.begin();
							(lightPathVertex < lightPathVertices.end()) &&
							(lightPathVertex->depth + eyeVertex.depth + 1 <= engine->maxEyePathDepth);
							++lightPathVertex)
						ConnectVertices(time, eyeVertex, *lightPathVertex, eyeSampleResult,
								sampler->GetSample(sampleOffset + 6));

					assert (eyeSampleResult.IsValid());
				}

				//--------------------------------------------------------------
				// Build the next vertex path ray
				//--------------------------------------------------------------

				if (!Bounce(time, sampler, sampleOffset + 7, &eyeVertex, &eyeRay, true))
					break;
				
				isTransmittedEyePath = isTransmittedEyePath && (eyeVertex.bsdfEvent & TRANSMIT);

#ifdef WIN32
				// Work around Windows bad scheduling
                std::this_thread::yield();
#endif
			}
		}
		
		assert (SampleResult::IsAllValid(sampleResults));

		// Variance clamping
		if (varianceClamping.hasClamping()) {
			for(u_int i = 0; i < sampleResults.size(); ++i)
				varianceClamping.Clamp(engine->GetFilm(), sampleResults[i]);

			assert (SampleResult::IsAllValid(sampleResults));
		}

		sampler->NextSample(sampleResults);

		// Check halt conditions
		if (engine->GetFilm().GetConvergence() == 1.f)
			break;

		if (photonGICache) {
			const u_int spp = engine->GetFilm().GetTotalEyeSampleCount() / engine->GetFilm().GetPixelCount();
			photonGICache->Update(threadIndex, spp);
                }
	} // ~for

	threadDone = true;

	// This is done to stop threads pending on barrier wait
	// inside engine->photonGICache->Update(). This can happen when an
	// halt condition is satisfied.
	if (photonGICache)
		photonGICache->FinishUpdate(threadIndex);

#ifndef NDEBUG
	SLG_LOG("[BiDirCPURenderThread::" << threadIndex << "] Rendering thread halted");
#endif
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
