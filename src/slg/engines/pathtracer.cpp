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


#include "luxrays/usings.h"
#include "luxrays/utils/properties.h"
#include "slg/lights/light.h"
#include "slg/usings.h"
#include "slg/engines/pathtracer.h"
#include "slg/engines/caches/photongi/photongicache.h"
#include "slg/samplers/metropolis.h"
#include "slg/utils/varianceclamping.h"
#include <boost/algorithm/string.hpp>
#include <cstring>

#include "slg/cameras/camera.h"

// The code shared with the PathOCL kernels, compiled through the OpenCL
// emulation (see luxrays/devices/cpudevice_oclemul.h)
namespace slg { namespace ocl { namespace {
#include "luxrays/devices/cpudevice_oclemul.h"
#include "slg/engines/pathtracer_funcs.cl"
#include "slg/materials/subsurface_funcs.cl"

float3 ToFloat3(const luxrays::Spectrum &s) {
	return MAKE_FLOAT3(s.c[0], s.c[1], s.c[2]);
}

} } }

using namespace std;
using namespace luxrays;
using namespace slg;

//------------------------------------------------------------------------------
// PathTracer
//------------------------------------------------------------------------------

PathTracerThreadState::PathTracerThreadState(IntersectionDeviceRef dev,
		const SamplerUPtr& eSampler,
		const SamplerUPtr& lSampler,
		SceneConstRef scn, FilmRef flm,
		const VarianceClamping *varClamping,
		const bool useFilmSplat) : device(dev),
		eyeSampler(eSampler), lightSampler(lSampler), scene(scn), film(flm),
		varianceClamping(varClamping) {
	// Initialize Eye SampleResults
	eyeSampleResults.resize(1);
	PathTracer::InitEyeSampleResults(film, eyeSampleResults, useFilmSplat);

	eyeSampleCount = 0.0;
	// Using 1.0 instead of 0.0 to avoid a division by zero
	lightSampleCount = 1.0;
}

PathTracerThreadState::~PathTracerThreadState() {
}

//------------------------------------------------------------------------------
// PathTracer
//------------------------------------------------------------------------------

const Film::FilmChannels PathTracer::eyeSampleResultsChannels({
	Film::RADIANCE_PER_PIXEL_NORMALIZED, Film::ALPHA, Film::DEPTH,
	Film::POSITION, Film::GEOMETRY_NORMAL, Film::SHADING_NORMAL, Film::MATERIAL_ID,
	Film::DIRECT_DIFFUSE, Film::DIRECT_DIFFUSE_REFLECT, Film::DIRECT_DIFFUSE_TRANSMIT,
	Film::DIRECT_GLOSSY, Film::DIRECT_GLOSSY_REFLECT, Film::DIRECT_GLOSSY_TRANSMIT,
	Film::EMISSION,
	Film::INDIRECT_DIFFUSE, Film::INDIRECT_DIFFUSE_REFLECT, Film::INDIRECT_DIFFUSE_TRANSMIT,
	Film::INDIRECT_GLOSSY, Film::INDIRECT_GLOSSY_REFLECT, Film::INDIRECT_GLOSSY_TRANSMIT,
	Film::INDIRECT_SPECULAR, Film::INDIRECT_SPECULAR_REFLECT, Film::INDIRECT_SPECULAR_TRANSMIT,
	Film::DIRECT_SHADOW_MASK, Film::INDIRECT_SHADOW_MASK, Film::UV, Film::RAYCOUNT,
	Film::IRRADIANCE, Film::OBJECT_ID, Film::SAMPLECOUNT, Film::CONVERGENCE,
	Film::MATERIAL_ID_COLOR, Film::ALBEDO, Film::AVG_SHADING_NORMAL, Film::NOISE
});

const Film::FilmChannels PathTracer::lightSampleResultsChannels({
	Film::RADIANCE_PER_SCREEN_NORMALIZED
}); 

PathTracer::PathTracer() : pixelFilterDistribution(nullptr),
		photonGICache(nullptr) {
}

PathTracer::~PathTracer() {
	delete pixelFilterDistribution;
}

void PathTracer::InitPixelFilterDistribution(const FilterUPtr& pixelFilter) {
	// Compile sample distribution
	delete pixelFilterDistribution;
	pixelFilterDistribution = new FilterDistribution(pixelFilter, 64);
}

void  PathTracer::DeletePixelFilterDistribution() {
	delete pixelFilterDistribution;
	pixelFilterDistribution = NULL;
}

void PathTracer::InitEyeSampleResults(FilmConstRef film, vector<SampleResult> &sampleResults,
		const bool useFilmSplat) {
	SampleResult &sampleResult = sampleResults[0];

	sampleResult.Init(&eyeSampleResultsChannels, film.GetRadianceGroupCount());
	sampleResult.useFilmSplat = useFilmSplat;
}

void PathTracer::ResetEyeSampleResults(vector<SampleResult> &sampleResults) {
	SampleResult &sampleResult = sampleResults[0];

	// Set to 0.0 all result colors
	sampleResult.emission = Spectrum();
	for (u_int i = 0; i < sampleResult.radiance.Size(); ++i)
		sampleResult.radiance[i] = Spectrum();
	sampleResult.directDiffuseReflect = Spectrum();
	sampleResult.directDiffuseTransmit = Spectrum();
	sampleResult.directGlossyReflect = Spectrum();
	sampleResult.directGlossyTransmit = Spectrum();
	sampleResult.indirectDiffuseReflect = Spectrum();
	sampleResult.indirectDiffuseTransmit = Spectrum();
	sampleResult.indirectGlossyReflect = Spectrum();
	sampleResult.indirectGlossyTransmit = Spectrum();
	sampleResult.indirectSpecularReflect = Spectrum();
	sampleResult.indirectSpecularTransmit = Spectrum();
	sampleResult.directShadowMask = 1.f;
	sampleResult.indirectShadowMask = 1.f;
	sampleResult.irradiance = Spectrum();
	sampleResult.albedo = Spectrum();
	sampleResult.isHoldout = false;

	sampleResult.rayCount = 0.f;
}

//------------------------------------------------------------------------------
// RenderEyeSample methods
//------------------------------------------------------------------------------

PathTracer::DirectLightResult PathTracer::DirectLightSampling(
		luxrays::IntersectionDeviceRef device, SceneConstRef scene,
		const float time,
		const float u0, const float u1, const float u2,
		const float u3, const float u4,
		const EyePathInfo &pathInfo, const Spectrum &pathThroughput,
		const BSDF &bsdf, SampleResult *sampleResult,
		const bool useBSDFEVal) const {
	if (!bsdf.IsDelta()) {
		// Select the light strategy to use
		auto& lightStrategy =
			bsdf.IsShadowCatcherOnlyInfiniteLights() ?
			scene.GetLightSources().GetInfiniteLightStrategy() :
			scene.GetLightSources().GetIlluminateLightStrategy();

		// Pick a light source to sample
		const Normal landingNormal = bsdf.hitPoint.intoObject ? bsdf.hitPoint.shadeN : -bsdf.hitPoint.shadeN;
		float lightPickPdf;
		auto light = lightStrategy.SampleLights(
			scene,
			u0,
			bsdf.hitPoint.p,
			landingNormal,
			bsdf.IsVolume(),
			&lightPickPdf
		);

		float learnContribution = 0.f;
		if (light) {
			Ray shadowRay;
			float directPdfW;
			Spectrum lightRadiance = light->Illuminate(
				scene, bsdf, time, u1, u2, u3, shadowRay, directPdfW
			);
			verify (!lightRadiance.IsNaN() && !lightRadiance.IsInf());

			if (!lightRadiance.Black()) {
				verify (!isnan(directPdfW) && !isinf(directPdfW));

				BSDFEvent event;
				float bsdfPdfW;
				Spectrum bsdfEval;
				
				if (useBSDFEVal)
					bsdfEval = bsdf.Evaluate(shadowRay.d, &event, &bsdfPdfW);
				else {
					// This is used by BAKECPU and must be aligned with its BSDF sampling
					bsdfEval = Spectrum(Dot(shadowRay.d, bsdf.hitPoint.shadeN) * INV_PI);
					bsdfPdfW = INV_TWOPI;
					event = DIFFUSE | REFLECT;
				}
				verify (!bsdfEval.IsNaN() && !bsdfEval.IsInf());

				if (!bsdfEval.Black() &&
						(!hybridBackForwardEnable ||
						!pathInfo.IsCausticPath(event, bsdf.GetUnregularizedGlossiness(), hybridBackForwardGlossinessThreshold))) {
					verify (!isnan(bsdfPdfW) && !isinf(bsdfPdfW));
					
					// Create a new PathDepthInfo for the path to the light source
					PathDepthInfo directLightDepthInfo = pathInfo.depth;
					directLightDepthInfo.IncDepths(PathDepthInfo::GetDepthEvent(bsdf, event));
					
					RayHit shadowRayHit;
					BSDF shadowBsdf;
					Spectrum connectionThroughput;
					// Create a new PathVolumeInfo for the path to the light source:
					// a light on the other side of the surface is seen through
					// its interior volume, or from inside it (as the path would
					// be after sampling the same direction)
					PathVolumeInfo volInfo = pathInfo.volume;
					if (!bsdf.IsVolume() && (event & TRANSMIT))
						volInfo.Update(event, bsdf);
					// Check if the light source is visible
					if (!scene.Intersect(IntersectionDevicePtr(&device), EYE_RAY | SHADOW_RAY, &volInfo, u4, &shadowRay,
							&shadowRayHit, &shadowBsdf, &connectionThroughput, nullptr,
							nullptr, true)) {
						// Add the light contribution only if it is not a shadow catcher
						// (because, if the light is visible, the material will be
						// transparent in the case of a shadow catcher).

						if (!bsdf.IsShadowCatcher()) {
							// I'm ignoring volume emission because it is not sampled in
							// direct light step.
							const float directLightSamplingPdfW = directPdfW * lightPickPdf;
							const float factor = 1.f / directLightSamplingPdfW;

							if ((directLightDepthInfo.GetRRDepth() >= rrDepth) && (bsdfPdfW > 0.f)) {
								// Russian Roulette: the BSDF technique continues the path
								// with the probability RenderEyePath() computes from the
								// sample weight of the direction, f cos / pdf, and the
								// path throughput (the MIS weight of a BSDF hit on the
								// light uses the same product, see EyePathInfo::lastBSDFPdfW)
								const float etaScale = pathInfo.etaScale *
										(((rrType != slg::ocl::RR_BOUNCE) && (event & TRANSMIT)) ? bsdf.GetRefractionEtaScale() : 1.f);
								bsdfPdfW *= RussianRouletteProb(bsdfEval / bsdfPdfW, pathThroughput, etaScale);
							}

							// Account for material transparency
							bsdfPdfW *= light->GetAvgPassThroughTransparency();

							// MIS between direct light sampling and BSDF sampling
							//
							// Note: I have to avoid MIS on the last path vertex
							const bool misEnabled = !sampleResult->lastPathVertex &&
								(light->IsEnvironmental() || light->IsIntersectable()) &&
								CheckDirectHitVisibilityFlags(*light, directLightDepthInfo, event) &&
								!shadowBsdf.hitPoint.throughShadowTransparency;

							const float weight = misEnabled ? PowerHeuristic(directLightSamplingPdfW, bsdfPdfW) : 1.f;
							const Spectrum incomingRadiance = bsdfEval * (weight * factor) * connectionThroughput * lightRadiance;

							sampleResult->AddDirectLight(light->GetID(), event, pathThroughput, incomingRadiance, 1.f);
							learnContribution = RGBColor(incomingRadiance.c[0], incomingRadiance.c[1], incomingRadiance.c[2]).Y();

							// The first path vertex is not handled by AddDirectLight(). This is valid
							// for irradiance AOV only if it is not a SPECULAR material.
							//
							// Note: irradiance samples the light sources only here (i.e. no
							// direct hit, no MIS, it would be useless)
							//
							// Note: RR is ignored here because it can not happen on first path vertex
							if ((sampleResult->firstPathVertex) && !(bsdf.GetEventTypes() & SPECULAR))
								sampleResult->irradiance =
										(INV_PI * fabsf(Dot(bsdf.hitPoint.shadeN, shadowRay.d)) *
										factor) * connectionThroughput * lightRadiance;
						}

						lightStrategy.AddSample(bsdf.hitPoint.p, landingNormal, bsdf.IsVolume(),
								light->lightSceneIndex, lightPickPdf, learnContribution);
						return ILLUMINATED;
					} else {
						lightStrategy.AddSample(bsdf.hitPoint.p, landingNormal, bsdf.IsVolume(),
								light->lightSceneIndex, lightPickPdf, 0.f);
						return SHADOWED;
					}
				}
			}
		}

		// A sample that added nothing (no light picked, or a black one)
		lightStrategy.AddSample(bsdf.hitPoint.p, landingNormal, bsdf.IsVolume(),
				light ? light->lightSceneIndex : NULL_INDEX, lightPickPdf, 0.f);
	}

	return NOT_VISIBLE;
}

bool PathTracer::CheckDirectHitVisibilityFlags(LightSourceConstRef lightSource, const PathDepthInfo &depthInfo,
		const BSDFEvent lastBSDFEvent) const {
	if (depthInfo.depth == 0)
		return true;

	if ((lastBSDFEvent & DIFFUSE) && lightSource.IsVisibleIndirectDiffuse())
		return true;
	if ((lastBSDFEvent & GLOSSY) && lightSource.IsVisibleIndirectGlossy())
		return true;
	if ((lastBSDFEvent & SPECULAR) && lightSource.IsVisibleIndirectSpecular())
		return true;

	return false;
}

void PathTracer::DirectHitFiniteLight(SceneConstRef scene,
		const EyePathInfo &pathInfo,
		const Spectrum &pathThroughput, const Ray &ray,
		const float distance, const BSDF &bsdf,
		SampleResult *sampleResult) const {

	auto lightSource = bsdf.GetLightSource();

	// Check if the light source is visible according the settings
	if (!CheckDirectHitVisibilityFlags(*lightSource, pathInfo.depth, pathInfo.lastBSDFEvent) ||
			// If the material is shadow transparent, Direct Light sampling
			// will take care of transporting all emitted light
			bsdf.hitPoint.throughShadowTransparency)
		return;

	float directPdfA;
	const Spectrum emittedRadiance = bsdf.GetEmittedRadiance(&directPdfA);

	if (!emittedRadiance.Black()) {
		float weight;
		if (!(pathInfo.lastBSDFEvent & SPECULAR)) {
			auto& lightStrategy = scene.GetLightSources().GetIlluminateLightStrategy();
			const float lightPickProb = lightStrategy.SampleLightPdf(
				*lightSource,
				ray.o, pathInfo.lastShadeN, pathInfo.lastFromVolume);

			// This is a specific check to avoid fireflies with DLSC
			if ((lightPickProb == 0.f) && lightSource->IsDirectLightSamplingEnabled() &&
					(lightStrategy.GetType() == TYPE_DLS_CACHE))
				return;

			// The pdf of the light sampling from the origin of the ray: the
			// last scattering vertex (Scene::Intersect() moves mint, not the
			// origin, past the pass-through surfaces)
			const float directPdfW = bsdf.GetTriangleLightSource()->GetDirectPdfW(bsdf.hitPoint,
					ray.o, distance, AbsDot(bsdf.hitPoint.fixedDir, bsdf.hitPoint.shadeN));

			// MIS between BSDF sampling and direct light sampling
			weight = PowerHeuristic(pathInfo.lastBSDFPdfW * lightSource->GetAvgPassThroughTransparency(), directPdfW * lightPickProb);
		} else
			weight = 1.f;

		sampleResult->AddEmission(bsdf.GetLightID(), pathThroughput, weight * emittedRadiance);
	}
}

void PathTracer::DirectHitInfiniteLight(SceneConstRef scene,
		const EyePathInfo &pathInfo, const Spectrum &pathThroughput,
		const Ray &ray, const BSDF *bsdf, SampleResult *sampleResult) const {
	// If the material is shadow transparent, Direct Light sampling
	// will take care of transporting all emitted light
	if (bsdf && bsdf->hitPoint.throughShadowTransparency)
		return;

	for(EnvLightSource& envLight: scene.GetLightSources().GetEnvLightSources()) {
		// Check if the light source is visible according the settings
		if (!CheckDirectHitVisibilityFlags(envLight, pathInfo.depth, pathInfo.lastBSDFEvent))
			continue;

		float directPdfW;
		const Spectrum envRadiance = envLight.GetRadiance(scene, bsdf, -ray.d, &directPdfW);
		if (!envRadiance.Black()) {
			float weight;
			if (!(pathInfo.lastBSDFEvent & SPECULAR)) {
				const float lightPickProb = scene.GetLightSources().GetIlluminateLightStrategy().
						SampleLightPdf(envLight, ray.o, pathInfo.lastShadeN, pathInfo.lastFromVolume);

				// MIS between BSDF sampling and direct light sampling
				weight = PowerHeuristic(pathInfo.lastBSDFPdfW, directPdfW * lightPickProb);
			} else
				weight = 1.f;

			sampleResult->AddEmission(envLight.GetID(), pathThroughput, weight * envRadiance);
		}
	}	
}

void PathTracer::GenerateEyeRay(CameraConstRef camera, FilmConstRef film, Ray &eyeRay,
		PathVolumeInfo &volInfo, Sampler& sampler, SampleResult &sampleResult) const {
	const float filmX = sampler.GetSample(0);
	const float filmY = sampler.GetSample(1);

	// Use fast pixel filtering, like the one used in TILEPATH.

	const u_int *subRegion = film.GetSubRegion();
	sampleResult.pixelX = Min(Floor2UInt(filmX), subRegion[1]);
	sampleResult.pixelY = Min(Floor2UInt(filmY), subRegion[3]);
	assert (sampleResult.pixelX >= subRegion[0]);
	assert (sampleResult.pixelX <= subRegion[1]);
	assert (sampleResult.pixelY >= subRegion[2]);
	assert (sampleResult.pixelY <= subRegion[3]);

	const float uSubPixelX = filmX - sampleResult.pixelX;
	const float uSubPixelY = filmY - sampleResult.pixelY;

	// Sample according the pixel filter distribution
	float distX, distY;
	pixelFilterDistribution->SampleContinuous(uSubPixelX, uSubPixelY, &distX, &distY);

	sampleResult.filmX = sampleResult.pixelX + .5f + distX;
	sampleResult.filmY = sampleResult.pixelY + .5f + distY;

	const float timeSample = sampler.GetSample(IDX_EYE_TIME(eyeSampleSize));
	const float time = camera.GenerateRayTime(timeSample);

	camera.GenerateRay(time, sampleResult.filmX, sampleResult.filmY, &eyeRay, &volInfo,
		sampler.GetSample(IDX_DOF_X(eyeSampleSize)), sampler.GetSample(IDX_DOF_Y(eyeSampleSize)));
}

//------------------------------------------------------------------------------
// Random-walk subsurface scattering
//------------------------------------------------------------------------------

bool PathTracer::SubsurfaceRandomWalk(IntersectionDeviceRef device,
		SceneConstRef scene, const float u, const Vector &sampledDir,
		const float time, const float passThrough, const PathVolumeInfo &volInfo,
		BSDF &bsdf, Spectrum &pathThroughput) const {
	Spectrum color, radius;
	float anisotropy;
	bsdf.GetSubsurfaceWalkParams(&color, &radius, &anisotropy);

	// The outward normals at the entry: the side of the eye
	const Normal geometryN = (Dot(bsdf.hitPoint.fixedDir, bsdf.hitPoint.geometryN) > 0.f) ?
		bsdf.hitPoint.geometryN : -bsdf.hitPoint.geometryN;
	if (Dot(sampledDir, geometryN) >= 0.f)
		return false;
	const Normal shadeN = (Dot(bsdf.hitPoint.shadeN, geometryN) > 0.f) ?
		bsdf.hitPoint.shadeN : -bsdf.hitPoint.shadeN;

	const Point entryP = bsdf.GetRayOrigin(sampledDir);
	ocl::SSSWalk walk;
	if (!ocl::SSSWalk_Init(&walk, ocl::make_float3(entryP.x, entryP.y, entryP.z),
			ocl::make_float3(shadeN.x, shadeN.y, shadeN.z),
			ocl::make_float3(sampledDir.x, sampledDir.y, sampledDir.z),
			ocl::ToFloat3(color), ocl::ToFloat3(radius), anisotropy))
		return false;

	// Decorrelated from the other users of the same sample
	uint32_t seed;
	memcpy(&seed, &u, sizeof(seed));
	TauswortheRandomGenerator rng(seed ^ 0xdeadbeefu);

	const SceneObject *object = &bsdf.GetSceneObject();
	for (;;) {
		const float u0 = rng.floatValue();
		const float u1 = rng.floatValue();
		const float u2 = rng.floatValue();
		const float u3 = rng.floatValue();
		const float u4 = rng.floatValue();
		const float u5 = rng.floatValue();
		ocl::SSSWalk_PrepareStep(&walk, u0, u1, u2, u3, u4, u5);

		// The first hit on the walked object up to tmax: the walk continues
		// through the other objects
		const Point o(walk.P.x, walk.P.y, walk.P.z);
		Ray ray(o, Vector(walk.D.x, walk.D.y, walk.D.z), 0.f, walk.tmax, time);
		ray.mint = (walk.bounce == 0) ? MachineEpsilon::E(o) : 0.f;
		RayHit rayHit;
		bool hitObject = false;
		while (device.TraceRay(&ray, &rayHit)) {
			if (&scene.GetObjects().GetSceneObject(rayHit.meshIndex) == object) {
				hitObject = true;
				break;
			}
			ray.mint = rayHit.t + MachineEpsilon::E(ray(rayHit.t));
			if (ray.mint >= ray.maxt)
				break;
		}

		const int result = ocl::SSSWalk_FinishStep(&walk, hitObject, rayHit.t);
		if (result == SSSWALK_END)
			return false;
		if (result == SSSWALK_EXIT) {
			const Spectrum weight(walk.throughput.x, walk.throughput.y, walk.throughput.z);
			if (weight.Black() || weight.IsNaN() || weight.IsInf())
				return false;
			pathThroughput *= weight;

			// The exit vertex
			bsdf.Init(false, false, scene, ray, rayHit, passThrough, &volInfo);
			bsdf.hitPoint.subsurface = HITPOINT_SSS_EXIT;

			return true;
		}
	}
}

//------------------------------------------------------------------------------
// RenderEyePath methods
//------------------------------------------------------------------------------

void PathTracer::RenderEyePath(IntersectionDeviceRef device,
		SceneConstRef scene, Sampler& sampler, EyePathInfo &pathInfo,
		Ray &eyeRay,  const luxrays::Spectrum &eyeTroughput,
		vector<SampleResult> &sampleResults) const {
	// To keep track of the number of rays traced
	const double deviceRayCount = device.GetTotalRaysCount();

	// This is used by light strategy
	pathInfo.lastShadeN = Normal(eyeRay.d);

	SampleResult &sampleResult = sampleResults[0];
	bool photonGIShowIndirectPathMixUsed = false;
	bool photonGICausticCacheUsed = false;
	bool photonGICacheEnabledOnLastHit = false;
	bool albedoToDo = true;
	sampleResult.albedo = Spectrum(); // Just in case albedoToDo is never true
	sampleResult.shadingNormal = Normal();
	Spectrum pathThroughput(eyeTroughput);
	BSDF bsdf;
	// The vertex is the exit point of a subsurface random walk: bsdf is
	// already initialized and there is no ray to trace
	bool subsurfaceExit = false;
	for (;;) {
		sampleResult.firstPathVertex = (pathInfo.depth.depth == 0);
		const u_int sampleOffset = eyeSampleBootSize + pathInfo.depth.depth * eyeSampleStepSize;

		RayHit eyeRayHit;
		const float passThrough = sampler.GetSample(sampleOffset + IDX_PASSTHROUGH);
		bool hit;
		if (subsurfaceExit) {
			hit = true;
			eyeRayHit.t = 0.f;
		} else {
			Spectrum connectionThroughput;
			hit = scene.Intersect(
					IntersectionDevicePtr(&device),
					EYE_RAY | (sampleResult.firstPathVertex ? CAMERA_RAY : INDIRECT_RAY),
					&pathInfo.volume, passThrough,
					&eyeRay, &eyeRayHit, &bsdf, &connectionThroughput,
					&pathThroughput, &sampleResult);
			pathThroughput *= connectionThroughput;
			// Note: pass-through check is done inside Scene::Intersect()
		}

		const bool checkDirectLightHit =
				// Avoid to render caustic path if hybridBackForwardEnable
				(!hybridBackForwardEnable || !pathInfo.IsCausticPath()) &&
				// Avoid to render caustic path if PhotonGI caustic cache is enabled
				(!photonGICache ||
					photonGICache->IsDirectLightHitVisible(pathInfo, photonGICausticCacheUsed));

		if (!hit) {
			// Nothing was hit, look for env. lights
			if ((!(forceBlackBackground && pathInfo.isPassThroughPath) || !pathInfo.isPassThroughPath) &&
					checkDirectLightHit) {
				DirectHitInfiniteLight(scene, pathInfo, pathThroughput,
						eyeRay, sampleResult.firstPathVertex ? nullptr : &bsdf,
						&sampleResult);
			}

			if (sampleResult.firstPathVertex) {
				sampleResult.alpha = 0.f;
				sampleResult.depth = numeric_limits<float>::infinity();
				sampleResult.position = Point(
						numeric_limits<float>::infinity(),
						numeric_limits<float>::infinity(),
						numeric_limits<float>::infinity());
				sampleResult.geometryNormal = Normal();
				sampleResult.shadingNormal = Normal();
				sampleResult.materialID = 0;
				sampleResult.objectID = 0;
				sampleResult.uv = UV(numeric_limits<float>::infinity(),
						numeric_limits<float>::infinity());
			} else if (!sampleResult.isHoldout && pathInfo.isTransmittedPath) {
				// I set to 0.0 also the alpha all purely transmitted paths hitting nothing
				sampleResult.alpha = 0.f;
			}
			break;
		}

		// Something was hit

		if (albedoToDo && bsdf.IsAlbedoEndPoint(albedoSpecularSetting, albedoSpecularGlossinessThreshold)) {
			sampleResult.albedo = pathThroughput * bsdf.Albedo();
			sampleResult.shadingNormal = bsdf.hitPoint.shadeN;
			albedoToDo = false;
		}

		if (sampleResult.firstPathVertex) {
			// The alpha value can be changed if the material is a shadow catcher (see below)
			sampleResult.alpha = bsdf.IsHoldout() ? 0.f : 1.f;
			sampleResult.depth = eyeRayHit.t;
			sampleResult.position = bsdf.hitPoint.p;
			sampleResult.geometryNormal = bsdf.hitPoint.geometryN;
			sampleResult.materialID = bsdf.GetMaterialID();
			sampleResult.objectID = bsdf.GetObjectID();
			sampleResult.uv = bsdf.hitPoint.GetUV(0);
			sampleResult.isHoldout = bsdf.IsHoldout();
		}
		// Path regularization: after the first non-specular bounce the GGX
		// lobes of the vertex get a roughness floor and its delta lobes become
		// rough (see HitPoint::regularization)
		bsdf.hitPoint.regularization = (regularizationEnable && pathInfo.anyNonSpecularBounce) ?
			regularizationRoughness : 0.f;
		// The base lobe of a subsurface material enters the object for the
		// walk while the subsurface depth lasts. The walk is then the base's
		// diffuse lobe (unless it has another one), out of the diffuse depth.
		// Beyond the subsurface depth, and on the last vertex (no next vertex
		// for the walk), the base is the Lambertian stand-in.
		const bool walkEntry = !subsurfaceExit && bsdf.HasSubsurfaceWalk() &&
				pathInfo.depth.CanStartSubsurfaceWalk(maxPathDepth);
		BSDFEvent eventTypes = bsdf.GetEventTypes();
		if (walkEntry && !bsdf.HasDiffuseBesideSubsurfaceWalk())
			eventTypes &= ~DIFFUSE;

		sampleResult.lastPathVertex = pathInfo.depth.IsLastPathVertex(maxPathDepth,
				PathDepthInfo::GetDepthEvent(bsdf, eventTypes));
		if (walkEntry && !sampleResult.lastPathVertex)
			bsdf.hitPoint.subsurface = HITPOINT_SSS_ENTRY;

		//----------------------------------------------------------------------
		// Check if it is a baked material
		//----------------------------------------------------------------------

		if (bsdf.HasBakeMap(COMBINED)) {
			sampleResult.radiance[0] += pathThroughput * bsdf.GetBakeMapValue();
			break;
		} else if (bsdf.HasBakeMap(LIGHTMAP)) {
			sampleResult.radiance[0] += pathThroughput * bsdf.Albedo() * bsdf.GetBakeMapValue();
			break;
		}

		//----------------------------------------------------------------------
		// Check if it is a light source and I have to add light emission
		//----------------------------------------------------------------------

		if (!subsurfaceExit && bsdf.IsLightSource() && checkDirectLightHit) {
			DirectHitFiniteLight(scene, pathInfo, pathThroughput,
					eyeRay, eyeRayHit.t, bsdf, &sampleResult);
		}

		//----------------------------------------------------------------------
		// Check if I can use the photon cache
		//----------------------------------------------------------------------

		if (photonGICache && !subsurfaceExit) {
			// The cache doesn't have the random walk of the subsurface materials
			const bool isPhotonGIEnabled = photonGICache->IsPhotonGIEnabled(bsdf) &&
					(bsdf.hitPoint.subsurface != HITPOINT_SSS_ENTRY);

			// Check if one of the debug modes is enabled
			if (photonGICache->GetDebugType() == PhotonGIDebugType::PGIC_DEBUG_SHOWINDIRECT) {
				if (isPhotonGIEnabled) {
					const SpectrumGroup *group = photonGICache->GetIndirectRadiance(bsdf);
					if (group)
						sampleResult.radiance += *group;
				}
				break;
			} else if (photonGICache->GetDebugType() == PhotonGIDebugType::PGIC_DEBUG_SHOWCAUSTIC) {
				if (isPhotonGIEnabled)
					sampleResult.radiance += photonGICache->ConnectWithCausticPaths(bsdf);
				break;
			} else if (photonGICache->GetDebugType() == PhotonGIDebugType::PGIC_DEBUG_SHOWINDIRECTPATHMIX) {
				// Check if the cache is enabled for this material
				if (isPhotonGIEnabled) {
					if (photonGICacheEnabledOnLastHit &&
							(eyeRayHit.t > photonGICache->GetIndirectUsageThreshold(pathInfo.lastBSDFEvent,
								pathInfo.lastGlossiness,
								passThrough))) {
						sampleResult.radiance[0] = Spectrum(0.f, 0.f, 1.f);
						photonGIShowIndirectPathMixUsed = true;
						break;
					}

					photonGICacheEnabledOnLastHit = true;
				} else
					photonGICacheEnabledOnLastHit = false;
			} else {
				// Check if the cache is enabled for this material
				if (isPhotonGIEnabled) {
					// TODO: add support for AOVs (possible ?)

					if (photonGICache->IsCausticEnabled() && (!hybridBackForwardEnable || pathInfo.depth.depth != 0)) {
						const SpectrumGroup causticRadiance = photonGICache->ConnectWithCausticPaths(bsdf);

						if (!causticRadiance.Black()) {
							sampleResult.radiance.AddWeighted(pathThroughput, causticRadiance);
							photonGICausticCacheUsed = true;
						}
					}

					if (photonGICache->IsIndirectEnabled() && photonGICacheEnabledOnLastHit &&
							(eyeRayHit.t > photonGICache->GetIndirectUsageThreshold(pathInfo.lastBSDFEvent,
								pathInfo.lastGlossiness,
								// I hope to not introduce strange sample correlations
								// by using passThrough here
								passThrough))) {
						const SpectrumGroup *group = photonGICache->GetIndirectRadiance(bsdf);
						if (group)
							sampleResult.radiance.AddWeighted(pathThroughput, *group);
						// I can terminate the path, all done
						break;
					}

					photonGICacheEnabledOnLastHit = true;
				} else
					photonGICacheEnabledOnLastHit = false;
			}
		}

		//------------------------------------------------------------------
		// Direct light sampling
		//------------------------------------------------------------------

		// I avoid to do DL on the last vertex otherwise it introduces a lot of
		// noise because I can not use MIS.
		// I handle as a special case when the path vertex is both the first
		// and the last: I do direct light sampling without MIS. The exit of a
		// subsurface walk too: the walk's light comes from there (the entry
		// has no light sampling of the base), so the depths it reaches don't
		// make the material black.
		if (sampleResult.lastPathVertex && !sampleResult.firstPathVertex && !subsurfaceExit)
			break;

		const DirectLightResult directLightResult = DirectLightSampling(
				device, scene,
				eyeRay.time,
				sampler.GetSample(sampleOffset + IDX_DIRECTLIGHT_X),
				sampler.GetSample(sampleOffset + IDX_DIRECTLIGHT_Y),
				sampler.GetSample(sampleOffset + IDX_DIRECTLIGHT_Z),
				sampler.GetSample(sampleOffset + IDX_DIRECTLIGHT_W),
				sampler.GetSample(sampleOffset + IDX_DIRECTLIGHT_A),
				pathInfo, 
				pathThroughput, bsdf, &sampleResult);

		if (sampleResult.lastPathVertex)
			break;

		//------------------------------------------------------------------
		// Build the next vertex path ray
		//------------------------------------------------------------------

		Vector sampledDir;
		float cosSampledDir;
		Spectrum bsdfSample;
		float bsdfPdfW;
		BSDFEvent bsdfEvent;
		if (bsdf.IsShadowCatcher() && (directLightResult != SHADOWED)) {
			bsdfSample = bsdf.ShadowCatcherSample(&sampledDir, &bsdfPdfW, &cosSampledDir, &bsdfEvent);

			if (sampleResult.firstPathVertex) {
				// In this case I have also to set the value of the alpha channel to 0.0
				sampleResult.alpha = 0.f;
			}
		} else {
			const Spectrum &shadowTransparency = bsdf.GetPassThroughShadowTransparency();
			if (!sampleResult.firstPathVertex && !shadowTransparency.Black() && !pathInfo.IsSpecularPath()) {
				sampledDir = -bsdf.hitPoint.fixedDir;
				bsdfSample = shadowTransparency;
				bsdfPdfW = pathInfo.lastBSDFPdfW;
				cosSampledDir = -1.f;
				bsdfEvent = pathInfo.lastBSDFEvent;
			} else {
				bsdfSample = bsdf.Sample(&sampledDir,
						sampler.GetSample(sampleOffset + IDX_BSDF_X),
						sampler.GetSample(sampleOffset + IDX_BSDF_Y),
						&bsdfPdfW, &cosSampledDir, &bsdfEvent);
				pathInfo.isPassThroughPath = false;
			}
		}

		verify (!bsdfSample.IsNaN() && !bsdfSample.IsInf() && !bsdfSample.IsNeg());
		if (bsdfSample.Black())
			break;

		if (sampleResult.firstPathVertex)
			sampleResult.firstPathVertexEvent = bsdfEvent;

		pathInfo.AddVertex(bsdf, bsdfEvent, bsdfPdfW, hybridBackForwardGlossinessThreshold,
				rrType != slg::ocl::RR_BOUNCE, regularizationRoughness);

		// Russian Roulette
		float rrProb = 1.f;
		if (pathInfo.UseRR(rrDepth)) {
			rrProb = RussianRouletteProb(bsdfSample, pathThroughput, pathInfo.etaScale);
			if (rrProb < sampler.GetSample(sampleOffset + IDX_RR))
				break;

			// Increase path contribution
			bsdfSample /= rrProb;

			// The pdf of the BSDF technique, for the MIS weight of a hit on
			// a light, includes the Russian Roulette
			pathInfo.lastBSDFPdfW *= rrProb;
		}

		pathThroughput *= bsdfSample;
		verify (!pathThroughput.IsNaN() && !pathThroughput.IsInf());

		// This is valid for irradiance AOV only if it is not a SPECULAR material and
		// first path vertex. Set or update sampleResult.irradiancePathThroughput
		if (sampleResult.firstPathVertex) {
			if (!(bsdf.GetEventTypes() & SPECULAR))
				sampleResult.irradiancePathThroughput = INV_PI * AbsDot(bsdf.hitPoint.shadeN, sampledDir) / rrProb;
			else
				sampleResult.irradiancePathThroughput = Spectrum();
		} else
			sampleResult.irradiancePathThroughput *= bsdfSample;

		if (bsdfEvent & SUBSURFACE) {
			// The random walk to the exit point, the next vertex. Its random
			// numbers come from the pass-through sample of that vertex, which
			// has no ray to trace.
			const u_int exitSampleOffset = eyeSampleBootSize + pathInfo.depth.depth * eyeSampleStepSize;
			if (!SubsurfaceRandomWalk(device, scene,
					sampler.GetSample(exitSampleOffset + IDX_PASSTHROUGH), sampledDir,
					eyeRay.time, sampler.GetSample(exitSampleOffset + IDX_PASSTHROUGH),
					pathInfo.volume, bsdf, pathThroughput))
				break;

			subsurfaceExit = true;
			continue;
		}
		subsurfaceExit = false;

		eyeRay.Update(bsdf.GetRayOrigin(sampledDir), sampledDir);
	}

	sampleResult.rayCount += static_cast<float>(device.GetTotalRaysCount() - deviceRayCount);

	if (sampleResult.isHoldout) {
		sampleResult.radiance.Clear();
		sampleResult.albedo = Spectrum();
	}

	if (photonGICache && (photonGICache->GetDebugType() == PhotonGIDebugType::PGIC_DEBUG_SHOWINDIRECTPATHMIX) &&
			!photonGIShowIndirectPathMixUsed)
		sampleResult.radiance[0] = Spectrum(1.f, 0.f, 0.f);
}

//------------------------------------------------------------------------------
// RenderEyeSample
//------------------------------------------------------------------------------

void PathTracer::RenderEyeSample(
	IntersectionDeviceRef device,
	SceneConstRef scene, FilmConstRef film,
	Sampler& sampler,
	vector<SampleResult> &sampleResults
) const {
	ResetEyeSampleResults(sampleResults);

	EyePathInfo pathInfo;
	Ray eyeRay;
	GenerateEyeRay(scene.GetCamera(), film, eyeRay, pathInfo.volume, sampler, sampleResults[0]);

	RenderEyePath(device, scene, sampler, pathInfo, eyeRay, Spectrum(1.f), sampleResults);
}

//------------------------------------------------------------------------------
// RenderLightSample methods
//------------------------------------------------------------------------------

SampleResult &PathTracer::AddLightSampleResult(vector<SampleResult> &sampleResults,
		FilmConstRef film) {
	const u_int size = sampleResults.size();
	sampleResults.resize(size + 1);

	SampleResult &sampleResult = sampleResults[size];
	sampleResult.Init(&lightSampleResultsChannels, film.GetRadianceGroupCount());

	return sampleResult;
}

void PathTracer::ConnectToEye(IntersectionDeviceRef device,
		SceneConstRef scene,
		FilmConstRef film, const float time,
		const float u0, const float u1, const float u2,
		const LightSource &light, const BSDF &bsdf, 
		const Spectrum &flux, const LightPathInfo &pathInfo,
		vector<SampleResult> &sampleResults) const {
	// I don't connect camera invisible objects with the eye
	if (bsdf.IsCameraInvisible() || bsdf.IsDelta())
		return;

	float filmX, filmY;
	bool sampleSuccess;
	Ray eyeRay;

	Vector eyeDir;
	float eyeDistance = 0;
	Point lensPoint = pathInfo.lensPoint;
    if (scene.GetCamera().GetType() == Camera::ORTHOGRAPHIC){
		// Orthographic camera need to be handled separately,
		// lensPoint can not be pre-calculated in this case
		Point p = bsdf.hitPoint.p;
		eyeDir = scene.GetCamera().GetDir();
		// calculate distance from vertex to camera plane
		const float D = -eyeDir.x*lensPoint.x - eyeDir.y*lensPoint.y - eyeDir.z*lensPoint.z;
		eyeDistance = eyeDir.x*p.x + eyeDir.y*p.y + eyeDir.z*p.z + D;
		eyeDistance = fabsf(eyeDistance);

		eyeRay = Ray(bsdf.hitPoint.p, eyeDir,
			0.f,
			eyeDistance,
			time);
		sampleSuccess = scene.GetCamera().ProjectToImage(&eyeRay, &filmX, &filmY);
	} else {
		eyeDir = Vector(bsdf.hitPoint.p - lensPoint);
		eyeDistance = eyeDir.Length();
		eyeDir /= eyeDistance;

		eyeRay = Ray(lensPoint, eyeDir,
			0.f,
			eyeDistance,
			time);
		sampleSuccess = scene.GetCamera().GetSamplePosition(&eyeRay, &filmX, &filmY);
	}

	if (sampleSuccess) {
		BSDFEvent event;
		const Spectrum bsdfEval = bsdf.Evaluate(-eyeDir, &event);

		if (!bsdfEval.Black()) {
			// I have to flip the direction of the traced ray because
			// the information inside PathVolumeInfo are about the path from
			// the light toward the camera (i.e. ray.o would be in the wrong
			// place).
			Ray traceRay(bsdf.GetRayOrigin(-eyeRay.d), -eyeRay.d,
					eyeDistance - eyeRay.maxt,
					eyeDistance - eyeRay.mint,
					time);
			traceRay.UpdateMinMaxWithEpsilon();
			RayHit traceRayHit;

			BSDF bsdfConn;
			Spectrum connectionThroughput;
			// Create a new PathVolumeInfo for the path to the camera: the
			// camera on the other side of the surface is seen through its
			// interior volume, or from inside it
			PathVolumeInfo volInfo = pathInfo.volume;
			if (!bsdf.IsVolume() && (event & TRANSMIT))
				volInfo.Update(event, bsdf);
			if (!scene.Intersect(
					luxrays::make_observer<IntersectionDevice>(device),
					LIGHT_RAY | CAMERA_RAY,
					&volInfo, u0, &traceRay, &traceRayHit, &bsdfConn,
					&connectionThroughput)) {
				// Nothing was hit, the light path vertex is visible

				float fluxToRadianceFactor;
				scene.GetCamera().GetPDF(eyeRay, eyeDistance, filmX, filmY, nullptr, &fluxToRadianceFactor);

				SampleResult &sampleResult = AddLightSampleResult(sampleResults, film);
				sampleResult.filmX = filmX;
				sampleResult.filmY = filmY;

				sampleResult.pixelX = Floor2UInt(filmX);
				sampleResult.pixelY = Floor2UInt(filmY);

#if !defined(NDEBUG)
				const u_int *subRegion = film.GetSubRegion();
#endif
				assert (sampleResult.pixelX >= subRegion[0]);
				assert (sampleResult.pixelX <= subRegion[1]);
				assert (sampleResult.pixelY >= subRegion[2]);
				assert (sampleResult.pixelY <= subRegion[3]);

				sampleResult.isCaustic = pathInfo.IsCausticPath(event, bsdf.GetGlossiness(), hybridBackForwardGlossinessThreshold);

				// Add radiance from the light source
				sampleResult.radiance[light.GetID()] = connectionThroughput * flux * fluxToRadianceFactor * bsdfEval;
			}
		}
	}
}

//------------------------------------------------------------------------------
// RenderLightSample
//------------------------------------------------------------------------------

void PathTracer::RenderLightSample(IntersectionDeviceRef device,
		SceneConstRef scene, FilmConstRef film,
		Sampler& sampler, vector<SampleResult> &sampleResults,
		const ConnectToEyeCallBackType &ConnectToEyeCallBack) const {
	sampleResults.clear();

	Spectrum lightPathFlux;

	const float timeSample = sampler.GetSample(8);
	const float time = scene.GetCamera().GenerateRayTime(timeSample);

	// Select one light source
	float lightPickPdf;
	auto light = scene.GetLightSources().GetEmitLightStrategy().
			SampleLights(scene, sampler.GetSample(0), &lightPickPdf);

	if (light) {
		// Initialize the light path
		Ray nextEventRay;
		float lightEmitPdfW;
		lightPathFlux = light->Emit(scene,
				time, sampler.GetSample(1), sampler.GetSample(2),
				sampler.GetSample(3), sampler.GetSample(4), sampler.GetSample(5),
				nextEventRay, lightEmitPdfW);

		if (lightPathFlux.Black())
			return;

		lightPathFlux /= lightEmitPdfW * lightPickPdf;
		assert (!lightPathFlux.IsNaN() && !lightPathFlux.IsInf());

		LightPathInfo pathInfo;

		/*
		// Sample a point on the camera lens
		if (!scene.GetCamera().SampleLens(time, sampler.GetSample(6), sampler.GetSample(7),
				&pathInfo.lensPoint))
			return;
		*/

		//----------------------------------------------------------------------
		// Trace the light path
		//----------------------------------------------------------------------

		while (pathInfo.depth.depth < maxPathDepth.depth) {
			const u_int sampleOffset = lightSampleBootSize +  pathInfo.depth.depth * lightSampleStepSize;

			RayHit nextEventRayHit;
			BSDF bsdf;
			Spectrum connectionThroughput;
			const bool hit = scene.Intersect(
				luxrays::make_observer(device),
				LIGHT_RAY | INDIRECT_RAY,
				&pathInfo.volume, sampler.GetSample(sampleOffset),
				&nextEventRay, &nextEventRayHit, &bsdf,
				&connectionThroughput
			);
			if (!hit) {
				// Ray lost in space...
				break;
			}

			// Check if it is something with a not black shadow transparency
			// and stop if it has. Direct light sampling will take care of
			// this kind of paths.
			if (!bsdf.GetPassThroughShadowTransparency().Black() & !bsdf.GetPassThroughShadowTransparencyOverride())
				break;

			// Something was hit

			lightPathFlux *= connectionThroughput;

			//--------------------------------------------------------------
			// Try to connect the light path vertex with the eye
			//--------------------------------------------------------------

			scene.GetCamera().SampleLens(time, sampler.GetSample(6), sampler.GetSample(7),
				&pathInfo.lensPoint);

			if (ConnectToEyeCallBack){
				ConnectToEyeCallBack(pathInfo, bsdf, light->GetID(), lightPathFlux, sampleResults);
			} else {
				ConnectToEye(device, scene, film,
						nextEventRay.time,
						sampler.GetSample(sampleOffset + 1),
						sampler.GetSample(sampleOffset + 2),
						sampler.GetSample(sampleOffset + 3),
						*light, bsdf, lightPathFlux, pathInfo, sampleResults);
			}

			if (pathInfo.depth.depth == maxPathDepth.depth - 1)
				break;

			//--------------------------------------------------------------
			// Build the next vertex path ray
			//--------------------------------------------------------------

			float bsdfPdf = 0.f;
			Vector sampledDir;
			BSDFEvent bsdfEvent;
			float cosSampleDir;
			Spectrum bsdfSample = bsdf.Sample(&sampledDir,
						sampler.GetSample(sampleOffset + 4),
						sampler.GetSample(sampleOffset + 5),
					&bsdfPdf, &cosSampleDir, &bsdfEvent);
			if (bsdfSample.Black())
				break;

			pathInfo.AddVertex(bsdf, bsdfEvent, hybridBackForwardGlossinessThreshold);

			// If it isn't anymore a (nearly) specular path, I can stop
			if (hybridBackForwardEnable && !pathInfo.IsSpecularPath() &&
					// This condition is added to "stabilize" Metropolis sampler
					// used in light tracing part of hybrid rendering. In this case
					// I render also some not-caustic sample to make an "easy"
					// computation of the avg. image luminance.
					(pathInfo.depth.GetRRDepth() > 1))
				break;

			// Russian Roulette
			if (pathInfo.UseRR(rrDepth)) {
				// Russian Roulette
				const float rrProb = RenderEngine::RussianRouletteProb(bsdfSample, rrImportanceCap);
				if (rrProb < sampler.GetSample(sampleOffset + 6))
					break;

				// Increase path contribution
				bsdfSample /= rrProb;
			}

			lightPathFlux *= bsdfSample;
			verify (!lightPathFlux.IsNaN() && !lightPathFlux.IsInf());

			nextEventRay.Update(bsdf.GetRayOrigin(sampledDir), sampledDir);
		}
	}
}

//------------------------------------------------------------------------------
// RenderSample
//------------------------------------------------------------------------------

bool PathTracer::HasToRenderEyeSample(PathTracerThreadState &state) const {
	// Check if I have to trace an eye or light path
	if (hybridBackForwardEnable) {
		const double ratio = state.eyeSampleCount / state.lightSampleCount;
		if ((hybridBackForwardPartition == 1.f) ||
				(ratio < hybridBackForwardPartition)) {
			// Trace an eye path
			state.eyeSampleCount += 1.0;
			return true;
		} else {
			// Trace a light path
			state.lightSampleCount += 1.0;
			return false;
		}
	} else {
		state.eyeSampleCount += 1.0;
		return true;
	}
}

void PathTracer::ApplyVarianceClamp(const PathTracerThreadState &state,
		vector<SampleResult> &sampleResults) const {
	// Variance clamping
	if (state.varianceClamping->hasClamping()) {
		for(u_int i = 0; i < sampleResults.size(); ++i) {
			SampleResult &sampleResult = sampleResults[i];

			// I clamp only eye paths samples (variance clamping would cut
			// SDS path values due to high scale of PSR samples)
			if (sampleResult.HasChannel(Film::RADIANCE_PER_PIXEL_NORMALIZED))
				state.varianceClamping->Clamp(state.GetFilm(), sampleResult);
		}
	}
}

void PathTracer::RenderSample(PathTracerThreadState &state) const {

	auto Render = [&]() {

		// Check if I have to trace an eye or light path
		if (HasToRenderEyeSample(state)) {
			// Trace an eye path
			auto& sampler = state.GetEyeSampler();
			auto& sampleResults = state.GetEyeSampleResults();
			RenderEyeSample(
				state.device,
				state.scene,
				state.GetFilm(),
				sampler,
				sampleResults
			);
			return std::make_tuple(std::ref(sampler), std::ref(sampleResults));
		} else {
			// Otherwise trace a light path
			auto& sampler = state.GetLightSampler();
			auto& sampleResults = state.GetLightSampleResults();
			RenderLightSample(
				state.device,
				state.scene,
				state.GetFilm(),
				sampler,
				sampleResults
			);
			return std::make_tuple(std::ref(sampler), std::ref(sampleResults));
		}
	};  // lambda

	// Render sample
	auto [sampler, sampleResults] = Render();
	assert(&sampleResults == &state.GetEyeSampleResults() || &sampleResults == &state.GetLightSampleResults());

	// Apply variance clamping
	ApplyVarianceClamp(state, sampleResults);

	sampler.NextSample(sampleResults);
}

//------------------------------------------------------------------------------
// ParseOptions
//------------------------------------------------------------------------------

// The path depths, as Arnold and Cycles: total, diffuse, glossy (rough and
// sharp reflections), transmission and subsurface. Older settings:
//  - path.pathdepth.specular (the sharp reflections, and the refractions
//    before path.pathdepth.transmission): the glossy depth is at least it,
//    and it is the transmission depth when that isn't set;
//  - path.maxdepth alone: every depth.
PathDepthInfo PathTracer::GetMaxPathDepth(luxrays::PropertiesConstRef cfg,
		const luxrays::Properties &defaultProps) {
	PathDepthInfo maxPathDepth;

	if (cfg.IsDefined("path.maxdepth") &&
			!cfg.IsDefined("path.pathdepth.total") &&
			!cfg.IsDefined("path.pathdepth.diffuse") &&
			!cfg.IsDefined("path.pathdepth.glossy") &&
			!cfg.IsDefined("path.pathdepth.specular")) {
		const u_int maxDepth = Max(0, cfg.Get("path.maxdepth").Get<int>());
		maxPathDepth.depth = maxDepth;
		maxPathDepth.diffuseDepth = maxDepth;
		maxPathDepth.glossyDepth = maxDepth;
		maxPathDepth.transmissionDepth = maxDepth;
		maxPathDepth.subsurfaceDepth = maxDepth;
		return maxPathDepth;
	}

	maxPathDepth.depth = Max(0, cfg.Get(defaultProps.Get("path.pathdepth.total")).Get<int>());
	maxPathDepth.diffuseDepth = Max(0, cfg.Get(defaultProps.Get("path.pathdepth.diffuse")).Get<int>());
	maxPathDepth.glossyDepth = Max(0, cfg.Get(defaultProps.Get("path.pathdepth.glossy")).Get<int>());
	maxPathDepth.transmissionDepth = Max(0, cfg.Get(defaultProps.Get("path.pathdepth.transmission")).Get<int>());
	if (cfg.IsDefined("path.pathdepth.specular")) {
		const u_int specularDepth = Max(0, cfg.Get("path.pathdepth.specular").Get<int>());
		maxPathDepth.glossyDepth = Max(maxPathDepth.glossyDepth, specularDepth);
		if (!cfg.IsDefined("path.pathdepth.transmission"))
			maxPathDepth.transmissionDepth = specularDepth;
	}
	// The walks counted in the diffuse depth before
	maxPathDepth.subsurfaceDepth = cfg.IsDefined("path.pathdepth.subsurface") ?
		Max(0, cfg.Get("path.pathdepth.subsurface").Get<int>()) : maxPathDepth.diffuseDepth;

	return maxPathDepth;
}

// The resolved depths: the older path.maxdepth and path.pathdepth.specular are
// not written back
Properties PathTracer::MaxPathDepthToProperties(luxrays::PropertiesConstRef cfg,
		const luxrays::Properties &defaultProps) {
	const PathDepthInfo maxPathDepth = GetMaxPathDepth(cfg, defaultProps);

	Properties props;
	props <<
			Property("path.pathdepth.total")(maxPathDepth.depth) <<
			Property("path.pathdepth.diffuse")(maxPathDepth.diffuseDepth) <<
			Property("path.pathdepth.glossy")(maxPathDepth.glossyDepth) <<
			Property("path.pathdepth.transmission")(maxPathDepth.transmissionDepth);
	// No default value: the diffuse depth when it isn't set
	if (cfg.IsDefined("path.pathdepth.subsurface"))
		props << Property("path.pathdepth.subsurface")(maxPathDepth.subsurfaceDepth);

	return props;
}

void PathTracer::ParseOptions(
	luxrays::PropertiesConstRef cfg,
	const luxrays::Properties &defaultProps
) {
	// Path depth settings
	maxPathDepth = GetMaxPathDepth(cfg, defaultProps);

	// Russian Roulette settings
	rrType = String2RussianRouletteType(cfg.Get(defaultProps.Get("path.russianroulette.type")).Get<string>());
	rrDepth = (u_int)Max(1, cfg.Get(defaultProps.Get("path.russianroulette.depth")).Get<int>());
	rrImportanceCap = Clamp(cfg.Get(defaultProps.Get("path.russianroulette.cap")).Get<double>(), 0.0, 1.0);

	// Clamping settings
	// clamping.radiance.maxvalue is the old radiance clamping, now converted in variance clamping
	sqrtVarianceClampMaxValue = cfg.Get(Property("path.clamping.radiance.maxvalue")(0.0)).Get<double>();
	if (cfg.IsDefined("path.clamping.variance.maxvalue"))
		sqrtVarianceClampMaxValue = cfg.Get(defaultProps.Get("path.clamping.variance.maxvalue")).Get<double>();
	sqrtVarianceClampMaxValue = Max(0.f, sqrtVarianceClampMaxValue);

	forceBlackBackground = cfg.Get(defaultProps.Get("path.forceblackbackground.enable")).Get<bool>();
	
	hybridBackForwardEnable = cfg.Get(defaultProps.Get("path.hybridbackforward.enable")).Get<bool>();
	// hybridBackForwardGlossinessThreshold is used by LIGHTCPU when PSR is enabled
	// so I have always to set the value
	hybridBackForwardGlossinessThreshold = .05f;
	if (hybridBackForwardEnable) {
		hybridBackForwardPartition = Clamp(cfg.Get(defaultProps.Get("path.hybridbackforward.partition")).Get<double>(), 0.0, 1.0);
		hybridBackForwardGlossinessThreshold = Clamp(cfg.Get(defaultProps.Get("path.hybridbackforward.glossinessthreshold")).Get<double>(), 0.0, 1.0);
	}

	// Path regularization settings. Only the eye paths are regularized: with
	// the light tracing on, the light paths render the caustics seen through a
	// diffuse vertex unblurred, and the eye paths, which drop them (classified
	// without the regularization), regularize the rest.
	regularizationEnable = cfg.Get(defaultProps.Get("path.regularization.enable")).Get<bool>();
	regularizationRoughness = Clamp(cfg.Get(defaultProps.Get("path.regularization.roughness")).Get<float>(), 0.f, 1.f);
	if (!(regularizationRoughness > 0.f))
		regularizationEnable = false;

	// Albedo AOV settings
	albedoSpecularSetting = String2AlbedoSpecularSetting(cfg.Get(defaultProps.Get("path.albedospecular.type")).Get<string>());
	albedoSpecularGlossinessThreshold = Max(cfg.Get(defaultProps.Get("path.albedospecular.glossinessthreshold")).Get<double>(), 0.0);

	// Update eye sample size (the layout is in sampler_types.cl)
	eyeSampleBootSize = EYE_SAMPLE_BOOT_SIZE;
	eyeSampleStepSize = VERTEX_SAMPLE_SIZE;
	eyeSampleSize = 
		eyeSampleBootSize + // The pixel position
		(maxPathDepth.depth + 1) * eyeSampleStepSize + // For each path vertex
		EYE_SAMPLE_TAIL_SIZE; // The camera time and lens position
	
	// Update light sample size
	lightSampleBootSize = 9;
	lightSampleStepSize = 7;
	lightSampleSize = 
		lightSampleBootSize + // To generate eye ray
		maxPathDepth.depth * lightSampleStepSize; // For each path vertex
}

//------------------------------------------------------------------------------
// Static methods used by RenderEngineRegistry
//------------------------------------------------------------------------------

PropertiesUPtr PathTracer::ToProperties(const Properties &cfg) {
	auto props_ptr = std::make_unique<Properties>();
	auto& props = *props_ptr;

	props << MaxPathDepthToProperties(cfg, *GetDefaultProps());

	props <<
			cfg.Get(GetDefaultProps()->Get("path.hybridbackforward.enable")) <<
			cfg.Get(GetDefaultProps()->Get("path.hybridbackforward.partition")) <<
			cfg.Get(GetDefaultProps()->Get("path.hybridbackforward.glossinessthreshold")) <<
			cfg.Get(GetDefaultProps()->Get("path.russianroulette.type")) <<
			cfg.Get(GetDefaultProps()->Get("path.russianroulette.depth")) <<
			cfg.Get(GetDefaultProps()->Get("path.russianroulette.cap")) <<
			cfg.Get(GetDefaultProps()->Get("path.regularization.enable")) <<
			cfg.Get(GetDefaultProps()->Get("path.regularization.roughness")) <<
			cfg.Get(GetDefaultProps()->Get("path.clamping.variance.maxvalue")) <<
			cfg.Get(GetDefaultProps()->Get("path.forceblackbackground.enable")) <<
			cfg.Get(GetDefaultProps()->Get("path.albedospecular.type")) <<
			cfg.Get(GetDefaultProps()->Get("path.albedospecular.glossinessthreshold")) <<
			*Sampler::ToProperties(cfg);

	return props_ptr;
}

PropertiesUPtr PathTracer::GetDefaultProps() {
	auto props = std::make_unique<Properties>();
	*props <<
			Property("path.hybridbackforward.enable")(false) <<
			Property("path.hybridbackforward.partition")(0.8) <<
			Property("path.hybridbackforward.glossinessthreshold")(.05f) <<
			Property("path.pathdepth.total")(6) <<
			Property("path.pathdepth.diffuse")(4) <<
			Property("path.pathdepth.glossy")(4) <<
			Property("path.pathdepth.transmission")(6) <<
			Property("path.russianroulette.type")("BOUNCE") <<
			Property("path.russianroulette.depth")(3) <<
			Property("path.russianroulette.cap")(.5f) <<
			Property("path.regularization.enable")(false) <<
			Property("path.regularization.roughness")(.1f) <<
			Property("path.clamping.variance.maxvalue")(0.f) <<
			Property("path.forceblackbackground.enable")(false) <<
			Property("path.albedospecular.type")("REFLECT_TRANSMIT") <<
			Property("path.albedospecular.glossinessthreshold")(.05f);

	return props;
}

//------------------------------------------------------------------------------
// Russian roulette
//------------------------------------------------------------------------------

slg::ocl::RussianRouletteType PathTracer::String2RussianRouletteType(const string &type) {
	const string t = boost::to_upper_copy(type);
	if (t == "BOUNCE")
		return slg::ocl::RR_BOUNCE;
	else if (t == "PATH")
		return slg::ocl::RR_PATH;
	else if (t == "PATH_SQRT")
		return slg::ocl::RR_PATH_SQRT;
	else
		throw runtime_error("Unknown Russian roulette type: " + type);
}

string PathTracer::RussianRouletteType2String(const slg::ocl::RussianRouletteType type) {
	switch (type) {
		case slg::ocl::RR_BOUNCE:
			return "BOUNCE";
		case slg::ocl::RR_PATH:
			return "PATH";
		case slg::ocl::RR_PATH_SQRT:
			return "PATH_SQRT";
		default:
			throw runtime_error("Unknown Russian roulette type: " + to_string((int)type));
	}
}

float PathTracer::RussianRouletteProb(const Spectrum &bounceWeight,
		const Spectrum &pathThroughput, const float etaScale) const {
	return slg::ocl::PathTracer_RussianRouletteProb(rrType, rrImportanceCap,
			slg::ocl::ToFloat3(bounceWeight), slg::ocl::ToFloat3(pathThroughput), etaScale);
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
