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

#include "slg/bsdf/bsdf.h"
#include "slg/scene/scene.h"
#include "slg/lights/sunlight.h"
#include "slg/lights/data/sunspect.h"

using namespace std;
using namespace luxrays;
using namespace slg;

//------------------------------------------------------------------------------
// SunLight
//------------------------------------------------------------------------------

SunLight::SunLight() : color(1.f), localSunDir(0.f, 0.f, 1.f), turbidity(2.2f), relSize(1.f) {
}

void SunLight::Preprocess() {
	EnvLightSource::Preprocess();

	absoluteSunDir = Normalize(lightToWorld * localSunDir);
	CoordinateSystem(absoluteSunDir, &x, &y);

	// Values from NASA Solar System Exploration page
	// http://solarsystem.nasa.gov/planets/profile.cfm?Object=Sun&Display=Facts&System=Metric
	const float sunRadius = 695500.f;
	const float sunMeanDistance = 149600000.f;

	const float sunSize = relSize * sunRadius;
	if (sunSize <= sunMeanDistance) {
		sin2ThetaMax = sunSize / sunMeanDistance;
		sin2ThetaMax *= sin2ThetaMax;
		cosThetaMax = sqrtf(1.f - sin2ThetaMax);
	} else {
		cosThetaMax = 0.f;
		sin2ThetaMax = 1.f;
	}
	
	Vector wh = Normalize(absoluteSunDir);
	absoluteTheta = SphericalTheta(wh);
	absolutePhi = SphericalPhi(wh);

	// NOTE - lordcrc - sun_k_oWavelengths contains 64 elements, while sun_k_oAmplitudes contains 65?!?
	IrregularSPD k_oCurve(sun_k_oWavelengths, sun_k_oAmplitudes, 64);
	IrregularSPD k_gCurve(sun_k_gWavelengths, sun_k_gAmplitudes, 4);
	IrregularSPD k_waCurve(sun_k_waWavelengths, sun_k_waAmplitudes, 13);

	RegularSPD solCurve(sun_solAmplitudes, 380, 750, 38);  // every 5 nm

	float beta = 0.04608365822050f * turbidity - 0.04586025928522f;
	float tauR, tauA, tauO, tauG, tauWA;

	float m = 1.f / (cosf(absoluteTheta) + 0.00094f * powf(1.6386f - absoluteTheta, 
		-1.253f));  // Relative Optical Mass

	int i;
	float lambda;
	// NOTE - lordcrc - SPD stores data internally, no need for Ldata to stick around
	float Ldata[91];
	for(i = 0, lambda = 350.f; i < 91; ++i, lambda += 5.f) {
			// Rayleigh Scattering
		tauR = expf(-m * 0.008735f * powf(lambda / 1000.f, -4.08f));
			// Aerosol (water + dust) attenuation
			// beta - amount of aerosols present
			// alpha - ratio of small to large particle sizes. (0:4,usually 1.3)
		const float alpha = 1.3f;
		tauA = expf(-m * beta * powf(lambda / 1000.f, -alpha));  // lambda should be in um
			// Attenuation due to ozone absorption
			// lOzone - amount of ozone in cm(NTP)
		const float lOzone = .35f;
		tauO = expf(-m * k_oCurve.Sample(lambda) * lOzone);
			// Attenuation due to mixed gases absorption
		tauG = expf(-1.41f * k_gCurve.Sample(lambda) * m / powf(1.f +
			118.93f * k_gCurve.Sample(lambda) * m, 0.45f));
			// Attenuation due to water vapor absorption
			// w - precipitable water vapor in centimeters (standard = 2)
		const float w = 2.0f;
		tauWA = expf(-0.2385f * k_waCurve.Sample(lambda) * w * m /
		powf(1.f + 20.07f * k_waCurve.Sample(lambda) * w * m, 0.45f));

		Ldata[i] = (solCurve.Sample(lambda) * tauR * tauA * tauO * tauG * tauWA);
	}

	RegularSPD LSPD(Ldata, 350, 800, 91);
	color = ((temperatureScale * gain) / (relSize * relSize)) *
			workingColorSpace.FromRec709(ColorSystem::DefaultColorSystem.ToRGBConstrained(LSPD.ToXYZ()).Clamp());
}

void SunLight::GetPreprocessedData(float *absoluteSunDirData,
		float *xData, float *yData,
		float *absoluteThetaData, float *absolutePhiData,
		float *VData, float *cosThetaMaxData, float *sin2ThetaMaxData) const {
	if (absoluteSunDirData) {
		absoluteSunDirData[0] = absoluteSunDir.x;
		absoluteSunDirData[1] = absoluteSunDir.y;
		absoluteSunDirData[2] = absoluteSunDir.z;
	}

	if (xData) {
		xData[0] = x.x;
		xData[1] = x.y;
		xData[2] = x.z;
	}

	if (yData) {
		yData[0] = y.x;
		yData[1] = y.y;
		yData[2] = y.z;
	}

	if (absoluteThetaData)
		*absoluteThetaData = absoluteTheta;

	if (absolutePhiData)
		*absolutePhiData = absolutePhi;

	if (VData)
		*VData = V;

	if (cosThetaMaxData)
		*cosThetaMaxData = cosThetaMax;

	if (sin2ThetaMaxData)
		*sin2ThetaMaxData = sin2ThetaMax;
}

// The cone of the sun is about 180 float ulps wide around cos(thetaMax) =
// 0.99999 at relsize 1: 1 - cosThetaMax cancels (0.4% too large at relsize 1,
// 1.5% at 0.5), so the cone is sampled, and its pdf computed, from
// 1 - cos(thetaMax) = sin^2(thetaMax) / (1 + cos(thetaMax))
static float SunConeOneMinusCos(const float cosThetaMax, const float sin2ThetaMax) {
	return sin2ThetaMax / (1.f + cosThetaMax);
}

static float SunConePdf(const float oneMinusCosThetaMax) {
	return 1.f / (2.f * M_PI * oneMinusCosThetaMax);
}

static Vector SunConeSample(const float u0, const float u1, const float oneMinusCosThetaMax,
		const Vector &x, const Vector &y, const Vector &z) {
	const float u0x = oneMinusCosThetaMax * u0;
	const float cosTheta = 1.f - u0x;
	const float sinTheta = sqrtf(Max(0.f, u0x * (2.f - u0x)));
	const float phi = u1 * 2.f * M_PI;

	return cosf(phi) * sinTheta * x + sinf(phi) * sinTheta * y + cosTheta * z;
}

float SunLight::GetPower(SceneConstRef scene) const {
	const float envRadius = GetEnvRadius(scene);

	return color.Y() * (M_PI * envRadius * envRadius) * 2.f * M_PI * sin2ThetaMax;
}

Spectrum SunLight::Emit(SceneConstRef scene,
		const float time, const float u0, const float u1,
		const float u2, const float u3, const float passThroughEvent,
		Ray &ray, float &emissionPdfW,
		float *directPdfA, float *cosThetaAtLight) const {
	const Point worldCenter = scene.GetDataSet().GetBSphere().center;
	const float envRadius = GetEnvRadius(scene);

	// Set ray origin and direction for infinite light ray
	float d1, d2;
	ConcentricSampleDisk(u0, u1, &d1, &d2);
	const Point rayOrig = worldCenter + envRadius * (absoluteSunDir + d1 * x + d2 * y);
	const float oneMinusCosThetaMax = SunConeOneMinusCos(cosThetaMax, sin2ThetaMax);
	if (oneMinusCosThetaMax == 0.f)
		return Spectrum();
	const Vector rayDir = -SunConeSample(u2, u3, oneMinusCosThetaMax, x, y, absoluteSunDir);

	const float uniformConePdf = SunConePdf(oneMinusCosThetaMax);
	emissionPdfW = uniformConePdf / (M_PI * envRadius * envRadius);

	if (directPdfA)
		*directPdfA = uniformConePdf;

	if (cosThetaAtLight)
		*cosThetaAtLight = Dot(absoluteSunDir, -rayDir);

	ray.Update(rayOrig, rayDir, time);

	return color;
}

Spectrum SunLight::Illuminate(SceneConstRef scene, const BSDF &bsdf,
		const float time, const float u0, const float u1, const float passThroughEvent,
        Ray &shadowRay, float &directPdfW,
		float *emissionPdfW, float *cosThetaAtLight) const {
	const float oneMinusCosThetaMax = SunConeOneMinusCos(cosThetaMax, sin2ThetaMax);
	if (oneMinusCosThetaMax == 0.f)
		return Spectrum();
	const Vector shadowRayDir = SunConeSample(u0, u1, oneMinusCosThetaMax, x, y, absoluteSunDir);

	// The direction is in the cone by construction: testing it again against
	// cosThetaMax dropped the samples the float rounding put on the edge
	// (0.3% of them at relsize 1, a device dependent loss)
	const float cosAtLight = Dot(absoluteSunDir, shadowRayDir);

	const Point worldCenter = scene.GetDataSet().GetBSphere().center;
	const float envRadius = GetEnvRadius(scene);

	const Point shadowRayOrig = bsdf.GetRayOrigin(shadowRayDir);
	const Vector toCenter(worldCenter - shadowRayOrig);
	const float centerDistanceSquared = Dot(toCenter, toCenter);
	// Outside the sphere of the environment: see ConstantInfiniteLight::Illuminate()
	if (centerDistanceSquared > envRadius * envRadius)
		return Spectrum();
	const float approach = Dot(toCenter, shadowRayDir);
	const float shadowRayDistance = approach + sqrtf(Max(0.f, envRadius * envRadius -
		centerDistanceSquared + approach * approach));

	const float uniformConePdf = SunConePdf(oneMinusCosThetaMax);
	directPdfW = uniformConePdf;

	if (cosThetaAtLight)
		*cosThetaAtLight = cosAtLight;

	if (emissionPdfW)
		*emissionPdfW =  uniformConePdf / (M_PI * envRadius * envRadius);

	shadowRay = Ray(shadowRayOrig, shadowRayDir, 0.f, shadowRayDistance, time);

	return color;
}

Spectrum SunLight::GetRadiance(SceneConstRef scene,
		const BSDF *bsdf, const Vector &dir,
		float *directPdfA, float *emissionPdfW) const {
	const float xD = Dot(-dir, x);
	const float yD = Dot(-dir, y);
	const float zD = Dot(-dir, absoluteSunDir);
	if ((cosThetaMax == 1.f) || (zD < 0.f) || ((xD * xD + yD * yD) > sin2ThetaMax))
		return Spectrum();

	const float uniformConePdf = SunConePdf(SunConeOneMinusCos(cosThetaMax, sin2ThetaMax));
	if (directPdfA)
		*directPdfA = uniformConePdf;

	if (emissionPdfW) {
		const float envRadius = GetEnvRadius(scene);
		*emissionPdfW = uniformConePdf / (M_PI * envRadius * envRadius);
	}

	return color;
}

PropertiesUPtr SunLight::ToProperties(const ImageMapCache &imgMapCache, const bool useRealFileName) const {
	const string prefix = "scene.lights." + GetName();
	PropertiesUPtr props = EnvLightSource::ToProperties(imgMapCache, useRealFileName);

	props->Set(Property(prefix + ".type")("sun"));
	props->Set(Property(prefix + ".dir")(localSunDir));
	props->Set(Property(prefix + ".turbidity")(turbidity));
	props->Set(Property(prefix + ".relsize")(relSize));

	return props;
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
