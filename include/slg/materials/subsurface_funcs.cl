#line 2 "subsurface_funcs.cl"

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
// Random-walk subsurface scattering, the walk of Blender Cycles
// (kernel/integrator/subsurface_random_walk.h, Copyright 2011-2022 Blender
// Foundation, Apache-2.0):
//
//  - Chiang, Kutz & Burley 2016, "Practical and Controllable Subsurface
//    Scattering for Production Path Tracing": a pure scattering random walk
//    in the medium below the surface, the color channel chosen per step
//    with MIS (balance heuristic) over the channels;
//  - the van de Hulst mapping of the color (the multiple-scattering albedo)
//    to the single-scattering albedo with anisotropy (d'Eon, "A Hitchhiker's
//    Guide to Multiple Scattering", Eq. 53.7), the radius is the mean free
//    path per channel;
//  - Dwivedi guiding (Krivanek & d'Eon 2014, Meng, Hanika & Dachsbacher 2016,
//    d'Eon & Krivanek 2020): the direction is sampled toward the entry surface
//    (or away from it, toward an opposite interface found by the first step)
//    and the distance stretched to match, combined with the classic
//    Henyey-Greenstein sampling by MIS;
//  - the similarity relation after the 9th bounce (Wrenninge, Villemin & Hery):
//    isotropic scattering with the reduced coefficients;
//  - the dielectric boundary of relative IOR eta: a walk that reaches the
//    surface is reflected back into the medium with the Fresnel reflectance
//    of its direction (SSSWalk_InternalReflection()) and leaves otherwise.
//
// The walk is split in two pure functions per step so that the CPU (a loop
// with a ray cast between them) and the GPU (one wavefront iteration per step,
// the ray cast by the trace kernel) run the same code:
//  SSSWalk_PrepareStep(): channel, direction and distance of the next step, the
//                         ray to trace;
//  SSSWalk_FinishStep():  the throughput update from the trace result; the
//                         walk continues, leaves the medium (the ray hit the
//                         surface of the same object) or ends.
// The random numbers are inputs, so the caller decides where they come from.
//------------------------------------------------------------------------------

// The walks of a dense, nearly white medium are heavy-tailed (the walks
// longer than n bounces go as 1 / sqrt(n)), so a hard limit loses energy:
// 13% of a white sphere of radius 100 mean free paths at 256 bounces, 2% at
// 4096. Past SSS_RR_BOUNCES bounces the walk goes on with the probability
// b / (b + 1) at the bounce b (Russian roulette, times the throughput when it
// is below 1): it reaches n bounces with the probability SSS_RR_BOUNCES / n,
// so a long walk costs some SSS_RR_BOUNCES more bounces on average and its
// weight grows only linearly. The energy lost is then that of the walks
// longer than SSS_MAX_BOUNCES (tilepathoclthread.cpp has its value).
#define SSS_RR_BOUNCES 256
#define SSS_MAX_BOUNCES 65536
#define SSS_SIMILARITY_LEVEL 9
#define SSS_THROUGHPUT_EPSILON 1e-6f

typedef struct {
	// The float3 first, the scalars after (the size of the GPU task buffer,
	// see GPUTaskSSSWalk)

	// The current ray: origin and direction (the length is tmax)
	float3 P, D;
	// The entry point and the outward normal there (the guiding direction)
	float3 entryP, N;

	float3 throughput;
	// The medium (bounce <= SSS_SIMILARITY_LEVEL) and its similar isotropic
	// version (after)
	float3 sigmaT, sigmaS, sigmaTStar, sigmaSStar;

	// The state of the current step, kept for SSSWalk_FinishStep()
	float3 channelPdf;

	float tmax;
	float anisotropy, diffusionLength, phaseLog, guidedFractionOrg;
	float t;
	float guidedFraction, backwardFraction;
	float forwardPdfFactor, backwardPdfFactor, forwardStretching, backwardStretching;
	float oppositeDistance;
	// The IOR of the medium over the exterior (1: no boundary reflection)
	float eta;
	// The step starts on the surface, after an internal reflection: no
	// scattering at its origin, and the ray skips the surface
	int bounce, haveOpposite, fromSurface;
} SSSWalk;

#if defined(SLG_OPENCL_KERNEL)
// The walk of a PathOCL task between two iterations (MK_RT_SSS_WALK): the
// host allocates GPUTASK_SSSWALK_SIZE bytes per task
typedef struct {
	SSSWalk walk;
	Seed seed;
	// The walked object (RayHit::meshIndex)
	unsigned int meshIndex;
} GPUTaskSSSWalk;

typedef char GPUTaskSSSWalkSizeCheck[(sizeof(GPUTaskSSSWalk) <= GPUTASK_SSSWALK_SIZE) ? 1 : -1];
#endif

//------------------------------------------------------------------------------
// Coefficients
//------------------------------------------------------------------------------

// The van de Hulst mapping: the single-scattering albedo of the color (the
// multiple-scattering albedo) with the anisotropy g
OPENCL_FORCE_INLINE float SSS_SingleScatteringAlbedo(const float color, const float g) {
	const float c = clamp(color, 0.f, 1.f);
	const float s = 4.20863f * c - sqrt(9.59217f + 41.6808f * c + 17.7126f * c * c) + 4.09712f;
	const float sSq = s * s;
	const float denom = 1.f - g * sSq;
	const float alpha = (denom != 0.f) ? (1.f - sSq) / denom : 0.f;

	return clamp(alpha, 0.f, 0.999999f);
}

// Eq. 67 of d'Eon & Krivanek 2020
OPENCL_FORCE_INLINE float SSS_DiffusionLength(const float alpha) {
	return 1.f / sqrt(1.f - pow(alpha, 2.44294f - .0215813f * alpha + .578637f / alpha));
}

OPENCL_FORCE_INLINE float SSS_EvalPhaseDwivedi(const float v, const float phaseLog, const float cosTheta) {
	return 1.f / ((v - cosTheta) * phaseLog);
}

OPENCL_FORCE_INLINE float SSS_SamplePhaseDwivedi(const float v, const float phaseLog, const float u) {
	return v - (v + 1.f) * exp(-u * phaseLog);
}

OPENCL_FORCE_INLINE float SSS_PhaseHG(const float cosTheta, const float g) {
	const float g2 = g * g;
	const float d = 1.f + g2 - 2.f * g * cosTheta;

	return (1.f - g2) / (4.f * M_PI_F * d * sqrt(d));
}

// Two unit vectors orthogonal to the unit vector v
OPENCL_FORCE_INLINE void SSS_Orthonormals(const float3 v, float3 *t, float3 *b) {
	if (fabs(v.x) > fabs(v.y)) {
		const float invLen = 1.f / sqrt(v.x * v.x + v.z * v.z);
		*t = MAKE_FLOAT3(-v.z * invLen, 0.f, v.x * invLen);
	} else {
		const float invLen = 1.f / sqrt(v.y * v.y + v.z * v.z);
		*t = MAKE_FLOAT3(0.f, v.z * invLen, -v.y * invLen);
	}
	*b = MAKE_FLOAT3(v.y * t->z - v.z * t->y, v.z * t->x - v.x * t->z, v.x * t->y - v.y * t->x);
}

OPENCL_FORCE_INLINE float3 SSS_DirectionFromCosine(const float3 axis, const float cosTheta, const float u) {
	const float sinTheta = sqrt(fmax(0.f, 1.f - cosTheta * cosTheta));
	const float phi = 2.f * M_PI_F * u;
	float3 t, b;
	SSS_Orthonormals(axis, &t, &b);

	return sinTheta * cos(phi) * t + sinTheta * sin(phi) * b + cosTheta * axis;
}

// Henyey-Greenstein around the propagation direction D (g > 0 is forward)
OPENCL_FORCE_INLINE float3 SSS_SamplePhaseHG(const float3 D, const float g,
		const float u0, const float u1, float *pdf) {
	float cosTheta;
	if (fabs(g) < 1e-3f)
		cosTheta = 1.f - 2.f * u0;
	else {
		const float k = (1.f - g * g) / (1.f - g + 2.f * g * u0);
		cosTheta = clamp((1.f + g * g - k * k) / (2.f * g), -1.f, 1.f);
	}
	*pdf = SSS_PhaseHG(cosTheta, g);

	return SSS_DirectionFromCosine(D, cosTheta, u1);
}

OPENCL_FORCE_INLINE float3 SSS_Exp(const float3 v) {
	return MAKE_FLOAT3(exp(v.x), exp(v.y), exp(v.z));
}

OPENCL_FORCE_INLINE float SSS_Channel(const float3 v, const int c) {
	return (c == 0) ? v.x : ((c == 1) ? v.y : v.z);
}

//------------------------------------------------------------------------------
// The walk
//------------------------------------------------------------------------------

// Starts a walk at the entry point P (outward normal N) into the direction D,
// with the color (multiple-scattering albedo), the radius (mean free path per
// channel), the anisotropy and the IOR of the medium over the exterior. False
// when the medium can't be walked.
OPENCL_FORCE_INLINE bool SSSWalk_Init(SSSWalk *w, const float3 P, const float3 N, const float3 D,
		const float3 color, const float3 radius, const float anisotropy, const float eta) {
	const float g = clamp(anisotropy, -.9f, .9f);
	float3 alpha = MAKE_FLOAT3(SSS_SingleScatteringAlbedo(color.x, g),
			SSS_SingleScatteringAlbedo(color.y, g), SSS_SingleScatteringAlbedo(color.z, g));
	w->sigmaT = MAKE_FLOAT3(1.f / fmax(radius.x, 1e-16f), 1.f / fmax(radius.y, 1e-16f), 1.f / fmax(radius.z, 1e-16f));

	// With low albedos the diffusion length goes to 1 and the guided phase to
	// infinity: raise alpha and lower the throughput to match (as Cycles)
	w->throughput = MAKE_FLOAT3(1.f, 1.f, 1.f);
	const float minAlpha = .2f;
	if (alpha.x < minAlpha) { w->throughput.x *= alpha.x / minAlpha; alpha.x = minAlpha; }
	if (alpha.y < minAlpha) { w->throughput.y *= alpha.y / minAlpha; alpha.y = minAlpha; }
	if (alpha.z < minAlpha) { w->throughput.z *= alpha.z / minAlpha; alpha.z = minAlpha; }
	w->sigmaS = w->sigmaT * alpha;

	// The highest albedo of the three is the safe one for the guiding
	w->diffusionLength = SSS_DiffusionLength(fmax(alpha.x, fmax(alpha.y, alpha.z)));
	if (!(w->diffusionLength > 1.f) || !(w->diffusionLength < 1e30f))
		return false;
	w->phaseLog = log((w->diffusionLength + 1.f) / (w->diffusionLength - 1.f));

	w->anisotropy = g;
	w->guidedFractionOrg = 1.f - fmax(.5f, pow(fabs(g), .125f));
	w->sigmaSStar = w->sigmaS * (1.f - g);
	w->sigmaTStar = w->sigmaT - w->sigmaS + w->sigmaSStar;

	w->P = P;
	w->D = D;
	w->entryP = P;
	w->N = N;
	w->bounce = 0;
	w->haveOpposite = 0;
	w->oppositeDistance = 0.f;
	w->eta = eta;
	w->fromSurface = 0;

	return true;
}

// Prepares the step: the ray (w->P, w->D) to trace up to w->tmax. The random
// numbers: the channel, the distance, the guiding strategy, the guiding
// direction (forward or backward) and the two of the scattering direction.
OPENCL_FORCE_INLINE void SSSWalk_PrepareStep(SSSWalk *w, const float uChannel, const float uDistance,
		const float uGuide, const float uGuideDir, const float uScatter0, const float uScatter1) {
	// The medium of this bounce (the similarity relation after a few bounces)
	const bool similar = (w->bounce > SSS_SIMILARITY_LEVEL);
	const float g = similar ? 0.f : w->anisotropy;
	const float3 sigmaT = similar ? w->sigmaTStar : w->sigmaT;
	// After an internal reflection the direction is the reflected one: only
	// the distance is sampled, without guiding
	w->guidedFraction = w->fromSurface ? 0.f : (similar ? .75f : w->guidedFractionOrg);

	// The color channel, in proportion to throughput * single-scattering albedo
	// (the albedo of the medium, also after the similarity relation, as Cycles)
	const float3 alpha = w->sigmaS / w->sigmaT;
	const float3 weights = MAKE_FLOAT3(fabs(w->throughput.x * alpha.x), fabs(w->throughput.y * alpha.y),
			fabs(w->throughput.z * alpha.z));
	const float sumWeights = weights.x + weights.y + weights.z;
	w->channelPdf = (sumWeights > 0.f) ? weights / sumWeights : MAKE_FLOAT3(1.f / 3.f, 1.f / 3.f, 1.f / 3.f);
	const int channel = (uChannel < w->channelPdf.x) ? 0 : ((uChannel < w->channelPdf.x + w->channelPdf.y) ? 1 : 2);
	float sampleSigmaT = SSS_Channel(sigmaT, channel);

	w->backwardFraction = 0.f;
	w->forwardPdfFactor = 0.f;
	w->forwardStretching = 1.f;
	w->backwardPdfFactor = 0.f;
	w->backwardStretching = 1.f;

	// The first step goes into the entry direction, the step after an
	// internal reflection into the reflected one: classic distance sampling
	if ((w->bounce > 0) && !w->fromSurface) {
		const bool guided = (uGuide < w->guidedFraction);

		// Guiding away from the entry surface, toward the opposite interface
		// found by the first step, more likely the closer it is
		bool guideBackward = false;
		if (w->haveOpposite) {
			const float3 offset = w->P - w->entryP;
			const float x = clamp(-dot(offset, w->N), 0.f, w->oppositeDistance);
			w->backwardFraction = 1.f / (1.f + exp((w->oppositeDistance - 2.f * x) / w->diffusionLength));
			guideBackward = (uGuideDir < w->backwardFraction);
		}

		float cosTheta, hgPdf;
		if (guided) {
			cosTheta = SSS_SamplePhaseDwivedi(w->diffusionLength, w->phaseLog, uScatter0);
			if (guideBackward)
				cosTheta = -cosTheta;
			const float3 newD = SSS_DirectionFromCosine(w->N, cosTheta, uScatter1);
			hgPdf = SSS_PhaseHG(dot(w->D, newD), g);
			w->D = newD;
		} else {
			const float3 newD = SSS_SamplePhaseHG(w->D, g, uScatter0, uScatter1, &hgPdf);
			cosTheta = dot(newD, w->N);
			w->D = newD;
		}

		// The guided phase pdfs over the classic one
		w->forwardPdfFactor = (.5f * M_1_PI_F) * SSS_EvalPhaseDwivedi(w->diffusionLength, w->phaseLog, cosTheta) / hgPdf;
		w->backwardPdfFactor = (.5f * M_1_PI_F) * SSS_EvalPhaseDwivedi(w->diffusionLength, w->phaseLog, -cosTheta) / hgPdf;

		w->forwardStretching = 1.f - cosTheta / w->diffusionLength;
		w->backwardStretching = 1.f + cosTheta / w->diffusionLength;
		if (guided)
			sampleSigmaT *= guideBackward ? w->backwardStretching : w->forwardStretching;
	}

	// The distance
	w->t = -log(1.f - fmin(uDistance, .99999994f)) / sampleSigmaT;

	// The first step looks further, for an opposite interface
	w->tmax = (w->bounce == 0) ? fmax(w->t, 10.f / fmin(w->sigmaT.x, fmin(w->sigmaT.y, w->sigmaT.z))) : w->t;
}

// The pdf of the distance t (with the stretched coefficients) per channel
OPENCL_FORCE_INLINE float3 SSS_DistancePdf(const float3 sigmaT, const float t, const bool hit, float3 *transmittance) {
	const float3 T = SSS_Exp(-sigmaT * t);
	if (transmittance)
		*transmittance = T;

	return hit ? T : sigmaT * T;
}

#define SSSWALK_CONTINUE 0
#define SSSWALK_EXIT 1
#define SSSWALK_END 2

// Finishes the step with the result of tracing (w->P, w->D) up to w->tmax:
// hit tells whether the surface of the walked object was hit, at the distance
// hitT. SSSWALK_EXIT: w->P is the exit point (on the surface) and
// w->throughput the weight of the walk. uSurvive is the random number of the
// Russian roulette of the long walks.
OPENCL_FORCE_INLINE int SSSWalk_FinishStep(SSSWalk *w, const bool hitSurface, const float hitT,
		const float uSurvive) {
	// The ray of the step has been traced
	w->fromSurface = 0;

	const bool similar = (w->bounce > SSS_SIMILARITY_LEVEL);
	const float3 sigmaT = similar ? w->sigmaTStar : w->sigmaT;
	const float3 sigmaS = similar ? w->sigmaSStar : w->sigmaS;

	bool hit = hitSurface;
	float t = w->t;
	if (w->bounce == 0) {
		// The first step: an opposite interface within the longer look
		if (hit) {
			w->haveOpposite = 1;
			const float3 exitOffset = w->P + hitT * w->D - w->entryP;
			w->oppositeDistance = -dot(exitOffset, w->N);
		}
		// ...and a real hit only within the sampled distance
		hit = hit && (hitT < w->t);
	}
	if (hit)
		t = hitT;

	// Move to the scattering point or to the exit point
	w->P = w->P + t * w->D;

	float3 transmittance;
	float3 pdf = SSS_DistancePdf(sigmaT, t, hit, &transmittance);
	if (w->bounce > 0) {
		float3 guidedPdf = SSS_DistancePdf(w->forwardStretching * sigmaT, t, hit, NULL);
		if (w->haveOpposite) {
			// MIS between the forward and the backward guiding
			const float3 backPdf = SSS_DistancePdf(w->backwardStretching * sigmaT, t, hit, NULL);
			guidedPdf = (1.f - w->backwardFraction) * guidedPdf * w->forwardPdfFactor +
					w->backwardFraction * backPdf * w->backwardPdfFactor;
		} else
			guidedPdf *= w->forwardPdfFactor;

		// MIS between the classic and the guided sampling
		pdf = (1.f - w->guidedFraction) * pdf + w->guidedFraction * guidedPdf;
	}

	// MIS over the channels (balance heuristic): up to nine estimators
	const float pdfSum = dot(w->channelPdf, pdf);
	if (!(pdfSum > 0.f))
		return SSSWALK_END;
	w->throughput *= (hit ? transmittance : sigmaS * transmittance) / pdfSum;

	++w->bounce;
	if (hit)
		return SSSWALK_EXIT;
	if (fmax(w->throughput.x, fmax(w->throughput.y, w->throughput.z)) < SSS_THROUGHPUT_EPSILON)
		return SSSWALK_END;
	if (w->bounce >= SSS_RR_BOUNCES) {
		if (w->bounce >= SSS_MAX_BOUNCES)
			return SSSWALK_END;
		const float survival = (w->bounce / (w->bounce + 1.f)) *
				fmin(1.f, fmax(w->throughput.x, fmax(w->throughput.y, w->throughput.z)));
		if (uSurvive >= survival)
			return SSSWALK_END;
		w->throughput /= survival;
	}

	return SSSWALK_CONTINUE;
}

// The dielectric Fresnel reflectance of the medium of relative IOR eta (over
// the exterior) seen from inside, at the cosine cosI: 1 at the total internal
// reflection
OPENCL_FORCE_INLINE float SSS_InternalFresnel(const float cosI, const float eta) {
	const float sinT2 = eta * eta * fmax(0.f, 1.f - cosI * cosI);
	if (sinT2 >= 1.f)
		return 1.f;
	const float cosT = sqrt(1.f - sinT2);
	const float rs = (eta * cosI - cosT) / (eta * cosI + cosT);
	const float rp = (cosI - eta * cosT) / (cosI + eta * cosT);

	return .5f * (rs * rs + rp * rp);
}

// At the exit point of SSSWALK_EXIT, with the surface normal N there (either
// side) and a random number: true when the walk is reflected back into the
// medium, with the probability of the Fresnel reflectance of its direction
// (the throughput doesn't change: the walk leaves with the probability of
// the transmittance). w->D is then the reflected direction and the walk goes
// on from w->P with SSSWalk_PrepareStep().
OPENCL_FORCE_INLINE bool SSSWalk_InternalReflection(SSSWalk *w, const float3 N, const float u) {
	if (w->eta == 1.f)
		return false;

	// The normal on the side the walk leaves to
	const float cosN = dot(w->D, N);
	const float3 n = (cosN > 0.f) ? N : -N;
	const float cosI = fabs(cosN);
	if (!(cosI > 0.f))
		return false;

	if (u >= SSS_InternalFresnel(cosI, w->eta))
		return false;

	w->D = w->D - (2.f * cosI) * n;
	w->fromSurface = 1;

	return true;
}
