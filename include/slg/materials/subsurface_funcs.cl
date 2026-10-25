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
//    The color stays the albedo of the slab with that boundary (the OpenPBR
//    definition): the walk takes the color SSS_BoundaryColor() of it.
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

// BEGIN GENERATED by build-system/tools/subsurface_boundary_table.py, do not edit.
//
// The effective transmittance T[eta][C] of the boundary of the walk (see
// SSS_BoundaryColor() and the generator for the definitions and the grids)
#define SSS_BOUNDARY_N_C 33
#define SSS_BOUNDARY_N_ETA 16
#define SSS_BOUNDARY_ETA_MAX 3.0f

__constant float SSS_BoundaryTable[16 * 33] = {
	1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f,
	1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f,
	1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f,
	1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f,
	1.f,
	0.952839f, 0.957438f, 0.962037f, 0.964517f, 0.966799f, 0.968889f, 0.970798f, 0.97254f,
	0.974126f, 0.975571f, 0.976885f, 0.978081f, 0.979172f, 0.980169f, 0.981085f, 0.981931f,
	0.982719f, 0.98346f, 0.984163f, 0.984839f, 0.985495f, 0.986134f, 0.986759f, 0.987365f,
	0.987945f, 0.988481f, 0.988943f, 0.98926f, 0.989289f, 0.989289f, 0.989289f, 0.989289f,
	0.989289f,
	0.857626f, 0.870343f, 0.883061f, 0.887144f, 0.890995f, 0.894631f, 0.898071f, 0.901332f,
	0.904429f, 0.907374f, 0.910178f, 0.912851f, 0.915401f, 0.917836f, 0.920163f, 0.922388f,
	0.924517f, 0.926556f, 0.92851f, 0.930386f, 0.932191f, 0.933935f, 0.935631f, 0.937294f,
	0.938946f, 0.940616f, 0.942342f, 0.944152f, 0.94605f, 0.94605f, 0.94605f, 0.94605f,
	0.94605f,
	0.729023f, 0.754331f, 0.77964f, 0.785876f, 0.791708f, 0.797188f, 0.80236f, 0.807263f,
	0.811925f, 0.816369f, 0.820614f, 0.824674f, 0.828559f, 0.832281f, 0.835847f, 0.839266f,
	0.842548f, 0.845703f, 0.84874f, 0.851669f, 0.854495f, 0.857224f, 0.859856f, 0.86239f,
	0.864824f, 0.867153f, 0.869363f, 0.871397f, 0.873136f, 0.873136f, 0.873136f, 0.873136f,
	0.873136f,
	0.605752f, 0.637146f, 0.66854f, 0.675286f, 0.681695f, 0.6878f, 0.693631f, 0.699209f,
	0.704554f, 0.70968f, 0.7146f, 0.719327f, 0.723872f, 0.728248f, 0.732466f, 0.736536f,
	0.74047f, 0.744281f, 0.747982f, 0.751589f, 0.755117f, 0.758581f, 0.761994f, 0.765365f,
	0.768698f, 0.772f, 0.775284f, 0.77858f, 0.781991f, 0.781991f, 0.781991f, 0.781991f,
	0.781991f,
	0.500116f, 0.528738f, 0.55736f, 0.565372f, 0.57283f, 0.579778f, 0.586267f, 0.592346f,
	0.598057f, 0.603438f, 0.608525f, 0.613348f, 0.617937f, 0.622316f, 0.62651f, 0.63054f,
	0.634423f, 0.638177f, 0.641815f, 0.645349f, 0.64879f, 0.652149f, 0.655436f, 0.658661f,
	0.661837f, 0.664966f, 0.668034f, 0.670987f, 0.673717f, 0.673717f, 0.673717f, 0.673717f,
	0.673717f,
	0.41628f, 0.440583f, 0.464887f, 0.472088f, 0.478823f, 0.485104f, 0.490971f, 0.496467f,
	0.501632f, 0.506503f, 0.511108f, 0.515474f, 0.519621f, 0.523569f, 0.527333f, 0.530926f,
	0.534362f, 0.53765f, 0.540802f, 0.543831f, 0.54675f, 0.549574f, 0.552318f, 0.554993f,
	0.55761f, 0.560175f, 0.562696f, 0.565183f, 0.567663f, 0.567663f, 0.567663f, 0.567663f,
	0.567663f,
	0.336387f, 0.35788f, 0.379373f, 0.386633f, 0.393223f, 0.399231f, 0.404739f, 0.409816f,
	0.414521f, 0.418902f, 0.422997f, 0.426838f, 0.43045f, 0.433853f, 0.437067f, 0.440105f,
	0.442984f, 0.445715f, 0.448313f, 0.45079f, 0.453159f, 0.455429f, 0.45761f, 0.459707f,
	0.461721f, 0.463651f, 0.465494f, 0.467239f, 0.468864f, 0.468864f, 0.468864f, 0.468864f,
	0.468864f,
	0.270901f, 0.288983f, 0.307064f, 0.313461f, 0.319185f, 0.324366f, 0.329096f, 0.33344f,
	0.337445f, 0.341146f, 0.344571f, 0.347746f, 0.350694f, 0.353439f, 0.356003f, 0.358404f,
	0.360661f, 0.362788f, 0.3648f, 0.366706f, 0.368516f, 0.370233f, 0.371865f, 0.373414f,
	0.37489f, 0.376304f, 0.377669f, 0.379002f, 0.380336f, 0.380336f, 0.380336f, 0.380336f,
	0.380336f,
	0.219144f, 0.234677f, 0.250209f, 0.255966f, 0.260939f, 0.265302f, 0.269181f, 0.272665f,
	0.275822f, 0.278701f, 0.281339f, 0.283767f, 0.286009f, 0.288085f, 0.290014f, 0.291811f,
	0.29349f, 0.295066f, 0.296549f, 0.297948f, 0.299271f, 0.300521f, 0.3017f, 0.302802f,
	0.30382f, 0.304739f, 0.305544f, 0.306225f, 0.306782f, 0.306782f, 0.306782f, 0.306782f,
	0.306782f,
	0.173826f, 0.187461f, 0.201095f, 0.206549f, 0.211131f, 0.215058f, 0.218472f, 0.221475f,
	0.224141f, 0.226525f, 0.228673f, 0.230618f, 0.232388f, 0.234005f, 0.235488f, 0.236851f,
	0.238109f, 0.239275f, 0.240359f, 0.241372f, 0.242322f, 0.243218f, 0.244063f, 0.244862f,
	0.245617f, 0.246329f, 0.246997f, 0.247626f, 0.248224f, 0.248224f, 0.248224f, 0.248224f,
	0.248224f,
	0.139195f, 0.150787f, 0.162379f, 0.167084f, 0.170896f, 0.174076f, 0.176783f, 0.179125f,
	0.181176f, 0.182991f, 0.184613f, 0.186073f, 0.187397f, 0.188605f, 0.189714f, 0.190737f,
	0.191684f, 0.192564f, 0.193383f, 0.194145f, 0.194857f, 0.195524f, 0.196153f, 0.19675f,
	0.19732f, 0.197867f, 0.198395f, 0.1989f, 0.199363f, 0.199363f, 0.199363f, 0.199363f,
	0.199363f,
	0.111925f, 0.121695f, 0.131465f, 0.135444f, 0.138587f, 0.14116f, 0.143321f, 0.145171f,
	0.146776f, 0.148184f, 0.149427f, 0.150532f, 0.151517f, 0.152399f, 0.153191f, 0.153903f,
	0.154546f, 0.15513f, 0.155662f, 0.156149f, 0.156595f, 0.157004f, 0.157378f, 0.157719f,
	0.158029f, 0.158311f, 0.15856f, 0.158764f, 0.158902f, 0.158902f, 0.158902f, 0.158902f,
	0.158902f,
	0.0904673f, 0.098478f, 0.106489f, 0.109676f, 0.112147f, 0.114141f, 0.115793f, 0.117187f,
	0.118377f, 0.119407f, 0.120307f, 0.121101f, 0.121809f, 0.122443f, 0.123015f, 0.123533f,
	0.124004f, 0.124434f, 0.124828f, 0.125192f, 0.125532f, 0.125851f, 0.126156f, 0.126447f,
	0.126727f, 0.126996f, 0.127256f, 0.127506f, 0.127683f, 0.127683f, 0.127683f, 0.127683f,
	0.127683f,
	0.0725131f, 0.0792017f, 0.0858904f, 0.088512f, 0.0904577f, 0.0919835f, 0.0932318f, 0.0942855f,
	0.095194f, 0.0959881f, 0.0966873f, 0.097305f, 0.0978512f, 0.0983345f, 0.0987626f, 0.0991423f,
	0.0994799f, 0.0997804f, 0.100048f, 0.100287f, 0.1005f, 0.10069f, 0.100862f, 0.101017f,
	0.101162f, 0.1013f, 0.101433f, 0.101525f, 0.101222f, 0.101222f, 0.101222f, 0.101222f,
	0.101222f,
	0.0590709f, 0.0646361f, 0.0702012f, 0.0723201f, 0.0738311f, 0.074968f, 0.0758627f, 0.076593f,
	0.0772052f, 0.0777282f, 0.0781809f, 0.0785769f, 0.078926f, 0.0792364f, 0.0795141f, 0.0797643f,
	0.0799906f, 0.0801956f, 0.080381f, 0.0805474f, 0.0806958f, 0.0808267f, 0.0809409f, 0.0810403f,
	0.0811271f, 0.0811994f, 0.0812271f, 0.0810083f, 0.0796036f, 0.0796036f, 0.0796036f, 0.0796036f,
	0.0796036f
};
// END GENERATED

// The color to walk with under the boundary of relative IOR eta, so that a
// semi-infinite slab of the medium returns what it returns for the color
// with an index-matched boundary: the internal reflections send most of the
// light that reaches the boundary back for another walk (a slab of color 0.5
// returns 0.28 at eta 1.5 without this). T is the effective transmittance of
// the boundary, tabulated by the generator above from walks in a slab; above
// SSS_BOUNDARY_ETA_MAX it takes the value of SSS_BOUNDARY_ETA_MAX.
OPENCL_FORCE_INLINE float SSS_BoundaryColor(const float color, const float eta) {
	const float c = clamp(color, 0.f, 1.f);
	if (!(eta > 1.f) || !(c > 0.f) || (c >= 1.f))
		return c;

	const float fe = sqrt((fmin(eta, SSS_BOUNDARY_ETA_MAX) - 1.f) / (SSS_BOUNDARY_ETA_MAX - 1.f)) *
			(float)(SSS_BOUNDARY_N_ETA - 1);
	const uint ie = min((uint)fe, (uint)(SSS_BOUNDARY_N_ETA - 2));
	const float te = fe - (float)ie;
	const float fc = c * (float)(SSS_BOUNDARY_N_C - 1);
	const uint ic = min((uint)fc, (uint)(SSS_BOUNDARY_N_C - 2));
	const float tc = fc - (float)ic;
	__constant const float *t0 = &SSS_BoundaryTable[ie * SSS_BOUNDARY_N_C + ic];
	__constant const float *t1 = t0 + SSS_BOUNDARY_N_C;
	const float T = (1.f - te) * ((1.f - tc) * t0[0] + tc * t0[1]) +
			te * ((1.f - tc) * t1[0] + tc * t1[1]);

	return c / (T + c * (1.f - T));
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
	float3 alpha = MAKE_FLOAT3(SSS_SingleScatteringAlbedo(SSS_BoundaryColor(color.x, eta), g),
			SSS_SingleScatteringAlbedo(SSS_BoundaryColor(color.y, eta), g),
			SSS_SingleScatteringAlbedo(SSS_BoundaryColor(color.z, eta), g));
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
