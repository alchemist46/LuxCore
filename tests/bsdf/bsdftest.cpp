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

// bsdftest: consistency checks of the CPU material BSDFs, without rendering.
//
// For every test case (a material with constant parameters) and every
// configuration (eye or light path, front or back face, a few view angles)
// it draws samples with Material::Sample() and checks:
//
//  1. value: Sample() returns f |cos light| / pdf and Evaluate() returns
//     f |cos light| (both on eye and light paths: BSDF::Evaluate()/Sample()
//     apply the adjoint cosine swap on top, see bsdf.cpp), so
//     value * pdf must equal Evaluate();
//  2. pdf: the pdf of Sample() must equal the directPdfW of Evaluate();
//  3. reverse pdf: the reversePdfW of Evaluate() and of Pdf() must equal the
//     directPdfW of Evaluate() with the fixed and the sampled direction
//     exchanged, and Pdf()'s directPdfW must equal Evaluate()'s;
//  4. chi-square: the histogram of the sampled directions must match the
//     integral of Evaluate()'s directPdfW over each bin (only where the BSDF
//     is not zero; Mitsuba/pbrt style);
//  5. albedo: the mean of the sample values is the directional albedo
//     E(mu) of the fixed direction. Cases with lossless parameters declare
//     the expected value (1 for a white furnace), checked on the front face
//     of eye paths.
//
// Delta lobes (SPECULAR events) skip 1-4 and count in 5 only.
//
// Usage: bsdftest [--filter <substring>] [--samples <n>] [--verbose] [--list] [--dump <n>]
// The exit code is the number of failing cases (0 = all passed).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "luxcore/luxcore.h"
#include "luxrays/utils/properties.h"
#include "slg/scene/scene.h"
#include "slg/materials/material.h"
#include "slg/bsdf/hitpoint.h"

using namespace std;
using namespace luxrays;
using namespace slg;

namespace {

//------------------------------------------------------------------------------
// Test cases
//------------------------------------------------------------------------------

struct TestCase {
	string name;
	// Scene properties defining the material "m" (plus helpers)
	string props;
	// Expected directional albedo on the front face of eye paths, < 0 if none
	float expectAlbedo;
	// Run the chi-square test (off for very peaked lobes, where the bin
	// quadrature is not accurate enough)
	bool chi2;
};

const float NO_EXPECT = -1.f;

// Metal2 with F ~ 1 at every angle (the furnace "white metal")
const string WHITE_FRESNEL =
	"scene.textures.fr.type = fresnelconst\n"
	"scene.textures.fr.n = 0.001 0.001 0.001\n"
	"scene.textures.fr.k = 1000 1000 1000\n";

string Rough(const float a) {
	return "scene.materials.m.uroughness = " + to_string(a) + "\n"
		"scene.materials.m.vroughness = " + to_string(a) + "\n";
}

string Tag(const float a) {
	char buf[32];
	snprintf(buf, sizeof(buf), "a%.2f", a);
	return buf;
}

vector<TestCase> BuildTestCases() {
	vector<TestCase> cases;
	const float roughs[] = { 0.05f, 0.2f, 0.5f, 1.f };

	// Lambert and EON
	cases.push_back({ "matte", "scene.materials.m.type = matte\nscene.materials.m.kd = 1 1 1\n", 1.f, true });
	for (const float s : { 0.3f, 0.7f, 1.f })
		cases.push_back({ "roughmatte-s" + to_string(s).substr(0, 3),
			"scene.materials.m.type = roughmatte\nscene.materials.m.kd = 1 1 1\n"
			"scene.materials.m.sigma = " + to_string(s) + "\nscene.materials.m.multiscattering = 1\n", 1.f, true });

	// Delta and thin dielectrics
	cases.push_back({ "mirror", "scene.materials.m.type = mirror\nscene.materials.m.kr = 1 1 1\n", 1.f, false });
	// Glass: the eye-path transmission is in radiance form (eta^2), so its
	// albedo is not 1; the furnace render of a closed sphere checks it
	cases.push_back({ "glass", "scene.materials.m.type = glass\nscene.materials.m.kr = 1 1 1\n"
		"scene.materials.m.kt = 1 1 1\nscene.materials.m.interiorior = 1.5\n", NO_EXPECT, false });
	cases.push_back({ "archglass", "scene.materials.m.type = archglass\nscene.materials.m.kr = 1 1 1\n"
		"scene.materials.m.kt = 1 1 1\nscene.materials.m.interiorior = 1.5\n", 1.f, false });

	for (const float a : roughs) {
		cases.push_back({ "roughglass-" + Tag(a), "scene.materials.m.type = roughglass\n"
			"scene.materials.m.kr = 1 1 1\nscene.materials.m.kt = 1 1 1\n"
			"scene.materials.m.interiorior = 1.5\nscene.materials.m.multiscattering = 1\n" + Rough(a),
			NO_EXPECT, a >= 0.2f });

		cases.push_back({ "metal2-" + Tag(a), WHITE_FRESNEL + "scene.materials.m.type = metal2\n"
			"scene.materials.m.fresnel = fr\nscene.materials.m.multiscattering = 1\n" + Rough(a),
			1.f, a >= 0.2f });

		// The bare coating lobe with F = 1
		cases.push_back({ "glossy2-coat-" + Tag(a), "scene.materials.m.type = glossy2\n"
			"scene.materials.m.kd = 0 0 0\nscene.materials.m.ks = 1 1 1\nscene.materials.m.index = 0 0 0\n"
			"scene.materials.m.multiscattering = 1\n" + Rough(a), 1.f, a >= 0.2f });
		// A white base under a clear (4%) coating
		cases.push_back({ "glossy2-white-" + Tag(a), "scene.materials.m.type = glossy2\n"
			"scene.materials.m.kd = 1 1 1\nscene.materials.m.ks = 0.04 0.04 0.04\nscene.materials.m.index = 0 0 0\n"
			"scene.materials.m.multiscattering = 1\n" + Rough(a), 1.f, a >= 0.2f });

		cases.push_back({ "glossycoating-white-" + Tag(a),
			"scene.materials.b.type = matte\nscene.materials.b.kd = 1 1 1\n"
			"scene.materials.m.type = glossycoating\nscene.materials.m.base = b\n"
			"scene.materials.m.ks = 0.04 0.04 0.04\nscene.materials.m.index = 0 0 0\n"
			"scene.materials.m.multiscattering = 1\n" + Rough(a), 1.f, a >= 0.2f });

		// Reflection kd plus transmission kt (1 - kd): lossless in total
		cases.push_back({ "glossytranslucent-" + Tag(a), "scene.materials.m.type = glossytranslucent\n"
			"scene.materials.m.kd = 0.5 0.5 0.5\nscene.materials.m.kt = 1 1 1\n"
			"scene.materials.m.ks = 0.04 0.04 0.04\nscene.materials.m.ks_bf = 0.04 0.04 0.04\n"
			"scene.materials.m.multiscattering = 1\n" + Rough(a) +
			"scene.materials.m.uroughness_bf = " + to_string(a) + "\n"
			"scene.materials.m.vroughness_bf = " + to_string(a) + "\n", 1.f, a >= 0.2f });

		// Disney: white metal, white dielectric, clear coat on the metal
		const string disneyRough = "scene.materials.m.roughness = " + to_string(sqrtf(a)) + "\n";
		cases.push_back({ "disney-metal-" + Tag(a), "scene.materials.m.type = disney\n"
			"scene.materials.m.basecolor = 1 1 1\nscene.materials.m.metallic = 1\n"
			"scene.materials.m.specular = 1\nscene.materials.m.multiscattering = 1\n" + disneyRough,
			1.f, a >= 0.2f });
		cases.push_back({ "disney-dielectric-" + Tag(a), "scene.materials.m.type = disney\n"
			"scene.materials.m.basecolor = 1 1 1\nscene.materials.m.metallic = 0\n"
			"scene.materials.m.specular = 0.5\nscene.materials.m.multiscattering = 1\n" + disneyRough,
			1.f, a >= 0.2f });
		cases.push_back({ "disney-clearcoat-" + Tag(a), "scene.materials.m.type = disney\n"
			"scene.materials.m.basecolor = 1 1 1\nscene.materials.m.metallic = 1\n"
			"scene.materials.m.specular = 1\nscene.materials.m.clearcoat = 1\n"
			"scene.materials.m.clearcoatgloss = 0.5\nscene.materials.m.multiscattering = 1\n" + disneyRough,
			1.f, a >= 0.2f });
	}

	// Carpaint: three white lobes summing to 1 (exposes the lobe sampling)
	cases.push_back({ "carpaint-3lobes", "scene.materials.m.type = carpaint\n"
		"scene.materials.m.kd = 0 0 0\n"
		"scene.materials.m.ks1 = 0.333 0.333 0.333\nscene.materials.m.ks2 = 0.333 0.333 0.333\n"
		"scene.materials.m.ks3 = 0.334 0.334 0.334\n"
		"scene.materials.m.r1 = 1\nscene.materials.m.r2 = 1\nscene.materials.m.r3 = 1\n"
		"scene.materials.m.m1 = 0.25\nscene.materials.m.m2 = 0.45\nscene.materials.m.m3 = 0.7\n"
		"scene.materials.m.multiscattering = 1\n", 1.f, true });
	cases.push_back({ "carpaint-preset", "scene.materials.m.type = carpaint\n"
		// No chi-square: the preset's sharp first lobe is beyond the bin quadrature
		"scene.materials.m.preset = \"ford f8\"\n", NO_EXPECT, false });

	// Cloth: pure Lambert yarn (lossless), and the default preset (no target)
	cases.push_back({ "cloth-diffuse", "scene.materials.m.type = cloth\nscene.materials.m.preset = denim\n"
		"scene.materials.m.weft_kd = 1 1 1\nscene.materials.m.warp_kd = 1 1 1\n"
		"scene.materials.m.weft_ks = 0 0 0\nscene.materials.m.warp_ks = 0 0 0\n", 1.f, true });
	cases.push_back({ "cloth-denim", "scene.materials.m.type = cloth\nscene.materials.m.preset = denim\n",
		NO_EXPECT, true });

	cases.push_back({ "velvet", "scene.materials.m.type = velvet\nscene.materials.m.kd = 1 1 1\n",
		NO_EXPECT, true });

	cases.push_back({ "mattetranslucent", "scene.materials.m.type = mattetranslucent\n"
		"scene.materials.m.kr = 0.5 0.5 0.5\nscene.materials.m.kt = 1 1 1\n", 1.f, true });
	for (const float s : { 0.f, 0.5f, 1.f })
		cases.push_back({ "roughmattetranslucent-s" + to_string(s).substr(0, 3),
			"scene.materials.m.type = roughmattetranslucent\n"
			"scene.materials.m.kr = 0.5 0.5 0.5\nscene.materials.m.kt = 1 1 1\n"
			"scene.materials.m.sigma = " + to_string(s) + "\n", 1.f, true });

	// Combinators of lossless materials
	cases.push_back({ "mix", WHITE_FRESNEL +
		"scene.materials.a.type = matte\nscene.materials.a.kd = 1 1 1\n"
		"scene.materials.b.type = metal2\nscene.materials.b.fresnel = fr\n"
		"scene.materials.b.uroughness = 0.3\nscene.materials.b.vroughness = 0.3\n"
		"scene.materials.b.multiscattering = 1\n"
		"scene.materials.m.type = mix\nscene.materials.m.material1 = a\n"
		"scene.materials.m.material2 = b\nscene.materials.m.amount = 0.5\n", 1.f, true });
	cases.push_back({ "twosided",
		"scene.materials.a.type = matte\nscene.materials.a.kd = 1 1 1\n"
		"scene.materials.b.type = matte\nscene.materials.b.kd = 1 1 1\n"
		"scene.materials.m.type = twosided\nscene.materials.m.frontmaterial = a\n"
		"scene.materials.m.backmaterial = b\n", 1.f, true });

	return cases;
}

//------------------------------------------------------------------------------
// Statistics
//------------------------------------------------------------------------------

struct RelErrStats {
	double maxRel = 0.0;
	u_int count = 0;
	u_int bad = 0;

	void Add(const float a, const float b, const float tol = 1e-2f) {
		const float denom = max(fabsf(b), 1e-4f);
		const double rel = fabsf(a - b) / denom;
		maxRel = max(maxRel, rel);
		++count;
		if (rel > tol)
			++bad;
	}

	void Add(const Spectrum &a, const Spectrum &b, const float tol = 1e-2f) {
		float worst = 0.f;
		u_int worstIndex = 0;
		for (u_int i = 0; i < 3; ++i) {
			const float rel = fabsf(a.c[i] - b.c[i]) / max(fabsf(b.c[i]), 1e-4f);
			if (rel > worst) {
				worst = rel;
				worstIndex = i;
			}
		}
		Add(a.c[worstIndex], b.c[worstIndex], tol);
	}

	// A handful of samples at the float noise floor are tolerated
	bool Failed() const { return (count > 0) && (bad > max(2u, count / 1000)); }
};

// Upper tail probability of the chi-square distribution (Wilson-Hilferty)
double ChiSquareQ(const double x, const double dof) {
	if (dof <= 0.0)
		return 1.0;
	const double k = 2.0 / (9.0 * dof);
	const double z = (pow(x / dof, 1.0 / 3.0) - (1.0 - k)) / sqrt(k);
	return 0.5 * erfc(z / sqrt(2.0));
}

//------------------------------------------------------------------------------
// One configuration
//------------------------------------------------------------------------------

struct Config {
	bool fromLight;
	bool front;
	float mu;
};

struct ConfigResult {
	Spectrum albedo;
	RelErrStats value, pdf, revPdf, pdfFunc;
	u_int nanCount = 0;
	double chi2P = -1.0; // < 0: not run
	u_int deltaCount = 0;
	u_int wrongSideCount = 0;
};

// --dump <n>: print the first n samples that fail the value or pdf check
u_int dumpMax = 0;
u_int dumpCount = 0;

const u_int CHI2_COS_BINS = 20;
const u_int CHI2_PHI_BINS = 40;
const u_int CHI2_QUAD = 12;

Vector SphericalDir(const float cosTheta, const float phi) {
	const float sinTheta = sqrtf(max(0.f, 1.f - cosTheta * cosTheta));
	return Vector(sinTheta * cosf(phi), sinTheta * sinf(phi), cosTheta);
}

void SetupHitPoint(HitPoint &hp, const Vector &fixedDir, const bool fromLight) {
	hp.Init();
	hp.fixedDir = fixedDir;
	hp.p = Point(0.f, 0.f, 0.f);
	hp.geometryN = hp.interpolatedN = hp.shadeN = Normal(0.f, 0.f, 1.f);
	hp.defaultUV = UV(.3f, .7f);
	hp.dpdu = Vector(1.f, 0.f, 0.f);
	hp.dpdv = Vector(0.f, 1.f, 0.f);
	hp.dndu = hp.dndv = Normal(0.f, 0.f, 0.f);
	hp.triangleIndex = 0;
	hp.triangleBariCoord1 = hp.triangleBariCoord2 = 0.f;
	hp.localToWorld = Transform();
	hp.fromLight = fromLight;
	hp.intoObject = (fixedDir.z > 0.f);
}

// The side test of BSDF::Evaluate() (bsdf.cpp): a material without TRANSMIT
// events is never evaluated for a pair of directions on opposite sides, one
// without REFLECT never on the same side. The materials rely on it (matte
// returns its lobe on both sides), so the test applies it as well.
bool SideAllowed(const Material &mat, const Vector &a, const Vector &b) {
	const BSDFEvent events = mat.GetEventTypes();
	const float side = a.z * b.z;
	return !(((side > 0.f) && !(events & REFLECT)) ||
			((side < 0.f) && !(events & TRANSMIT)));
}

// Evaluate() with (fixed, generated) mapped to (light, eye) by the mode
Spectrum EvaluateFixedGenerated(const Material &mat, const HitPoint &hp,
		const Vector &fixedDir, const Vector &generatedDir,
		BSDFEvent *event, float *directPdfW, float *reversePdfW) {
	const Vector &lightDir = hp.fromLight ? fixedDir : generatedDir;
	const Vector &eyeDir = hp.fromLight ? generatedDir : fixedDir;
	*event = NONE;
	*directPdfW = 0.f;
	*reversePdfW = 0.f;
	if (!SideAllowed(mat, fixedDir, generatedDir))
		return Spectrum();
	return mat.Evaluate(hp, lightDir, eyeDir, event, directPdfW, reversePdfW);
}

ConfigResult RunConfig(const Material &mat, const Config &cfg, const u_int samples,
		const bool chi2, mt19937 &rng) {
	ConfigResult res;
	uniform_real_distribution<float> uni(0.f, 1.f);

	const float phiFixed = 0.7f;
	Vector fixedDir = SphericalDir(cfg.mu, phiFixed);
	if (!cfg.front)
		fixedDir.z = -fixedDir.z;

	HitPoint hp;
	SetupHitPoint(hp, fixedDir, cfg.fromLight);

	vector<double> histogram(CHI2_COS_BINS * CHI2_PHI_BINS, 0.0);

	Spectrum albedoSum;
	for (u_int i = 0; i < samples; ++i) {
		const float u0 = uni(rng);
		const float u1 = uni(rng);
		const float passThrough = uni(rng);
		hp.passThroughEvent = passThrough;

		Vector sampledDir;
		float pdfW = 0.f;
		BSDFEvent event = NONE;
		const Spectrum value = mat.Sample(hp, fixedDir, &sampledDir, u0, u1, passThrough, &pdfW, &event);

		if (value.IsNaN() || value.IsInf() || isnan(pdfW)) {
			++res.nanCount;
			continue;
		}
		if (value.Black() || (pdfW <= 0.f))
			continue;

		// A sample on a side the material has no events for is dropped by
		// the side test of the path tracers, so it is counted as a
		// rejection here too
		if (!(event & SPECULAR) && !SideAllowed(mat, fixedDir, sampledDir)) {
			++res.wrongSideCount;
			continue;
		}

		albedoSum += value;

		if (event & SPECULAR) {
			++res.deltaCount;
			continue;
		}

		const float absCosSampled = fabsf(sampledDir.z);
		if (absCosSampled < 1e-3f)
			continue;

		// 1-2: value and pdf against Evaluate()
		BSDFEvent evalEvent;
		float directPdfW, reversePdfW;
		const Spectrum evalValue = EvaluateFixedGenerated(mat, hp, fixedDir, sampledDir,
				&evalEvent, &directPdfW, &reversePdfW);
		const u_int badBefore = res.value.bad + res.pdf.bad;
		res.value.Add(value * pdfW, evalValue);
		res.pdf.Add(pdfW, directPdfW);
		if ((res.value.bad + res.pdf.bad > badBefore) && (dumpCount < dumpMax)) {
			++dumpCount;
			printf("  dump: fixed (%.4f %.4f %.4f) sampled (%.4f %.4f %.4f) event %d: "
					"sample value*pdf %.5g pdf %.5g | evaluate %.5g pdf %.5g event %d\n",
					fixedDir.x, fixedDir.y, fixedDir.z, sampledDir.x, sampledDir.y, sampledDir.z,
					static_cast<int>(event), (value * pdfW).c[0], pdfW, evalValue.c[0], directPdfW,
					static_cast<int>(evalEvent));
		}

		// 3: reverse pdf against the swapped evaluation and against Pdf()
		HitPoint hpSwapped;
		SetupHitPoint(hpSwapped, sampledDir, cfg.fromLight);
		hpSwapped.passThroughEvent = passThrough;
		BSDFEvent swappedEvent;
		float swappedDirectPdfW, swappedReversePdfW;
		EvaluateFixedGenerated(mat, hpSwapped, sampledDir, fixedDir,
				&swappedEvent, &swappedDirectPdfW, &swappedReversePdfW);
		res.revPdf.Add(reversePdfW, swappedDirectPdfW);

		const Vector &lightDir = cfg.fromLight ? fixedDir : sampledDir;
		const Vector &eyeDir = cfg.fromLight ? sampledDir : fixedDir;
		float pdfFuncDirect = 0.f, pdfFuncReverse = 0.f;
		mat.Pdf(hp, lightDir, eyeDir, &pdfFuncDirect, &pdfFuncReverse);
		res.pdfFunc.Add(pdfFuncDirect, directPdfW);
		res.pdfFunc.Add(pdfFuncReverse, reversePdfW);

		// 4: histogram (equal solid angle bins: uniform in cos(theta) and phi)
		if (chi2) {
			const u_int cb = min(CHI2_COS_BINS - 1, static_cast<u_int>((sampledDir.z + 1.f) * .5f * CHI2_COS_BINS));
			float phi = atan2f(sampledDir.y, sampledDir.x);
			if (phi < 0.f)
				phi += 2.f * M_PI;
			const u_int pb = min(CHI2_PHI_BINS - 1, static_cast<u_int>(phi / (2.f * M_PI) * CHI2_PHI_BINS));
			histogram[cb * CHI2_PHI_BINS + pb] += 1.0;
		}
	}
	res.albedo = albedoSum / static_cast<float>(samples);

	// The chi-square test makes sense only for a non-delta BSDF
	if (chi2 && (res.deltaCount == 0) && (res.pdf.count > 1000)) {
		// Expected counts: the integral of Evaluate()'s pdf over each bin,
		// where the BSDF is not zero (the samples that Sample() rejects carry
		// no value and are not in the histogram either)
		vector<double> expected(histogram.size(), 0.0);
		const double dCos = 2.0 / CHI2_COS_BINS;
		const double dPhi = 2.0 * M_PI / CHI2_PHI_BINS;
		for (u_int cb = 0; cb < CHI2_COS_BINS; ++cb) {
			for (u_int pb = 0; pb < CHI2_PHI_BINS; ++pb) {
				double sum = 0.0;
				for (u_int qi = 0; qi < CHI2_QUAD; ++qi) {
					const double cosTheta = -1.0 + (cb + (qi + .5) / CHI2_QUAD) * dCos;
					for (u_int qj = 0; qj < CHI2_QUAD; ++qj) {
						const double phi = (pb + (qj + .5) / CHI2_QUAD) * dPhi;
						const Vector dir = SphericalDir(static_cast<float>(cosTheta), static_cast<float>(phi));
						if (fabsf(dir.z) < 1e-3f)
							continue;
						BSDFEvent e;
						float directPdfW, reversePdfW;
						const Spectrum f = EvaluateFixedGenerated(mat, hp, fixedDir, dir, &e, &directPdfW, &reversePdfW);
						if (!f.Black())
							sum += directPdfW;
					}
				}
				expected[cb * CHI2_PHI_BINS + pb] = sum * (dCos / CHI2_QUAD) * (dPhi / CHI2_QUAD) * samples;
			}
		}

		// Pool the bins with fewer than 5 expected samples
		double chi2Stat = 0.0, pooledObs = 0.0, pooledExp = 0.0;
		u_int bins = 0;
		for (size_t i = 0; i < histogram.size(); ++i) {
			if (expected[i] < 5.0) {
				pooledObs += histogram[i];
				pooledExp += expected[i];
			} else {
				const double d = histogram[i] - expected[i];
				chi2Stat += d * d / expected[i];
				++bins;
			}
		}
		if (pooledExp >= 5.0) {
			const double d = pooledObs - pooledExp;
			chi2Stat += d * d / pooledExp;
			++bins;
		} else if (pooledObs > 20.0) {
			// Samples where Evaluate() says the pdf is (nearly) zero
			chi2Stat = 1e30;
		}
		res.chi2P = (bins > 1) ? ChiSquareQ(chi2Stat, bins - 1) : 1.0;

		// --dump also prints the worst bins of a failed chi-square test
		if ((dumpMax > 0) && (res.chi2P < 1e-4)) {
			vector<size_t> order(histogram.size());
			for (size_t i = 0; i < order.size(); ++i)
				order[i] = i;
			auto contribution = [&](const size_t i) {
				return (expected[i] >= 5.0) ? (histogram[i] - expected[i]) * (histogram[i] - expected[i]) / expected[i] : 0.0;
			};
			sort(order.begin(), order.end(), [&](const size_t a, const size_t b) {
				return contribution(a) > contribution(b);
			});
			printf("  chi2 dump: pooled obs %.0f exp %.1f\n", pooledObs, pooledExp);
			for (u_int k = 0; k < min<u_int>(dumpMax, order.size()); ++k) {
				const size_t i = order[k];
				const u_int cb = i / CHI2_PHI_BINS, pb = i % CHI2_PHI_BINS;
				printf("  chi2 dump: cos [%.2f %.2f] phi [%.2f %.2f] obs %.0f exp %.1f\n",
						-1.0 + cb * 2.0 / CHI2_COS_BINS, -1.0 + (cb + 1) * 2.0 / CHI2_COS_BINS,
						pb * 2.0 * M_PI / CHI2_PHI_BINS, (pb + 1) * 2.0 * M_PI / CHI2_PHI_BINS,
						histogram[i], expected[i]);
			}
		}
	}

	return res;
}

//------------------------------------------------------------------------------
// Main
//------------------------------------------------------------------------------

string Fmt(const RelErrStats &s) {
	if (s.count == 0)
		return "      -      ";
	char buf[64];
	snprintf(buf, sizeof(buf), "%.1e/%u", s.maxRel, s.bad);
	return buf;
}

const double CHI2_SIGNIFICANCE = 1e-4;
const float ALBEDO_TOLERANCE = 0.01f;

} // namespace

int main(int argc, char *argv[]) {
	string filter;
	u_int samples = 200000;
	bool verbose = false;
	bool listOnly = false;
	for (int i = 1; i < argc; ++i) {
		if (!strcmp(argv[i], "--filter") && (i + 1 < argc))
			filter = argv[++i];
		else if (!strcmp(argv[i], "--samples") && (i + 1 < argc))
			samples = static_cast<u_int>(atoi(argv[++i]));
		else if (!strcmp(argv[i], "--verbose"))
			verbose = true;
		else if (!strcmp(argv[i], "--dump") && (i + 1 < argc))
			dumpMax = static_cast<u_int>(atoi(argv[++i]));
		else if (!strcmp(argv[i], "--list"))
			listOnly = true;
		else {
			fprintf(stderr, "Usage: %s [--filter <substring>] [--samples <n>] [--verbose] [--list] [--dump <n>]\n", argv[0]);
			return 1;
		}
	}

	luxcore::Init([](const char *) {});

	const vector<TestCase> cases = BuildTestCases();
	const Config configs[] = {
		{ false, true, .95f }, { false, true, .6f }, { false, true, .25f }, { false, true, .08f },
		{ false, false, .6f }, { false, false, .25f },
		{ true, true, .6f }, { true, true, .25f },
		{ true, false, .6f }, { true, false, .25f }
	};

	printf("%-34s %-5s %-5s %4s  %-8s %-13s %-13s %-13s %-13s %-9s %s\n",
			"case", "path", "face", "mu", "albedo", "value", "pdf", "revpdf", "Pdf()", "chi2 p", "result");

	int failedCases = 0;
	for (const TestCase &tc : cases) {
		if (!filter.empty() && (tc.name.find(filter) == string::npos))
			continue;
		if (listOnly) {
			printf("%s\n", tc.name.c_str());
			continue;
		}

		const string sceneProps =
			"scene.camera.type = perspective\n"
			"scene.camera.lookat.orig = 0 -5 0\n"
			"scene.camera.lookat.target = 0 0 0\n" + tc.props;

		unique_ptr<Scene> scene;
		try {
			auto props = make_unique<Properties>();
			props->SetFromString(sceneProps);
			scene = make_unique<Scene>();
			scene->Parse(props);
		} catch (const exception &e) {
			printf("%-34s parse error: %s\n", tc.name.c_str(), e.what());
			++failedCases;
			continue;
		}
		const Material &mat = scene->GetMaterials().GetMaterial("m");

		mt19937 rng(12345);
		bool caseFailed = false;
		for (const Config &cfg : configs) {
			const bool chi2 = tc.chi2 && ((cfg.mu == .6f) || (cfg.mu == .25f));
			const ConfigResult r = RunConfig(mat, cfg, samples, chi2, rng);

			const float albedo = (r.albedo.c[0] + r.albedo.c[1] + r.albedo.c[2]) / 3.f;
			vector<string> fails;
			if (r.nanCount > 0)
				fails.push_back("nan");
			if (r.value.Failed())
				fails.push_back("value");
			if (r.pdf.Failed())
				fails.push_back("pdf");
			if (r.revPdf.Failed())
				fails.push_back("revpdf");
			if (r.pdfFunc.Failed())
				fails.push_back("Pdf()");
			if ((r.chi2P >= 0.0) && (r.chi2P < CHI2_SIGNIFICANCE))
				fails.push_back("chi2");
			if ((tc.expectAlbedo >= 0.f) && !cfg.fromLight && cfg.front &&
					(fabsf(albedo - tc.expectAlbedo) > ALBEDO_TOLERANCE))
				fails.push_back("albedo");

			string result = "ok";
			if (!fails.empty()) {
				result = "FAIL(";
				for (size_t i = 0; i < fails.size(); ++i)
					result += (i ? "," : "") + fails[i];
				result += ")";
				caseFailed = true;
			}

			if (verbose || !fails.empty() || (!cfg.fromLight && cfg.front)) {
				char chi2Buf[16];
				if (r.chi2P >= 0.0)
					snprintf(chi2Buf, sizeof(chi2Buf), "%.2e", r.chi2P);
				else
					snprintf(chi2Buf, sizeof(chi2Buf), "-");
				printf("%-34s %-5s %-5s %4.2f  %-8.4f %-13s %-13s %-13s %-13s %-9s %s\n",
						tc.name.c_str(), cfg.fromLight ? "light" : "eye", cfg.front ? "front" : "back",
						cfg.mu, albedo, Fmt(r.value).c_str(), Fmt(r.pdf).c_str(), Fmt(r.revPdf).c_str(),
						Fmt(r.pdfFunc).c_str(), chi2Buf, result.c_str());
			}
		}

		if (caseFailed)
			++failedCases;
	}

	if (!listOnly)
		printf("\n%d failing case(s)\n", failedCases);

	return min(failedCases, 100);
}
