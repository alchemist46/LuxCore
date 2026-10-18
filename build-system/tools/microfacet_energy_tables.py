#!/usr/bin/env python3
# Copyright 1998-2026 by authors (see AUTHORS.txt)
#
# This file is part of LuxCoreRender.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generates include/slg/materials/microfacet_energy_tables.cl: the directional
albedo tables used by the multiple-scattering energy compensation of the GGX
microfacet lobes (Kulla & Conty 2017, "Revisiting Physically Based Shading at
Imageworks"; Turquin 2019, "Practical Multiple Scattering Compensation for
Microfacet Models"), the way Cycles 4.0 does it.

Every table is the hemispherical-directional albedo of a lobe with F = 1:

    E(mu, alpha) = integral over the hemisphere of f(wo, wi) |cos(theta_i)| dwi

with mu = cos(theta_o), computed by sampling the half vector from the
distribution of the visible normals of wo (Dupuy & Benyoub 2023), so that the
estimator is G2(wo, wi) / G1(wo) with a negligible variance:

- GGXEnergy_E_HC[rough][mu]: the reflection lobe D G2 / (4 cos_o cos_i) with
  the height-correlated Smith G2 (metal2, roughglass, carpaint and the glossy
  coatings with distribution = ggx).
- GGXEnergy_E_SEP[rough][mu]: the same with the separable Smith G2 = G1 G1
  (the Disney material).
- GGXEnergy_Eavg_*[rough]: the cosine-weighted average of E over mu,
  Eavg = 2 integral E(mu) mu dmu.
- GGXEnergy_SchlickB_HC[rough][mu]: the albedo of the height-correlated lobe
  weighted by the Schlick term (1 - cos(wo, h))^5, so that the albedo of the
  lobe with the Schlick Fresnel of F0 is F0 (E - B) + B (the coatings of
  glossy2, glossycoating and glossytranslucent).
- GGXEnergy_CoatAvg_HC[4][rough]: the cosine-weighted averages P, Q, R, S of
  E - B, B, (E - B) (1 - E) / E and B (1 - E) / E: the average albedo of the
  energy-compensated Schlick lobe is F0 P + Q + Fms (F0 R + S).
- GGXEnergy_GlassTable[eta][rough][mu] and GGXEnergy_GlassInvTable[1/eta][rough][mu]
  (12 x 16 x 16):
  the albedo of the full rough dielectric interface (Fresnel-weighted
  reflection plus Walter's transmission, in the energy form used by the
  roughglass material) seen from the side with the relative IOR eta >= 1 and
  eta <= 1 respectively; the lossless interface needs no Eavg.
- GGXEnergy_DielectricTable[eta][rough][mu] (12 x 16 x 16): the albedo of the
  height-correlated reflection lobe weighted by the exact dielectric Fresnel of
  the relative IOR eta >= 1, without the transmission (the specular reflection
  over the opaque base of the openpbr material).

The eta axes of the glass and dielectric tables are uniform in
sqrt((eta - 1) / 2): near eta = 1 (water, a base under a coat of a similar
IOR, the inside of a glass near the critical angle) the albedo is far from
linear in eta (an axis linear in eta was 3% off inside a glass of IOR 1.5).
These three tables are stored as 16-bit fixed point (the albedos are in
[0, 1], the step is 1.5e-5), so that all the __constant tables of the kernels
stay within the 64 KB of CUDA's constant memory.

The tables are indexed by rough = sqrt(alpha) (the roughness of the LuxCore
materials is alpha itself, the table resolution is spent where E changes),
mu = cos(theta_o) and eta in [1, 3] (the inverse table by 1 / eta in [1, 3]),
all on uniform grids (eta as above) with the end points included and linear
interpolation.
The albedo at mu = 0 is evaluated at mu = 1e-3 (its limit is finite).

Run it from the source tree root (it takes about a minute):

    python3 build-system/tools/microfacet_energy_tables.py
"""

import os
import sys
import numpy as np

N_ROUGH = 32
N_MU = 32
N_GLASS = 16
N_GLASS_ETA = 12
N_DIEL_ETA = 12
ETA_MIN, ETA_MAX = 1.0, 3.0
MU_MIN = 1e-3
STRATA = 256  # STRATA^2 half vectors per table entry
EAVG_MU_SAMPLES = 256

OUTPUT = os.path.join("include", "slg", "materials", "microfacet_energy_tables.cl")


def sample_points():
    i = (np.arange(STRATA) + 0.5) / STRATA
    u0, u1 = np.meshgrid(i, i)
    return u0.ravel(), u1.ravel()


U0, U1 = sample_points()


def lambda_smith(alpha, wz, wx, wy):
    """Smith Lambda of the isotropic GGX for a unit vector (|wz| > 0)."""
    tan2 = (wx * wx + wy * wy) / (wz * wz)
    return (-1.0 + np.sqrt(1.0 + alpha * alpha * tan2)) * 0.5


def sample_vndf(alpha, wo):
    """Dupuy & Benyoub 2023: half vectors distributed as the visible normals of
    wo (wo.z > 0), one per (U0, U1) pair."""
    ws = np.array([alpha * wo[0], alpha * wo[1], wo[2]])
    ws /= np.linalg.norm(ws)
    phi = 2.0 * np.pi * U0
    z = (1.0 - U1) * (1.0 + ws[2]) - ws[2]
    s = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
    hx = s * np.cos(phi) + ws[0]
    hy = s * np.sin(phi) + ws[1]
    hz = z + ws[2]
    h = np.stack([alpha * hx, alpha * hy, np.maximum(hz, 0.0)], axis=0)
    h /= np.maximum(np.linalg.norm(h, axis=0), 1e-30)
    return h


def g2_over_g1(alpha, wo, wi, separable):
    lo = lambda_smith(alpha, wo[2], wo[0], wo[1])
    li = lambda_smith(alpha, wi[2], wi[0], wi[1])
    if separable:
        return 1.0 / (1.0 + li)
    return (1.0 + lo) / (1.0 + lo + li)


def albedo_reflect(alpha, mu, separable):
    """Directional albedo of the F = 1 reflection lobe."""
    if alpha == 0.0:
        return 1.0
    wo = np.array([np.sqrt(max(0.0, 1.0 - mu * mu)), 0.0, mu])
    h = sample_vndf(alpha, wo)
    woh = wo[0] * h[0] + wo[1] * h[1] + wo[2] * h[2]
    wi = 2.0 * woh * h - wo[:, None]
    valid = wi[2] > 0.0
    wiz = np.where(valid, wi[2], 1.0)
    weight = g2_over_g1(alpha, wo, (wi[0], wi[1], wiz), separable)
    return float(np.mean(np.where(valid, weight, 0.0)))


def albedo_reflect_schlick(alpha, mu):
    """(E, B): the albedo of the F = 1 height-correlated reflection lobe and of
    the same lobe weighted by the Schlick term (1 - cos(wo, h))^5."""
    if alpha == 0.0:
        return 1.0, (1.0 - mu) ** 5
    wo = np.array([np.sqrt(max(0.0, 1.0 - mu * mu)), 0.0, mu])
    h = sample_vndf(alpha, wo)
    woh = wo[0] * h[0] + wo[1] * h[1] + wo[2] * h[2]
    wi = 2.0 * woh * h - wo[:, None]
    valid = wi[2] > 0.0
    wiz = np.where(valid, wi[2], 1.0)
    weight = np.where(valid, g2_over_g1(alpha, wo, (wi[0], wi[1], wiz), False), 0.0)
    schlick = (1.0 - np.clip(woh, 0.0, 1.0)) ** 5
    return float(np.mean(weight)), float(np.mean(weight * schlick))


def fresnel_dielectric(cos_i, eta):
    """Unpolarized Fresnel reflectance; eta is n_t / n_i for the side of cos_i
    (cos_i > 0). Total internal reflection gives 1."""
    sin2_t = (1.0 - cos_i * cos_i) / (eta * eta)
    tir = sin2_t >= 1.0
    cos_t = np.sqrt(np.clip(1.0 - sin2_t, 0.0, 1.0))
    r_par = (eta * cos_i - cos_t) / (eta * cos_i + cos_t)
    r_perp = (cos_i - eta * cos_t) / (cos_i + eta * cos_t)
    f = 0.5 * (r_par * r_par + r_perp * r_perp)
    return np.where(tir, 1.0, f), cos_t, tir


def albedo_glass(alpha, mu, eta):
    """Albedo of the rough dielectric interface (reflection plus transmission,
    energy form) for wo on the side with the relative IOR eta = n_other / n_wo."""
    wo = np.array([np.sqrt(max(0.0, 1.0 - mu * mu)), 0.0, mu])
    if alpha == 0.0:
        f, _, _ = fresnel_dielectric(np.array([mu]), eta)
        return 1.0  # F + (1 - F): a smooth interface is lossless
    h = sample_vndf(alpha, wo)
    woh = np.clip(wo[0] * h[0] + wo[1] * h[1] + wo[2] * h[2], 0.0, 1.0)
    f, cos_t, tir = fresnel_dielectric(woh, eta)

    # Reflection
    wr = 2.0 * woh * h - wo[:, None]
    r_valid = wr[2] > 0.0
    wrz = np.where(r_valid, wr[2], 1.0)
    r_weight = g2_over_g1(alpha, wo, (wr[0], wr[1], wrz), False)
    result = np.where(r_valid, f * r_weight, 0.0)

    # Transmission: wt = (woh / eta - cos_t) h - wo / eta
    k = woh / eta - cos_t
    wt = k * h - wo[:, None] / eta
    t_valid = (~tir) & (wt[2] < 0.0)
    wtz = np.where(t_valid, -wt[2], 1.0)
    t_weight = g2_over_g1(alpha, wo, (wt[0], wt[1], wtz), False)
    result += np.where(t_valid, (1.0 - f) * t_weight, 0.0)
    return float(np.mean(result))


def albedo_dielectric_reflect(alpha, mu, eta):
    """Albedo of the reflection lobe weighted by the exact Fresnel of the
    relative IOR eta >= 1 (no transmission)."""
    if alpha == 0.0:
        f, _, _ = fresnel_dielectric(np.array([mu]), eta)
        return float(f[0])
    wo = np.array([np.sqrt(max(0.0, 1.0 - mu * mu)), 0.0, mu])
    h = sample_vndf(alpha, wo)
    woh = np.clip(wo[0] * h[0] + wo[1] * h[1] + wo[2] * h[2], 0.0, 1.0)
    f, _, _ = fresnel_dielectric(woh, eta)
    wr = 2.0 * woh * h - wo[:, None]
    valid = wr[2] > 0.0
    wrz = np.where(valid, wr[2], 1.0)
    weight = g2_over_g1(alpha, wo, (wr[0], wr[1], wrz), False)
    return float(np.mean(np.where(valid, f * weight, 0.0)))


def rough_grid(n):
    return np.arange(n) / (n - 1)


def mu_grid(n):
    return np.maximum(np.arange(n) / (n - 1), MU_MIN)


def reflect_tables(separable):
    E = np.zeros((N_ROUGH, N_MU))
    Eavg = np.zeros(N_ROUGH)
    mus_avg = (np.arange(EAVG_MU_SAMPLES) + 0.5) / EAVG_MU_SAMPLES
    for i, rough in enumerate(rough_grid(N_ROUGH)):
        alpha = rough * rough
        for j, mu in enumerate(mu_grid(N_MU)):
            E[i, j] = albedo_reflect(alpha, mu, separable)
        Eavg[i] = 2.0 * np.mean([albedo_reflect(alpha, mu, separable) * mu for mu in mus_avg])
        print(f"  {'separable' if separable else 'height-correlated'} rough {rough:.3f}: "
              f"E(1) {E[i, -1]:.4f} E(0.5) {E[i, N_MU // 2]:.4f} Eavg {Eavg[i]:.4f}", file=sys.stderr)
    return E, Eavg


def coat_tables():
    B = np.zeros((N_ROUGH, N_MU))
    avg = np.zeros((4, N_ROUGH))
    mus_avg = (np.arange(EAVG_MU_SAMPLES) + 0.5) / EAVG_MU_SAMPLES
    for i, rough in enumerate(rough_grid(N_ROUGH)):
        alpha = rough * rough
        for j, mu in enumerate(mu_grid(N_MU)):
            B[i, j] = albedo_reflect_schlick(alpha, mu)[1]
        EB = np.array([albedo_reflect_schlick(alpha, mu) for mu in mus_avg])
        E, Bm = np.maximum(EB[:, 0], 1e-4), EB[:, 1]
        w = 2.0 * mus_avg
        avg[0, i] = np.mean((E - Bm) * w)
        avg[1, i] = np.mean(Bm * w)
        avg[2, i] = np.mean((E - Bm) * (1.0 - E) / E * w)
        avg[3, i] = np.mean(Bm * (1.0 - E) / E * w)
        print(f"  coat rough {rough:.3f}: B(1) {B[i, -1]:.4f} B(0.5) {B[i, N_MU // 2]:.4f} "
              f"P {avg[0, i]:.4f} Q {avg[1, i]:.4f} R {avg[2, i]:.4f} S {avg[3, i]:.4f}", file=sys.stderr)
    return B, avg


def eta_grid(n):
    """Uniform in sqrt((eta - eta_min) / (eta_max - eta_min))."""
    x = np.linspace(0.0, 1.0, n)
    return ETA_MIN + (ETA_MAX - ETA_MIN) * x * x


def glass_tables(inverse):
    E = np.zeros((N_GLASS_ETA, N_GLASS, N_GLASS))
    for k, e in enumerate(eta_grid(N_GLASS_ETA)):
        eta = 1.0 / e if inverse else e
        for i, rough in enumerate(rough_grid(N_GLASS)):
            alpha = rough * rough
            for j, mu in enumerate(mu_grid(N_GLASS)):
                E[k, i, j] = albedo_glass(alpha, mu, eta)
        print(f"  glass {'1/' if inverse else ''}eta {e:.3f}: E(rough 1, mu 1) {E[k, -1, -1]:.4f} "
              f"E(rough 0.5, mu 0.5) {E[k, N_GLASS // 2, N_GLASS // 2]:.4f}", file=sys.stderr)
    return E


def dielectric_tables():
    E = np.zeros((N_DIEL_ETA, N_GLASS, N_GLASS))
    for k, eta in enumerate(eta_grid(N_DIEL_ETA)):
        for i, rough in enumerate(rough_grid(N_GLASS)):
            alpha = rough * rough
            for j, mu in enumerate(mu_grid(N_GLASS)):
                E[k, i, j] = albedo_dielectric_reflect(alpha, mu, eta)
        print(f"  dielectric eta {eta:.3f}: E(rough 0, mu 1) {E[k, 0, -1]:.4f} "
              f"E(rough 0.5, mu 0.5) {E[k, N_GLASS // 2, N_GLASS // 2]:.4f}", file=sys.stderr)
    return E


def literal(v):
    """A float literal valid in C++ and OpenCL C (always with a decimal point)."""
    t = f"{v:.6g}"
    if "." not in t and "e" not in t:
        t += "."
    return t + "f"


def emit_table(out, name, values, dims):
    flat = values.ravel()
    out.write(f"__constant float {name}[{' * '.join(str(d) for d in dims)}] = {{\n")
    for start in range(0, flat.size, 8):
        out.write("\t" + ", ".join(literal(v) for v in flat[start:start + 8]) + ",\n")
    out.write("};\n\n")


def emit_table_u16(out, name, values, dims):
    """Values in [0, 1] as 16-bit fixed point (v * 65535)."""
    flat = np.rint(np.clip(values.ravel(), 0.0, 1.0) * 65535.0).astype(np.int64)
    out.write(f"__constant ushort {name}[{' * '.join(str(d) for d in dims)}] = {{\n")
    for start in range(0, flat.size, 12):
        out.write("\t" + ", ".join(str(v) for v in flat[start:start + 12]) + ",\n")
    out.write("};\n\n")


def main():
    print("Reflection tables", file=sys.stderr)
    E_hc, Eavg_hc = reflect_tables(False)
    E_sep, Eavg_sep = reflect_tables(True)
    print("Coating tables", file=sys.stderr)
    B_coat, coat_avg = coat_tables()
    print("Glass tables", file=sys.stderr)
    G = glass_tables(False)
    Ginv = glass_tables(True)
    print("Dielectric reflection tables", file=sys.stderr)
    D = dielectric_tables()

    with open(OUTPUT, "w") as out:
        out.write("#line 2 \"microfacet_energy_tables.cl\"\n\n")
        out.write("/***************************************************************************\n")
        out.write(" * Copyright 1998-2026 by authors (see AUTHORS.txt)                        *\n")
        out.write(" *                                                                         *\n")
        out.write(" *   This file is part of LuxCoreRender.                                   *\n")
        out.write(" *                                                                         *\n")
        out.write(" * Licensed under the Apache License, Version 2.0 (the \"License\");         *\n")
        out.write(" * you may not use this file except in compliance with the License.        *\n")
        out.write(" * You may obtain a copy of the License at                                 *\n")
        out.write(" *                                                                         *\n")
        out.write(" *     http://www.apache.org/licenses/LICENSE-2.0                          *\n")
        out.write(" *                                                                         *\n")
        out.write(" * Unless required by applicable law or agreed to in writing, software     *\n")
        out.write(" * distributed under the License is distributed on an \"AS IS\" BASIS,       *\n")
        out.write(" * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.*\n")
        out.write(" * See the License for the specific language governing permissions and     *\n")
        out.write(" * limitations under the License.                                          *\n")
        out.write(" ***************************************************************************/\n\n")
        out.write("// GENERATED by build-system/tools/microfacet_energy_tables.py, do not edit.\n")
        out.write("//\n// Directional albedo tables of the GGX microfacet lobes with F = 1, used by\n")
        out.write("// the multiple-scattering energy compensation (see the generator for the\n")
        out.write("// definitions and the grids). The lookup functions are at the end.\n\n")
        out.write(f"#define GGXENERGY_N_ROUGH {N_ROUGH}\n")
        out.write(f"#define GGXENERGY_N_MU {N_MU}\n")
        out.write(f"#define GGXENERGY_N_GLASS {N_GLASS}\n")
        out.write(f"#define GGXENERGY_N_GLASS_ETA {N_GLASS_ETA}\n")
        out.write(f"#define GGXENERGY_N_DIEL_ETA {N_DIEL_ETA}\n")
        out.write(f"#define GGXENERGY_ETA_MIN {ETA_MIN}f\n")
        out.write(f"#define GGXENERGY_ETA_MAX {ETA_MAX}f\n\n")
        emit_table(out, "GGXEnergy_E_HC", E_hc, (N_ROUGH, N_MU))
        emit_table(out, "GGXEnergy_Eavg_HC", Eavg_hc, (N_ROUGH,))
        emit_table(out, "GGXEnergy_E_SEP", E_sep, (N_ROUGH, N_MU))
        emit_table(out, "GGXEnergy_Eavg_SEP", Eavg_sep, (N_ROUGH,))
        emit_table(out, "GGXEnergy_SchlickB_HC", B_coat, (N_ROUGH, N_MU))
        emit_table(out, "GGXEnergy_CoatAvg_HC", coat_avg, (4, N_ROUGH))
        emit_table_u16(out, "GGXEnergy_GlassTable", G, (N_GLASS_ETA, N_GLASS, N_GLASS))
        emit_table_u16(out, "GGXEnergy_GlassInvTable", Ginv, (N_GLASS_ETA, N_GLASS, N_GLASS))
        emit_table_u16(out, "GGXEnergy_DielectricTable", D, (N_DIEL_ETA, N_GLASS, N_GLASS))
        out.write(LOOKUP_CODE)
    print(f"Written {OUTPUT}", file=sys.stderr)


LOOKUP_CODE = r"""//------------------------------------------------------------------------------
// Table lookups (linear interpolation, the inputs clamped to the grids)
//------------------------------------------------------------------------------

OPENCL_FORCE_INLINE float GGXEnergy_Lerp1D(__constant const float *table, const uint n, const float x) {
	const float fx = clamp(x, 0.f, 1.f) * (float)(n - 1);
	const uint i0 = min((uint)fx, n - 2);
	const float t = fx - (float)i0;

	return (1.f - t) * table[i0] + t * table[i0 + 1];
}

OPENCL_FORCE_INLINE float GGXEnergy_Lerp2D(__constant const float *table, const uint n0, const uint n1,
		const float x0, const float x1) {
	const float fx0 = clamp(x0, 0.f, 1.f) * (float)(n0 - 1);
	const uint i0 = min((uint)fx0, n0 - 2);
	const float t0 = fx0 - (float)i0;

	return (1.f - t0) * GGXEnergy_Lerp1D(&table[i0 * n1], n1, x1) +
			t0 * GGXEnergy_Lerp1D(&table[(i0 + 1) * n1], n1, x1);
}

// The same for the 16-bit fixed point tables
OPENCL_FORCE_INLINE float GGXEnergy_Lerp1DU16(__constant const ushort *table, const uint n, const float x) {
	const float fx = clamp(x, 0.f, 1.f) * (float)(n - 1);
	const uint i0 = min((uint)fx, n - 2);
	const float t = fx - (float)i0;

	return ((1.f - t) * (float)table[i0] + t * (float)table[i0 + 1]) * (1.f / 65535.f);
}

OPENCL_FORCE_INLINE float GGXEnergy_Lerp2DU16(__constant const ushort *table, const uint n0, const uint n1,
		const float x0, const float x1) {
	const float fx0 = clamp(x0, 0.f, 1.f) * (float)(n0 - 1);
	const uint i0 = min((uint)fx0, n0 - 2);
	const float t0 = fx0 - (float)i0;

	return (1.f - t0) * GGXEnergy_Lerp1DU16(&table[i0 * n1], n1, x1) +
			t0 * GGXEnergy_Lerp1DU16(&table[(i0 + 1) * n1], n1, x1);
}

// The position of eta on the eta axes of the glass and dielectric tables,
// uniform in sqrt((eta - eta_min) / (eta_max - eta_min))
OPENCL_FORCE_INLINE float GGXEnergy_EtaAxis(const float eta) {
	const float e = clamp(eta, GGXENERGY_ETA_MIN, GGXENERGY_ETA_MAX);

	return sqrt((e - GGXENERGY_ETA_MIN) / (GGXENERGY_ETA_MAX - GGXENERGY_ETA_MIN));
}

// The albedo of the F = 1 GGX reflection lobe (alpha = (u, v) as in Microfacet_*,
// mu = |cos(theta)| of the viewing direction). separable selects the Disney
// masking (G1 G1) instead of the height-correlated one.
OPENCL_FORCE_INLINE float GGXEnergy_E(const float u, const float v, const float mu, const bool separable) {
	const float rough = sqrt(sqrt(u * v));

	return GGXEnergy_Lerp2D(separable ? GGXEnergy_E_SEP : GGXEnergy_E_HC,
			GGXENERGY_N_ROUGH, GGXENERGY_N_MU, rough, mu);
}

OPENCL_FORCE_INLINE float GGXEnergy_Eavg(const float u, const float v, const bool separable) {
	const float rough = sqrt(sqrt(u * v));

	return GGXEnergy_Lerp1D(separable ? GGXEnergy_Eavg_SEP : GGXEnergy_Eavg_HC,
			GGXENERGY_N_ROUGH, rough);
}

// The albedo of the rough dielectric interface seen from the side where the
// relative IOR (the other side over this side) is eta
OPENCL_FORCE_INLINE float GGXEnergy_GlassE(const float u, const float v, const float mu, const float eta) {
	const float rough = sqrt(sqrt(u * v));
	const bool inverse = (eta < 1.f);
	const float fe = GGXEnergy_EtaAxis(inverse ? 1.f / eta : eta) * (float)(GGXENERGY_N_GLASS_ETA - 1);
	const uint ie = min((uint)fe, (uint)(GGXENERGY_N_GLASS_ETA - 2));
	const float te = fe - (float)ie;
	__constant const ushort *table = inverse ? GGXEnergy_GlassInvTable : GGXEnergy_GlassTable;
	const uint stride = GGXENERGY_N_GLASS * GGXENERGY_N_GLASS;

	return (1.f - te) * GGXEnergy_Lerp2DU16(&table[ie * stride], GGXENERGY_N_GLASS, GGXENERGY_N_GLASS, rough, mu) +
			te * GGXEnergy_Lerp2DU16(&table[(ie + 1) * stride], GGXENERGY_N_GLASS, GGXENERGY_N_GLASS, rough, mu);
}

// The albedo of the GGX reflection lobe (height-correlated, no energy
// compensation) weighted by the exact Fresnel of the relative IOR eta >= 1,
// without the transmission (the specular reflection of the openpbr material)
OPENCL_FORCE_INLINE float GGXEnergy_DielectricE(const float u, const float v, const float mu, const float eta) {
	const float rough = sqrt(sqrt(u * v));
	const float fe = GGXEnergy_EtaAxis(eta) * (float)(GGXENERGY_N_DIEL_ETA - 1);
	const uint ie = min((uint)fe, (uint)(GGXENERGY_N_DIEL_ETA - 2));
	const float te = fe - (float)ie;
	const uint stride = GGXENERGY_N_GLASS * GGXENERGY_N_GLASS;

	return (1.f - te) * GGXEnergy_Lerp2DU16(&GGXEnergy_DielectricTable[ie * stride], GGXENERGY_N_GLASS, GGXENERGY_N_GLASS, rough, mu) +
			te * GGXEnergy_Lerp2DU16(&GGXEnergy_DielectricTable[(ie + 1) * stride], GGXENERGY_N_GLASS, GGXENERGY_N_GLASS, rough, mu);
}

// The coatings (glossy2, glossycoating, glossytranslucent): the GGX reflection
// lobe (height-correlated) with the Schlick Fresnel of F0, energy compensated
// as Microfacet_MSScale() does (the average Fresnel F0 + (1 - F0) / 21). The
// albedo of the compensated lobe seen from mu, and its cosine-weighted average
// over the hemisphere: the layering of a coating over its base scales the base
// by (1 - E(mu_i)) (1 - E(mu_o)) / (1 - Eavg), reciprocal and exact for a
// Lambertian base.
OPENCL_FORCE_INLINE float GGXEnergy_CoatFms(const float u, const float v, const float F0) {
	const float rough = sqrt(sqrt(u * v));
	const float Eavg = GGXEnergy_Lerp1D(GGXEnergy_Eavg_HC, GGXENERGY_N_ROUGH, rough);
	const float Fss = F0 + (1.f - F0) * (1.f / 21.f);

	return Fss * Eavg / fmax(1.f - Fss * (1.f - Eavg), 1e-4f);
}

OPENCL_FORCE_INLINE float GGXEnergy_CoatE(const float u, const float v, const float mu, const float F0) {
	const float rough = sqrt(sqrt(u * v));
	const float E = GGXEnergy_Lerp2D(GGXEnergy_E_HC, GGXENERGY_N_ROUGH, GGXENERGY_N_MU, rough, mu);
	const float B = GGXEnergy_Lerp2D(GGXEnergy_SchlickB_HC, GGXENERGY_N_ROUGH, GGXENERGY_N_MU, rough, mu);
	const float EF = F0 * (E - B) + B;

	return EF * (1.f + GGXEnergy_CoatFms(u, v, F0) * (1.f - E) / fmax(E, 1e-4f));
}

OPENCL_FORCE_INLINE float GGXEnergy_CoatEavg(const float u, const float v, const float F0) {
	const float rough = sqrt(sqrt(u * v));
	const float P = GGXEnergy_Lerp1D(&GGXEnergy_CoatAvg_HC[0 * GGXENERGY_N_ROUGH], GGXENERGY_N_ROUGH, rough);
	const float Q = GGXEnergy_Lerp1D(&GGXEnergy_CoatAvg_HC[1 * GGXENERGY_N_ROUGH], GGXENERGY_N_ROUGH, rough);
	const float R = GGXEnergy_Lerp1D(&GGXEnergy_CoatAvg_HC[2 * GGXENERGY_N_ROUGH], GGXENERGY_N_ROUGH, rough);
	const float S = GGXEnergy_Lerp1D(&GGXEnergy_CoatAvg_HC[3 * GGXENERGY_N_ROUGH], GGXENERGY_N_ROUGH, rough);

	return F0 * P + Q + GGXEnergy_CoatFms(u, v, F0) * (F0 * R + S);
}

// The multiple-scattering scale of a reflection lobe with the average Fresnel
// reflectance Fss (one channel): the lobe f_ss * F is multiplied by
// 1 + Fms * (1 - E) / E with Fms = Fss * Eavg / (1 - Fss * (1 - Eavg)) (Kulla &
// Conty 2017, as scaled by Turquin 2019 and Cycles). With Fss = 1 the albedo of
// the lobe becomes 1 for every viewing direction.
OPENCL_FORCE_INLINE float GGXEnergy_ReflectScale(const float E, const float Eavg, const float Fss) {
	const float Fms = Fss * Eavg / fmax(1.f - Fss * (1.f - Eavg), 1e-4f);

	return 1.f + Fms * (1.f - E) / fmax(E, 1e-4f);
}
"""


if __name__ == "__main__":
    main()
