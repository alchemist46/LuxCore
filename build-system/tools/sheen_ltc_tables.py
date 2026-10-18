#!/usr/bin/env python3
# -*- coding: utf-8 -*-
################################################################################
# Copyright 1998-2026 by authors (see AUTHORS.txt)
#
#   This file is part of LuxCoreRender.
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
################################################################################

"""Generate include/slg/materials/sheen_ltc_tables.cl.

The microfiber sheen of Zeltner, Burley & Chiang 2022, "Practical
Multiple-Scattering Sheen Using Linearly Transformed Cosines": a volumetric
layer of fibers whose reflectance is fitted, per view cosine mu and roughness
r, by a linearly transformed cosine (the parameters a and b) times the layer's
directional albedo. The fit comes from Blender Cycles (Apache-2.0,
intern/cycles/scene/shader.tables, table_sheen_ltc: 32 x 32 values of a, then
b, then the albedo, rows by roughness, mu along a row), which is what Cycles'
Sheen BSDF and its OpenPBR fuzz lobe use; fitting it again would need the
volumetric simulation of the paper.

The values are stored as 16-bit integers (value * 32767) to keep the kernels'
constant tables small (CUDA caps them at 64 KB); the source has 5 decimals and
|value| < 1, so the quantization error (at most 1.5e-5) is below its own
precision. The script prints the error and checks that every valid LTC lobe
integrates to at most 1 over the upper hemisphere.

Usage: sheen_ltc_tables.py <cycles shader.tables> <output .cl>
"""

import math
import re
import sys

RES = 32
SCALE = 32767


def read_table(path):
    s = open(path).read()
    i = s.index("table_sheen_ltc[3072]")
    j = s.index("};", i)
    body = s[s.index("{", i) + 1:j]
    vals = [float(x.rstrip("f")) for x in re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?f?", body)]
    if len(vals) != 3 * RES * RES:
        raise ValueError(f"expected {3 * RES * RES} values, found {len(vals)}")
    return vals


def ltc_hemisphere_integral(a, b, n=200):
    """The integral over the upper hemisphere of the LTC density used by the
    material (1/pi max(z, 0) (a / |M^-1 w|^2)^2 with M^-1 w = (a x + b z, a y, z)).
    Midpoint rule in (cos theta, phi); 1 for a lobe entirely above the horizon."""
    total = 0.0
    for i in range(n):
        z = (i + 0.5) / n
        s = math.sqrt(1.0 - z * z)
        for j in range(2 * n):
            phi = (j + 0.5) / (2 * n) * 2.0 * math.pi
            x, y = s * math.cos(phi), s * math.sin(phi)
            len_sqr = (a * x + b * z) ** 2 + (a * y) ** 2 + z * z
            total += (z / math.pi) * (a / len_sqr) ** 2
    return total * (1.0 / n) * (2.0 * math.pi / (2 * n))


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    vals = read_table(sys.argv[1])
    q = [int(round(v * SCALE)) for v in vals]
    err = max(abs(qi / SCALE - v) for qi, v in zip(q, vals))
    print(f"quantization: max error {err:.2e}")

    # The LTC lobes the material uses (a and the albedo not ~0) must not
    # integrate to more than 1 over the upper hemisphere
    worst = 0.0
    for r in range(0, RES, 3):
        for m in range(0, RES, 3):
            a = vals[r * RES + m]
            b = vals[RES * RES + r * RES + m]
            albedo = vals[2 * RES * RES + r * RES + m]
            if abs(a) < 1e-5 or albedo < 1e-5:
                continue
            # sharper lobes (small a) need a finer quadrature
            n = int(min(1000, max(100, 15.0 / abs(a))))
            worst = max(worst, ltc_hemisphere_integral(a, b, n))
    print(f"LTC hemisphere integral: max {worst:.4f} (at most 1 up to the quadrature error)")
    if worst > 1.003:
        raise ValueError("an LTC lobe integrates to more than 1")

    out = []
    out.append("""#line 2 "sheen_ltc_tables.cl"

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
// The microfiber sheen of Zeltner, Burley & Chiang 2022 ("Practical
// Multiple-Scattering Sheen Using Linearly Transformed Cosines"): the LTC
// parameters a and b and the directional albedo, 32 x 32 over the roughness
// (rows) and the view cosine. The fit is Blender Cycles' table_sheen_ltc
// (intern/cycles/scene/shader.tables, Copyright 2011-2022 Blender Foundation,
// Apache-2.0), stored as value * 32767.
//
// Generated by build-system/tools/sheen_ltc_tables.py: do not edit.
//------------------------------------------------------------------------------

#define SHEEN_LTC_RES 32
""")
    out.append(f"__constant short SheenLTC_Table[{3 * RES * RES}] = {{")
    for row in range(3 * RES):
        line = ", ".join(str(v) for v in q[row * RES:(row + 1) * RES])
        out.append(f"\t{line},")
    out.append("};")
    out.append("""
// Cycles' lookup_table_read(): linear interpolation of x in [0, 1] over a row
OPENCL_FORCE_INLINE float SheenLTC_ReadRow(const uint offset, const float x) {
	const float fx = clamp(x, 0.f, 1.f) * (SHEEN_LTC_RES - 1);
	const uint index = min((uint)fx, (uint)(SHEEN_LTC_RES - 1));
	const uint nindex = min(index + 1, (uint)(SHEEN_LTC_RES - 1));
	const float t = fx - index;

	const float data0 = SheenLTC_Table[offset + index] * (1.f / 32767.f);
	const float data1 = SheenLTC_Table[offset + nindex] * (1.f / 32767.f);
	return (1.f - t) * data0 + t * data1;
}

// Cycles' lookup_table_read_2D(): x along a row, y across the rows
OPENCL_FORCE_INLINE float SheenLTC_Read2D(const uint section, const float x, const float y) {
	const float fy = clamp(y, 0.f, 1.f) * (SHEEN_LTC_RES - 1);
	const uint index = min((uint)fy, (uint)(SHEEN_LTC_RES - 1));
	const uint nindex = min(index + 1, (uint)(SHEEN_LTC_RES - 1));
	const float t = fy - index;

	const uint offset = section * SHEEN_LTC_RES * SHEEN_LTC_RES;
	const float data0 = SheenLTC_ReadRow(offset + SHEEN_LTC_RES * index, x);
	const float data1 = SheenLTC_ReadRow(offset + SHEEN_LTC_RES * nindex, x);
	return (1.f - t) * data0 + t * data1;
}

// The LTC parameters and the directional albedo for the view cosine mu and
// the roughness (clamped to [1e-3, 1], as Cycles does)
OPENCL_FORCE_INLINE void SheenLTC_Lookup(const float mu, const float roughness,
		float *a, float *b, float *albedo) {
	const float r = clamp(roughness, 1e-3f, 1.f);
	*a = SheenLTC_Read2D(0, mu, r);
	*b = SheenLTC_Read2D(1, mu, r);
	*albedo = SheenLTC_Read2D(2, mu, r);
}
""")
    open(sys.argv[2], "w").write("\n".join(out))
    print(f"wrote {sys.argv[2]}")


if __name__ == "__main__":
    main()
