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

"""Generates include/slg/core/workingcolorspace_matrices.h: the matrices of the
working colour spaces (scene.colorspace.working = rec709 | rec2020 | acescg),
taken from Blender's config.ocio with OpenColorIO, so that LuxCore and Cycles
convert to the same primaries and white point to float precision (ACEScg goes
through OCIO's ACEScg_to_ACES2065-1 builtin, i.e. the ACES white point with
Blender's chromatic adaptation).

For every working space W the header holds:

- FromRec709[W]: Linear Rec.709 to W (the image maps, the gamma constants and
  the spectral paths, which compute Rec.709 values).
- ToRec709[W]: W to Linear Rec.709 (the standalone output).
- ToXYZ[W]: W to CIE XYZ D65; its second row holds the luminance weights.

Run it with an interpreter that has PyOpenColorIO, e.g. Blender's own:

    <blender>/5.2/python/bin/python3.13 build-system/tools/working_colorspace_matrices.py \\
        <blender>/5.2/datafiles/colormanagement/config.ocio
"""

import os
import sys

import PyOpenColorIO as OCIO

# (enum name, Blender's colour space name, interop id)
SPACES = [
    ("REC709", "Linear Rec.709", "lin_rec709_scene"),
    ("REC2020", "Linear Rec.2020", "lin_rec2020_scene"),
    ("ACESCG", "ACEScg", "lin_ap1_scene"),
]
XYZ_D65 = "Linear CIE-XYZ D65"

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "..", "include", "slg", "core", "workingcolorspace_matrices.h")

HEADER = """/***************************************************************************
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
"""


def matrix(cfg, src, dst):
    cpu = cfg.getProcessor(src, dst).getDefaultCPUProcessor()
    cols = [cpu.applyRGB([1.0 if i == j else 0.0 for i in range(3)]) for j in range(3)]
    return [[cols[c][r] for c in range(3)] for r in range(3)]


def source(path):
    # "blender-5.2.2-linux-x64/.../config.ocio" -> "Blender 5.2.2 config.ocio"
    for part in os.path.abspath(path).split(os.sep):
        if part.lower().startswith("blender-"):
            return "Blender " + part.split("-")[1] + " config.ocio"
    return os.path.basename(path)


def snap(m):
    # The identity (Rec.709 to Rec.709) must be exact
    return [[0.0 if abs(v) < 1e-12 else (1.0 if abs(v - 1.0) < 1e-12 else v) for v in row] for row in m]


def fmt(m, indent):
    rows = []
    for row in m:
        rows.append(indent + "{ " + ", ".join("%.9ef" % v for v in row) + " }")
    return ",\n".join(rows)


def table(name, mats, comment):
    out = ["// " + comment, "static const float %s[3][3][3] = {" % name]
    for (enum, csName, _), m in zip(SPACES, mats):
        out.append("\t// " + enum + " (" + csName + ")")
        out.append("\t{\n" + fmt(m, "\t\t") + "\n\t},")
    out.append("};")
    return "\n".join(out)


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    cfg = OCIO.Config.CreateFromFile(sys.argv[1])
    for _, csName, interopId in SPACES:
        cs = cfg.getColorSpace(csName)
        if cs is None:
            raise RuntimeError("Colour space not found in the config: " + csName)
        if cs.getInteropID() != interopId:
            raise RuntimeError("Unexpected interop id of " + csName + ": " + cs.getInteropID())

    fromRec709 = [snap(matrix(cfg, "Linear Rec.709", s[1])) for s in SPACES]
    toRec709 = [snap(matrix(cfg, s[1], "Linear Rec.709")) for s in SPACES]
    toXYZ = [matrix(cfg, s[1], XYZ_D65) for s in SPACES]

    with open(OUT, "w", newline="\n") as f:
        f.write(HEADER)
        f.write("""
// GENERATED by build-system/tools/working_colorspace_matrices.py, do not edit.
// Source: %s, OpenColorIO %s.
//
// Row-major matrices, applied as out = M * in, indexed by
// WorkingColorSpace::WorkingColorSpaceType.

#ifndef _SLG_WORKINGCOLORSPACE_MATRICES_H
#define\t_SLG_WORKINGCOLORSPACE_MATRICES_H

namespace slg {
namespace workingcolorspace {

""" % (source(sys.argv[1]), OCIO.__version__))
        f.write(table("FromRec709", fromRec709, "Linear Rec.709 to the working space") + "\n\n")
        f.write(table("ToRec709", toRec709, "The working space to Linear Rec.709") + "\n\n")
        f.write(table("ToXYZ", toXYZ, "The working space to CIE XYZ D65 (the second row holds the luminance weights)") + "\n\n")
        f.write("""}
}

#endif\t/* _SLG_WORKINGCOLORSPACE_MATRICES_H */
""")
    print("Written", os.path.normpath(OUT))


if __name__ == "__main__":
    main()
