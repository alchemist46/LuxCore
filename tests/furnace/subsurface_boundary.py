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

"""The albedo of the subsurface walk under a dielectric boundary.

The color of the walk (subsurface.color of openpbr, color of the subsurface
material) is the fraction of the light entering a semi-infinite slab of the
medium that comes out again (the OpenPBR definition), whatever the IOR of the
boundary, which reflects the walk back into the medium
(SSSWalk_InternalReflection(), SSS_BoundaryColor() in subsurface_funcs.cl).

A box 2000 mean free paths deep under the furnace's uniform white environment,
its face seen head-on: the pixel value is d = E + (1 - E) P, with E the
reflection of the boundary (the coating or the specular layer, energy
compensated) and P the albedo of the walk, so P = (d(C) - d(0)) / (1 - d(0)).
The script checks that P at each eta is P at eta 1 (the index-matched
boundary, where the van de Hulst mapping sets the albedo) within TOLERANCE,
for both materials.

Usage:
    python subsurface_boundary.py <engine> [spp=256] [size=32] [devices=01]
        [etas=1.33,1.5,2.0,3.0] [colors=0.2,0.5,0.8,0.9] [anisotropy=0]
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import furnace
import pyluxcore

TOLERANCE = 0.03
MFP = 0.001


def box(x0, x1, y0, y1, z0, z1):
    P, N, UV, F = [], [], [], []
    faces = [((0, -1, 0), [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)]),
             ((0, 1, 0), [(x0, y1, z0), (x0, y1, z1), (x1, y1, z1), (x1, y1, z0)]),
             ((-1, 0, 0), [(x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x0, y1, z0)]),
             ((1, 0, 0), [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)]),
             ((0, 0, -1), [(x0, y0, z0), (x0, y1, z0), (x1, y1, z0), (x1, y0, z0)]),
             ((0, 0, 1), [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)])]
    for n, vs in faces:
        b = len(P)
        P += vs
        N += [n] * 4
        UV += [(0, 0), (1, 0), (1, 1), (0, 1)]
        F += [(b, b + 1, b + 2), (b, b + 2, b + 3)]
    return P, F, N, UV


def material(kind, c, eta, g):
    if kind == "subsurface":
        return ("scene.materials.m.type = subsurface\n"
                f"scene.materials.m.color = {c} {c} {c}\n"
                f"scene.materials.m.radius = {MFP} {MFP} {MFP}\nscene.materials.m.scale = 1\n"
                f"scene.materials.m.anisotropy = {g}\n"
                f"scene.materials.m.ior = {eta}\nscene.materials.m.roughness = 0.3\n")
    return ("scene.materials.m.type = openpbr\nscene.materials.m.subsurface.weight = 1\n"
            f"scene.materials.m.subsurface.color = {c} {c} {c}\n"
            f"scene.materials.m.subsurface.radius = {MFP}\nscene.materials.m.subsurface.radiusscale = 1 1 1\n"
            f"scene.materials.m.subsurface.anisotropy = {g}\n"
            f"scene.materials.m.specular.weight = {1 if eta > 1 else 0}\n"
            f"scene.materials.m.specular.ior = {eta}\nscene.materials.m.specular.roughness = 0.3\n")


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(1)
    args = dict(a.split("=", 1) for a in sys.argv[2:])
    opts = {"engine": sys.argv[1], "filter": "", "spp": int(args.get("spp", 256)), "env": "inf",
            "size": int(args.get("size", 32)), "out": "", "devices": args.get("devices", "01"), "kv": []}
    etas = [float(v) for v in args.get("etas", "1.33,1.5,2.0,3.0").split(",")]
    colors = [float(v) for v in args.get("colors", "0.2,0.5,0.8,0.9").split(",")]
    g = float(args.get("anisotropy", 0))

    pyluxcore.Init(lambda s: None)
    slab = box(-3, 3, 0, 2, -3, 3)

    def albedos(kind, eta):
        d0 = furnace.analyze(furnace.render_case(opts, material(kind, 0, eta, g), slab))["disk"]
        return [(furnace.analyze(furnace.render_case(opts, material(kind, c, eta, g), slab))["disk"] - d0) /
                (1.0 - d0) for c in colors]

    print(f"{'material':11s} {'eta':>5s}  " + " ".join(f"{'P(' + str(c) + ')':>9s}" for c in colors) + "  result")
    failures = 0
    for kind in ("subsurface", "openpbr"):
        ref = albedos(kind, 1.0)
        print(f"{kind:11s} {1.0:5.2f}  " + " ".join(f"{p:9.4f}" for p in ref) + "  reference", flush=True)
        for eta in etas:
            ps = albedos(kind, eta)
            ok = all(abs(p - r) <= TOLERANCE * r for p, r in zip(ps, ref))
            failures += 0 if ok else 1
            print(f"{kind:11s} {eta:5.2f}  " + " ".join(f"{p:9.4f}" for p in ps) +
                  f"  {'ok' if ok else 'FAIL'}", flush=True)
    print(f"\n{failures} failing case(s)")
    pyluxcore.SetLogHandler(None)
    sys.exit(min(failures, 100))


if __name__ == "__main__":
    main()
