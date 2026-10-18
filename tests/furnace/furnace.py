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

"""White furnace of the materials.

A sphere (radius 1, smooth normals, UVs) seen by an orthographic camera under a
uniform white environment of radiance 1. With lossless parameters (every
reflectance and transmittance at its energy-conserving maximum) the sphere must
disappear: every pixel equals 1. A closed transmissive sphere is lossless too,
since the light that enters leaves again.

The script reports, per case: the mean over the disk r < 0.9 (the figure of the
earlier furnace runs), the mean of the pixels that are entirely on the sphere
in five view-cosine bands (the grazing band is where layered models fail),
the relative std of the pixels and the NaN count.

Environments (env=):
  inf       constantinfinite light: light sampling and BSDF sampling with MIS
  box       an inward-facing emissive box of radiance 1, light sampling + MIS
  boxnodls  the same box with its direct light sampling disabled: the
            BSDF-sampling-only estimator. inf/box and boxnodls must agree; a
            difference means Sample() and Evaluate() are inconsistent.

Usage:
  furnace.py <ENGINE> [filter] [spp=256] [env=inf] [size=96] [out=results.tsv]
             [devices=01] [k=v ...]

ENGINE is PATHCPU, BIDIRCPU, PATHOCL or TILEPATHOCL. "filter" selects the cases
whose name contains it, or any of several comma-separated substrings ("all" or
nothing for every case). Extra k=v pairs go to
the scene (scene.*) or to the render configuration. "--list" prints the cases.
"""

import math
import os
import sys
import time

import numpy as np
import pyluxcore

ROUGHS = [0.05, 0.2, 0.5, 1.0]

WHITE_FRESNEL = """
scene.textures.fr.type = fresnelconst
scene.textures.fr.n = 0.001 0.001 0.001
scene.textures.fr.k = 1000 1000 1000
"""


def rough(a, suffix=""):
    return (f"scene.materials.m.uroughness{suffix} = {a}\n"
            f"scene.materials.m.vroughness{suffix} = {a}\n")


def cases():
    """(name, material scene properties, expected furnace value or None)

    The same lossless parameter sets as tests/bsdf/bsdftest.cpp. The expected
    value is 1 wherever the model can be lossless; None where it cannot by
    construction (velvet, the default presets) and only the number is logged.
    """
    c = []
    c.append(("matte", "scene.materials.m.type = matte\nscene.materials.m.kd = 1 1 1\n", 1.0))
    for s in (0.0, 0.3, 0.7, 1.0):
        c.append((f"roughmatte-s{s}", "scene.materials.m.type = roughmatte\nscene.materials.m.kd = 1 1 1\n"
                  f"scene.materials.m.sigma = {s}\n", 1.0))
    c.append(("mirror", "scene.materials.m.type = mirror\nscene.materials.m.kr = 1 1 1\n", 1.0))
    c.append(("glass", "scene.materials.m.type = glass\nscene.materials.m.kr = 1 1 1\n"
              "scene.materials.m.kt = 1 1 1\nscene.materials.m.interiorior = 1.5\n", 1.0))
    c.append(("glass-disp", "scene.materials.m.type = glass\nscene.materials.m.kr = 1 1 1\n"
              "scene.materials.m.kt = 1 1 1\nscene.materials.m.interiorior = 1.5\n"
              "scene.materials.m.cauchyb = 0.01\n", 1.0))
    c.append(("archglass", "scene.materials.m.type = archglass\nscene.materials.m.kr = 1 1 1\n"
              "scene.materials.m.kt = 1 1 1\nscene.materials.m.interiorior = 1.5\n", 1.0))
    for a in ROUGHS:
        c.append((f"roughglass-a{a}", "scene.materials.m.type = roughglass\n"
                  "scene.materials.m.kr = 1 1 1\nscene.materials.m.kt = 1 1 1\n"
                  "scene.materials.m.interiorior = 1.5\n" + rough(a), 1.0))
        c.append((f"metal2-a{a}", WHITE_FRESNEL + "scene.materials.m.type = metal2\n"
                  "scene.materials.m.fresnel = fr\n" + rough(a), 1.0))
        c.append((f"glossy2-coat-a{a}", "scene.materials.m.type = glossy2\n"
                  "scene.materials.m.kd = 0 0 0\nscene.materials.m.ks = 1 1 1\n"
                  "scene.materials.m.index = 0 0 0\n" + rough(a), 1.0))
        c.append((f"glossy2-white-a{a}", "scene.materials.m.type = glossy2\n"
                  "scene.materials.m.kd = 1 1 1\nscene.materials.m.ks = 0.04 0.04 0.04\n"
                  "scene.materials.m.index = 0 0 0\n" + rough(a), 1.0))
        c.append((f"glossycoating-white-a{a}",
                  "scene.materials.b.type = matte\nscene.materials.b.kd = 1 1 1\n"
                  "scene.materials.m.type = glossycoating\nscene.materials.m.base = b\n"
                  "scene.materials.m.ks = 0.04 0.04 0.04\nscene.materials.m.index = 0 0 0\n"
                  + rough(a), 1.0))
        c.append((f"glossycoating-metal-a{a}", WHITE_FRESNEL +
                  "scene.materials.b.type = metal2\nscene.materials.b.fresnel = fr\n"
                  "scene.materials.b.uroughness = 0.3\nscene.materials.b.vroughness = 0.3\n"
                  "scene.materials.m.type = glossycoating\nscene.materials.m.base = b\n"
                  "scene.materials.m.ks = 0.04 0.04 0.04\nscene.materials.m.index = 0 0 0\n"
                  + rough(a), None))
        c.append((f"glossytranslucent-a{a}", "scene.materials.m.type = glossytranslucent\n"
                  "scene.materials.m.kd = 0.5 0.5 0.5\nscene.materials.m.kt = 1 1 1\n"
                  "scene.materials.m.ks = 0.04 0.04 0.04\nscene.materials.m.ks_bf = 0.04 0.04 0.04\n"
                  + rough(a) + rough(a, "_bf"), 1.0))
        dr = f"scene.materials.m.roughness = {math.sqrt(a):.4f}\n"
        c.append((f"disney-metal-a{a}", "scene.materials.m.type = disney\n"
                  "scene.materials.m.basecolor = 1 1 1\nscene.materials.m.metallic = 1\n"
                  "scene.materials.m.specular = 1\n" + dr, 1.0))
        c.append((f"disney-dielectric-a{a}", "scene.materials.m.type = disney\n"
                  "scene.materials.m.basecolor = 1 1 1\nscene.materials.m.metallic = 0\n"
                  "scene.materials.m.specular = 0.5\n" + dr, 1.0))
        c.append((f"disney-clearcoat-a{a}", "scene.materials.m.type = disney\n"
                  "scene.materials.m.basecolor = 1 1 1\nscene.materials.m.metallic = 1\n"
                  "scene.materials.m.specular = 1\nscene.materials.m.clearcoat = 1\n"
                  "scene.materials.m.clearcoatgloss = 0.5\n" + dr, 1.0))
    c.append(("carpaint-3lobes", "scene.materials.m.type = carpaint\nscene.materials.m.kd = 0 0 0\n"
              "scene.materials.m.ks1 = 0.333 0.333 0.333\nscene.materials.m.ks2 = 0.333 0.333 0.333\n"
              "scene.materials.m.ks3 = 0.334 0.334 0.334\n"
              "scene.materials.m.r1 = 1\nscene.materials.m.r2 = 1\nscene.materials.m.r3 = 1\n"
              "scene.materials.m.m1 = 0.25\nscene.materials.m.m2 = 0.45\nscene.materials.m.m3 = 0.7\n", 1.0))
    c.append(("carpaint-preset", "scene.materials.m.type = carpaint\n"
              "scene.materials.m.preset = \"ford f8\"\n", None))
    c.append(("cloth-diffuse", "scene.materials.m.type = cloth\nscene.materials.m.preset = denim\n"
              "scene.materials.m.weft_kd = 1 1 1\nscene.materials.m.warp_kd = 1 1 1\n"
              "scene.materials.m.weft_ks = 0 0 0\nscene.materials.m.warp_ks = 0 0 0\n", 1.0))
    c.append(("cloth-denim", "scene.materials.m.type = cloth\nscene.materials.m.preset = denim\n", None))
    for r in (0.2, 0.5, 1.0):
        c.append((f"velvet-r{r}", "scene.materials.m.type = velvet\nscene.materials.m.kd = 1 1 1\n"
                  f"scene.materials.m.roughness = {r}\n", None))
    c.append(("mattetranslucent", "scene.materials.m.type = mattetranslucent\n"
              "scene.materials.m.kr = 0.5 0.5 0.5\nscene.materials.m.kt = 1 1 1\n", 1.0))
    for s in (0.0, 0.5, 1.0):
        c.append((f"roughmattetranslucent-s{s}", "scene.materials.m.type = roughmattetranslucent\n"
                  "scene.materials.m.kr = 0.5 0.5 0.5\nscene.materials.m.kt = 1 1 1\n"
                  f"scene.materials.m.sigma = {s}\n", 1.0))
    c.append(("mix", WHITE_FRESNEL +
              "scene.materials.a.type = matte\nscene.materials.a.kd = 1 1 1\n"
              "scene.materials.b.type = metal2\nscene.materials.b.fresnel = fr\n"
              "scene.materials.b.uroughness = 0.3\nscene.materials.b.vroughness = 0.3\n"
              "scene.materials.m.type = mix\nscene.materials.m.material1 = a\n"
              "scene.materials.m.material2 = b\nscene.materials.m.amount = 0.5\n", 1.0))
    c.append(("twosided", "scene.materials.a.type = matte\nscene.materials.a.kd = 1 1 1\n"
              "scene.materials.b.type = matte\nscene.materials.b.kd = 1 1 1\n"
              "scene.materials.m.type = twosided\nscene.materials.m.frontmaterial = a\n"
              "scene.materials.m.backmaterial = b\n", 1.0))
    c.append(("null", "scene.materials.m.type = null\n", 1.0))
    return c


def uv_sphere(n_theta=64, n_phi=128):
    """Smooth UV sphere of radius 1 with normals and UVs (cloth needs UVs)."""
    P, N, UV, F = [], [], [], []
    for i in range(n_theta + 1):
        th = math.pi * i / n_theta
        for j in range(n_phi + 1):
            ph = 2.0 * math.pi * j / n_phi
            n = (math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th))
            P.append(n)
            N.append(n)
            UV.append((j / n_phi, i / n_theta))
    row = n_phi + 1
    for i in range(n_theta):
        for j in range(n_phi):
            a = i * row + j
            b = a + 1
            c = a + row
            d = c + 1
            F.append((a, c, d))
            F.append((a, d, b))
    return P, F, N, UV


def parse_args(argv):
    if len(argv) < 2 or argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(1)
    if argv[1] == "--list":
        for name, _, expect in cases():
            print(f"{name:32s} expect {expect}")
        sys.exit(0)
    opts = {"engine": argv[1], "filter": "", "spp": 256, "env": "inf", "size": 96,
            "out": "", "devices": "01", "kv": []}
    for a in argv[2:]:
        if "=" not in a:
            opts["filter"] = "" if a == "all" else a
            continue
        k, v = a.split("=", 1)
        if k in ("spp", "size"):
            opts[k] = int(v)
        elif k in ("env", "out", "devices"):
            opts[k] = v
        else:
            opts["kv"].append((k.strip(), v.strip()))
    return opts


def render_case(opts, mat_props, sphere):
    W = H = opts["size"]
    sp = pyluxcore.Properties()
    sp.SetFromString("""
scene.camera.type = orthographic
scene.camera.lookat.orig = 0 -5 0
scene.camera.lookat.target = 0 0 0
scene.camera.up = 0 0 1
scene.camera.screenwindow = -1.05 1.05 -1.05 1.05
scene.materials.sky.type = matte
scene.materials.sky.kd = 0 0 0
scene.materials.sky.emission = 1 1 1
""" + mat_props)
    env = opts["env"]
    if env == "inf":
        sp.SetFromString("scene.lights.env.type = constantinfinite\nscene.lights.env.color = 1 1 1\n")
    elif env in ("box", "boxnodls"):
        L = 60.0
        bv = [(-L, -L, -L), (L, -L, -L), (L, L, -L), (-L, L, -L),
              (-L, -L, L), (L, -L, L), (L, L, L), (-L, L, L)]
        bf = [(0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6), (0, 5, 1), (0, 4, 5),
              (1, 6, 2), (1, 5, 6), (2, 7, 3), (2, 6, 7), (3, 4, 0), (3, 7, 4)]
        sp.Set(pyluxcore.Property("scene.objects.sky.vertices", [x for t in bv for x in t]))
        sp.Set(pyluxcore.Property("scene.objects.sky.faces", [x for t in bf for x in t]))
        sp.Set(pyluxcore.Property("scene.objects.sky.material", "sky"))
        if env == "boxnodls":
            sp.Set(pyluxcore.Property("scene.materials.sky.emission.directlightsampling.type", "DISABLED"))
    else:
        raise ValueError(f"unknown env {env}")

    cp = pyluxcore.Properties()
    cp.SetFromString(f"""
renderengine.type = {opts['engine']}
renderengine.seed = 3
sampler.type = SOBOL
sampler.sobol.adaptive.strength = 0
film.width = {W}
film.height = {H}
film.filter.type = NONE
film.imagepipelines.0.0.type = NOP
batch.haltspp = {opts['spp']}
batch.halttime = 0
batch.haltthreshold = -1
path.pathdepth.total = 48
path.pathdepth.diffuse = 48
path.pathdepth.glossy = 48
path.pathdepth.specular = 48
path.maxdepth = 48
light.maxdepth = 48
path.russianroulette.depth = 100
path.clamping.variance.maxvalue = 0
path.hybridbackforward.enable = 0
opencl.cpu.use = 0
opencl.gpu.use = 1
opencl.devices.select = {opts['devices']}
opencl.native.threads.count = 0
opencl.task.count = 4096
""")
    for k, v in opts["kv"]:
        (sp if k.startswith("scene.") else cp).Set(pyluxcore.Property(k, v))

    scene = pyluxcore.Scene()
    P, F, N, UV = sphere
    scene.DefineMesh("sph", P, F, N, UV, None, None)
    scene.Parse(sp)
    scene.Parse(pyluxcore.Properties().SetFromString(
        "scene.objects.s.shape = sph\nscene.objects.s.material = m\n"))

    session = pyluxcore.RenderSession(pyluxcore.RenderConfig(cp, scene))
    session.Start()
    while not session.HasDone():
        time.sleep(0.05)
        session.UpdateStats()
    session.Stop()
    img = np.zeros((H, W, 3), np.float32)
    session.GetFilm().GetOutputFloat(pyluxcore.FilmOutputType.RGB, img)
    return img


def analyze(img):
    H, W, _ = img.shape
    g = img.mean(axis=2)
    yy, xx = np.mgrid[0:H, 0:W]
    px = 2.1 / W
    cx = (xx + 0.5) * px - 1.05
    cy = (yy + 0.5) * px - 1.05
    r = np.hypot(cx, cy)
    res = {"nan": int(np.isnan(g).sum())}
    disk = g[r < 0.9]
    res["disk"] = float(np.nanmean(disk))
    res["relstd"] = float(np.nanstd(disk) / max(res["disk"], 1e-9))
    # pixels entirely on the sphere, binned by the view cosine at their centre
    inside = (r + px * math.sqrt(0.5)) < 1.0
    mu = np.sqrt(np.clip(1.0 - r * r, 0.0, 1.0))
    edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0001]
    res["bands"] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = inside & (mu >= lo) & (mu < hi)
        res["bands"].append(float(np.nanmean(g[sel])) if sel.any() else float("nan"))
    return res


def main():
    opts = parse_args(sys.argv)
    pyluxcore.Init(lambda s: None)
    sphere = uv_sphere()

    out = open(opts["out"], "a") if opts["out"] else None
    header = (f"{'case':32s} {'engine':11s} {'env':8s} {'disk':>7s}  "
              f"{'mu<.2':>6s} {'<.4':>6s} {'<.6':>6s} {'<.8':>6s} {'<=1':>6s}  {'relstd':>6s}  result")
    print(header)
    failures = 0
    for name, props, expect in cases():
        if opts["filter"] and not any(f in name for f in opts["filter"].split(",")):
            continue
        t0 = time.time()
        img = render_case(opts, props, sphere)
        r = analyze(img)
        if expect is None:
            verdict = "logged"
        else:
            tol = 0.01 if any(k in name for k in ("glass", "translucent")) else 0.005
            ok = abs(r["disk"] - expect) <= tol and r["nan"] == 0
            verdict = "ok" if ok else "FAIL"
            failures += 0 if ok else 1
        bands = " ".join(f"{b:6.3f}" for b in r["bands"])
        line = (f"{name:32s} {opts['engine']:11s} {opts['env']:8s} {r['disk']:7.4f}  {bands}  "
                f"{r['relstd']:6.3f}  {verdict}{'' if r['nan'] == 0 else ' nan=' + str(r['nan'])}"
                f"  ({time.time() - t0:.0f}s)")
        print(line, flush=True)
        if out:
            out.write("\t".join([name, opts["engine"], opts["env"], str(opts["spp"]),
                                 f"{r['disk']:.5f}"] + [f"{b:.5f}" for b in r["bands"]] +
                                [f"{r['relstd']:.4f}", str(r["nan"]), verdict]) + "\n")
            out.flush()
    if out:
        out.close()
    print(f"\n{failures} failing case(s)")
    # Avoid the exit-time crash with a Python log handler (known open issue)
    pyluxcore.SetLogHandler(None)
    sys.exit(min(failures, 100))


if __name__ == "__main__":
    main()
