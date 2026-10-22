#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Authors (see AUTHORS.txt)
#
# SPDX-License-Identifier: Apache-2.0

"""CPU/GPU parity, one scene element at a time (the elements of the working
colour space test scene: Fresnel metals, dispersive glass with a film, an image
map, sun, sky2, light and emission temperatures).

cpugpu.py OUTDIR [cases=a,b] [spp=256] [size=96] [cfg=k = v;k = v]: renders each case with
PATHCPU and PATHOCL (HIP), two seeds each, and prints the mean per channel of
each engine and the GPU/CPU ratio with its noise (from the seed spread).
"""
import math
import os
import sys
import time

import numpy as np
import pyluxcore

outdir = sys.argv[1]
opts = dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a)
SPP = int(opts.get("spp", 256))
N = int(opts.get("size", 96))
os.makedirs(outdir, exist_ok=True)


def image_file():
    """A saturated 8-bit gradient for the image map case."""
    path = os.path.join(outdir, "grad.png").replace(os.sep, "/")
    if not os.path.exists(path):
        from PIL import Image
        x = np.linspace(0, 1, 64)
        img = np.stack(list(np.meshgrid(x, x)) + [np.full((64, 64), 0.3)], axis=2)
        Image.fromarray((img * 255).astype(np.uint8)).save(path)
    return path


# The floor and a sphere; each case sets the sphere's material and the lights
SKY = "scene.lights.env.type = constantinfinite\nscene.lights.env.color = 0.5 0.5 0.5\n"
CASES = {
    "gold": SKY + "scene.textures.gold.type = fresnelpreset\nscene.textures.gold.name = gold\n"
            "scene.materials.s.type = metal2\nscene.materials.s.fresnel = gold\n"
            "scene.materials.s.uroughness = 0.2\nscene.materials.s.vroughness = 0.2\n",
    "fconst-red": SKY + "scene.textures.fc.type = fresnelconst\nscene.textures.fc.n = 0.0981 0.0981 0.0981\n"
            "scene.textures.fc.k = 3.3536 3.3536 3.3536\nscene.materials.s.type = metal2\nscene.materials.s.fresnel = fc\n"
            "scene.materials.s.uroughness = 0.2\nscene.materials.s.vroughness = 0.2\n",
    "fconst-green": SKY + "scene.textures.fc.type = fresnelconst\nscene.textures.fc.n = 0.3737 0.3737 0.3737\n"
            "scene.textures.fc.k = 2.526 2.526 2.526\nscene.materials.s.type = metal2\nscene.materials.s.fresnel = fc\n"
            "scene.materials.s.uroughness = 0.2\nscene.materials.s.vroughness = 0.2\n",
    "fconst-blue": SKY + "scene.textures.fc.type = fresnelconst\nscene.textures.fc.n = 1.5706 1.5706 1.5706\n"
            "scene.textures.fc.k = 1.7678 1.7678 1.7678\nscene.materials.s.type = metal2\nscene.materials.s.fresnel = fc\n"
            "scene.materials.s.uroughness = 0.2\nscene.materials.s.vroughness = 0.2\n",
    "cauchymetal": SKY + "scene.textures.fc.type = fresnelcauchy\nscene.textures.fc.a = 1.5\n"
            "scene.textures.fc.b = 0.01\nscene.materials.s.type = metal2\nscene.materials.s.fresnel = fc\n"
            "scene.materials.s.uroughness = 0.2\nscene.materials.s.vroughness = 0.2\n",
    "glassdispfilm": SKY + "scene.materials.s.type = glass\nscene.materials.s.interiorior = 1.5\n"
            "scene.materials.s.cauchyb = 0.01\nscene.materials.s.filmthickness = 400\n"
            "scene.materials.s.filmior = 1.33\n",
    "imagemap": SKY + "scene.textures.im.type = imagemap\nscene.textures.im.file = {img}\n"
            "scene.textures.im.gamma = 2.2\nscene.materials.s.type = matte\nscene.materials.s.kd = im\n",
    "sun": "scene.lights.sun.type = sun\nscene.lights.sun.dir = 0.3 -0.4 1\nscene.lights.sun.gain = 0.01 0.01 0.01\n"
            "scene.materials.s.type = matte\nscene.materials.s.kd = 0.6 0.6 0.6\n",
    "sky2": "scene.lights.sky.type = sky2\nscene.lights.sky.dir = 0.3 -0.4 0.6\nscene.lights.sky.gain = 0.0001 0.0001 0.0001\n"
            "scene.materials.s.type = matte\nscene.materials.s.kd = 0.6 0.6 0.6\n",
    "pointtemp": "scene.lights.pt.type = point\nscene.lights.pt.position = 1.5 -1.5 2.5\n"
            "scene.lights.pt.temperature = 3000\nscene.lights.pt.gain = 20 20 20\n"
            "scene.materials.s.type = matte\nscene.materials.s.kd = 0.6 0.6 0.6\n",
    "emittemp": "scene.materials.s.type = matte\nscene.materials.s.kd = 0.6 0.6 0.6\n"
            "scene.materials.lamp.type = matte\nscene.materials.lamp.kd = 0 0 0\n"
            "scene.materials.lamp.emission = 5 5 5\nscene.materials.lamp.emission.temperature = 3000\n"
            "scene.objects.lamp.shape = lampquad\nscene.objects.lamp.material = lamp\n",
}


def sphere(n_theta=32, n_phi=64):
    P, F = [], []
    for i in range(n_theta + 1):
        th = math.pi * i / n_theta
        for j in range(n_phi + 1):
            ph = 2 * math.pi * j / n_phi
            P.append((math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), 1 + math.cos(th)))
    row = n_phi + 1
    for i in range(n_theta):
        for j in range(n_phi):
            a = i * row + j
            F.append((a, a + row, a + row + 1))
            F.append((a, a + row + 1, a + 1))
    return P, F


def render(engine, case, seed):
    scene = pyluxcore.Scene()
    P, F = sphere()
    scene.DefineMesh("sph", P, F, P, None, None, None)
    scene.DefineMesh("floor", [(-6, -6, 0), (6, -6, 0), (6, 6, 0), (-6, 6, 0)], [(0, 1, 2), (0, 2, 3)],
                     None, None, None, None)
    # A lamp facing down, above the sphere
    scene.DefineMesh("lampquad", [(-1, -1, 4), (-1, 1, 4), (1, 1, 4), (1, -1, 4)], [(0, 1, 2), (0, 2, 3)],
                     None, None, None, None)
    uv = [((p[0] + 1) / 2, (p[2]) / 2) for p in P]
    scene.DefineMesh("sphuv", P, F, P, uv, None, None)
    sp = pyluxcore.Properties()
    sp.SetFromString("""
scene.camera.lookat.orig = 0 -7 3
scene.camera.lookat.target = 0 0 0.8
scene.camera.fieldofview = 40
scene.materials.floor.type = matte
scene.materials.floor.kd = 0.5 0.5 0.5
scene.objects.floor.shape = floor
scene.objects.floor.material = floor
scene.objects.sph.material = s
""" + CASES[case].format(img=image_file()) +
                     ("scene.objects.sph.shape = sphuv\n" if case == "imagemap" else "scene.objects.sph.shape = sph\n"))
    scene.Parse(sp)
    cp = pyluxcore.Properties()
    cp.SetFromString(f"""
renderengine.type = {engine}
renderengine.seed = {seed}
sampler.type = SOBOL
film.width = {N}
film.height = {N}
batch.haltspp = {SPP}
path.pathdepth.total = 8
path.pathdepth.diffuse = 8
path.pathdepth.glossy = 8
path.pathdepth.specular = 8
lightstrategy.lighttree.learning.enable = 0
path.clamping.variance.maxvalue = 0
opencl.devices.select = {select}
film.opencl.enable = 0
opencl.cpu.use = 0
opencl.native.threads.count = 0
""")
    # cfg=a=b;c=d: extra render configuration properties
    if opts.get("cfg"):
        cp.SetFromString(opts["cfg"].replace(";", "\n"))
    session = pyluxcore.RenderSession(pyluxcore.RenderConfig(cp, scene))
    session.Start()
    while not session.HasDone():
        time.sleep(0.05)
        session.UpdateStats()
    session.Stop()
    img = np.zeros((N, N, 3), np.float32)
    session.GetFilm().GetOutputFloat(pyluxcore.FilmOutputType.RGB, img)
    return img


pyluxcore.Init(lambda s: None)
devices = pyluxcore.GetOpenCLDeviceList()
select = "".join("1" if d[1] == "HIP_GPU" else "0" for d in devices)
cases = opts["cases"].split(",") if "cases" in opts else list(CASES)
print("%-14s %-26s %-26s %-22s %s" % ("case", "PATHCPU mean RGB", "PATHOCL mean RGB", "GPU/CPU", "noise (1 sigma)"), flush=True)
for case in cases:
    m = {}
    for e in ("PATHCPU", "PATHOCL"):
        m[e] = np.array([render(e, case, s).reshape(-1, 3).mean(axis=0) for s in (1, 2)], np.float64)
    c, g = m["PATHCPU"].mean(axis=0), m["PATHOCL"].mean(axis=0)
    ratio = g / c
    # The noise of each mean from the two seeds (std of the mean of 2 = |a-b|/2)
    sc = np.abs(m["PATHCPU"][0] - m["PATHCPU"][1]) / 2 / c
    sg = np.abs(m["PATHOCL"][0] - m["PATHOCL"][1]) / 2 / g
    noise = np.sqrt(sc ** 2 + sg ** 2)
    print("%-14s %-26s %-26s %-22s %s" % (case, " ".join("%.5f" % v for v in c), " ".join("%.5f" % v for v in g),
          " ".join("%.4f" % v for v in ratio), " ".join("%.4f" % v for v in noise)), flush=True)
