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

"""Render the material test scenes with one engine for a CPU/GPU comparison.

  parity.py <ENGINE> <out.npz> [scene ...] [spp=512] [devices=01] [k=v ...]

Renders pyunittests/resources/scenes/simple/mat-<scene>-area.cfg (every
mat-*-area scene when none is given) at the size of the suite (150x100), SOBOL
without adaptivity, a fixed seed and a NOP pipeline, and saves the RGB image of
each scene in <out.npz>. Compare two runs with paritycmp.py.

PATHOCL runs with opencl.task.count = 4096: with the default 512K tasks the
small film stops with most paths in flight and is biased dark (see "Known open
issues" in MODERNIZATION_PLAN.md).

Run it from the repository root (the scenes use paths relative to it).
"""

import glob
import os
import sys
import time

import numpy as np
import pyluxcore

SCENE_DIR = "pyunittests/resources/scenes/simple"


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    engine, out = sys.argv[1], sys.argv[2]
    spp, devices, scenes, kv = 512, "01", [], []
    for a in sys.argv[3:]:
        if a.startswith("spp="):
            spp = int(a[4:])
        elif a.startswith("devices="):
            devices = a[8:]
        elif "=" in a:
            k, v = a.split("=", 1)
            kv.append((k.strip(), v.strip()))
        else:
            scenes.append(a)
    if not scenes:
        scenes = sorted(os.path.basename(p)[4:-9] for p in glob.glob(f"{SCENE_DIR}/mat-*-area.cfg"))

    pyluxcore.Init(lambda s: None)
    results = {}
    for name in scenes:
        cfg_file = f"{SCENE_DIR}/mat-{name}-area.cfg"
        props = pyluxcore.Properties(cfg_file)
        props.Set(pyluxcore.Properties().SetFromString(f"""
renderengine.type = {engine}
renderengine.seed = 5
sampler.type = SOBOL
sampler.sobol.adaptive.strength = 0
film.width = 150
film.height = 100
film.imagepipelines.0.0.type = NOP
batch.haltspp = {spp}
batch.halttime = 0
batch.haltthreshold = -1
opencl.cpu.use = 0
opencl.gpu.use = 1
opencl.devices.select = {devices}
opencl.native.threads.count = 0
opencl.task.count = 4096
"""))
        for k, v in kv:
            props.Set(pyluxcore.Property(k, v))
        t0 = time.time()
        config = pyluxcore.RenderConfig(props)
        session = pyluxcore.RenderSession(config)
        session.Start()
        while not session.HasDone():
            time.sleep(0.05)
            session.UpdateStats()
        session.Stop()
        film = session.GetFilm()
        img = np.zeros((film.GetHeight(), film.GetWidth(), 3), np.float32)
        film.GetOutputFloat(pyluxcore.FilmOutputType.RGB, img)
        results[name] = img
        print(f"{name:28s} {engine:12s} mean {img.mean():.5f}  nan {int(np.isnan(img).sum())}"
              f"  ({time.time() - t0:.0f}s)", flush=True)
    np.savez_compressed(out, **results)
    pyluxcore.SetLogHandler(None)


if __name__ == "__main__":
    main()
