#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Authors (see AUTHORS.txt)
#
# SPDX-License-Identifier: Apache-2.0

"""BN: RMSE after OIDN at low spp, blue-noise pixel decorrelation on and off.

python tests/benchmark/bn_oidn.py <cfg> <size WxH> <out dir>
Renders a 1024 spp reference (no denoiser), then 1/4/16 spp with
sampler.sobol.bluenoise.enable 0 and 1 (two seeds each), raw and denoised.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pyluxcore

REPO = Path(r"C:\Users\Ricardo\Documents\GitHub\LuxCore")


def render(cfg, size, spp, seed, bluenoise, denoise):
    props = pyluxcore.Properties(str(cfg))
    devices = pyluxcore.GetOpenCLDeviceList()
    select = "".join("1" if d[1] == "HIP_GPU" else "0" for d in devices)
    w, h = size.split("x")
    props.Set(pyluxcore.Properties().SetFromString(f"""
        renderengine.type = PATHOCL
        sampler.type = SOBOL
        sampler.sobol.adaptive.strength = 0
        sampler.sobol.bluenoise.enable = {bluenoise}
        film.opencl.enable = 0
        opencl.native.threads.count = 0
        opencl.devices.select = {select}
        batch.halttime = 0
        batch.haltthreshold = 0
        batch.haltspp = {spp}
        renderengine.seed = {seed}
        film.width = {w}
        film.height = {h}
        """))
    # One pipeline: the denoiser (with its albedo and normal AOVs) or nothing
    for k in list(props.GetAllNames("film.imagepipelines")) + list(props.GetAllNames("film.outputs")):
        props.Delete(k)
    pipe = "film.imagepipelines.0.0.type = INTEL_OIDN\n" if denoise else "film.imagepipelines.0.0.type = NOP\n"
    props.Set(pyluxcore.Properties().SetFromString(pipe + """
        film.outputs.0.type = RGB_IMAGEPIPELINE
        film.outputs.0.index = 0
        film.outputs.0.filename = x.exr
        film.outputs.1.type = ALBEDO
        film.outputs.1.filename = a.exr
        film.outputs.2.type = AVG_SHADING_NORMAL
        film.outputs.2.filename = n.exr
        """))
    pyluxcore.ClearFileNameResolverPaths()
    pyluxcore.AddFileNameResolverPath(str(cfg.parent))
    session = pyluxcore.RenderSession(pyluxcore.RenderConfig(props))
    session.Start()
    session.WaitForDone()
    session.Stop()
    film = session.GetFilm()
    img = np.zeros(film.GetWidth() * film.GetHeight() * 3, dtype=np.float32)
    film.GetOutputFloat(pyluxcore.FilmOutputType.RGB_IMAGEPIPELINE, img, 0)
    return img.reshape(film.GetHeight(), film.GetWidth(), 3)


def main():
    cfg = Path(sys.argv[1])
    cfg = cfg if cfg.is_absolute() else REPO / cfg
    size, out = sys.argv[2], Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    pyluxcore.Init(lambda m: None)
    os.chdir(REPO)
    ref_file = out / "ref.npy"
    if not ref_file.exists():
        np.save(ref_file, render(cfg, size, 1024, 99, 1, False))
    ref = np.load(ref_file)
    # Relative RMSE on a tone-mapped-like scale (x / (1 + x)), robust to fireflies
    tm = lambda x: np.clip(x, 0, None) / (1 + np.clip(x, 0, None))
    for spp in (1, 4, 16):
        for denoise in (False, True):
            for bn in (0, 1):
                errs = []
                for seed in (1, 2):
                    img = render(cfg, size, spp, seed, bn, denoise)
                    errs.append(np.sqrt(np.mean((tm(img) - tm(ref)) ** 2)))
                print(f"{cfg.stem} {spp:3d} spp {'oidn' if denoise else 'raw '} bluenoise {bn}: "
                      f"RMSE {np.mean(errs):.5f} ({errs[0]:.5f}/{errs[1]:.5f})", flush=True)


if __name__ == "__main__":
    main()
