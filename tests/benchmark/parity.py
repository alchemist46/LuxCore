#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Authors (see AUTHORS.txt)
#
# SPDX-License-Identifier: Apache-2.0

"""Equal-spp parity renders for the bit-identical changes (ROADMAP.md,
"Bit-identical items").

Renders each case with TILEPATHOCL at a fixed seed and spp per pixel (aa 8,
one pass, no light tree learning, no variance clamping: both depend on the
timing of the run) on the HIP device, and saves the float RGB output as .npy.
Then compare the directories of two builds:

    # The previous build, from its own venv
    out/venvbase/Scripts/python tests/benchmark/parity.py render out/parity/base
    # The new one
    out/venvtest/Scripts/python tests/benchmark/parity.py render out/parity/new
    python tests/benchmark/parity.py compare out/parity/base out/parity/new

A case is identical when the largest difference of a value relative to the
value (over the values above 1e-3 of the image mean) is under 1e-5. Two runs of
the same build differ by up to 1e-6: the samples of a pixel reach the film
through atomic adds in a different order. A real change moves pixels by 1e-3
and more.
"""

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# name: (scene .cfg, extra render config properties, extra scene properties)
CASES = {
    "cornell": ("scenes/cornell/cornell.cfg", "", ""),
    "luxball-hdr": ("scenes/luxball/luxball-hdr.cfg", "", ""),
    "classroom": ("scenes/classroom/classroom.cfg", "", ""),
    "kitchen": ("scenes/kitchen/kitchen.cfg", "", ""),
    "luxball-hdr-vol": ("scenes/luxball/luxball-hdr-vol.cfg", "", ""),
    "cornell-vol": ("scenes/cornell/cornell-vol.cfg", "", ""),
    "transparencyball": ("scenes/luxball/transparencyball.cfg", "", ""),
    "cornell-invis": ("scenes/cornell/cornell-invis.cfg", "", ""),
    # The Cornell box with a shadow-catcher floor
    "cornell-shadowcatcher": ("scenes/cornell/cornell.cfg", "",
                              "scene.materials.Khaki.shadowcatcher.enable = 1"),
    "luxball-shadowcatcher": ("scenes/luxball/luxball-shadowcatcher.cfg", "", ""),
    # Textures and the material interpreter
    "texball": ("scenes/luxball/texball.cfg", "", ""),
    "mix1ball": ("scenes/luxball/mix1ball.cfg", "", ""),
    "imagemap": ("scenes/luxball/luxball-imagemap-speed.cfg", "", ""),
    "glossycoat": ("scenes/luxball/luxball-glossycoat.cfg", "", ""),
}

PROPS = """
renderengine.type = TILEPATHOCL
renderengine.seed = 1
sampler.type = TILEPATHSAMPLER
tilepath.sampling.aa.size = 8
tile.multipass.enable = 0
tilepath.multipass.enable = 0
lightstrategy.lighttree.learning.enable = 0
path.clamping.variance.maxvalue = 0
batch.halttime = 0
batch.haltspp = 0
batch.haltthreshold = 0
film.opencl.enable = 0
opencl.native.threads.count = 0
opencl.cpu.use = 0
opencl.gpu.use = 1
"""


def render(out, cases, size, defines):
    import numpy as np
    import pyluxcore

    pyluxcore.Init(lambda msg: None)
    devices = pyluxcore.GetOpenCLDeviceList()
    select = "".join("1" if d[1] == "HIP_GPU" else "0" for d in devices)
    if "1" not in select:
        sys.exit("no HIP device")

    out.mkdir(parents=True, exist_ok=True)
    os.chdir(REPO_ROOT)
    for name in cases:
        cfg_file, cfg_extra, scene_extra = CASES[name] if name in CASES else (name, "", "")
        cfg = Path(cfg_file) if Path(cfg_file).is_absolute() else REPO_ROOT / cfg_file
        props = pyluxcore.Properties(str(cfg))
        props.Set(pyluxcore.Properties().SetFromString(PROPS))
        props.Set(pyluxcore.Properties().SetFromString(cfg_extra))
        props.Set(pyluxcore.Property("opencl.devices.select", select))
        props.Set(pyluxcore.Property("film.width", size[0]))
        props.Set(pyluxcore.Property("film.height", size[1]))
        for define in defines:
            props.Set(pyluxcore.Properties().SetFromString(define))

        pyluxcore.ClearFileNameResolverPaths()
        pyluxcore.AddFileNameResolverPath(str(cfg.parent))

        scene_props = pyluxcore.Properties(props.Get("scene.file").GetString())
        scene_props.Set(pyluxcore.Properties().SetFromString(scene_extra))
        scene = pyluxcore.Scene()
        scene.Parse(scene_props)

        start = time.perf_counter()
        session = pyluxcore.RenderSession(pyluxcore.RenderConfig(props, scene))
        session.Start()
        while not session.HasDone():
            time.sleep(0.1)
            session.UpdateStats()
        session.Stop()

        film = session.GetFilm()
        w, h = film.GetWidth(), film.GetHeight()
        img = np.zeros(w * h * 3, dtype=np.float32)
        film.GetOutputFloat(pyluxcore.FilmOutputType.RGB, img)
        np.save(out / f"{Path(name).stem}.npy", img.reshape(h, w, 3))
        print(f"{name}: {time.perf_counter() - start:.1f}s, mean {img.mean():.6f}", flush=True)


def compare(a, b, tolerance):
    import numpy as np

    status = 0
    for pa in sorted(a.glob("*.npy")):
        pb = b / pa.name
        if not pb.exists():
            print(f"{pa.stem}: missing in {b}")
            status = 1
            continue
        ia, ib = np.load(pa), np.load(pb)
        ia64, ib64 = ia.astype(np.float64), ib.astype(np.float64)
        mean = max(float(np.abs(ia64).mean()), 1e-30)
        diff = np.abs(ia64 - ib64)
        scale = np.maximum(np.abs(ia64), np.abs(ib64))
        significant = scale > 1e-3 * mean
        rel = float((diff[significant] / scale[significant]).max()) if significant.any() else 0.0
        changed = int((diff > 0).sum())
        flag = "identical" if rel < tolerance else "DIFFERENT"
        if rel >= tolerance:
            status = 1
        print(f"{pa.stem:<24} max relative diff {rel:.2e}, mean {ia.mean():.6f} -> {ib.mean():.6f} "
              f"({ib.mean() / max(ia.mean(), 1e-30) - 1:+.2e}), {changed} values differ: {flag}")
    return status


def main():
    parser = argparse.ArgumentParser(description="Equal-spp parity renders",
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render", help="render the cases to .npy files")
    r.add_argument("out", type=Path)
    r.add_argument("cases", nargs="*", help="case names or .cfg files (default: all): " + ", ".join(CASES))
    r.add_argument("--size", default="320x240", help="film size WxH (default: 320x240)")
    r.add_argument("-D", "--define", action="append", default=[],
                   help="additional render config property (i.e. -D 'path.maxdepth = 4')")
    c = sub.add_parser("compare", help="compare the renders of two builds")
    c.add_argument("a", type=Path)
    c.add_argument("b", type=Path)
    c.add_argument("--tolerance", type=float, default=1e-5,
                   help="largest relative difference of a value (default: 1e-5)")
    args = parser.parse_args()

    if args.cmd == "render":
        render(args.out.resolve(), args.cases or list(CASES),
               tuple(int(v) for v in args.size.split("x")), args.define)
    else:
        sys.exit(compare(args.a, args.b, args.tolerance))


if __name__ == "__main__":
    main()
