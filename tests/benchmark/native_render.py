#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Authors (see AUTHORS.txt)
#
# SPDX-License-Identifier: Apache-2.0

"""Renders of the native LuxCore benchmark scenes, for A/B runs between
pyluxcore builds (CLAUDE.md, "Test scenes: LuxCore native only").

The scene is rendered from its own render.cfg (materials, lights, engine
settings and path depths as the author made them), with only what a
measurement needs changed:

- the film scaled by --scale; two outputs: the linear radiance (RGB, for
  the numbers) and the scene's own first image pipeline without its denoiser
  steps (the look, to compare with the scene's reference.png);
- a fixed spp halt and seed, no halt threshold, the film on the host;
- the adaptive sampling off unless --keep-adaptive (it moves samples between
  pixels, which hides the per-sample noise);
- the device: the HIP GPUs (PATHOCL), the OpenCL GPUs, or the CPU (PATHCPU,
  with --engine PATHCPU).

Each render is saved as <name>.npy (the linear radiance, float32 RGB, top
row first) and <name>-look.npy (the pipeline image, uint8 RGB), with a JSON
log: the setup time (scene parse, caches, kernel compile: the
RenderSession.Start() call), the render time and the engine's samples/s. The
script only uses the pyluxcore API of the 2.x releases, so the same file runs
with upstream pyluxcore 2.11, the pre-optimization build and HEAD.

    out/venvbaked/Scripts/python.exe tests/benchmark/native_render.py \\
        --scene LynxSpider --out out/native/head/LynxSpider --scale 0.5 --spp 64 256

Scenes: the names of scenes/<Name>/<Name>/LuxCoreScene/render.cfg, or a path
to any render.cfg.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pyluxcore

REPO_ROOT = Path(__file__).resolve().parents[2]
DENOISERS = {"INTEL_OIDN", "OPTIX_DENOISER", "BCD_DENOISER"}
SCENES = ("CannelleEtFromage", "DanishMood", "LuxCore2.1Benchmark", "LynxSpider", "OrangeJuice")

parser = argparse.ArgumentParser(prog="native_render.py")
parser.add_argument("--scene", required=True, help="scene name (%s) or a render.cfg" % ", ".join(SCENES))
parser.add_argument("--out", required=True, help="output directory")
parser.add_argument("--tag", default="run", help="prefix of the output files")
parser.add_argument("--spp", type=int, nargs="+", default=[64])
parser.add_argument("--repeats", type=int, default=2, help="renders per spp value, each with its own seed")
parser.add_argument("--seed", type=int, default=1, help="seed of the first render")
parser.add_argument("--scale", type=float, default=0.5, help="film size relative to the scene's")
parser.add_argument("--engine", help="render engine (default: the scene's)")
parser.add_argument("--device", choices=("hip", "opencl", "cpu"), default="hip",
                    help="GPU devices of PATHOCL; cpu = no GPU (use with --engine PATHCPU)")
parser.add_argument("--keep-adaptive", action="store_true", help="keep the scene's adaptive sampling")
parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="extra render config property (repeatable)")
parser.add_argument("--scene-set", action="append", default=[], metavar="KEY=VALUE",
                    help="scene property set after the scene file is read, e.g. to switch a light off (repeatable)")
args = parser.parse_args()

if args.scene in SCENES:
    cfg = REPO_ROOT / "scenes" / args.scene / args.scene / "LuxCoreScene" / "render.cfg"
else:
    cfg = Path(args.scene).resolve()
out = Path(args.out).resolve()
out.mkdir(parents=True, exist_ok=True)

log_file = open(out / (args.tag + ".luxcore.log"), "w", encoding="utf-8")
pyluxcore.Init(lambda msg: (log_file.write(msg + "\n"), log_file.flush()))


def device_selection(kind):
    devices = pyluxcore.GetOpenCLDeviceList()
    if kind == "hip":
        sel = ["1" if "HIP" in d[1] and "GPU" in d[1] else "0" for d in devices]
    else:
        sel = ["1" if d[1] == "OPENCL_GPU" else "0" for d in devices]
    if "1" not in sel:
        sys.exit(f"no {kind} device in {devices}")
    return "".join(sel), [d[0] for d, s in zip(devices, sel) if s == "1"]


def config_props(spp, seed):
    props = pyluxcore.Properties(str(cfg))
    w = max(1, round(props.Get("film.width").GetInt() * args.scale))
    h = max(1, round(props.Get("film.height").GetInt() * args.scale))
    # The first image pipeline (numbered 0 or 000), renamed 0, without its denoisers
    pipeline = pyluxcore.Properties()
    names = props.GetAllNamesRE(r"^film\.imagepipelines\.0+\.")
    skip = {n.split(".")[3] for n in names
            if n.endswith(".type") and props.Get(n).GetString() in DENOISERS}
    for n in names:
        parts = n.split(".")
        if parts[3] not in skip:
            pipeline.Set(pyluxcore.Property(".".join(parts[:2] + ["0"] + parts[3:]), props.Get(n).GetValues()))
    for prefix in ("film.imagepipelines.", "film.outputs.", "batch.haltthreshold", "periodicsave."):
        props.DeleteAll(props.GetAllNamesRE("^" + prefix.replace(".", r"\.")))
    props.Set(pyluxcore.Property("film.width", w))
    props.Set(pyluxcore.Property("film.height", h))
    props.Set(pipeline)
    props.Set(pyluxcore.Property("film.outputs.0.type", "RGB"))
    props.Set(pyluxcore.Property("film.outputs.0.filename", "native-rgb.exr"))
    props.Set(pyluxcore.Property("film.outputs.1.type", "RGB_IMAGEPIPELINE"))
    props.Set(pyluxcore.Property("film.outputs.1.index", 0))
    props.Set(pyluxcore.Property("film.outputs.1.filename", "native-look.png"))
    props.Set(pyluxcore.Property("film.opencl.enable", False))
    props.Set(pyluxcore.Property("batch.haltspp", spp))
    props.Set(pyluxcore.Property("batch.halttime", 0))
    props.Set(pyluxcore.Property("renderengine.seed", seed))
    if args.engine:
        props.Set(pyluxcore.Property("renderengine.type", args.engine))
    if not args.keep_adaptive:
        props.Set(pyluxcore.Property("sampler.sobol.adaptive.strength", 0.0))
    if args.device != "cpu":
        sel, names = device_selection(args.device)
        props.Set(pyluxcore.Property("opencl.devices.select", sel))
        props.Set(pyluxcore.Property("opencl.cpu.use", False))
        props.Set(pyluxcore.Property("opencl.gpu.use", True))
        props.Set(pyluxcore.Property("opencl.native.threads.count", 0))
        log["devices"] = names
    else:
        log["devices"] = [f"CPU, {os.cpu_count()} threads"]
    for kv in args.set:
        key, value = kv.split("=", 1)
        props.SetFromString(f"{key.strip()} = {value.strip()}")
    return props, w, h


log = {"scene": str(cfg), "pyluxcore": pyluxcore.Version(), "pyluxcore_file": pyluxcore.__file__,
       "scale": args.scale, "keep_adaptive": args.keep_adaptive, "set": args.set,
       "scene_set": args.scene_set, "runs": []}


def render(name, spp, seed):
    props, w, h = config_props(spp, seed)
    os.chdir(cfg.parent)
    t0 = time.perf_counter()
    if args.scene_set:
        # Parse() defines each object again from the given properties only:
        # start from the object's whole definition in the scene file
        scene_file = props.Get("scene.file").GetString()
        scene = pyluxcore.Scene(scene_file)
        scene_props = pyluxcore.Properties(scene_file)
        edits = pyluxcore.Properties()
        for kv in args.scene_set:
            key = kv.split("=", 1)[0].strip()
            edits.Set(scene_props.GetAllProperties(".".join(key.split(".")[:3]) + "."))
        for kv in args.scene_set:
            key, value = kv.split("=", 1)
            edits.SetFromString(f"{key.strip()} = {value.strip()}")
        scene.Parse(edits)
        config = pyluxcore.RenderConfig(props, scene)
    else:
        config = pyluxcore.RenderConfig(props)
    session = pyluxcore.RenderSession(config)
    session.Start()
    t1 = time.perf_counter()
    while True:
        time.sleep(0.02)
        session.UpdateStats()
        if session.HasDone():
            break
    t2 = time.perf_counter()
    stats = session.GetStats()
    film = session.GetFilm()
    # The film's first row is the bottom of the image
    img = np.zeros((h, w, 3), dtype=np.float32)
    film.GetOutputFloat(pyluxcore.FilmOutputType.RGB, img)
    img = img[::-1].copy()
    look = np.zeros((h, w, 3), dtype=np.float32)
    film.GetOutputFloat(pyluxcore.FilmOutputType.RGB_IMAGEPIPELINE, look, 0)
    look = (np.clip(look[::-1], 0, 1) * 255 + 0.5).astype(np.uint8)
    session.Stop()
    passes = stats.Get("stats.renderengine.pass").GetInt()
    samplesec = stats.Get("stats.renderengine.total.samplesec").GetFloat()
    raysec = stats.Get("stats.renderengine.performance.total").GetFloat()
    npy = out / (name + ".npy")
    np.save(npy, img)
    np.save(out / (name + "-look.npy"), look)
    run = {"name": name, "spp": spp, "seed": seed, "setup_s": t1 - t0, "render_s": t2 - t1,
           "pass": passes, "samplesec": samplesec, "raysec": raysec,
           # Rays traced per sample (camera, shadow and bounce rays): the path length
           "rays_per_sample": raysec * (t2 - t1) / max(1, passes * w * h),
           "resolution": [w, h], "file": npy.name,
           "mean": [float(x) for x in img.reshape(-1, 3).mean(axis=0)],
           "nonfinite": int((~np.isfinite(img)).sum())}
    log["engine"] = props.Get("renderengine.type").GetString()
    log["runs"].append(run)
    print(f"[native] {name}: {spp} spp seed {seed} setup {t1 - t0:.2f} s render {t2 - t1:.2f} s "
          f"{samplesec / 1e6:.2f} Msamples/s mean {run['mean']}", flush=True)
    return run


seed = args.seed
for spp in args.spp:
    for r in range(args.repeats):
        render(f"{args.tag}-spp{spp}-r{r}", spp, seed)
        seed += 1

with open(out / (args.tag + ".json"), "w") as f:
    json.dump(log, f, indent=1)
print("[native] done", out / (args.tag + ".json"))
