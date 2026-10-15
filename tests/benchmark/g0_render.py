# SPDX-FileCopyrightText: 2026 Authors (see AUTHORS.txt)
#
# SPDX-License-Identifier: Apache-2.0

"""G0 renders: one engine, one scene prepared by g0_prepare.py, on the HIP GPU.

The renders go through Blender, as a user would run them (Cycles, or
BlendLuxCore with PATHOCL). The process renders:

1. one render at 1 spp: its wall time is the time to the first pixel (scene
   sync or export, kernel load or compilation, BVH build). Run with an empty
   XDG_CACHE_HOME for the cold kernel cache;
2. --baseline-runs renders at 1 spp: the fixed cost of a render;
3. --repeats renders at each --spp, each with its own seed.

Each image is saved as a linear float32 RGB .npy with a JSON log of the wall
times; g0_analyze.py turns them into RMSE-against-time curves. The engine
settings are those of the prepared .blend; only the spp, the seed, the
resolution percentage and the devices are set here.

    blender -b --factory-startup --disable-autoexec out/g0/cornell/cornell-g0.blend \\
        --python-exit-code 1 --python tests/benchmark/g0_render.py -- \\
        --engine LUXCORE --out out/g0/results/cornell/luxcore --spp 16 32 64 \\
        --addon ~/GitHub/BlendLuxCore --site-packages out/venvs/<venv>/lib64/python3.13/site-packages
"""

import argparse
import json
import os
import statistics
import sys
import time

import bpy
import numpy as np

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
parser = argparse.ArgumentParser(prog="g0_render.py")
parser.add_argument("--engine", required=True, choices=("CYCLES", "LUXCORE"))
parser.add_argument("--out", required=True, help="output directory")
parser.add_argument("--tag", default="run", help="prefix of the output files")
parser.add_argument("--spp", type=int, nargs="*", default=[])
parser.add_argument("--repeats", type=int, default=2)
parser.add_argument("--baseline-runs", type=int, default=3)
parser.add_argument("--seed", type=int, default=1, help="seed of the first render")
parser.add_argument("--percentage", type=int, default=100, help="resolution percentage")
parser.add_argument("--cycles-hiprt", action="store_true", help="HIPRT for Cycles (off by default)")
parser.add_argument("--no-hiprt", action="store_true", help="LuxCore BVH instead of HIPRT")
parser.add_argument("--addon", help="BlendLuxCore source directory (LUXCORE)")
parser.add_argument("--site-packages", help="site-packages with pyluxcore (LUXCORE)")
parser.add_argument("--bounces", type=int,
                    help="override the path length: Cycles bounces = B, LuxCore path depth = B + 1 "
                         "(LuxCore counts vertices, Cycles bounces); 0 = direct light only")
parser.add_argument("--luxcore-set", action="append", default=[], metavar="KEY=VALUE",
                    help="extra LuxCore config property, e.g. path.russianroulette.depth=5 (repeatable)")
args = parser.parse_args(argv)

os.makedirs(args.out, exist_ok=True)
scene = bpy.context.scene
scene.render.resolution_percentage = args.percentage
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.image_settings.color_depth = "32"
scene.render.image_settings.exr_codec = "ZIP"
scene.render.use_compositing = False
scene.render.use_sequencer = False
log = {"engine": args.engine, "blend": bpy.data.filepath, "blender": bpy.app.version_string,
       "resolution": [scene.render.resolution_x * args.percentage // 100,
                      scene.render.resolution_y * args.percentage // 100],
       "xdg_cache_home": os.environ.get("XDG_CACHE_HOME"), "runs": []}

if args.engine == "CYCLES":
    scene.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "HIP"
    prefs.get_devices()
    for d in prefs.devices:
        d.use = d.type == "HIP"
    if hasattr(prefs, "use_hiprt"):
        prefs.use_hiprt = args.cycles_hiprt
    used = [d.name for d in prefs.devices if d.use]
    if not used:
        sys.exit("[G0] no HIP device for Cycles")
    log["devices"] = used
    log["cycles_hiprt"] = args.cycles_hiprt
    scene.cycles.device = "GPU"
    scene.cycles.time_limit = 0
    # Keep the BVH and the kernels between the renders of this process, as the
    # LuxCore kernel cache does; the fixed cost per render is measured anyway
    scene.render.use_persistent_data = True

    if args.bounces is not None:
        cy = scene.cycles
        cy.max_bounces = cy.diffuse_bounces = cy.glossy_bounces = args.bounces
        cy.transmission_bounces = cy.transparent_max_bounces = args.bounces
        log["bounces"] = args.bounces

    def set_render(spp, seed):
        scene.cycles.samples = spp
        scene.cycles.seed = seed
        scene.cycles.use_animated_seed = False
else:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import g0_blc
    restore_manifest = g0_blc.load(args.addon, args.site_packages)
    import pyluxcore
    log["pyluxcore"] = pyluxcore.Version()
    log["pyluxcore_file"] = pyluxcore.__file__
    scene.render.engine = "LUXCORE"
    if args.bounces is not None:
        # Set in the render config directly: the addon exports its depths + 1
        # (except specular) and cannot go below one bounce
        args.luxcore_set += [f"path.pathdepth.{k}={args.bounces + 1}"
                             for k in ("total", "diffuse", "glossy", "specular")]
        log["bounces"] = args.bounces
    if args.luxcore_set:
        # Properties the addon does not expose: added to the render config it creates
        extra = pyluxcore.Properties()
        for kv in args.luxcore_set:
            key, value = kv.split("=", 1)
            extra.SetFromString(f"{key.strip()} = {value.strip()}")
        log["luxcore_set"] = args.luxcore_set
        render_config = pyluxcore.RenderConfig

        def render_config_with_extra(*a):
            if a:
                a[0].Set(extra)
            return render_config(*a)
        pyluxcore.RenderConfig = render_config_with_extra
    prefs = g0_blc.preferences()
    prefs.gpu_backend = "HIP"
    prefs.use_hiprt = not args.no_hiprt
    devices = scene.luxcore.devices
    devices.use_native_cpu = False
    devices.update_devices_if_necessary()
    for d in devices.devices:
        d.enabled = d.type == "HIP_GPU"
    used = [d.name for d in devices.devices if d.enabled]
    if not used:
        sys.exit("[G0] no HIP device for LuxCore")
    log["devices"] = used
    log["selection"] = devices.devices_to_selection_string()
    log["hiprt"] = not args.no_hiprt
    tonemapper = scene.camera.data.luxcore.imagepipeline.tonemapper
    tonemapper.type = "TONEMAP_LINEAR"
    tonemapper.use_autolinear = False
    tonemapper.linear_scale = 1
    if scene.luxcore.config.sobol_adaptive_strength > 0:
        sys.exit("[G0] adaptive sampling is on: prepare the scene with g0_prepare.py")
    halt = scene.luxcore.halt
    halt.enable = True
    halt.use_time = False
    halt.use_noise_thresh = False
    halt.use_light_samples = False
    halt.use_samples = True
    scene.luxcore.config.use_animated_seed = False

    def set_render(spp, seed):
        halt.samples = spp
        scene.luxcore.config.seed = seed

print("[G0] devices", log["devices"])


def render(name, spp, seed):
    set_render(spp, seed)
    exr = os.path.join(args.out, name + ".exr")
    scene.render.filepath = exr
    t0 = time.perf_counter()
    bpy.ops.render.render(write_still=True)
    wall = time.perf_counter() - t0
    img = bpy.data.images.load(exr, check_existing=False)
    w, h = img.size
    px = np.empty(w * h * img.channels, dtype=np.float32)
    img.pixels.foreach_get(px)
    bpy.data.images.remove(img)
    os.remove(exr)
    npy = os.path.join(args.out, name + ".npy")
    np.save(npy, px.reshape(h, w, -1)[..., :3])
    run = {"name": name, "spp": spp, "seed": seed, "wall_s": wall, "file": os.path.basename(npy)}
    log["runs"].append(run)
    print(f"[G0] {name}: {spp} spp seed {seed} in {wall:.2f} s", flush=True)
    return run


seed = args.seed
first = render(f"{args.tag}-first", 1, seed)
seed += 1
log["first_render_s"] = first["wall_s"]
base = [render(f"{args.tag}-base{i}", 1, seed + i)["wall_s"] for i in range(args.baseline_runs)]
seed += args.baseline_runs
log["baseline_s"] = statistics.median(base) if base else None
for spp in args.spp:
    for r in range(args.repeats):
        render(f"{args.tag}-spp{spp}-r{r}", spp, seed)
        seed += 1

with open(os.path.join(args.out, args.tag + ".json"), "w") as f:
    json.dump(log, f, indent=1)
if args.engine == "LUXCORE":
    restore_manifest()
print("[G0] done", os.path.join(args.out, args.tag + ".json"))
