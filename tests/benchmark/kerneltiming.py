#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Authors (see AUTHORS.txt)
#
# SPDX-License-Identifier: Apache-2.0

"""GPU time per kernel of PATHOCL on the HIP devices.

Renders each scene at a fixed spp with LUXRAYS_HIP_KERNEL_TIMING=1 (a pair of
hipEvents around every kernel launch of HIPDevice, reported by the render
thread when it ends) and prints, per scene, the time per sample after the
first samples and the share of each kernel. The scenes are the renderbench.py
set or .cfg files (i.e. a g0_export.py export).

Each scene runs in its own process with the default pyluxcore.Init(): a Python
log handler never sees the render thread log, which holds the report.

Examples:

    # The six benchmark scenes at 256 spp, 640x480
    python3 tests/benchmark/kerneltiming.py

    # A g0_export.py export at its own resolution, with and without textures
    python3 tests/benchmark/kerneltiming.py --size 0 \\
        out/g0/classroom/export/classroom-g0_LuxCore/00001/render.cfg \\
        -D 'scene.file = scene-flat.scn'
"""

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))


def render_one(cfg, spp, size, defines):
    """Child process: one render, the log on stdout."""
    import pyluxcore

    pyluxcore.Init()
    devices = pyluxcore.GetOpenCLDeviceList()
    select = "".join("1" if d[1] == "HIP_GPU" else "0" for d in devices)
    if "1" not in select:
        sys.exit("no HIP device")

    cfg = Path(cfg) if Path(cfg).is_absolute() else REPO_ROOT / cfg
    props = pyluxcore.Properties(str(cfg))
    props.Set(pyluxcore.Properties().SetFromString(f"""
        renderengine.type = PATHOCL
        sampler.type = SOBOL
        sampler.sobol.adaptive.strength = 0
        film.opencl.enable = 0
        opencl.native.threads.count = 0
        opencl.cpu.use = 0
        opencl.gpu.use = 1
        opencl.devices.select = {select}
        batch.halttime = 0
        batch.haltthreshold = 0
        batch.haltspp = {spp}
        """))
    if size:
        props.Set(pyluxcore.Property("film.width", size[0]))
        props.Set(pyluxcore.Property("film.height", size[1]))
    for define in defines:
        props.Set(pyluxcore.Properties().SetFromString(define))

    pyluxcore.ClearFileNameResolverPaths()
    pyluxcore.AddFileNameResolverPath(str(cfg.parent))
    # The scene.file paths of the repository scenes are relative to its root
    os.chdir(REPO_ROOT)

    session = pyluxcore.RenderSession(pyluxcore.RenderConfig(props))
    session.Start()
    first = None
    while not session.HasDone():
        time.sleep(0.05)
        session.UpdateStats()
        if first is None and session.GetStats().Get("stats.renderengine.total.samplecount").GetFloat() > 0:
            first = time.perf_counter()
    end = time.perf_counter()
    samples = session.GetStats().Get("stats.renderengine.total.samplecount").GetFloat()
    session.Stop()
    print(f"[KT] samples {samples:.0f} time {end - first:.4f}", flush=True)


def run(cfg, args):
    cmd = [sys.executable, __file__, "--child", cfg, "--spp", str(args.spp), "--size", args.size]
    for define in args.define or []:
        cmd += ["-D", define]
    env = dict(os.environ, LUXRAYS_HIP_KERNEL_TIMING="1")
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, errors="replace")
    out = proc.stdout + proc.stderr
    if args.log:
        with open(args.log, "a") as f:
            f.write(out)
    m = re.search(r"\[KT\] samples (\S+) time (\S+)", out)
    if proc.returncode != 0 or not m:
        tail = "\n".join(out.strip().splitlines()[-5:])
        return None, f"FAILED (exit {proc.returncode}):\n{tail}"

    kernels = re.findall(r"^\s+(\S+): ([\d.]+)ms \(([\d.]+)%\) in (\d+) launches", out, re.M)
    resources = re.search(r"\[AdvancePaths_MK_RT\] (\d+) VGPRs, (\d+) bytes of scratch", out)
    return {
        "ns_per_sample": float(m.group(2)) * 1e9 / float(m.group(1)),
        "kernels": [(k, float(ms), float(share), int(n)) for k, ms, share, n in kernels],
        "resources": resources.groups() if resources else None,
    }, None


def main():
    from renderbench import SCENES

    parser = argparse.ArgumentParser(description="PATHOCL GPU time per kernel (HIP)",
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=__doc__)
    parser.add_argument("scenes", nargs="*",
                        help="renderbench.py scene names or .cfg files (default: the renderbench.py set)")
    parser.add_argument("--spp", type=int, default=256, help="spp of each render (default: 256)")
    parser.add_argument("--size", default="640x480",
                        help="film size WxH, or 0 to keep the one of the .cfg (default: 640x480)")
    parser.add_argument("-D", "--define", action="append",
                        help="additional property applied to all the renderings (i.e. -D 'path.maxdepth = 4')")
    parser.add_argument("--log", help="append the LuxCore log of every render to this file")
    parser.add_argument("--child", help=argparse.SUPPRESS)
    args = parser.parse_args()

    size = None if args.size == "0" else tuple(int(v) for v in args.size.split("x"))
    if args.child:
        render_one(args.child, args.spp, size, args.define or [])
        return

    status = 0
    for scene in args.scenes or list(SCENES):
        cfg = SCENES.get(scene, scene)
        result, error = run(cfg, args)
        if error:
            print(f"{scene}: {error}", flush=True)
            status = 1
            continue
        res = result["resources"]
        res = f", AdvancePaths_MK_RT {res[0]} VGPRs, {res[1]} B scratch" if res else ""
        print(f"{scene}: {result['ns_per_sample']:.1f} ns/sample{res}", flush=True)
        for name, ms, share, launches in result["kernels"]:
            if share >= 0.1:
                print(f"    {name:<34} {share:5.1f}%  {ms / launches:8.3f} ms x {launches}", flush=True)
    sys.exit(status)


if __name__ == "__main__":
    main()
