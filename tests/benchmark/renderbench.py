#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Authors (see AUTHORS.txt)
#
# SPDX-License-Identifier: Apache-2.0

"""Render throughput benchmark.

Renders a fixed set of scenes with a set of render engines and devices and
reports, for each combination:
- the setup time (scene preprocessing, kernel compilation, etc.: the time to
  the first sample);
- the samples/sec, measured after a warm-up, over a fixed time window;
- optionally (--threshold) the time to reach a noise threshold (the
  convergence test of the film).

The results are written in JSON and can be compared with a previous run
(--baseline) to detect performance regressions.

Examples:

    # All the scenes, CPU only
    python3 tests/benchmark/renderbench.py --engines PATHCPU

    # The GPU engines on the HIP devices only, saving the results
    python3 tests/benchmark/renderbench.py --engines PATHOCL --devices HIP_GPU \\
        --output hip.json

    # Compare with a previous run and fail on a regression above 5%
    python3 tests/benchmark/renderbench.py --engines PATHCPU --baseline old.json \\
        --max-regression 5
"""

import argparse
import json
import os
import platform
import re
import statistics
import sys
import time
from pathlib import Path

import pyluxcore

REPO_ROOT = Path(__file__).resolve().parents[2]

# The benchmark scene set: each scene stresses a different part of the
# renderer. The paths are relative to the repository root (as the scene.file
# property of the .cfg files).
SCENES = {
    # Simple diffuse interior, area light
    "cornell": "scenes/cornell/cornell.cfg",
    # HDR environment light, glossy and specular materials
    "luxball-hdr": "scenes/luxball/luxball-hdr.cfg",
    # Interior lit by the sun through a portal, ~100K triangles
    "classroom": "scenes/classroom/classroom.cfg",
    # Interior lit by many area lights
    "kitchen": "scenes/kitchen/kitchen.cfg",
    # Many lights (light sampling strategies)
    "manylights": "scenes/manylights/render.cfg",
    # Heterogeneous volume
    "luxball-hdr-vol": "scenes/luxball/luxball-hdr-vol.cfg",
}

# Properties applied to all the renderings, to make them comparable
COMMON_PROPS = """
film.width = 640
film.height = 480
batch.halttime = 0
batch.haltspp = 0
batch.haltthreshold = 0
film.noiseestimation.warmup = 8
"""

ENGINE_PROPS = {
    "PATHCPU": "sampler.type = SOBOL",
    "BIDIRCPU": "sampler.type = SOBOL",
    # Only the selected GPU devices: no native CPU threads
    "PATHOCL": """
        sampler.type = SOBOL
        opencl.native.threads.count = 0
        opencl.cpu.use = 0
        opencl.gpu.use = 1
        """,
}

POLL_PERIOD = 0.25


def log(msg):
    print(msg, flush=True)


def get_stat(stats, name, default=0.0):
    if stats.IsDefined(name):
        return stats.Get(name).GetFloat()
    return default


def devices_select(device_filter):
    """Build the opencl.devices.select string for the devices whose type or
    name match the regular expression."""
    devices = pyluxcore.GetOpenCLDeviceList()
    selected = [re.search(device_filter, f"{d[1]} {d[0]}") is not None for d in devices]
    names = [f"{d[0]} ({d[1]})" for d, s in zip(devices, selected) if s]
    return "".join("1" if s else "0" for s in selected), names


def render(scene_cfg, engine, args, device_select):
    props = pyluxcore.Properties(str(REPO_ROOT / scene_cfg))
    props.Set(pyluxcore.Properties().SetFromString(COMMON_PROPS))
    props.Set(pyluxcore.Properties().SetFromString(ENGINE_PROPS.get(engine, "")))
    props.Set(pyluxcore.Property("renderengine.type", engine))
    if device_select is not None:
        props.Set(pyluxcore.Property("opencl.devices.select", device_select))
    if args.threshold:
        props.Set(pyluxcore.Property("batch.haltthreshold", args.threshold))
    for define in args.define or []:
        props.Set(pyluxcore.Properties().SetFromString(define))

    pyluxcore.ClearFileNameResolverPaths()
    pyluxcore.AddFileNameResolverPath(str((REPO_ROOT / scene_cfg).parent))

    # The scene.file paths are relative to the repository root
    cwd = os.getcwd()
    os.chdir(REPO_ROOT)
    try:
        start = time.perf_counter()
        config = pyluxcore.RenderConfig(props)
        session = pyluxcore.RenderSession(config)
        session.Start()

        def update():
            session.UpdateStats()
            return session.GetStats()

        # Setup: wait for the first samples
        stats = update()
        while get_stat(stats, "stats.renderengine.total.samplecount") == 0:
            if time.perf_counter() - start > args.timeout:
                raise RuntimeError("timeout waiting for the first samples")
            time.sleep(POLL_PERIOD)
            stats = update()
        setup_time = time.perf_counter() - start

        # Warm-up (caches, GPU clocks, etc.)
        time.sleep(args.warmup)

        # Throughput window
        stats = update()
        t0 = time.perf_counter()
        count0 = get_stat(stats, "stats.renderengine.total.samplecount")
        time.sleep(args.time)
        stats = update()
        t1 = time.perf_counter()
        count1 = get_stat(stats, "stats.renderengine.total.samplecount")

        result = {
            "setup_time": setup_time,
            "samples_sec": (count1 - count0) / (t1 - t0),
            "rays_sec": get_stat(stats, "stats.renderengine.total.raysec"),
            "triangles": int(get_stat(stats, "stats.dataset.trianglecount")),
        }

        # Time to the noise threshold
        if args.threshold:
            while not session.HasDone():
                if time.perf_counter() - start > args.timeout:
                    raise RuntimeError("timeout waiting for the noise threshold")
                time.sleep(POLL_PERIOD)
                session.UpdateStats()
            result["threshold_time"] = time.perf_counter() - start - setup_time

        session.Stop()
        return result
    finally:
        os.chdir(cwd)


def machine_info():
    desc = pyluxcore.GetPlatformDesc()
    return {
        "luxcore": pyluxcore.Version(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "compile": {
            k: desc.Get(k).GetString() for k in desc.GetAllNames("compile.")
        },
        "devices": [f"{d[0]} ({d[1]})" for d in pyluxcore.GetOpenCLDeviceList()],
    }


def compare(results, baseline, max_regression):
    """Compare the samples/sec with the baseline ones, return the regressions."""
    regressions = []
    base = {r["key"]: r for r in baseline["results"]}
    log("")
    log(f"{'Benchmark':<40} {'Baseline':>12} {'Current':>12} {'Delta':>8}")
    for r in results:
        b = base.get(r["key"])
        if b is None or b["samples_sec"] == 0:
            continue
        delta = 100.0 * (r["samples_sec"] / b["samples_sec"] - 1.0)
        flag = ""
        if delta < -max_regression:
            regressions.append(r["key"])
            flag = "  REGRESSION"
        log(f"{r['key']:<40} {b['samples_sec'] / 1e6:>10.3f}M {r['samples_sec'] / 1e6:>10.3f}M "
            f"{delta:>+7.1f}%{flag}")
    return regressions


def markdown(results):
    lines = [
        "| Benchmark | Setup (s) | Samples/sec | Rays/sec | Threshold (s) |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in results:
        threshold = f"{r['threshold_time']:.1f}" if "threshold_time" in r else "-"
        lines.append(f"| {r['key']} | {r['setup_time']:.1f} | {r['samples_sec'] / 1e6:.3f}M | "
                     f"{r['rays_sec'] / 1e6:.2f}M | {threshold} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="LuxCore render throughput benchmark",
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=__doc__)
    parser.add_argument("--scenes", default=",".join(SCENES),
                        help="comma separated list of scenes (default: all): " + ", ".join(SCENES))
    parser.add_argument("--engines", default="PATHCPU",
                        help="comma separated list of render engines (default: PATHCPU)")
    parser.add_argument("--devices", default="GPU",
                        help="regular expression selecting the devices of the *OCL engines, matched "
                             "against '<type> <name>' (i.e. HIP_GPU, CUDA_GPU, OPENCL_GPU, a device "
                             "name; default: GPU)")
    parser.add_argument("--time", type=float, default=20.0,
                        help="length of the samples/sec measurement window in seconds (default: 20)")
    parser.add_argument("--warmup", type=float, default=5.0,
                        help="warm-up time after the first samples in seconds (default: 5)")
    parser.add_argument("--repeat", type=int, default=1,
                        help="number of runs of each benchmark, the median is reported (default: 1)")
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="also measure the time to reach this noise threshold (i.e. 0.05)")
    parser.add_argument("--timeout", type=float, default=900.0,
                        help="maximum time of a single rendering in seconds (default: 900)")
    parser.add_argument("-D", "--define", action="append",
                        help="additional property applied to all the renderings (i.e. -D 'path.maxdepth = 4')")
    parser.add_argument("--output", help="write the results to this JSON file")
    parser.add_argument("--markdown", help="write the results as a Markdown table to this file")
    parser.add_argument("--baseline", help="compare the samples/sec with this previous JSON result")
    parser.add_argument("--max-regression", type=float, default=0.0,
                        help="with --baseline, exit with an error if a samples/sec drops more than "
                             "this percentage (default: 0, never fail)")
    parser.add_argument("--verbose", action="store_true", help="print the LuxCore log")
    args = parser.parse_args()

    pyluxcore.Init((lambda msg: print(msg, flush=True)) if args.verbose else (lambda msg: None))

    scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
    for s in scenes:
        if s not in SCENES:
            parser.error(f"unknown scene '{s}', the available ones are: {', '.join(SCENES)}")
    engines = [e.strip() for e in args.engines.split(",") if e.strip()]

    info = machine_info()
    log(f"LuxCore {info['luxcore']} on {info['platform']}, {info['cpu_count']} hardware threads")

    results = []
    for engine in engines:
        device_select, device_names, device_tag = None, [], "cpu"
        if engine.endswith("OCL"):
            device_select, device_names = devices_select(args.devices)
            if not device_names:
                log(f"Skipping {engine}: no device matches '{args.devices}'")
                continue
            device_tag = args.devices
            log(f"{engine} devices: {', '.join(device_names)}")

        for scene in scenes:
            key = f"{scene}/{engine}/{device_tag}"
            runs = []
            for i in range(args.repeat):
                try:
                    runs.append(render(SCENES[scene], engine, args, device_select))
                except Exception as e:
                    log(f"{key}: FAILED: {e}")
                    break
            if not runs:
                continue

            result = {
                "key": key,
                "scene": scene,
                "engine": engine,
                "devices": device_names,
                "runs": runs,
            }
            for metric in ("setup_time", "samples_sec", "rays_sec", "threshold_time"):
                if metric in runs[0]:
                    result[metric] = statistics.median(r[metric] for r in runs)
            result["triangles"] = runs[0]["triangles"]
            results.append(result)

            threshold = f", threshold in {result['threshold_time']:.1f}s" if "threshold_time" in result else ""
            log(f"{key}: setup {result['setup_time']:.1f}s, {result['samples_sec'] / 1e6:.3f}M samples/sec, "
                f"{result['rays_sec'] / 1e6:.2f}M rays/sec{threshold}")

    output = {
        "machine": info,
        "settings": {k: v for k, v in vars(args).items() if k not in ("output", "markdown", "baseline")},
        "results": results,
    }
    if args.output:
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
    if args.markdown:
        with open(args.markdown, "w") as f:
            f.write(markdown(results))

    status = 0
    if args.baseline:
        with open(args.baseline) as f:
            baseline = json.load(f)
        regressions = compare(results, baseline, args.max_regression)
        if regressions and args.max_regression > 0:
            log(f"Performance regression above {args.max_regression}%: {', '.join(regressions)}")
            status = 1

    if len(results) < len(scenes) * len(engines):
        status = 1

    pyluxcore.SetLogHandler(None)
    sys.exit(status)


if __name__ == "__main__":
    main()
