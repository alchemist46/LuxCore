# SPDX-FileCopyrightText: 2026 Authors (see AUTHORS.txt)
#
# SPDX-License-Identifier: Apache-2.0

"""The viewport's film path in pyluxcore alone (LuxCore ROADMAP E6): the
RTPATHOCL frame time and the cost of a draw's GetOutputFloat(), without
Blender.

Loads a render config saved by viewport_timing.py --save-config (the addon's
exact viewport config and scene), sets the film size, and runs three phases:

- idle: nobody reads the film; the frame time alone;
- draw: GetOutputFloat(RGB_IMAGEPIPELINE) in a loop at --draw-interval, as the
  addon's view_draw() does; its duration and the frame time meanwhile;
- paused: the same reads with the session paused.

    blender -b --factory-startup --python tests/benchmark/viewport_film.py -- \\
        --site-packages <site-packages with pyluxcore> \\
        --config out/viewport/cornell-ocl-ps1/renderconfig.bcf --size 1920x1080

Writes a JSON summary to stdout (and to --json).
"""

import argparse
import json
import os
import sys
import time

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
parser = argparse.ArgumentParser()
parser.add_argument("--site-packages")
parser.add_argument("--config", required=True, help="renderconfig.bcf")
parser.add_argument("--size", action="append", default=None,
                    help="film WxH, repeatable (default: the saved size)")
parser.add_argument("--seconds", type=float, default=4.0, help="length of each phase")
parser.add_argument("--draw-interval", type=float, default=0.0,
                    help="seconds between two reads in the draw phase (0: back to back)")
parser.add_argument("--set", action="append", default=[],
                    help="extra config property, name=value (e.g. film.opencl.enable=0)")
parser.add_argument("--scene-set", action="append", default=[],
                    help="scene property, name=value (e.g. scene.materials.m.type=matte)")
parser.add_argument("--json")
args = parser.parse_args(argv)
if args.site_packages:
    sys.path.insert(0, args.site_packages)

import numpy as np
import pyluxcore

pyluxcore.Init()
clock = time.perf_counter


def stats(values):
    if not values:
        return None
    v = np.array(values) * 1000
    return {"n": len(v), "mean": round(float(v.mean()), 2), "median": round(float(np.median(v)), 2),
            "p90": round(float(np.percentile(v, 90)), 2), "max": round(float(v.max()), 2)}


def frames(session, seconds, read=None, interval=0.0):
    """Run for seconds, reading the film with read() every interval if given;
    returns the frame times and the read durations"""
    frame_times, reads, passes = [], [], []
    end = clock() + seconds
    last_read = 0.0
    while clock() < end:
        if read is not None and clock() - last_read >= interval:
            start = clock()
            read()
            reads.append(clock() - start)
            last_read = start
        else:
            time.sleep(0.005)
        session.UpdateStats()
        s = session.GetStats()
        passes.append((clock(), s.Get("stats.renderengine.pass").GetInt()))
        if s.IsDefined("stats.rtpathocl.frame.time"):
            frame_times.append(s.Get("stats.rtpathocl.frame.time").GetFloat())
    (t0, p0), (t1, p1) = passes[0], passes[-1]
    return {"frame_time_ms": stats(frame_times[len(frame_times) // 2:]),
            "passes_per_s": round((p1 - p0) / (t1 - t0), 2) if t1 > t0 else None,
            "read_ms": stats(reads)}


def run(size):
    config = pyluxcore.RenderConfig.LoadFile(args.config)
    props = pyluxcore.Properties()
    if size:
        w, h = (int(x) for x in size.split("x"))
        props.Set(pyluxcore.Property("film.width", w))
        props.Set(pyluxcore.Property("film.height", h))
    for item in args.set:
        name, value = item.split("=", 1)
        props.Set(pyluxcore.Property(name, value))
    if props.GetSize():
        config.Parse(props)
    if args.scene_set:
        scene_props = pyluxcore.Properties()
        for item in args.scene_set:
            name, value = item.split("=", 1)
            scene_props.Set(pyluxcore.Property(name, value))
        config.GetScene().Parse(scene_props)
    cfg = config.GetProperties()
    w = cfg.Get("film.width").GetInt()
    h = cfg.Get("film.height").GetInt()
    result = {"size": f"{w}x{h}", "engine": cfg.Get("renderengine.type").GetString(),
              "film.opencl.enable": cfg.Get("film.opencl.enable").GetString()
              if cfg.IsDefined("film.opencl.enable") else None,
              "set": args.set, "scene_set": args.scene_set}
    session = pyluxcore.RenderSession(config)
    start = clock()
    session.Start()
    result["start_s"] = round(clock() - start, 3)
    buffer = np.zeros(w * h * 3, dtype=np.float32)
    film = session.GetFilm()

    def read():
        film.GetOutputFloat(pyluxcore.FilmOutputType.RGB_IMAGEPIPELINE, buffer, 0, True)

    # Past the resolution-reduction preview
    frames(session, 1.0)
    result["idle"] = frames(session, args.seconds)
    result["draw"] = frames(session, args.seconds, read, args.draw_interval)
    session.Pause()
    time.sleep(0.3)
    result["paused"] = frames(session, min(args.seconds, 2.0), read, args.draw_interval)
    session.Resume()
    session.Stop()
    return result


results = [run(size) for size in (args.size or [None])]
text = json.dumps(results, indent=1)
print(text, flush=True)
if args.json:
    with open(args.json, "w") as f:
        f.write(text)
