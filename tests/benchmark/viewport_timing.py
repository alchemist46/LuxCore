# SPDX-FileCopyrightText: 2026 Authors (see AUTHORS.txt)
#
# SPDX-License-Identifier: Apache-2.0

"""Viewport latency in a GUI Blender (LuxCore ROADMAP E6, BlendLuxCore
ROADMAP 1.12): one engine, device and pixel size per process.

A GUI Blender is needed (background mode has no viewport). The script drives
the 3D view from a timer and measures:

- per view_draw() call (LuxCore only): the total main-thread time and its
  parts (the change check, the scene edit, WaitNewFrame(), UpdateStats(),
  GetOutputFloat(), the numpy and gpu.Buffer copies, the texture draw,
  get_pretty_stats()), by wrapping the addon's functions from here, so any
  addon commit can be measured; whether the draw had new samples; with an
  addon that reads the film in a thread (FrameBuffer.start_read()), "film" is
  the main-thread part that shows a finished read, and the reads themselves
  are timed apart ("reads");
- draws/s, the stat stats.rtpathocl.frame.time;
- the main-thread stalls, from the gaps of a timer that runs every loop;
- the time to the first rendered frame on screen and the main-thread stall
  meanwhile;
- the camera-move latency: from setting the view to the first draw whose
  screen probe is closer to the new view's converged image than to the old
  one's, both while rendering and after the halt (paused: each move is
  labelled by the state of the render when it is made);
- the resize latency (LuxCore only): from leaving the full-screen area to the
  first draw of a film from the new session;
- with --object-moves (LuxCore only): the view_update() that applies a move
  of a mesh object, its scene edit and how many lights it exported again.

The probe reads a band of the drawn region back in a POST_PIXEL handler, so it
measures what is on screen and works for Cycles too (--engine CYCLES, the A/B
on scenes/cornell/Cornell Box Cycles.blend; no addon needed then).

    blender.exe --factory-startup --window-geometry 0 0 1600 1000 \\
        --python tests/benchmark/viewport_timing.py -- \\
        --blend "scenes/cornell/Cornell Box Luxcore.blend" --addon ../BlendLuxCore --site-packages <site-packages with pyluxcore> \\
        --device OCL --pixel-size 1 --out out/viewport/cornell-ocl-ps1

Writes <out>/timing.json and prints a summary. The viewport denoiser is off
unless --denoise: it changes the image at the halt and the latency probe
compares images.
"""

import argparse
import json
import math
import os
import sys
import threading
import time

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
parser = argparse.ArgumentParser()
parser.add_argument("--blend", required=True)
parser.add_argument("--engine", default="LUXCORE", choices=("LUXCORE", "CYCLES"))
parser.add_argument("--addon", help="BlendLuxCore source tree (LUXCORE)")
parser.add_argument("--site-packages", help="site-packages with pyluxcore (LUXCORE)")
parser.add_argument("--device", default="OCL", choices=("OCL", "CPU"),
                    help="viewport device: OCL is RTPATHOCL on the HIP GPU, CPU is RTPATHCPU")
parser.add_argument("--pixel-size", default="1", choices=("1", "2", "4", "8"))
parser.add_argument("--halt", type=int, default=6, help="viewport halt time (s)")
parser.add_argument("--first-pause-timeout", type=float, default=120.0,
                    help="seconds past the halt to wait for the first pause (a cold HIP kernel compile takes ~6 min)")
parser.add_argument("--moves", type=int, default=4, help="camera moves while rendering")
parser.add_argument("--paused-moves", type=int, default=3,
                    help="camera moves after the halt (at least 2: the first two set the references)")
parser.add_argument("--move-gap", type=float, default=1.5,
                    help="seconds of rendering before each move while rendering")
parser.add_argument("--move-angle", type=float, default=8.0,
                    help="yaw of the second view around the camera's up axis (degrees)")
parser.add_argument("--cycles-samples", type=int, default=1024, help="Cycles viewport samples")
parser.add_argument("--denoise", action="store_true", help="keep the viewport denoiser on")
parser.add_argument("--no-resize", action="store_true", help="skip the resize test")
parser.add_argument("--object-moves", type=int, default=0,
                    help="moves of a mesh object (LuxCore): the view_update() time and its lights")
parser.add_argument("--object", help="the object --object-moves moves (default: the first mesh)")
parser.add_argument("--save-config", action="store_true",
                    help="save the session's render config with its scene at the end "
                    "(renderconfig.bcf, for viewport_film.py)")
parser.add_argument("--out", required=True)
args = parser.parse_args(argv)
args.out = os.path.abspath(args.out)
os.makedirs(args.out, exist_ok=True)

import numpy as np

if args.engine == "LUXCORE":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import g0_blc
    restore_manifest = g0_blc.load(args.addon, args.site_packages)
else:
    restore_manifest = lambda: None

import bpy
import gpu
from mathutils import Matrix

bpy.context.preferences.use_preferences_save = False
# The factory UI, which has a 3D view
bpy.ops.wm.open_mainfile(filepath=os.path.abspath(args.blend), load_ui=False)
clock = time.perf_counter
T0 = clock()
LOG = open(os.path.join(args.out, "timing.log"), "w")


def log(*a):
    s = " ".join(str(x) for x in a)
    print("[V]", s, flush=True)
    LOG.write(s + "\n")
    LOG.flush()


def now():
    return clock() - T0


# Records, all times in seconds since T0
draws = []      # view_draw() calls (LuxCore): dict of timings
probes = []     # after each engine draw: (t, region size, probe or None, read time)
screen_draws = []  # POST_PIXEL calls
ticks = []      # main-thread timer ticks
events = []     # (t, name, details)
calls = []      # (t, name, duration): session create/start, scene edits
reads = []      # (t, duration): GetOutputFloat() calls outside the main thread
updates = []    # view_update() calls (LuxCore): dict of timings
state = {"current": None, "engine": None}


def event(name, **details):
    events.append((now(), name, details))
    log(f"{now():7.3f}", name, details or "")


# Main-thread time of each view_draw() and of its parts (LuxCore)

def timed(key, function):
    """A part of view_draw(); a call from another thread goes to reads"""
    def wrapper(*a, **k):
        main = threading.current_thread() is threading.main_thread()
        record = state["current"] if main else None
        start = clock()
        try:
            return function(*a, **k)
        finally:
            if record is not None:
                record[key] = record.get(key, 0.0) + clock() - start
            elif not main:
                reads.append((start - T0, clock() - start))
    return wrapper


def counted(key, function):
    """A part of view_draw() or view_update() timed and counted"""
    timed_function = timed(key, function)

    def wrapper(*a, **k):
        record = state["current"]
        if record is not None and threading.current_thread() is threading.main_thread():
            record[key + "_n"] = record.get(key + "_n", 0) + 1
        return timed_function(*a, **k)
    return wrapper


def timed_call(name, function):
    """A call outside view_draw(), e.g. the session start in its thread"""
    def wrapper(*a, **k):
        start = clock()
        try:
            return function(*a, **k)
        finally:
            calls.append((start - T0, name, clock() - start))
    return wrapper


def instrument_luxcore():
    import pyluxcore
    viewport = g0_blc.submodule("engine.viewport")
    render_utils = g0_blc.submodule("utils.render")
    framebuffer = g0_blc.submodule("draw.viewport").FrameBuffer
    exporter = g0_blc.submodule("export").Exporter

    if hasattr(framebuffer, "start_read"):
        # The film is read in a thread: the main thread shows a finished read
        framebuffer._show_read = timed("film", framebuffer._show_read)
    else:
        framebuffer.update = timed("film", framebuffer.update)
    framebuffer.draw = timed("draw", framebuffer.draw)
    exporter.get_viewport_changes = timed("changes", exporter.get_viewport_changes)
    exporter.update = timed("edit", exporter.update)
    render_utils.get_pretty_stats = timed("pretty", render_utils.get_pretty_stats)
    session = pyluxcore.RenderSession
    session.WaitNewFrame = timed("wait", session.WaitNewFrame)
    session.UpdateStats = timed("stats", session.UpdateStats)
    session.Start = timed_call("Start", session.Start)
    session.BeginSceneEdit = timed("begin_edit", session.BeginSceneEdit)
    session.EndSceneEdit = timed("end_edit", session.EndSceneEdit)
    pyluxcore.Film.GetOutputFloat = timed("getoutput", pyluxcore.Film.GetOutputFloat)
    exporter.create_session = timed_call("create_session", exporter.create_session)
    # A config change on the same scene (BlendLuxCore 1.5; older addons have no
    # restart_session())
    if hasattr(exporter, "restart_session"):
        exporter.restart_session = timed_call("restart_session", exporter.restart_session)

    light = g0_blc.submodule("export.light")
    light.convert_light = counted("lights", light.convert_light)

    view_update = viewport.view_update

    def view_update_timed(engine, context, depsgraph, *a, **k):
        record = {"t": now()}
        state["current"] = record
        start = clock()
        try:
            view_update(engine, context, depsgraph, *a, **k)
        finally:
            record["total"] = clock() - start
            state["current"] = None
            updates.append(record)

    viewport.view_update = view_update_timed

    view_draw = viewport.view_draw

    def view_draw_timed(engine, context, depsgraph):
        record = {"t": now()}
        state["current"] = record
        state["engine"] = engine
        start = clock()
        try:
            view_draw(engine, context, depsgraph)
        finally:
            record["total"] = clock() - start
            state["current"] = None
            # The engine's draw target is still bound
            take_probe(context.region, 0, 0)
            try:
                session = engine.session
                starting = bool(engine.starting_session)
            except ReferenceError:
                session, starting = None, True
            record["session"] = id(session) if session is not None else None
            record["starting"] = starting
            if session is not None and not starting:
                if "config_saved" not in state:
                    # The render config the addon exported, for viewport_film.py
                    state["config_saved"] = True
                    props = session.GetRenderConfig().GetProperties()
                    with open(os.path.join(args.out, "renderconfig.txt"), "w") as f:
                        f.write(props.ToString())
                try:
                    stats = session.GetStats()
                    record["samples"] = stats.Get("stats.renderengine.total.samplecount").GetFloat()
                    if stats.IsDefined("stats.rtpathocl.frame.time"):
                        record["frame_time"] = stats.Get("stats.rtpathocl.frame.time").GetFloat()
                    record["paused"] = session.IsInPause()
                except Exception:
                    pass
            draws.append(record)

    viewport.view_draw = view_draw_timed


# Screen probe: a band through the middle of the region, averaged in blocks

BAND = 16
probe_buffer = {}


def read_probe(region, x0, y0):
    """x0, y0: the region's origin in the bound framebuffer"""
    w, h = region.width, region.height
    y = h // 2 - BAND // 2
    blocks = w // BAND
    if blocks < 4 or y < 0:
        return None
    size = (blocks * BAND, BAND)
    if probe_buffer.get("size") != size:
        probe_buffer["size"] = size
        probe_buffer["buffer"] = gpu.types.Buffer("FLOAT", size[0] * size[1] * 4)
    buffer = probe_buffer["buffer"]
    fb = gpu.state.active_framebuffer_get()
    fb.read_color(x0, y0 + y, size[0], size[1], 4, 0, "FLOAT", data=buffer)
    try:
        pixels = np.asarray(buffer, dtype=np.float32)
    except Exception:
        pixels = np.array(buffer.to_list(), dtype=np.float32)
    pixels = pixels.reshape(BAND, blocks, BAND, 4)[..., :3]
    return pixels.mean(axis=(0, 2)).ravel()


def take_probe(region, x0, y0):
    start = clock()
    try:
        probe = read_probe(region, x0, y0)
    except Exception as error:
        if "probe_error" not in state:
            state["probe_error"] = str(error)
            log("probe failed:", error)
        probe = None
    probes.append((now(), (region.width, region.height), probe, clock() - start))


def post_pixel():
    # POST_PIXEL draws into the overlay framebuffer, without the engine's
    # image: it only counts the region's redraws
    screen_draws.append(now())


def paint_cursor(*_):
    # Cycles has no Python draw hook: a paint cursor draws into the window
    # framebuffer after the regions were composited, while the mouse is over
    # the region (warped there at the start)
    _, _, region = view3d()
    if region is not None:
        take_probe(region, region.x, region.y)


def tick():
    ticks.append(now())
    return 0.0 if state.get("running", True) else None


# Views

def view3d():
    win = bpy.context.window_manager.windows[0]
    for area in win.screen.areas:
        if area.type == "VIEW_3D":
            region = [r for r in area.regions if r.type == "WINDOW"][0]
            return win, area, region
    return win, None, None


def set_view(name):
    _, area, _ = view3d()
    rv3d = area.spaces.active.region_3d
    rv3d.view_perspective = "PERSP"
    rv3d.view_matrix = state["views"][name]
    area.tag_redraw()
    state["view"] = name


def toggle_full_area():
    win, area, region = view3d()
    with bpy.context.temp_override(window=win, area=area, region=region):
        bpy.ops.screen.screen_full_area(use_hide_panels=True)


def setup_scene():
    scene = bpy.context.scene
    camera = scene.camera
    base = camera.matrix_world.normalized()
    yaw = Matrix.Rotation(math.radians(args.move_angle), 4, "Y")
    state["views"] = {"A": base.inverted(), "B": (base @ yaw).inverted()}
    info = {"blend": bpy.data.filepath, "blender": bpy.app.version_string, "engine": args.engine}

    if args.engine == "CYCLES":
        scene.render.engine = "CYCLES"
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "HIP"
        prefs.get_devices()
        for d in prefs.devices:
            d.use = d.type == "HIP"
        info["devices"] = [d.name for d in prefs.devices if d.use]
        scene.cycles.device = "GPU"
        scene.cycles.preview_samples = args.cycles_samples
        scene.cycles.use_preview_denoising = args.denoise
        info["preview_samples"] = args.cycles_samples
    else:
        scene.render.engine = "LUXCORE"
        vp = scene.luxcore.viewport
        vp.device = args.device
        vp.pixel_size = args.pixel_size
        vp.halt_time = args.halt
        vp.use_denoiser = args.denoise
        if args.device == "OCL":
            prefs = g0_blc.preferences()
            prefs.gpu_backend = "HIP"
            devices = scene.luxcore.devices
            devices.use_native_cpu = False
            devices.update_devices_if_necessary()
            for d in devices.devices:
                d.enabled = d.type == "HIP_GPU"
            info["devices"] = [d.name for d in devices.devices if d.enabled]
            if not info["devices"]:
                raise RuntimeError("no HIP device for LuxCore")
            info["film_device"] = prefs.film_device
        info.update(device=args.device, pixel_size=args.pixel_size, halt=args.halt)
    info["denoise"] = args.denoise
    return info


# The plan: a generator yielding the delay before its next step

def wait_paused(timeout):
    """Yield until the LuxCore session is paused (Cycles: the halt time)"""
    start = now()
    if args.engine == "CYCLES":
        while now() - start < args.halt:
            yield 0.1
        return
    while now() - start < timeout:
        engine = state["engine"]
        try:
            if engine is not None and engine.session is not None and engine.session.IsInPause():
                return
        except ReferenceError:
            pass
        yield 0.1
    event("pause_timeout")


def session_paused():
    """Whether the render is paused (Cycles: the halt time since the last move)"""
    if args.engine == "CYCLES":
        return now() - state.get("last_move", 0.0) >= args.halt
    try:
        engine = state["engine"]
        return bool(engine and engine.session and engine.session.IsInPause())
    except ReferenceError:
        return False


def move():
    source = state["view"]
    target = "A" if source == "B" else "B"
    kind = "paused" if session_paused() else "running"
    set_view(target)
    state["last_move"] = now()
    event("move", kind=kind, source=source, target=target)


def plan():
    yield 1.0
    toggle_full_area()
    yield 1.0
    win, area, region = view3d()
    space = area.spaces.active
    space.overlay.show_overlays = False
    space.show_gizmo = False
    space.lens = bpy.context.scene.camera.data.lens
    space.clip_start = bpy.context.scene.camera.data.clip_start
    space.clip_end = bpy.context.scene.camera.data.clip_end
    info = setup_scene()
    set_view("A")
    space.shading.type = "SOLID"
    bpy.types.SpaceView3D.draw_handler_add(post_pixel, (), "WINDOW", "POST_PIXEL")
    if args.engine == "CYCLES":
        bpy.context.window_manager.draw_cursor_add(paint_cursor, (), "VIEW_3D", "WINDOW")
        win.cursor_warp(region.x + region.width // 2, region.y + region.height // 4)
    bpy.app.timers.register(tick, first_interval=0.0)
    info["region"] = (region.width, region.height)
    event("setup", **info)
    yield 1.5

    event("rendered", view="A")
    space.shading.type = "RENDERED"
    yield 1.0
    yield from wait_paused(args.halt + args.first_pause_timeout)
    yield 0.5
    event("ref", view="A")
    yield 0.3

    move()
    yield 1.0
    yield from wait_paused(args.halt + 30)
    yield 0.5
    event("ref", view="B")
    yield 0.3

    # The first move resumes the render, the next ones come while it runs
    move()
    for _ in range(args.moves):
        yield args.move_gap
        move()
    for _ in range(args.paused_moves - 2):
        yield 1.0
        yield from wait_paused(args.halt + 30)
        yield 0.5
        move()
    yield 1.5

    if args.object_moves:
        objects = [o for o in bpy.context.scene.objects if o.type == "MESH" and o.visible_get()]
        obj = bpy.data.objects[args.object] if args.object else objects[0]
        size = max(obj.dimensions) or 1.0
        for i in range(args.object_moves):
            obj.location.x += (0.01 if i % 2 == 0 else -0.01) * size
            event("object_move", object=obj.name, kind="paused" if session_paused() else "running")
            yield args.move_gap
        yield 1.0

    if not args.no_resize:
        toggle_full_area()
        event("resize")
        yield 5.0
        toggle_full_area()
        event("resize")
        yield 5.0

    event("done")


steps = plan()


def step():
    try:
        return next(steps)
    except StopIteration:
        finish()
    except Exception:
        import traceback
        log(traceback.format_exc())
        finish()
    return None


# Analysis

def distance(a, b):
    return float(np.abs(a - b).mean())


def probe_at(t):
    """The last probe drawn before t"""
    best = None
    for p in probes:
        if p[0] > t:
            break
        if p[2] is not None:
            best = p
    return best


def stats(values):
    if not values:
        return None
    v = np.array(values)
    return {"n": len(v), "mean": float(v.mean()), "median": float(np.median(v)),
            "p90": float(np.percentile(v, 90)), "max": float(v.max())}


def ms(summary):
    if summary is None:
        return None
    return {k: (round(x * 1000, 2) if k != "n" else x) for k, x in summary.items()}


def gaps_in(t0, t1):
    t = [x for x in ticks if t0 <= x <= t1]
    if len(t) < 2:
        return {"max": None, "blocked": None}
    g = np.diff(t)
    return {"max_ms": round(float(g.max()) * 1000, 1),
            "blocked_ms": round(float(g[g > 0.03].sum()) * 1000, 1),
            "ticks": len(t)}


def analyze():
    result = {"args": vars(args), "events": [(round(t, 4), n, d) for t, n, d in events]}
    by_name = {}
    for t, name, details in events:
        by_name.setdefault(name, []).append((t, details))
    refs = {d["view"]: probe_at(t) for t, d in by_name.get("ref", [])}
    refs = {k: v[2] for k, v in refs.items() if v is not None}
    t_end = by_name["done"][0][0] if "done" in by_name else now()

    # Time to the first frame
    t_rendered = by_name["rendered"][0][0]
    # What the region showed before: the solid view (Cycles), or LuxCore's
    # first draw, before its session started
    solid = probe_at(t_rendered) or next(
        (p for p in probes if p[0] > t_rendered and p[2] is not None), None)
    first = {}
    if "A" in refs and solid is not None:
        for t, size, p, _ in probes:
            if t <= t_rendered or p is None:
                continue
            if distance(p, refs["A"]) < 0.5 * distance(solid[2], refs["A"]):
                first["screen_s"] = round(t - t_rendered, 3)
                break
    film_draws = [d for d in draws if d["t"] > t_rendered and "film" in d]
    if film_draws:
        first["film_s"] = round(film_draws[0]["t"] + film_draws[0]["total"] - t_rendered, 3)
    t_first = t_rendered + first.get("screen_s", first.get("film_s", 0))
    first["main_thread"] = gaps_in(t_rendered, t_first + 0.2)
    first["calls"] = [(round(t - t_rendered, 3), n, round(d, 3)) for t, n, d in calls
                      if t >= t_rendered - 0.5]
    result["first_frame"] = first

    # Camera moves
    moves = []
    for t, d in by_name.get("move", []):
        source, target = refs.get(d["source"]), refs.get(d["target"])
        entry = {"kind": d["kind"], "t": round(t, 3)}
        if source is None or target is None:
            moves.append(entry)
            continue
        drawn = 0
        for tp, size, p, _ in probes:
            if tp <= t or p is None:
                continue
            drawn += 1
            if distance(p, target) < distance(p, source):
                entry["latency_ms"] = round((tp - t) * 1000, 1)
                entry["draws"] = drawn
                break
        entry["main_thread"] = gaps_in(t, t + entry.get("latency_ms", 1000) / 1000)
        moves.append(entry)
    result["moves"] = moves
    for kind in ("running", "paused"):
        lat = [m["latency_ms"] / 1000 for m in moves if m["kind"] == kind and "latency_ms" in m]
        result[f"move_{kind}"] = ms(stats(lat))

    # Resize: to the first film drawn from a session created after the change
    resizes = []
    for t, _ in by_name.get("resize", []):
        entry = {"t": round(t, 3)}
        new_sessions = [c for c in calls if c[1] == "Start" and c[0] > t]
        if new_sessions:
            t_start = new_sessions[0][0]
            later = [d for d in draws if d["t"] > t_start + new_sessions[0][2] and "film" in d]
            if later:
                entry["latency_ms"] = round((later[0]["t"] + later[0]["total"] - t) * 1000, 1)
            entry["export_ms"] = [round(c[2] * 1000, 1) for c in calls
                                  if c[1] == "create_session" and t < c[0] < t_start]
            entry["restart_ms"] = [round(c[2] * 1000, 1) for c in calls
                                   if c[1] == "restart_session" and t < c[0] < t_start]
            entry["start_ms"] = round(new_sessions[0][2] * 1000, 1)
        entry["main_thread"] = gaps_in(t, t + entry.get("latency_ms", 3000) / 1000)
        resizes.append(entry)
    result["resizes"] = resizes

    # Object moves: the view_update() that applies each one
    object_moves = []
    for t, details in by_name.get("object_move", []):
        later = [u for u in updates if u["t"] >= t and "edit" in u]
        if later:
            u = later[0]
            entry = {"kind": details.get("kind"), "delay_ms": round((u["t"] - t) * 1000, 1)}
            for key in ("total", "edit", "begin_edit", "end_edit", "lights"):
                entry[key + "_ms"] = round(u.get(key, 0.0) * 1000, 2)
            entry["lights"] = u.get("lights_n", 0)
            object_moves.append(entry)
    result["object_moves"] = object_moves
    if object_moves:
        result["object_move"] = {key: ms(stats([m[key + "_ms"] / 1000 for m in object_moves]))
                                 for key in ("total", "edit", "end_edit", "lights")}

    # Steady state while rendering (before the first halt) and while paused
    t_ref_a = by_name["ref"][0][0] if by_name.get("ref") else t_end
    t_first_film = film_draws[0]["t"] if film_draws else t_first
    windows = {"rendering": (t_first_film + 0.2, t_first_film + min(args.halt - 1.0, 4.0))}
    if args.engine == "LUXCORE":
        paused = [d["t"] for d in draws if d.get("paused") and d["t"] < t_ref_a]
        if paused:
            windows["paused"] = (paused[0] + 0.2, t_ref_a)
    else:
        windows["paused"] = (t_ref_a - 0.6, t_ref_a)
    steady = {}
    for name, (t0, t1) in windows.items():
        if t1 <= t0:
            continue
        span = t1 - t0
        w = {"span_s": round(span, 2),
             "screen_draws_per_s": round(sum(1 for t in screen_draws if t0 <= t < t1) / span, 1),
             "probe_ms": ms(stats([p[3] for p in probes if t0 <= p[0] < t1])),
             "main_thread": gaps_in(t0, t1)}
        ds = [d for d in draws if t0 <= d["t"] < t1]
        if ds:
            w["view_draws_per_s"] = round(len(ds) / span, 1)
            parts = {}
            for key in ("total", "changes", "edit", "wait", "stats", "film", "getoutput",
                        "draw", "pretty"):
                parts[key] = ms(stats([d.get(key, 0.0) for d in ds]))
            film = [d["film"] - d.get("getoutput", 0.0) for d in ds if "film" in d]
            parts["copies"] = ms(stats(film))
            w["view_draw_ms"] = parts
            new = 0
            last = None
            for d in ds:
                if "samples" in d:
                    if last is not None and d["samples"] != last:
                        new += 1
                    last = d["samples"]
            w["draws_with_new_samples"] = round(new / len(ds), 3)
            frame_times = [d["frame_time"] for d in ds if "frame_time" in d]
            if frame_times:
                w["frame_time_ms"] = ms(stats(frame_times))
        thread_reads = [d for t, d in reads if t0 <= t < t1]
        if thread_reads:
            w["thread_reads_per_s"] = round(len(thread_reads) / span, 1)
            w["thread_read_ms"] = ms(stats(thread_reads))
        steady[name] = w
    result["steady"] = steady
    return result


def finish():
    state["running"] = False
    engine = state["engine"]
    if args.save_config and engine is not None:
        try:
            engine.session.GetRenderConfig().Save(os.path.join(args.out, "renderconfig.bcf"))
            log("saved renderconfig.bcf")
        except Exception as error:
            log("saving the render config failed:", error)
    try:
        result = analyze()
    except Exception:
        import traceback
        log(traceback.format_exc())
        result = {"error": "analysis failed"}
    with open(os.path.join(args.out, "timing.json"), "w") as f:
        json.dump(result, f, indent=1, default=str)
    sizes = [len(p[2]) for p in probes if p[2] is not None]
    length = max(set(sizes), key=sizes.count) if sizes else 0
    kept = [p for p in probes if p[2] is not None and len(p[2]) == length]
    np.savez_compressed(os.path.join(args.out, "probes.npz"),
                        t=np.array([p[0] for p in kept]),
                        probe=np.array([p[2] for p in kept]),
                        ticks=np.array(ticks))
    with open(os.path.join(args.out, "draws.json"), "w") as f:
        json.dump(draws, f)
    with open(os.path.join(args.out, "reads.json"), "w") as f:
        json.dump(reads, f)
    for key in ("first_frame", "move_running", "move_paused", "moves", "resizes", "object_move",
                "object_moves", "steady"):
        if key in result:
            log(key, json.dumps(result[key], default=str))
    restore_manifest()
    bpy.ops.wm.quit_blender()


if args.engine == "LUXCORE":
    instrument_luxcore()
bpy.app.timers.register(step, first_interval=2.0, persistent=True)
