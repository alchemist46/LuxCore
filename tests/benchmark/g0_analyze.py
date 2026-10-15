#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Authors (see AUTHORS.txt)
#
# SPDX-License-Identifier: Apache-2.0

"""G0 analysis: Cycles against LuxCore from the renders of g0_render.py.

Layout: <dir>/<engine>/{run,ref,cold}.json and the .npy images, with
<engine> = cycles and luxcore; ref and cold are optional. For each engine:

- the time per sample is the slope of a least-squares fit of the wall time
  against the spp (the intercept is the fixed cost of a render), so the run
  needs at least two spp values;
- the noise of each spp value comes from its two renders with different
  seeds: the variance of one render is mean((a - b)^2) / 2, relative to the
  squared mean of the image. No reference is needed;
- the efficiency is 1 / (relative variance * render time): the speed factor
  at equal error is the ratio of the efficiencies.

With a reference (ref.json), the error against it checks the bias: for an
unbiased engine, MSE against the reference = variance + the reference's own
variance, so the "bias check" column stays near 1.

    python3 tests/benchmark/g0_analyze.py out/g0/results/cornell
"""

import argparse
import json
import os
import statistics

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("dir")
parser.add_argument("--min-spp", type=int, default=32, help="smallest spp used for the efficiency")
parser.add_argument("--spp", type=int, nargs="+", help="use only these spp values (to match the engines)")
parser.add_argument("--equal-time", type=float, default=10.0, help="render time of the equal-time RMSE (s)")
parser.add_argument("--target", type=float, default=0.02, help="relative RMSE of the time-to-error figure")
args = parser.parse_args()


def load_log(engine_dir, tag):
    path = os.path.join(engine_dir, tag + ".json")
    return json.load(open(path)) if os.path.exists(path) else None


def analyze(engine):
    d = os.path.join(args.dir, engine)
    run, ref_log, cold = (load_log(d, t) for t in ("run", "ref", "cold"))
    runs = [r for r in run["runs"] if "-spp" in r["name"] and (not args.spp or r["spp"] in args.spp)]
    spp = np.array([r["spp"] for r in runs], dtype=np.float64)
    if len(set(spp)) < 2:
        raise SystemExit(f"{engine}: the run needs at least two spp values")
    slope, intercept = np.polyfit(spp, [r["wall_s"] for r in runs], 1)

    images = {}
    for r in runs:
        images.setdefault(r["spp"], []).append(np.load(os.path.join(d, r["file"])).astype(np.float64))
    if any(len(v) < 2 for v in images.values()):
        raise SystemExit(f"{engine}: every spp value needs two renders (--repeats 2)")
    best = sum(images[max(images)]) / len(images[max(images)])

    ref = ref_spp = None
    if ref_log:
        ref_run = max(ref_log["runs"], key=lambda r: r["spp"])
        ref = np.load(os.path.join(d, ref_run["file"])).astype(np.float64)
        ref_spp = ref_run["spp"]
    norm = (ref if ref is not None else best).mean() ** 2

    rows = []
    for s, imgs in sorted(images.items()):
        var = ((imgs[0] - imgs[1]) ** 2).mean() / 2 / norm
        t = s * slope
        row = {"spp": s, "render_s": t, "relRMSE": var ** 0.5, "var_x_spp": var * s,
               "efficiency": 1 / (var * t)}
        if ref is not None:
            mse = statistics.mean(((i - ref) ** 2).mean() for i in imgs) / norm
            row["bias_check"] = mse / (var * (1 + s / ref_spp))
        rows.append(row)
    used = [r["efficiency"] for r in rows if r["spp"] >= args.min_spp] or [r["efficiency"] for r in rows]
    return {
        "engine": engine, "resolution": run["resolution"], "devices": run.get("devices"),
        "s_per_spp": slope, "fixed_s": intercept, "baseline_s": run.get("baseline_s"),
        "first_render_warm_s": run["first_render_s"],
        "first_render_cold_s": cold["first_render_s"] if cold else None,
        "ref_spp": ref_spp, "rows": rows, "efficiency": statistics.median(used),
        "image": ref if ref is not None else best,
    }


res = {e: analyze(e) for e in ("cycles", "luxcore") if os.path.isdir(os.path.join(args.dir, e))}
report = {}
for e, r in res.items():
    print(f"\n== {e}: {r['resolution'][0]}x{r['resolution'][1]} on {', '.join(r['devices'] or [])}")
    base = f" (1-spp baseline {r['baseline_s']:.2f} s)" if r["baseline_s"] is not None else ""
    print(f"   {r['s_per_spp'] * 1000:.2f} ms/spp, fixed cost {r['fixed_s']:.2f} s per render{base}")
    cold = r["first_render_cold_s"]
    print(f"   first render: warm {r['first_render_warm_s']:.2f} s, cold "
          + (f"{cold:.2f} s" if cold is not None else "not measured"))
    has_ref = r["ref_spp"] is not None
    print("   reference: " + (f"{r['ref_spp']} spp" if has_ref else "none (no bias check)"))
    print("     spp  render_s  relRMSE   var*spp   efficiency" + ("  bias check" if has_ref else ""))
    for row in r["rows"]:
        print(f"   {row['spp']:5d} {row['render_s']:9.2f} {row['relRMSE']:8.4f} {row['var_x_spp']:9.4f} "
              f"{row['efficiency']:12.1f}" + (f"  {row['bias_check']:10.2f}" if has_ref else ""))
    eq = (1 / (r["efficiency"] * args.equal_time)) ** 0.5
    ttt = 1 / (r["efficiency"] * args.target ** 2)
    print(f"   efficiency {r['efficiency']:.1f}: relRMSE {eq:.4f} after {args.equal_time:g} s, "
          f"{ttt:.1f} s to relRMSE {args.target:g}")
    report[e] = {k: v for k, v in r.items() if k != "image"}
    report[e].update({"equal_time_relRMSE": eq, "time_to_target_s": ttt})

if len(res) == 2:
    c, l = res["cycles"], res["luxcore"]
    factor = c["efficiency"] / l["efficiency"]
    time_ratio = l["s_per_spp"] / c["s_per_spp"]
    ratio = l["image"].mean(axis=(0, 1)) / c["image"].mean(axis=(0, 1))
    print(f"\n== Cycles is {factor:.2f}x as fast as LuxCore at equal error "
          f"(time per sample ratio {time_ratio:.2f}, noise per sample ratio {factor / time_ratio:.2f})")
    print(f"   brightness LuxCore/Cycles (RGB): {', '.join(f'{x:.3f}' for x in ratio)}")
    report["cycles_over_luxcore"] = factor
    report["time_per_sample_ratio"] = time_ratio
    report["brightness_ratio"] = ratio.tolist()

try:
    from PIL import Image
    for e, r in res.items():
        x = np.clip(r["image"] / max(r["image"].mean() * 4, 1e-9), 0, 1)
        x = np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)
        Image.fromarray((x[::-1] * 255 + 0.5).astype(np.uint8)).save(os.path.join(args.dir, f"image-{e}.png"))
except ImportError:
    pass

with open(os.path.join(args.dir, "report.json"), "w") as f:
    json.dump(report, f, indent=1)
