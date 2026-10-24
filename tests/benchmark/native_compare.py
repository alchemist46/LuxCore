#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Authors (see AUTHORS.txt)
#
# SPDX-License-Identifier: Apache-2.0

"""Compare two sets of native_render.py renders of the same scene (two
builds, or two settings): A is the baseline, B the candidate.

For each spp value present in both:
- brightness: the ratio of the image means (RGB), and the spread of the
  ratio over the image: both images are smoothed with a box filter (--blur
  pixels) so that the noise does not count, and the ratio of the smoothed
  images is summarized by its 5/50/95 percentiles over the pixels that are
  not near black. A change of the look shows as a wide spread or a median
  away from 1, noise does not;
- noise (with two renders per spp): the variance of one render from the seed
  pair, mean((a - b)^2) / 2, relative to the squared mean, times the spp;
- speed: the render time per sample (from the JSON logs).

With --png, writes the looks (the scene's pipeline images) of A and B side
by side with the map of the smoothed ratio B/A (blue darker, red brighter,
full scale at a factor 2).

    python tests/benchmark/native_compare.py out/native/b0/LynxSpider out/native/head/LynxSpider \\
        --png out/native/LynxSpider-b0-head.png
"""

import argparse
import json
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("a", help="baseline render directory")
parser.add_argument("b", help="candidate render directory")
parser.add_argument("--tag", default="run")
parser.add_argument("--blur", type=int, default=9, help="box filter size for the ratio map (pixels)")
parser.add_argument("--png", help="write the looks and the ratio map")
parser.add_argument("--label-a", default="A")
parser.add_argument("--label-b", default="B")
args = parser.parse_args()


def load(d):
    d = Path(d)
    log = json.load(open(d / (args.tag + ".json")))
    by_spp = {}
    for r in log["runs"]:
        by_spp.setdefault(r["spp"], []).append(r)
    return d, log, by_spp


def box(img, k):
    # Separable box filter through cumulative sums, edges clamped
    pad = k // 2
    p = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    c = np.cumsum(np.cumsum(p, axis=0), axis=1)
    c = np.pad(c, ((1, 0), (1, 0), (0, 0)))
    h, w = img.shape[:2]
    return (c[k:k + h, k:k + w] - c[:h, k:k + w] - c[k:k + h, :w] + c[:h, :w]) / (k * k)


def lum(img):
    return img @ np.array([0.2126, 0.7152, 0.0722])


da, la, sa = load(args.a)
db, lb, sb = load(args.b)
print(f"A: {la['pyluxcore']} {la.get('engine')} {la.get('devices')}  ({da})")
print(f"B: {lb['pyluxcore']} {lb.get('engine')} {lb.get('devices')}  ({db})")
last = None
for spp in sorted(set(sa) & set(sb)):
    ia = [np.load(da / r["file"]).astype(np.float64) for r in sa[spp]]
    ib = [np.load(db / r["file"]).astype(np.float64) for r in sb[spp]]
    ma, mb = ia[0].reshape(-1, 3).mean(axis=0), ib[0].reshape(-1, 3).mean(axis=0)
    fa, fb = box(np.mean(ia, axis=0), args.blur), box(np.mean(ib, axis=0), args.blur)
    la_, lb_ = lum(fa), lum(fb)
    mask = la_ > 0.02 * la_.mean()
    ratio = lb_[mask] / la_[mask]
    p5, p50, p95 = np.percentile(ratio, [5, 50, 95])
    line = (f"{spp:5d} spp  mean B/A {mb[0] / ma[0]:.3f} {mb[1] / ma[1]:.3f} {mb[2] / ma[2]:.3f}"
            f"  smoothed ratio p5/p50/p95 {p5:.3f}/{p50:.3f}/{p95:.3f}")
    for name, imgs, runs in (("A", ia, sa[spp]), ("B", ib, sb[spp])):
        # The spp actually rendered: a build that halts from the host overshoots
        done = [r.get("pass") or spp for r in runs]
        t = np.median([r["render_s"] / n for r, n in zip(runs, done)]) * 1e3
        line += f"  {name}: {t:.2f} ms/spp"
        if min(done) != spp or max(done) != spp:
            line += f" ({min(done)}-{max(done)} spp done)"
        if len(imgs) >= 2:
            var = np.mean((lum(imgs[0]) - lum(imgs[1])) ** 2) / 2 / lum(imgs[0]).mean() ** 2
            line += f" var*spp {var * np.mean(done[:2]):.4f}"
        nonfinite = sum(r.get("nonfinite", 0) for r in runs)
        if nonfinite:
            line += f" NONFINITE {nonfinite}"
    print(line)
    last = (spp, fa, fb)

if args.png and last:
    from PIL import Image, ImageDraw
    spp, fa, fb = last
    looks = [Image.fromarray(np.load(d / (s[spp][0]["file"][:-4] + "-look.npy"))) for d, s in ((da, sa), (db, sb))]
    r = np.log2(np.clip(lum(fb), 1e-6, None) / np.clip(lum(fa), 1e-6, None))
    r = np.clip(r, -1, 1)
    heat = np.zeros(r.shape + (3,))
    heat[..., 0] = np.clip(r, 0, 1)
    heat[..., 2] = np.clip(-r, 0, 1)
    heat[..., 1] = 1 - np.abs(r)
    heat *= 0.85
    heat[lum(fa) < 0.02 * lum(fa).mean()] = 0.15
    tiles = looks + [Image.fromarray((heat * 255).astype(np.uint8))]
    h = 480
    tiles = [t.resize((round(t.width * h / t.height), h)) for t in tiles]
    c = Image.new("RGB", (sum(t.width for t in tiles) + 16, h), (40, 40, 40))
    x = 0
    d = ImageDraw.Draw(c)
    for t, label in zip(tiles, (args.label_a, args.label_b, "B/A (blue darker, red brighter, x2 full)")):
        c.paste(t, (x, 0))
        d.text((x + 6, 6), f"{label} {spp} spp" if label in (args.label_a, args.label_b) else label, fill=(255, 255, 0))
        x += t.width + 8
    c.save(args.png)
    print("wrote", args.png)
