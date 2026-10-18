#!/usr/bin/env python3
# -*- coding: utf-8 -*-
################################################################################
# Copyright 1998-2026 by authors (see AUTHORS.txt)
#
#   This file is part of LuxCoreRender.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

"""Compare two parity.py runs (e.g. PATHCPU against PATHOCL).

  paritycmp.py <a.npz> <b.npz> [tolerance=0.01]

Per scene: the ratio of the image means, and the median and maximum relative
difference of 8x8 pixel block means (the Monte Carlo noise averages out in a
block, a bias does not; the denominator is floored at 0.05 so dark blocks do
not dominate). A scene fails when the mean ratio or the median block
difference is off by more than the tolerance.
"""

import sys

import numpy as np


def blocks(img, n=8):
    h, w = (img.shape[0] // n) * n, (img.shape[1] // n) * n
    g = img[:h, :w].mean(axis=2)
    return g.reshape(h // n, n, w // n, n).mean(axis=(1, 3))


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
    tol = float(sys.argv[3].split("=")[-1]) if len(sys.argv) > 3 else 0.01
    failures = 0
    print(f"{'scene':28s} {'mean b/a':>9s} {'median blk':>10s} {'max blk':>8s}  result")
    for name in sorted(set(a.files) & set(b.files)):
        ia, ib = a[name], b[name]
        ratio = ib.mean() / max(ia.mean(), 1e-9)
        ba, bb = blocks(ia), blocks(ib)
        rel = np.abs(bb - ba) / np.maximum(ba, 0.05)
        med, mx = float(np.median(rel)), float(rel.max())
        ok = abs(ratio - 1) <= tol and med <= tol
        failures += 0 if ok else 1
        print(f"{name:28s} {ratio:9.4f} {med:10.4f} {mx:8.4f}  {'ok' if ok else 'FAIL'}")
    print(f"\n{failures} failing scene(s)")
    sys.exit(min(failures, 100))


if __name__ == "__main__":
    main()
