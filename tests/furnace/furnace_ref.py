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

"""Reference furnace values of the bare GGX lobe without energy compensation.

The hemispherical directional albedo E(mu) of the lobe with F = 1 (GGX,
height-correlated Smith), averaged as furnace.py averages its "disk" value: an
orthographic view of a sphere, pixels with r < 0.9 (mu = sqrt(1 - r^2)).
A single-scattering furnace (metal2 or glossy2 coating with the
compensation off) must reproduce these numbers; with the compensation on it
must give 1.

Expected output: alpha 0.05/0.2/0.5/1 -> about 1.001/0.927/0.687/0.370.
"""

import numpy as np

rng = np.random.default_rng(0)


def E(f_cos, n=2_000_000):
    """f_cos(wo, wi) -> f * cos_i; hemisphere integral by uniform sampling."""
    z = rng.random(n)
    phi = 2 * np.pi * rng.random(n)
    s = np.sqrt(1 - z * z)
    wi = np.stack([s * np.cos(phi), s * np.sin(phi), z], -1)
    return lambda wo: f_cos(wo, wi).mean() * 2 * np.pi


def ggx(alpha):
    def D(h):
        return 1 / (np.pi * alpha * alpha * ((h[:, 0] ** 2 + h[:, 1] ** 2) / alpha ** 2 + h[:, 2] ** 2) ** 2)

    def Lam(w):
        return (-1 + np.sqrt(1 + alpha * alpha * (w[..., 0] ** 2 + w[..., 1] ** 2) / w[..., 2] ** 2)) / 2

    def f_cos(wo, wi):
        h = wi + wo
        h /= np.linalg.norm(h, axis=1, keepdims=True)
        G2 = 1 / (1 + Lam(wo) + Lam(wi))
        return D(h) * G2 / (4 * wo[2])
    return f_cos


# mu distribution of the pixels of the disk r < 0.9 (uniform over the disk area)
r = 0.9 * np.sqrt(rng.random(20000))
mus = np.sqrt(1 - r * r)
mus = np.quantile(mus, np.linspace(0.005, 0.995, 40))
for alpha in [0.05, 0.2, 0.5, 1.0]:
    Ef = E(ggx(alpha))
    vals = [Ef(np.array([np.sqrt(1 - mu * mu), 0, mu])) for mu in mus]
    print(f"ggx alpha {alpha}: sphere furnace mean {np.mean(vals):.4f}   "
          f"E(normal) {Ef(np.array([0, 0, 1.])):.4f}  E(60deg) {Ef(np.array([np.sqrt(.75), 0, .5])):.4f}")
