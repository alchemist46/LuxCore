#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Authors (see AUTHORS.txt)
#
# SPDX-License-Identifier: Apache-2.0

"""A clean native LuxCore classroom for the speed and parity benchmarks.

The G0 classroom (the Blender demo classroom, exported by BlendLuxCore after
g0_prepare.py) is a good workload but a bad picture: g0_prepare.py degrades the
Cycles materials to what the addon's converter supports, so that Cycles and
LuxCore render the same (procedural textures become constants, coordinates and
mappings are dropped, RGB Curves are bypassed), and the converter turns every
Glossy BSDF into an extremely anisotropic metal2 (vroughness 0.05). The Cycles
vs LuxCore comparison needs that scene. A benchmark of LuxCore alone does not.

This script keeps the classroom geometry, the image maps, the lights, the
camera and the object list of the G0 export, and rewrites its materials from the
original Cycles node graphs (g0_materials.json, dumped by g0_dump_materials.py)
as native LuxCore materials and textures, under the same names (so every object
keeps its material):

- Diffuse BSDF -> matte; Glossy BSDF -> metal2 with a fresnelcolor, the
  reflectance of Cycles' Glossy BSDF (colour, no Fresnel), isotropic, u = v =
  roughness^2; Translucent -> mattetranslucent; Transparent -> null;
  Mix Shader -> mix (the Diffuse + Glossy mixes of 1-15% keep their shape);
- Image Texture -> imagemap (the exported image maps, with the Mapping node as
  uvscale/uvdelta); Mix (MIX, MULTIPLY, OVERLAY approximated as 2AB, LIGHTEN)
  -> mix/scale; Color Ramp -> band; RGB Curves -> power per channel (a gamma
  fitted to the curve); Hue/Saturation -> hsv; Bright/Contrast; Invert; Math
  (multiply, divide, power, greater/less than); Object Info Random ->
  objectidnormalized; Attribute Col -> hitpointcolor;
- Noise Texture (3D, fbm) -> fbm with a local mapping; Bump -> bumptex =
  scale(height, strength); a constant height gives no bump;
- the node groups (the coloured plastics), glass, emission, the window portal
  and the area light materials are kept as the addon exported them.

Run (no Blender needed; the export is the 00001 directory of g0_export.py /
the addon, the scene stays out of the repo next to it):

    python3 tests/benchmark/g0_clean.py \\
        C:/Users/Ricardo/g0-scenes/classroom/export2/classroom-g0_LuxCore/00001

It writes scene-clean.scn beside render.cfg; render it with

    python3 tests/benchmark/parity.py ... or kerneltiming.py --size 640x360 \\
        <export>/render.cfg -D 'scene.file = scene-clean.scn'
"""

import argparse
import collections
import json
import math
import re
import struct
import sys
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
PTR_LEN = 13                    # the pointer digits the addon appends to the material names
EXTRA_PREFIXES = ("emission", "transparency", "visibility", "shadowcatcher", "photongi", "holdout",
                  "bumpsamplingdistance", "id", "volume.")


class Unsupported(Exception):
    pass


# ---------------------------------------------------------------------------
# The export
# ---------------------------------------------------------------------------

def load_scene(path):
    """The scene as {materials, textures, other lines}; material and texture
    properties keep their file order (and their quotes)."""
    d = {"materials": collections.OrderedDict(), "textures": collections.OrderedDict(), "other": []}
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line.strip():
            continue
        k, v = line.split(" = ", 1)
        p = k.split(".")
        if p[0] == "scene" and p[1] in ("materials", "textures"):
            d[p[1]].setdefault(p[2], collections.OrderedDict())[".".join(p[3:])] = v
        else:
            d["other"].append(line)
    return d


def unquote(v):
    return v[1:-1] if len(v) >= 2 and v[0] == v[-1] == '"' else v


def toplevel_materials(scene):
    sub = set()
    for p in scene["materials"].values():
        for k in ("material1", "material2", "base", "matt", "mattetrans", "materialtrans"):
            if k in p:
                sub.add(unquote(p[k]))
    return [n for n in scene["materials"] if n not in sub]


def blender_name(exported):
    """Blender material name of an exported (top level) material name."""
    if len(exported) <= PTR_LEN or not exported[-PTR_LEN:].isdigit():
        return None
    return exported[:-PTR_LEN].replace("__", ".")


# ---------------------------------------------------------------------------
# Image maps: which exported file is which Blender image
# ---------------------------------------------------------------------------

def png_top_rows(path, rows=24):
    """(width, height, channels, mean of the first rows, 0..1 per channel) of an
    8 bit non interlaced PNG, by decoding only its first rows."""
    data = open(path, "rb").read()
    w, h, bitdepth, ctype, _, _, interlace = struct.unpack(">IIBBBBB", data[16:29])
    if bitdepth != 8 or interlace or ctype not in (0, 2, 6):
        return w, h, None, None
    ch = {0: 1, 2: 3, 6: 4}[ctype]
    pos = 8
    idat = []
    while pos < len(data):
        ln, typ = struct.unpack(">I4s", data[pos:pos + 8])
        if typ == b"IDAT":
            idat.append(data[pos + 8:pos + 8 + ln])
        pos += 12 + ln
    stride = w * ch
    rows = min(rows, h)
    dec = zlib.decompressobj()
    raw = dec.decompress(b"".join(idat), (stride + 1) * rows)
    prev = bytearray(stride)
    acc = [0] * ch
    n = 0
    for r in range(rows):
        line = raw[r * (stride + 1):(r + 1) * (stride + 1)]
        ft, cur = line[0], bytearray(line[1:])
        if ft == 1:
            for i in range(ch, stride):
                cur[i] = (cur[i] + cur[i - ch]) & 255
        elif ft == 2:
            for i in range(stride):
                cur[i] = (cur[i] + prev[i]) & 255
        elif ft == 3:
            for i in range(stride):
                left = cur[i - ch] if i >= ch else 0
                cur[i] = (cur[i] + ((left + prev[i]) >> 1)) & 255
        elif ft == 4:
            for i in range(stride):
                a = cur[i - ch] if i >= ch else 0
                b = prev[i]
                c = prev[i - ch] if i >= ch else 0
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                cur[i] = (cur[i] + pr) & 255
        prev = cur
        for x in range(0, w, max(1, w // 64)):
            for c in range(ch):
                acc[c] += cur[x * ch + c]
            n += 1
    return w, h, ch, [a / max(n, 1) / 255 for a in acc]


def match_images(blender, scene, export_dir):
    """{Blender image name: exported image file} from the image size and the mean
    of the top rows (only the images that the converted materials use)."""
    files = {}
    for t in scene["textures"].values():
        if unquote(t.get("type", "")) == "imagemap":
            files[unquote(t["file"])] = t
    cands = {}
    for f in sorted(files):
        if f.endswith(".png"):
            w, h, ch, mean = png_top_rows(export_dir / f)
            if mean:
                cands[f] = (w, h, mean)
    images = {}

    def walk(n):
        if n.get("type") == "ShaderNodeTexImage" and "image" in n:
            images[n["image"]] = n
        for s in n.get("in", {}).values():
            if s.get("link"):
                walk(s["link"])
    for m in blender.values():
        if m.get("surface", {}).get("link"):
            walk(m["surface"]["link"])
    result, used = {}, collections.defaultdict(list)
    for name, n in images.items():
        size = tuple(n.get("size", ()))
        bm = n.get("top_mean")
        best = []
        for f, (w, h, mean) in cands.items():
            if (w, h) != size or not bm:
                continue
            dist = sum(abs(a - b) for a, b in zip(mean[:3], bm[:3]))
            best.append((dist, f))
        best.sort()
        if best and best[0][0] < 0.05 and (len(best) == 1 or best[1][0] > best[0][0] * 1.5 + 0.002):
            result[name] = best[0][1]
            used[best[0][1]].append(name)
        elif best:
            print("g0_clean: image %s is ambiguous: %s" % (name, [(round(d, 4), f) for d, f in best[:3]]),
                  file=sys.stderr)
    for f, names in used.items():
        if len(names) > 1:
            print("g0_clean: %s matches %s" % (f, names), file=sys.stderr)
    return result, files


def vertex_colour_use(scene, export_dir):
    """{exported material name: [objects with vertex colours, objects without]}
    from the PLY headers of the objects that use the material."""
    has = {}
    use = collections.defaultdict(lambda: [0, 0])
    objs = collections.defaultdict(dict)
    for line in scene["other"]:
        if line.startswith("scene.objects."):
            k, v = line.split(" = ", 1)
            p = k.split(".")
            objs[p[2]][p[3]] = unquote(v)
    for d in objs.values():
        ply = d.get("ply")
        if not ply or "material" not in d:
            continue
        if ply not in has:
            head = open(export_dir / ply, "rb").read(2000).split(b"end_header")[0]
            has[ply] = b"property uchar red" in head or b"property float red" in head or b"red" in head
        use[d["material"]][0 if has[ply] else 1] += 1
    return use


# ---------------------------------------------------------------------------
# The converter
# ---------------------------------------------------------------------------

def num(x):
    return ("%.6g" % x)


def const_of(s):
    """The tuple of floats of a constant texture value, or None."""
    try:
        v = tuple(float(x) for x in s.split())
    except (ValueError, AttributeError):
        return None
    return v if len(v) in (1, 3) else None


def cstr(v):
    return num(v[0]) if len(v) == 1 else " ".join(num(x) for x in v)


class Converter:
    def __init__(self, blender, image_files, image_props, vcol, stats):
        self.blender = blender
        self.image_files = image_files
        self.image_props = image_props
        self.vcol = vcol
        self.stats = stats
        self.lines = []             # (kind, name, [(key, value)]) in creation order
        self.cache = {}
        self.memo = {}
        self.names = collections.Counter()
        self.mat = ""
        self.exported = ""

    # -- emission ----------------------------------------------------------
    def define(self, kind, base, props):
        """Add a texture or material (props: [(key, already formatted value)]),
        merging identical definitions of textures."""
        key = (kind, tuple(props))
        if kind == "textures" and key in self.cache:
            return self.cache[key]
        self.names[base] += 1
        name = "c_%s_%d" % (base, self.names[base]) if kind == "textures" else base
        self.lines.append((kind, name, props))
        if kind == "textures":
            self.cache[key] = name
            self.stats["tex:" + dict(props)["type"].strip('"')] += 1
        return name

    def tex(self, ttype, **kw):
        props = [("type", '"%s"' % ttype)]
        for k, v in kw.items():
            props.append((k, '"%s"' % v if isinstance(v, str) else num(v)))
        return self.define("textures", self.mat_short() + "_" + ttype, props)

    def mat_short(self):
        return re.sub(r"\W", "", self.mat)[:24]

    # -- constant folding helpers -----------------------------------------
    def t_scale(self, a, b):
        ca, cb = const_of(a), const_of(b)
        if ca and cb:
            if len(ca) == 1 and len(cb) == 3:
                ca = ca * 3
            if len(cb) == 1 and len(ca) == 3:
                cb = cb * 3
            return cstr(tuple(x * y for x, y in zip(ca, cb)))
        if ca == (1.0,):
            return b
        if cb == (1.0,):
            return a
        return self.tex("scale", texture1=a, texture2=b)

    def t_mix(self, amount, a, b):
        ca = const_of(amount)
        if ca == (0.0,) or a == b:
            return a
        if ca == (1.0,):
            return b
        return self.tex("mix", amount=amount, texture1=a, texture2=b)

    def t_add(self, a, b):
        ca, cb = const_of(a), const_of(b)
        if ca == (0.0,):
            return b
        if cb == (0.0,):
            return a
        return self.tex("add", texture1=a, texture2=b)

    def t_sub(self, a, b):
        return self.tex("subtract", texture1=a, texture2=b)

    def t_pow(self, a, e):
        ce = const_of(e)
        if ce == (1.0,):
            return a
        ca = const_of(a)
        if ca and ce and len(ce) == 1 and all(x >= 0 for x in ca):
            return cstr(tuple(x ** ce[0] for x in ca))
        return self.tex("power", base=a, exponent=e)

    # -- sockets -----------------------------------------------------------
    def value(self, s):
        """A node socket as a property value: a number, "r g b" or a texture name."""
        if not s.get("link"):
            v = s.get("v")
            if isinstance(v, list):
                return cstr(tuple(v[:3])) if len(v) >= 3 else num(v[0])
            if v is None:
                return "0"
            return num(float(v))
        n = s["link"]
        key = (self.mat, n["name"], s.get("out"))
        if key not in self.memo:
            self.memo[key] = self.node(n, s.get("out"))
        return self.memo[key]

    def inp(self, n, name, default="0"):
        s = n["in"].get(name)
        return self.value(s) if s is not None else default

    # -- vector inputs ------------------------------------------------------
    def vector(self, s):
        """(coordinates 'uv'|'gen'|'obj', mapping node type, scale, location)."""
        if not s.get("link"):
            return "gen", None, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0)
        n = s["link"]
        if n["type"] == "ShaderNodeMapping":
            base = self.vector(n["in"]["Vector"])
            sc = n["in"].get("Scale", {}).get("v") or [1, 1, 1]
            lo = n["in"].get("Location", {}).get("v") or [0, 0, 0]
            vt = n.get("vector_type", "POINT")
            if vt == "TEXTURE":
                inv = tuple(1.0 / x if x else 1.0 for x in sc[:3])
                return base[0], vt, inv, tuple(-l * i for l, i in zip(lo[:3], inv))
            return base[0], vt, tuple(sc[:3]), tuple(lo[:3])
        if n["type"] == "ShaderNodeTexCoord":
            out = s.get("out")
            return {"UV": "uv", "Generated": "gen", "Object": "obj"}.get(out, "uv"), None, (1.0,) * 3, (0.0,) * 3
        return "uv", None, (1.0,) * 3, (0.0,) * 3

    def uv_mapping(self, vec):
        """The uvmapping2d properties of a vector input (Blender's V runs up, the
        image maps of the export are flipped: v' = 1 - v)."""
        coord, vt, (sx, sy, _), (lx, ly, _) = vec
        if coord != "uv":
            sx = sy = 1.0
            lx = ly = 0.0
        return [("mapping.type", '"uvmapping2d"'), ("mapping.uvindex", "0"), ("mapping.rotation", "0"),
                ("mapping.uvscale", "%s %s" % (num(sx), num(-sy))),
                ("mapping.uvdelta", "%s %s" % (num(lx), num(1.0 - ly)))]

    def mapping3d(self, vec, scale):
        coord, vt, (sx, sy, sz), (lx, ly, lz) = vec
        d = (sx * scale, sy * scale, sz * scale)
        t = (lx * scale, ly * scale, lz * scale)
        m = "%s 0 0 0 0 %s 0 0 0 0 %s 0 %s %s %s 1" % (num(d[0]), num(d[1]), num(d[2]), num(t[0]), num(t[1]), num(t[2]))
        return [("mapping.type", '"localmapping3d"'), ("mapping.transformation", m)]

    # -- nodes -------------------------------------------------------------
    def node(self, n, out):
        t = n["type"]
        h = getattr(self, "n_" + t.replace("ShaderNode", ""), None)
        if h is None:
            self.stats["unsupported node:" + t] += 1
            return "0.5"
        return h(n, out)

    def n_TexImage(self, n, out):
        if out == "Alpha":
            return "1"
        f = self.image_files.get(n.get("image"))
        if not f:
            raise Unsupported("image %s" % n.get("image"))
        props = [("type", '"imagemap"'), ("file", '"%s"' % f)]
        src = self.image_props[f]
        for k in ("gain", "colorspace", "storage", "gamma"):
            if k in src:
                props.append((k, src[k]))
        props += self.uv_mapping(self.vector(n["in"].get("Vector", {})))
        return self.define("textures", self.mat_short() + "_image", props)

    def n_RGB(self, n, out):
        return cstr(tuple(n["v"][:3]))

    def n_Value(self, n, out):
        return num(n["v"])

    def n_Mix(self, n, out):
        fac = self.inp(n, "Factor", "0.5")
        a = self.inp(n, "A", "0.5")
        b = self.inp(n, "B", "0.5")
        bt = n.get("blend_type", "MIX")
        if bt == "MIX":
            return self.t_mix(fac, a, b)
        if bt == "MULTIPLY":
            return self.t_mix(fac, a, self.t_scale(a, b))
        if bt == "OVERLAY":      # 2AB is the overlay below 0.5, the usual range of the multiplied colour
            return self.t_mix(fac, a, self.t_scale(self.t_scale(a, b), "2"))
        if bt == "LIGHTEN":      # max(a, b) = a + (b - a) * (b > a)
            m = self.t_add(a, self.t_scale(self.t_sub(b, a), self.tex("greaterthan", texture1=b, texture2=a)))
            return self.t_mix(fac, a, m)
        if bt == "ADD":
            return self.t_mix(fac, a, self.t_add(a, b))
        self.stats["mix blend %s as MIX" % bt] += 1
        return self.t_mix(fac, a, b)

    n_MixRGB = n_Mix

    def n_Math(self, n, out):
        a = self.inp(n, "Value", "0.5")
        b = self.inp(n, "Value_", "0.5")
        op = n.get("operation")
        if op == "MULTIPLY":
            return self.t_scale(a, b)
        if op == "ADD":
            return self.t_add(a, b)
        if op == "SUBTRACT":
            return self.t_sub(a, b)
        if op == "DIVIDE":
            return self.tex("divide", texture1=a, texture2=b)
        if op == "POWER":
            return self.t_pow(a, b)
        if op == "GREATER_THAN":
            return self.tex("greaterthan", texture1=a, texture2=b)
        if op == "LESS_THAN":
            return self.tex("lessthan", texture1=a, texture2=b)
        self.stats["unsupported math:" + str(op)] += 1
        return a

    def n_Invert(self, n, out):
        c = self.inp(n, "Color", "0.5")
        fac = self.inp(n, "Fac", "1")
        return self.t_mix(fac, c, self.t_sub("1", c))

    def n_Gamma(self, n, out):
        return self.t_pow(self.inp(n, "Color", "0.5"), self.inp(n, "Gamma", "1"))

    def n_HueSaturation(self, n, out):
        c = self.inp(n, "Color", "0.5")
        fac = self.inp(n, "Factor", "1")
        h = self.tex("hsv", texture=c, hue=self.inp(n, "Hue", "0.5"), saturation=self.inp(n, "Saturation", "1"),
                     value=self.inp(n, "Value", "1"))
        return self.t_mix(fac, c, h)

    def n_BrightContrast(self, n, out):
        return self.tex("brightcontrast", texture=self.inp(n, "Color", "0.5"),
                        brightness=self.inp(n, "Bright", "0"), contrast=self.inp(n, "Contrast", "0"))

    def n_ValToRGB(self, n, out):
        ramp = n["ramp"]
        stops = ramp["stops"]
        props = [("type", '"band"'),
                 ("interpolation", '"none"' if ramp["interp"] == "CONSTANT" else '"linear"'),
                 ("amount", '"%s"' % self.inp(n, "Fac", "0.5"))]
        for i, st in enumerate(stops):
            props.append(("offset%d" % i, num(st[0])))
            props.append(("value%d" % i, cstr(tuple(st[1:4]))))
        return self.define("textures", self.mat_short() + "_band", props)

    @staticmethod
    def curve_gamma(points):
        """The exponent of x^g that fits the interior points of an RGB curve."""
        gs = []
        for x, y in points:
            if 0.02 < x < 0.98 and y > 0.005:
                gs.append(math.log(y) / math.log(x))
        if not gs:
            return 1.0
        gs.sort()
        return gs[len(gs) // 2]

    def n_RGBCurve(self, n, out):
        c = self.inp(n, "Color", "0.5")
        gr, gg, gb, gc = (self.curve_gamma(p) for p in n["curves"])
        g = (gr * gc, gg * gc, gb * gc)
        self.stats["rgb curve gamma"] += 1
        if max(g) - min(g) < 0.02:
            return self.t_pow(c, num(g[0]))
        ch = [self.t_pow(self.tex_split(c, i), num(g[i])) for i in range(3)]
        return self.tex("makefloat3", texture1=ch[0], texture2=ch[1], texture3=ch[2])

    def tex_split(self, c, i):
        cc = const_of(c)
        if cc:
            return num(cc[i if len(cc) == 3 else 0])
        props = [("type", '"splitfloat3"'), ("texture", '"%s"' % c), ("channel", str(i))]
        return self.define("textures", self.mat_short() + "_split", props)

    def n_TexNoise(self, n, out):
        vec = self.vector(n["in"].get("Vector", {}))
        scale = self.inp(n, "Scale", "5")
        cs = const_of(scale)
        detail = const_of(self.inp(n, "Detail", "2"))
        rough = const_of(self.inp(n, "Roughness", "0.5"))
        if not cs or not detail or not rough:
            raise Unsupported("noise with a linked scale")
        gen = self.gen_size.get(self.mat, 1.0) if vec[0] == "gen" else 1.0
        vec = (vec[0], vec[1], tuple(x / gen for x in vec[2]), vec[3])
        props = [("type", '"fbm"'), ("octaves", str(int(round(detail[0])) + 1)), ("roughness", num(rough[0]))]
        props += self.mapping3d(vec, cs[0])
        fbm = self.define("textures", self.mat_short() + "_fbm", props)
        # Blender's noise is normalised to 0..1 (amplitude sum of the octaves), the fbm is not
        amp = sum(rough[0] ** i for i in range(int(round(detail[0])) + 1))
        return self.t_add(self.t_scale(fbm, num(0.5 / amp)), "0.5")

    def n_TexWave(self, n, out):
        vec = self.vector(n["in"].get("Vector", {}))
        cs = const_of(self.inp(n, "Scale", "5")) or (5.0,)
        props = [("type", '"blender_wood"'), ("woodtype", '"bands"'), ("noisesize", num(0.25)),
                 ("turbulence", num(const_of(self.inp(n, "Distortion", "0")) and 5.0 or 0.0))]
        props += self.mapping3d(vec, cs[0] * 0.5)
        return self.define("textures", self.mat_short() + "_wood", props)

    def n_Attribute(self, n, out):
        idx = 1 if n.get("attribute_name") == "base" else 0
        with_col, without = self.vcol.get(self.exported, (1, 0))
        if not with_col:
            return "0"                  # Cycles reads a missing attribute as black, LuxCore as white
        props = [("type", '"hitpointcolor"'), ("dataindex", str(idx))]
        vcol = self.define("textures", self.mat_short() + "_vcol", props)
        # the exported vertex colours are the sRGB bytes; Cycles' Attribute node gives linear
        return self.t_pow(vcol, "2.2")

    def n_ObjectInfo(self, n, out):
        if out == "Random":
            return self.tex("objectidnormalized")
        if out == "Object Index":
            return self.tex("objectid")
        raise Unsupported("object info " + str(out))

    # -- bumps -------------------------------------------------------------
    def bump(self, s):
        """The bumptex of a Normal socket: scale(height, strength), or None."""
        if not s or not s.get("link"):
            return None
        n = s["link"]
        if n["type"] != "ShaderNodeBump":
            raise Unsupported("normal from " + n["type"])
        height = self.inp(n, "Height", "0")
        if const_of(height) is not None:
            return None                     # a constant height has no slope
        # Cycles' Strength blends the bumped normal in, its Distance is the height of 1.0
        # in scene units: the slope is about strength * distance * dheight/dx
        strength = self.t_scale(self.inp(n, "Strength", "1"), self.inp(n, "Distance", "1"))
        if n.get("invert"):
            strength = self.t_scale(strength, "-1")
        self.stats["bump"] += 1
        return self.t_scale(height, strength)

    # -- shaders -----------------------------------------------------------
    def shader(self, s, base):
        """A Shader socket as the name of a LuxCore material."""
        if not s.get("link"):
            return self.black()
        n = s["link"]
        t = n["type"]
        self.stats["shader:" + t] += 1
        if t == "ShaderNodeBsdfDiffuse":
            props = [("type", '"matte"'), ("kd", '"%s"' % self.inp(n, "Color", "0.8"))]
            b = self.bump(n["in"].get("Normal"))
            if b:
                props.append(("bumptex", '"%s"' % b))
            return self.define("materials", base, props)
        if t in ("ShaderNodeBsdfAnisotropic", "ShaderNodeBsdfGlossy"):
            fres = self.tex("fresnelcolor", kr=self.inp(n, "Color", "0.8"))
            rough = self.inp(n, "Roughness", "0.5")
            r2 = self.t_pow(rough, "2")
            props = [("type", '"metal2"'), ("fresnel", '"%s"' % fres), ("uroughness", '"%s"' % r2),
                     ("vroughness", '"%s"' % r2)]
            b = self.bump(n["in"].get("Normal"))
            if b:
                props.append(("bumptex", '"%s"' % b))
            return self.define("materials", base, props)
        if t == "ShaderNodeBsdfTranslucent":
            props = [("type", '"mattetranslucent"'), ("kr", '"0 0 0"'), ("kt", '"%s"' % self.inp(n, "Color", "0.8"))]
            b = self.bump(n["in"].get("Normal"))
            if b:
                props.append(("bumptex", '"%s"' % b))
            return self.define("materials", base, props)
        if t == "ShaderNodeBsdfTransparent":
            return self.define("materials", "c_null", [("type", '"null"')])
        if t == "ShaderNodeMixShader":
            fac = self.inp(n, "Fac", "0.5") if "Fac" in n["in"] else self.inp(n, "Factor", "0.5")
            m1 = self.shader(n["in"]["Shader"], base + "_a")
            m2 = self.shader(n["in"]["Shader_"], base + "_b")
            cf = const_of(fac)
            if cf == (0.0,):
                return m1
            if cf == (1.0,):
                return m2
            return self.define("materials", base, [("type", '"mix"'), ("material1", '"%s"' % m1),
                                                   ("material2", '"%s"' % m2), ("amount", '"%s"' % fac)])
        raise Unsupported(t)

    def black(self):
        return self.define("materials", "c_black", [("type", '"matte"'), ("kd", '"0 0 0"')])

    def convert(self, bname, exported, gen_size):
        """The new definitions of one material: the lines as created (the first
        of the materials is the top level one, named like the exported)."""
        self.mat = bname
        self.exported = exported
        self.gen_size = gen_size
        start = len(self.lines)
        info = self.blender[bname]
        if "surface" not in info or not info["surface"].get("link"):
            raise Unsupported("no surface")
        try:
            top = self.shader(info["surface"], exported)
        except Unsupported:
            del self.lines[start:]
            live = {n for _, n, _ in self.lines}
            self.cache = {k: v for k, v in self.cache.items() if v in live}
            raise
        new = self.lines[start:]
        if top != exported:             # a Mix Shader with a constant factor: the shader itself is the top level
            new = [(k, exported if n == top else n, p) for k, n, p in new]
        return exported, new


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("export", help="the export directory (the one with render.cfg, scene.scn, imagemap-*)")
    ap.add_argument("--materials", default=str(HERE / "g0_materials.json"),
                    help="the Blender dump (g0_dump_materials.py)")
    ap.add_argument("--scene", default="scene.scn", help="the exported scene to rewrite")
    ap.add_argument("--out", default="scene-clean.scn", help="the new scene file name, beside the old one")
    ap.add_argument("--only", help="convert only the materials whose name contains this (debugging)")
    args = ap.parse_args()

    export = Path(args.export)
    scene = load_scene(export / args.scene)
    blender = json.load(open(args.materials))
    image_files, image_props = match_images(blender, scene, export)
    stats = collections.Counter()
    conv = Converter(blender, image_files, image_props, vertex_colour_use(scene, export), stats)

    top = toplevel_materials(scene)
    converted, kept = {}, []
    for name in top:
        bname = blender_name(name)
        if bname not in blender or (args.only and args.only not in name):
            kept.append(name)
            stats["kept: no Blender material"] += 1
            continue
        try:
            main_name, lines = conv.convert(bname, name, {})
        except Unsupported as e:
            kept.append(name)
            stats["kept: " + str(e)] += 1
            print("g0_clean: %s kept as exported (%s)" % (name, e), file=sys.stderr)
            continue
        converted[name] = (main_name, lines)

    # What the kept materials still need of the export
    need_m, need_t = set(), set()

    def collect(n):
        if n in scene["materials"] and n not in need_m:
            need_m.add(n)
            for v in scene["materials"][n].values():
                collect(unquote(v))
        elif n in scene["textures"] and n not in need_t:
            need_t.add(n)
            for v in scene["textures"][n].values():
                collect(unquote(v))
    for n in kept:
        collect(n)

    out = list(scene["other"])
    for n in scene["textures"]:
        if n in need_t:
            out += ["scene.textures.%s.%s = %s" % (n, k, v) for k, v in scene["textures"][n].items()]
    for n in scene["materials"]:
        if n in need_m:
            out += ["scene.materials.%s.%s = %s" % (n, k, v) for k, v in scene["materials"][n].items()]
    defined = set()
    for name, (main_name, lines) in converted.items():
        old = scene["materials"][name]
        for kind, n, props in lines:
            if (kind, n) in defined:
                continue
            defined.add((kind, n))
            if kind == "materials" and n == name:
                props = props + [(k, v) for k, v in old.items() if k.startswith(EXTRA_PREFIXES)]
            for k, v in props:
                out.append("scene.%s.%s.%s = %s" % (kind, n, k, v))
    dest = export / args.out
    with open(dest, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(out) + "\n")
    print("g0_clean: %d materials converted, %d kept as exported -> %s" % (len(converted), len(kept), dest))
    for k in sorted(stats):
        print("  %-40s %d" % (k, stats[k]))


if __name__ == "__main__":
    main()
