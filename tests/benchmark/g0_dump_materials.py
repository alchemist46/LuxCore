# SPDX-FileCopyrightText: 2026 Authors (see AUTHORS.txt)
#
# SPDX-License-Identifier: Apache-2.0

"""Dump the shader graphs of a .blend as JSON, for g0_clean.py.

g0_clean.py rewrites the materials of the G0 classroom export as native
LuxCore materials from the ORIGINAL Cycles node graphs (the G0 export is made
from a copy degraded by g0_prepare.py). This script is the only part that
needs Blender. It writes, for every material of the file (library materials
included), the tree behind the Surface socket of the active output: node type,
the settings the converter reads, every input (linked sub-tree or default
value), the image size and a linear mean of its top rows (g0_clean.py uses it to find
the exported image map of each image), the colour ramps and RGB curves.

The result for the Blender demo classroom is committed as g0_materials.json
(next to this script), so the clean scene can be regenerated without Blender.

    blender -b --factory-startup --disable-autoexec classroom.blend \
        --python tests/benchmark/g0_dump_materials.py -- tests/benchmark/g0_materials.json
"""

import json
import sys

import bpy

EXTRA = ("blend_type", "operation", "distribution", "noise_dimensions", "noise_type", "data_type",
         "interpolation", "projection", "extension", "vector_type", "normalize", "wave_type", "use_clamp")


def val(v):
    if isinstance(v, (int, float, bool, str)):
        return v
    try:
        return [round(float(x), 5) for x in v]
    except TypeError:
        return None


def sock(s, depth):
    if s.is_linked:
        link = s.links[0]
        return {"link": node(link.from_node, depth + 1), "out": link.from_socket.name}
    return {"v": val(getattr(s, "default_value", None))}


def top_mean(image):
    """Mean (linear) of the top 24 rows of the picture (Blender stores rows bottom-up)."""
    w, h = image.size
    ch = image.channels
    rows = min(h, 24)
    px = image.pixels[(h - rows) * w * ch:]
    srgb = image.colorspace_settings.name == "sRGB"
    acc = [0.0] * ch
    n = 0
    step = max(1, (rows * w) // 4000)
    for i in range(0, rows * w, step):
        for c in range(ch):
            v = px[i * ch + c]
            if srgb and c < 3:       # the exported image maps hold linear values
                v = v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4
            acc[c] += v
        n += 1
    return [round(a / max(n, 1), 5) for a in acc]


def node(n, depth=0):
    while n.bl_idname == "NodeReroute" and n.inputs[0].is_linked:
        n = n.inputs[0].links[0].from_node
    d = {"type": n.bl_idname, "name": n.name}
    if depth > 14:
        d["truncated"] = True
        return d
    for a in EXTRA:
        if hasattr(n, a):
            d[a] = val(getattr(n, a))
    t = n.bl_idname
    if t == "ShaderNodeTexImage" and n.image:
        d["image"] = n.image.name
        d["size"] = list(n.image.size)
        d["channels"] = n.image.channels
        d["colorspace"] = n.image.colorspace_settings.name
        try:
            d["top_mean"] = top_mean(n.image)
        except Exception as e:
            d["top_mean_err"] = str(e)
    elif t == "ShaderNodeValToRGB":
        cr = n.color_ramp
        d["ramp"] = {"interp": cr.interpolation,
                     "stops": [[round(e.position, 5)] + [round(x, 5) for x in e.color] for e in cr.elements]}
    elif t == "ShaderNodeRGBCurve":
        d["curves"] = [[[round(p.location[0], 4), round(p.location[1], 4)] for p in c.points]
                       for c in n.mapping.curves]
    elif t == "ShaderNodeBump":
        d["invert"] = n.invert
    elif t == "ShaderNodeAttribute":
        d["attribute_name"] = n.attribute_name
    elif t == "ShaderNodeValue":
        d["v"] = round(n.outputs[0].default_value, 5)
    elif t == "ShaderNodeRGB":
        d["v"] = val(n.outputs[0].default_value)
    elif t == "ShaderNodeGroup":
        d["group"] = n.node_tree.name if n.node_tree else None
    d["in"] = {}
    for s in n.inputs:
        if not s.enabled:
            continue
        k = s.name
        while k in d["in"]:
            k += "_"            # the second Shader of a Mix Shader is "Shader_"
        d["in"][k] = sock(s, depth)
    return d


out = {}
for m in bpy.data.materials:
    if not m.use_nodes or not m.node_tree:
        out[m.name] = {"plain": val(m.diffuse_color)}
        continue
    outputs = [n for n in m.node_tree.nodes if n.bl_idname == "ShaderNodeOutputMaterial"]
    best = next((n for n in outputs if n.is_active_output and n.target in ("ALL", "CYCLES")),
                outputs[0] if outputs else None)
    e = {"library": m.library.filepath if m.library else None}
    if best is not None:
        e["surface"] = sock(best.inputs["Surface"], 0)
    out[m.name] = e

path = sys.argv[sys.argv.index("--") + 1]
with open(path, "w") as f:
    json.dump(out, f, separators=(",", ":"), sort_keys=True)
print("g0_dump_materials: %d materials -> %s" % (len(out), path))
