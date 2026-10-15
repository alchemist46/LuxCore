# SPDX-FileCopyrightText: 2026 Authors (see AUTHORS.txt)
#
# SPDX-License-Identifier: Apache-2.0

"""Prepare a .blend for the G0 Cycles/LuxCore A/B (MODERNIZATION_PLAN.md).

Both engines must render the same scene. BlendLuxCore exports the Cycles
material nodes through its reader (export/cycles_node_reader.py), which
supports only a subset of the Cycles nodes and modes. This script rewrites the
material graphs down to that subset, so that Cycles renders what LuxCore gets,
and aligns the render settings of the two engines:

- unsupported nodes: vector outputs (texture coordinates, mapping) are dropped
  (the textures fall back to the UV map, as the reader always does), RGB Curves
  and unsupported Math modes are bypassed, unsupported Mix modes become MIX,
  float/vector Mix nodes become color Mix nodes, procedural textures and
  attributes become a constant, the Sky Texture becomes a constant color
  (--sky-color), displacement is removed;
- the inputs the reader ignores are disconnected (image texture vector, bump
  normal and distance);
- every material uses its Cycles nodes in LuxCore, every light and the world
  use their Cycles settings, the lights are invisible to the camera (as
  BlendLuxCore exports the area lights);
- the same path depth in both engines, no clamping, no glossy filter, no
  adaptive sampling (the addon defaults to an adaptive strength of 0.9), no denoiser, Blackman-Harris 1.5 pixel filter in both;
- no scene animation (it would override the seed), only the combined pass,
  no transparent film.

The result is checked by running the addon's material converter on every
material: any warning fails the script.

Run with Blender (no render is started):

    blender -b --factory-startup --disable-autoexec SRC.blend --python-exit-code 1 \\
        --python tests/benchmark/g0_prepare.py -- --out DST.blend --depth 8 \\
        --addon ~/GitHub/BlendLuxCore \\
        --site-packages out/venvs/<venv>/lib64/python3.13/site-packages
"""

import argparse
import collections
import os
import re
import sys

import bpy

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
parser = argparse.ArgumentParser(prog="g0_prepare.py")
parser.add_argument("--out", required=True, help="output .blend")
parser.add_argument("--depth", type=int, required=True, help="path depth of both engines")
parser.add_argument("--scene", help="scene to keep (default: the active one)")
parser.add_argument("--addon", required=True, help="BlendLuxCore source directory")
parser.add_argument("--site-packages", required=True, help="site-packages with pyluxcore")
parser.add_argument("--sky-color", type=float, nargs=3, default=(0.12, 0.16, 0.22),
                    help="constant color replacing a Sky Texture")
parser.add_argument("--constant", type=float, default=0.5,
                    help="value replacing procedural textures and attributes")
args = parser.parse_args(argv)

# --- Load BlendLuxCore without its wheel loader ------------------------------

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g0_blc
restore_manifest = g0_blc.load(args.addon, args.site_packages)
material_export = g0_blc.submodule("export.material")
errorlog = g0_blc.submodule("utils.errorlog").LuxCoreErrorLog

# --- The node subset of the reader --------------------------------------------

reader_src = open(os.path.join(args.addon, "export", "cycles_node_reader.py")).read()
READER_TYPES = set(re.findall(r'"(ShaderNode\w+|NodeGroup\w+)"', reader_src))
READER_TYPES |= {"ShaderNodeOutputMaterial", "NodeReroute", "NodeFrame"}
MIX_BLENDS = {"MIX", "MULTIPLY", "ADD", "SUBTRACT", "DIVIDE"}
MATH_OPS = {"ADD", "SUBTRACT", "MULTIPLY", "DIVIDE", "GREATER_THAN", "LESS_THAN",
            "POWER", "ABSOLUTE", "ROUND", "MODULO"}
# Inputs the reader never reads: Cycles must not use them either
IGNORED_INPUTS = {
    "ShaderNodeTexImage": ("Vector",),
    "ShaderNodeTexChecker": ("Vector",),
    "ShaderNodeBump": ("Distance", "Normal"),
}

changes = collections.Counter()


def is_mix_rgba(node):
    return node.bl_idname == "ShaderNodeMixRGB" or (
        node.bl_idname == "ShaderNodeMix" and node.data_type == "RGBA")


def supported(node):
    if node.bl_idname not in READER_TYPES:
        return False
    if node.bl_idname == "ShaderNodeMix" and node.data_type != "RGBA":
        return False
    if is_mix_rgba(node) and node.blend_type not in MIX_BLENDS:
        return False
    if node.bl_idname == "ShaderNodeMath" and node.operation not in MATH_OPS:
        return False
    if node.bl_idname == "ShaderNodeGamma":
        # The reader converts Gamma only when an image texture feeds it
        color = node.inputs["Color"]
        return color.is_linked and color.links[0].from_node.bl_idname == "ShaderNodeTexImage"
    return True


# Outputs the reader reads, for the nodes that support only some of them
SUPPORTED_OUTPUTS = {
    "ShaderNodeNewGeometry": {"Position", "Normal"},
    "ShaderNodeObjectInfo": {"Object Index", "Material Index", "Random"},
}


def constant_node(tree, socket_type, value, location):
    if socket_type in ("RGBA", "SHADER"):
        n = tree.nodes.new("ShaderNodeRGB")
        v = list(value) if hasattr(value, "__len__") else [value] * 3
        n.outputs[0].default_value = (v + [1.0] * 4)[:3] + [1.0]
    else:
        n = tree.nodes.new("ShaderNodeValue")
        v = value
        if hasattr(v, "__len__"):
            v = sum(list(v)[:3]) / 3
        n.outputs[0].default_value = v
    n.location = location
    n.label = "G0 constant"
    return n.outputs[0]


def upstream_or_constant(tree, node, in_socket, to_socket):
    """The socket feeding in_socket, or a constant with its default value."""
    if in_socket.is_linked:
        return in_socket.links[0].from_socket
    return constant_node(tree, to_socket.type, in_socket.default_value, node.location)


def rewire_outputs(tree, node, source_for):
    """Replace every outgoing link of node with source_for(out_socket, to_socket)."""
    for out in node.outputs:
        for link in list(out.links):
            to_socket = link.to_socket
            tree.links.remove(link)
            if to_socket.type == "VECTOR":
                # Texture coordinates and normals fall back to the defaults (UV, shading normal)
                changes["vector input dropped"] += 1
                continue
            src = source_for(out, to_socket)
            if src is not None:
                tree.links.new(src, to_socket)


def as_color(v):
    if hasattr(v, "__len__"):
        return list(v)[:3] + [1.0]
    return [v, v, v, 1.0]


def convert_mix_to_rgba(tree, node):
    # Socket names resolve to the enabled socket of the current data type
    links_in = {s.name: s.links[0].from_socket for s in node.inputs if s.enabled and s.is_linked}
    values_in = {s.name: s.default_value for s in node.inputs
                 if s.enabled and not s.is_linked and hasattr(s, "default_value")}
    links_out = [l.to_socket for s in node.outputs if s.enabled for l in s.links]
    old_type = node.data_type
    node.data_type = "RGBA"
    node.blend_type = "MIX"
    for name, src in links_in.items():
        tree.links.new(src, node.inputs[name])
    for name, v in values_in.items():
        if name == "Factor":
            node.inputs[name].default_value = sum(v) / len(v) if hasattr(v, "__len__") else v
        else:
            node.inputs[name].default_value = as_color(v)
    for to_socket in links_out:
        tree.links.new(node.outputs["Result"], to_socket)
    changes[f"Mix {old_type} -> RGBA"] += 1


def fix_node(tree, node):
    t = node.bl_idname
    if t == "ShaderNodeMix" and node.data_type != "RGBA":
        convert_mix_to_rgba(tree, node)
    elif is_mix_rgba(node) and node.blend_type not in MIX_BLENDS:
        changes[f"Mix {node.blend_type} -> MIX"] += 1
        node.blend_type = "MIX"
    elif t == "ShaderNodeMath" and node.operation not in MATH_OPS:
        changes[f"Math {node.operation} bypassed"] += 1
        rewire_outputs(tree, node, lambda out, to: upstream_or_constant(tree, node, node.inputs[0], to))
        tree.nodes.remove(node)
    elif t == "ShaderNodeRGBCurve":
        changes["RGB Curves bypassed"] += 1
        rewire_outputs(tree, node, lambda out, to: upstream_or_constant(tree, node, node.inputs["Color"], to))
        tree.nodes.remove(node)
    elif t == "ShaderNodeGamma":
        color = node.inputs["Color"]
        src = color.links[0].from_socket if color.is_linked else None
        if src is None or src.node.bl_idname in ("ShaderNodeRGB", "ShaderNodeValue"):
            changes["Gamma folded into a constant"] += 1
            v = color.default_value if src is None else src.default_value
            g = node.inputs["Gamma"].default_value
            v = [max(c, 0.0) ** g for c in as_color(v)[:3]]
            rewire_outputs(tree, node, lambda out, to: constant_node(tree, to.type, v, node.location))
        else:
            changes["Gamma bypassed"] += 1
            rewire_outputs(tree, node, lambda out, to: upstream_or_constant(tree, node, color, to))
        tree.nodes.remove(node)
    elif t == "ShaderNodeDisplacement":
        changes["displacement removed"] += 1
        tree.nodes.remove(node)
    elif t == "ShaderNodeTexSky":
        changes["Sky Texture -> constant"] += 1
        rewire_outputs(tree, node, lambda out, to: constant_node(tree, to.type, args.sky_color, node.location))
        tree.nodes.remove(node)
    else:
        changes[f"{t} replaced (vector outputs dropped, others constant)"] += 1
        rewire_outputs(tree, node, lambda out, to: constant_node(tree, to.type, args.constant, node.location))
        tree.nodes.remove(node)


def fix_tree(tree, done, owner):
    if tree in done:
        return
    done.add(tree)
    for node in list(tree.nodes):
        if node.bl_idname == "ShaderNodeGroup" and node.node_tree:
            fix_tree(node.node_tree, done, owner)
    for node in list(tree.nodes):
        for name in IGNORED_INPUTS.get(node.bl_idname, ()):
            sock = node.inputs.get(name)
            if sock is not None and sock.is_linked:
                for link in list(sock.links):
                    tree.links.remove(link)
                changes[f"{node.bl_idname}.{name} disconnected"] += 1
        if node.bl_idname == "ShaderNodeOutputMaterial":
            for name in ("Displacement", "Volume"):
                sock = node.inputs.get(name)
                if sock is not None and sock.is_linked:
                    for link in list(sock.links):
                        tree.links.remove(link)
                    changes[f"output {name} disconnected"] += 1
    for node in list(tree.nodes):
        outputs = SUPPORTED_OUTPUTS.get(node.bl_idname)
        if outputs is None:
            continue
        for out in node.outputs:
            if out.name in outputs:
                continue
            for link in list(out.links):
                to_socket, to_node = link.to_socket, link.to_node
                if (out.name == "Backfacing" and to_node.bl_idname == "ShaderNodeMixShader"
                        and to_socket == to_node.inputs[0]):
                    # Keep the back-side shader; the caller flips the faces so that the
                    # old back side becomes the front (LuxCore mesh lights are one-sided)
                    back = to_node.inputs[2]
                    targets = [l.to_socket for o in to_node.outputs for l in o.links]
                    tree.links.remove(link)
                    for dst in targets:
                        if back.is_linked:
                            tree.links.new(back.links[0].from_socket, dst)
                    tree.nodes.remove(to_node)
                    flip.add(owner)
                    changes["Backfacing mix -> back shader, faces flipped"] += 1
                    continue
                tree.links.remove(link)
                if to_socket.type == "VECTOR":
                    changes["vector input dropped"] += 1
                else:
                    tree.links.new(constant_node(tree, to_socket.type, args.constant, node.location), to_socket)
                    changes[f"{node.bl_idname}.{out.name} -> constant"] += 1
    # Fixing a node can expose another one (e.g. a bypass): repeat until stable.
    # Gamma goes last, so that a constant feeding it is folded.
    for _ in range(10000):
        bad = [n for n in tree.nodes if not supported(n) and n.bl_idname != "ShaderNodeGroup"]
        if not bad:
            return
        bad.sort(key=lambda n: n.bl_idname == "ShaderNodeGamma")
        fix_node(tree, bad[0])
    raise RuntimeError(f"node tree {tree.name}: rewrite did not converge")


# --- Scene ---------------------------------------------------------------------

scene = bpy.data.scenes[args.scene] if args.scene else bpy.context.scene
for other in [s for s in bpy.data.scenes if s != scene]:
    bpy.data.scenes.remove(other)
if bpy.context.window:
    bpy.context.window.scene = scene

def linked_ids():
    return [i for coll in (getattr(bpy.data, a) for a in dir(bpy.data))
            if isinstance(coll, bpy.types.bpy_prop_collection)
            for i in coll if isinstance(i, bpy.types.ID) and i.library]


# Make the linked assets local, so their materials can be rewritten
for _ in range(4):
    linked = linked_ids()
    if not linked:
        break
    for i in linked:
        try:
            i.make_local()
        except Exception as e:
            print(f"[G0] make_local failed for {i!r}: {e}")
# What is left is kept only by a fake user (unused assets): drop it
for i in linked_ids():
    if i.users - int(i.use_fake_user) == 0 and not bpy.data.user_map(subset=[i])[i]:
        i.use_fake_user = False
bpy.data.orphans_purge(do_local_ids=False, do_linked_ids=True, do_recursive=True)
still_linked = [i.name_full for i in linked_ids()]
if still_linked:
    raise RuntimeError(f"still linked: {still_linked[:10]}")
for name in [lib.name for lib in bpy.data.libraries]:
    if name in bpy.data.libraries:
        bpy.data.libraries.remove(bpy.data.libraries[name])

done = set()
flip = set()
for mat in bpy.data.materials:
    if mat.node_tree:
        fix_tree(mat.node_tree, done, mat)
    mat.luxcore.use_cycles_nodes = True

import bmesh
flipped = set()
for ob in scene.objects:
    if ob.type != "MESH" or ob.data in flipped:
        continue
    indices = {i for i, slot in enumerate(ob.material_slots) if slot.material in flip}
    if not indices:
        continue
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    faces = [f for f in bm.faces if f.material_index in indices]
    bmesh.ops.reverse_faces(bm, faces=faces)
    bm.to_mesh(ob.data)
    bm.free()
    flipped.add(ob.data)
    print(f"[G0] flipped {len(faces)} faces of {ob.data.name}")

def blackbody_rgb(temperature):
    """Linear Rec.709 color of a black body, normalized to luminance 1
    (CIE 1931 matching functions, analytic fit of Wyman et al. 2013)."""
    import math
    def g(x, mu, s1, s2):
        return math.exp(-0.5 * ((x - mu) / (s1 if x < mu else s2)) ** 2)
    X = Y = Z = 0.0
    for nm in range(380, 781, 5):
        w = nm * 1e-9
        b = 1 / (w ** 5 * (math.exp(1.4388e-2 / (w * temperature)) - 1))
        X += b * (1.056 * g(nm, 599.8, 37.9, 31.0) + 0.362 * g(nm, 442.0, 16.0, 26.7) - 0.065 * g(nm, 501.1, 20.4, 26.2))
        Y += b * (0.821 * g(nm, 568.8, 46.9, 40.5) + 0.286 * g(nm, 530.9, 16.3, 31.1))
        Z += b * (1.217 * g(nm, 437.0, 11.8, 36.0) + 0.681 * g(nm, 459.0, 26.0, 13.8))
    X, Z = X / Y, Z / Y
    return [3.2406 * X - 1.5372 - 0.4986 * Z, -0.9689 * X + 1.8758 + 0.0415 * Z,
            0.0557 * X - 0.2040 + 1.0570 * Z]


# BlendLuxCore reads only the light color and energy (and an unlinked Emission
# strength) from the Cycles light nodes: fold the nodes into them, for Cycles too
for light in bpy.data.lights:
    if getattr(light, "use_nodes", False) and light.node_tree:
        out = light.node_tree.get_output_node("CYCLES")
        em = out.inputs["Surface"].links[0].from_node if out and out.inputs["Surface"].is_linked else None
        if em is not None and em.bl_idname == "ShaderNodeEmission":
            strength, color = em.inputs["Strength"], em.inputs["Color"]
            s = strength.default_value
            if strength.is_linked:
                src = strength.links[0].from_node
                if src.bl_idname == "ShaderNodeLightFalloff":
                    s = src.inputs["Strength"].default_value
                else:
                    raise RuntimeError(f"light {light.name}: strength from {src.bl_idname}")
            c = list(color.default_value)[:3]
            if color.is_linked:
                src = color.links[0].from_node
                if src.bl_idname == "ShaderNodeBlackbody":
                    c = blackbody_rgb(src.inputs["Temperature"].default_value)
                elif src.bl_idname == "ShaderNodeRGB":
                    c = list(src.outputs[0].default_value)[:3]
                else:
                    raise RuntimeError(f"light {light.name}: color from {src.bl_idname}")
            light.energy *= s
            light.color = [a * b for a, b in zip(light.color, c)]
            # Blender 5 ignores use_nodes = False: leave a neutral node tree instead
            for sock in (strength, color):
                for link in list(sock.links):
                    light.node_tree.links.remove(link)
            strength.default_value = 1
            color.default_value = (1, 1, 1, 1)
            changes["light nodes folded into energy and color"] += 1
        else:
            raise RuntimeError(f"light {light.name}: unsupported node tree")
    if not light.use_shadow:
        # LuxCore lights always cast shadows
        light.use_shadow = True
        changes["light shadows enabled"] += 1
    light.luxcore.use_cycles_settings = True

# Cycles renders the empty material slots with its default material, the addon
# with its own fallback: give them the Cycles default explicitly
default_mat = None
for ob in scene.objects:
    data = ob.data
    if ob.type not in {"MESH", "CURVE", "SURFACE", "META", "FONT"} or not hasattr(data, "materials"):
        continue
    if default_mat is None:
        default_mat = bpy.data.materials.new("G0 default")
        default_mat.use_nodes = True
        nodes = default_mat.node_tree.nodes
        for n in list(nodes):
            if n.bl_idname != "ShaderNodeOutputMaterial":
                nodes.remove(n)
        diffuse = nodes.new("ShaderNodeBsdfDiffuse")
        diffuse.inputs["Color"].default_value = (0.8, 0.8, 0.8, 1)
        default_mat.node_tree.links.new(diffuse.outputs[0], nodes["Material Output"].inputs["Surface"])
        default_mat.luxcore.use_cycles_nodes = True
    if len(ob.material_slots) == 0:
        data.materials.append(default_mat)
        changes["default material added"] += 1
    for slot in ob.material_slots:
        if slot.material is None:
            slot.material = default_mat
            changes["empty slot filled with the default material"] += 1
# BlendLuxCore exports the Cycles area lights invisible to the camera
for ob in scene.objects:
    if ob.type == "LIGHT":
        ob.visible_camera = False
if scene.world:
    scene.world.luxcore.use_cycles_settings = True

# Animation on the scene would override the render settings (the classroom
# demo keyframes cycles.seed, which freezes the noise pattern)
scene.animation_data_clear()

# Only the combined pass: both engines spend time on the others
for vl in scene.view_layers:
    for p in vl.bl_rna.properties:
        if p.identifier.startswith("use_pass_") and p.identifier != "use_pass_combined":
            try:
                if getattr(vl, p.identifier):
                    setattr(vl, p.identifier, False)
                    changes["render pass disabled"] += 1
            except (AttributeError, TypeError):
                pass
scene.render.film_transparent = False

# Cycles
cy = scene.cycles
cy.pixel_filter_type = "BLACKMAN_HARRIS"
cy.device = "GPU"
cy.use_adaptive_sampling = False
cy.use_denoising = False
cy.sample_clamp_direct = 0
cy.sample_clamp_indirect = 0
cy.blur_glossy = 0
cy.caustics_reflective = True
cy.caustics_refractive = True
cy.use_light_tree = True
cy.max_bounces = cy.diffuse_bounces = cy.glossy_bounces = cy.transmission_bounces = args.depth
cy.transparent_max_bounces = args.depth
cy.filter_width = 1.5
scene.view_settings.view_transform = "Standard"
scene.view_settings.look = "None"
scene.view_settings.exposure = 0

# LuxCore
lc = scene.luxcore.config
lc.engine = "PATH"
lc.device = "OCL"
lc.sampler_gpu = "SOBOL"
# The addon's default adaptive strength (0.9) moves samples between pixels
lc.sobol_adaptive_strength = 0
lc.filter_enabled = True
lc.filter = "BLACKMANHARRIS"
lc.filter_width = 1.5
lc.path.depth_total = lc.path.depth_diffuse = lc.path.depth_glossy = lc.path.depth_specular = args.depth
lc.path.use_clamping = False
halt = scene.luxcore.halt
halt.enable = True
halt.use_noise_thresh = False
halt.use_samples = False
halt.use_time = True
halt.time = 60

# --- Check with the addon's converter -------------------------------------------

depsgraph = bpy.context.evaluated_depsgraph_get()
used = set()
tris = 0
for oi in depsgraph.object_instances:
    ob = oi.object
    if ob.type == "MESH":
        ob.data.calc_loop_triangles()
        tris += len(ob.data.loop_triangles)
        used |= {s.material.original for s in ob.material_slots if s.material}

errorlog.errors.clear()
errorlog.warnings.clear()
for mat in sorted(used, key=lambda m: m.name):
    material_export.convert(None, depsgraph, mat, False, obj_name=mat.name)
problems = [f"{e.obj_name}: {e.message}" for e in errorlog.errors + errorlog.warnings]

print("[G0] scene", scene.name, "triangles", tris, "materials used", len(used))
for k, v in sorted(changes.items()):
    print(f"[G0]   {v:4d} x {k}")
for p in problems:
    print("[G0] converter:", p)

restore_manifest()

if problems:
    print("[G0] FAILED: the converter reported problems, not saving")
    sys.exit(1)

bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(args.out), relative_remap=True, compress=True)
print("[G0] saved", args.out)
