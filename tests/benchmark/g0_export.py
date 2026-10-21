# SPDX-FileCopyrightText: 2026 Authors (see AUTHORS.txt)
#
# SPDX-License-Identifier: Apache-2.0

"""Export a scene prepared by g0_prepare.py to LuxCore text files, with the
render settings of g0_render.py, for the measurements that need pyluxcore
alone (kerneltiming.py, renderbench-style A/B runs without Blender).

The export is BlendLuxCore's file saver (scene.scn and render.cfg in
<out>/<blend>_LuxCore/<frame>/). With --flat-textures it also writes
scene-flat.scn, where every image map is replaced by a constant of its mean
colour (linear, with the gamma, gain and channel of the map): the difference
in time per sample bounds what the texture work can gain (ROADMAP S0.2).

    blender -b --factory-startup --disable-autoexec out/g0/classroom/classroom-g0.blend \\
        --python-exit-code 1 --python tests/benchmark/g0_export.py -- \\
        --out out/g0/classroom/export --percentage 50 --flat-textures \\
        --addon ~/GitHub/BlendLuxCore --site-packages out/venvs/<venv>/lib64/python3.13/site-packages
"""

import argparse
import os
import sys

import bpy
import numpy as np

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
parser = argparse.ArgumentParser(prog="g0_export.py")
parser.add_argument("--out", required=True, help="output directory")
parser.add_argument("--percentage", type=int, default=100, help="resolution percentage")
parser.add_argument("--spp", type=int, default=64, help="halt spp written in render.cfg")
parser.add_argument("--flat-textures", action="store_true",
                    help="also write scene-flat.scn, the image maps replaced by their mean colour")
parser.add_argument("--addon", required=True, help="BlendLuxCore source directory")
parser.add_argument("--site-packages", required=True, help="site-packages with pyluxcore")
args = parser.parse_args(argv)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g0_blc
restore_manifest = g0_blc.load(args.addon, args.site_packages)
import pyluxcore

scene = bpy.context.scene
scene.render.engine = "LUXCORE"
scene.render.resolution_percentage = args.percentage

# As g0_render.py: the materials g0_prepare.py flagged are transparent to
# LuxCore's shadow rays
get_luxcore_name = g0_blc.submodule("utils").get_luxcore_name
shadow_transparent = [get_luxcore_name(m, False) for m in bpy.data.materials if m.get("g0_shadow_transparent")]
if shadow_transparent:
    scene_parse = pyluxcore.Scene.Parse

    def scene_parse_with_shadow_transparency(self, props):
        for name in shadow_transparent:
            if props.IsDefined(f"scene.materials.{name}.type"):
                props.Set(pyluxcore.Property(f"scene.materials.{name}.transparency.shadow", [1.0, 1.0, 1.0]))
                props.Set(pyluxcore.Property(f"scene.materials.{name}.transparency.shadowoverride", True))
        return scene_parse(self, props)
    pyluxcore.Scene.Parse = scene_parse_with_shadow_transparency

# The g0_render.py settings for the HIP GPU
scene.luxcore.config.device = "OCL"
prefs = g0_blc.preferences()
prefs.gpu_backend = "HIP"
prefs.use_hiprt = True
devices = scene.luxcore.devices
devices.use_native_cpu = False
devices.update_devices_if_necessary()
for d in devices.devices:
    d.enabled = d.type == "HIP_GPU"
tonemapper = scene.camera.data.luxcore.imagepipeline.tonemapper
tonemapper.type = "TONEMAP_LINEAR"
tonemapper.use_autolinear = False
tonemapper.linear_scale = 1
if scene.luxcore.config.sobol_adaptive_strength > 0:
    sys.exit("[G0] adaptive sampling is on: prepare the scene with g0_prepare.py")
halt = scene.luxcore.halt
halt.enable = True
halt.use_time = False
halt.use_noise_thresh = False
halt.use_light_samples = False
halt.use_samples = True
halt.samples = args.spp
scene.luxcore.config.use_animated_seed = False

out = os.path.abspath(args.out)
os.makedirs(out, exist_ok=True)
config = scene.luxcore.config
config.use_filesaver = True
config.filesaver_format = "TXT"
config.filesaver_path = out
bpy.ops.render.render()
restore_manifest()

root = None
for dirpath, dirnames, filenames in os.walk(out):
    if "render.cfg" in filenames:
        root = dirpath
if root is None:
    sys.exit(f"[G0] no render.cfg written in {out}")
print("[G0] exported to", os.path.join(root, "render.cfg"))

if args.flat_textures:
    props = pyluxcore.Properties(os.path.join(root, "scene.scn"))
    image_means = {}
    means = {}
    for name in props.GetAllUniqueSubNames("scene.textures"):
        prefix = name + "."
        if props.Get(prefix + "type").GetString() != "imagemap":
            continue
        path = props.Get(prefix + "file").GetString()
        if not os.path.isabs(path):
            path = os.path.join(root, path)
        gamma = props.Get(prefix + "gamma", [2.2]).GetFloat()
        gain = props.Get(prefix + "gain", [1.0]).GetFloat()
        channel = props.Get(prefix + "channel", ["default"]).GetString()
        key = (path, gamma, channel)
        if key not in image_means:
            img = bpy.data.images.load(path, check_existing=False)
            # The stored values: the map's gamma linearizes them
            img.colorspace_settings.name = "Non-Color"
            ch = img.channels
            px = np.empty(img.size[0] * img.size[1] * ch, dtype=np.float32)
            img.pixels.foreach_get(px)
            bpy.data.images.remove(img)
            px = px.reshape(-1, ch)
            if channel == "alpha":
                mean = [float(px[:, 3].mean()) if ch == 4 else 1.0] * 3
            else:
                rgb = px[:, :3] if ch >= 3 else np.repeat(px[:, :1], 3, axis=1)
                mean = [float(v) for v in np.power(np.clip(rgb, 0.0, None), gamma).mean(axis=0)]
                if channel not in ("default", "rgb"):
                    mean = [float(np.mean(mean))] * 3
            image_means[key] = mean
        means[name] = [gain * v for v in image_means[key]]

    # The order of the export (a texture is defined before its users), each
    # image map replaced by a constant where it was
    flat = pyluxcore.Properties()
    for k in props.GetAllNames():
        tex = ".".join(k.split(".")[:3])
        if k.startswith("scene.textures.") and tex in means:
            if k == tex + ".type":
                flat.Set(pyluxcore.Property(tex + ".type", "constfloat3"))
                flat.Set(pyluxcore.Property(tex + ".value", means[tex]))
            continue
        flat.Set(props.Get(k))
    with open(os.path.join(root, "scene-flat.scn"), "w") as f:
        f.write(flat.ToString())
    print(f"[G0] scene-flat.scn: {len(means)} image maps ({len(image_means)} images) replaced by their mean colour")
