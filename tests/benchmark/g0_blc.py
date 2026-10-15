# SPDX-FileCopyrightText: 2026 Authors (see AUTHORS.txt)
#
# SPDX-License-Identifier: Apache-2.0

"""Load BlendLuxCore from its source tree in a background Blender, with the
pyluxcore of a given site-packages and without the addon's wheel loader.
Used by the G0 scripts (g0_prepare.py, g0_render.py)."""

import os
import sys
import tempfile
import types

ADDON = "blendluxcore_g0"


def load(addon_dir, site_packages):
    """Enable the addon; returns a function restoring the addon's manifest,
    which the addon rewrites when it runs from its source tree."""
    import addon_utils

    manifest = os.path.join(addon_dir, "blender_manifest.toml")
    manifest_text = open(manifest).read() if os.path.exists(manifest) else None

    sys.path.insert(0, site_packages)
    parent = tempfile.mkdtemp(prefix="g0addon")
    os.symlink(os.path.abspath(addon_dir), os.path.join(parent, ADDON))
    sys.path.insert(0, parent)
    stub = types.ModuleType(ADDON + ".luxloader")
    stub.ensure_pyluxcore = lambda: None
    stub.__getattr__ = lambda name: (lambda *a, **k: None)
    sys.modules[ADDON + ".luxloader"] = stub
    addon_utils.enable(ADDON, default_set=True, handle_error=None)

    # The addon reads light.cycles.cast_shadow, which Blender 4 replaced with
    # light.use_shadow: without it every Cycles point, spot and sun light fails
    # to export. Map it as a fixed addon would.
    import bpy
    probe = bpy.data.lights.new("g0probe", "POINT")
    settings = type(probe.cycles)
    bpy.data.lights.remove(probe)
    if not hasattr(settings, "cast_shadow"):
        settings.cast_shadow = property(lambda self: self.id_data.use_shadow)

    def restore_manifest():
        if manifest_text is not None and open(manifest).read() != manifest_text:
            open(manifest, "w").write(manifest_text)
            print("[G0] restored", manifest)

    return restore_manifest


def submodule(name):
    from importlib import import_module
    return import_module(ADDON + "." + name)


def preferences():
    import bpy
    return bpy.context.preferences.addons[ADDON].preferences
