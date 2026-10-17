# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Keep the Blender addon in sync

Every time a new feature or option is added to the core (engines, materials, textures, lights, film, halt conditions, render config properties, and so on), check whether the BlendLuxCore addon ([alchemist46/BlendLuxCore](https://github.com/alchemist46/BlendLuxCore)) needs a matching update to expose it. Report the result even when no addon change is needed.

When a feature is done:

1. List the new scene/config properties it introduces, with their defaults.
2. Check whether the addon already has UI and export code for them.
3. If not, add it in the addon, following that repo's own CLAUDE.md. New properties are exported only when they differ from the default, so older `pyluxcore` builds keep working.
4. Test the addon against a freshly built `pyluxcore` wheel in Blender.

## Rebuild and install the addon after each commit

After every commit (in this repo or in the BlendLuxCore repo), rebuild the addon and install it into Blender, so the installed copy always matches the latest commit:

1. Set `version` in the addon's `blender_manifest.toml` to the current build version (the core's version in `build-system/build-settings.json`) and commit it if it changed.
2. Build the zip from a clean `git archive HEAD` copy of the addon, not from the working tree: the CMake configure wipes `wheels/` and rewrites the manifest `wheels` line. In the copy, run `cmake -S . -B <build> -DCMAKE_BUILD_TYPE=Release && cmake --build <build>`; the zip lands in `<build>/out/`.
3. Install it with `blender --command extension install-file -r user_default --enable <zip>`, after backing up the installed copy (`~/.config/blender/5.2/extensions/user_default/blendluxcore`).
4. Check the output says `BlendLuxCore <version> registered (with pyluxcore <version>)`.
5. A commit in this repo that changes the engine also needs a fresh `pyluxcore` wheel, which Blender loads from `path_to_wheel` in `blc_settings.json`. Rebuild the wheel first, or say that it was not rebuilt.
6. Do not bump `PYLUXCORE_VERSION` in the addon's `luxloader/__init__.py` until that version is on PyPI.

## Keep render measurements short

Use high sample counts only when a result needs them, so that iterations stay fast. Long references (thousands of spp) are rarely needed:

- Noise can be measured without a reference: render the same spp twice with different seeds; the variance of one render is `mean((a - b)^2) / 2`. The G0 scripts (`tests/benchmark/g0_run.sh`, `g0_analyze.py`) work this way, with 64 and 256 spp by default.
- Render a reference only to check bias, or to compare the look of two images, and keep it moderate (around 1024 spp). Do not repeat a bias check that already passed on the same scene and engine unless the change can affect the image.
- Start with a small resolution or a few spp to check that a setup works, before any long run.
- Compare noise figures at matching spp: with the Sobol sampler, variance times spp falls as the spp grows.
