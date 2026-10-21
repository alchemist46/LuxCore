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
5. Do not bump `PYLUXCORE_VERSION` in the addon's `luxloader/__init__.py` until that version is on PyPI.

## Keep the bug list

[BUGS.md](BUGS.md) in the repo root is the list of known bugs and issues that are not fixed yet. The open work items (features, speed and quality work) are in [ROADMAP.md](ROADMAP.md); the older plans with the measurements of the finished work are in `docs/history/`. Every bug or issue found during a task that is not resolved right away goes into it, so it can be fixed later:

1. Add an entry as soon as the problem is found, before moving on, even when it is unrelated to the current task or only suspected.
2. Each entry has a short title, the date found, the area (engine, material, film, addon, build, tests, and so on), what happens, how to reproduce it (scene, engine, command, spp), and what is already known about the cause or ruled out.
3. Do not add a problem that is fixed in the same task.
4. When an entry is fixed, remove it from the list and mention the entry in the fix commit message.
5. Mention new entries in the final report of the task.
6. Commit changes to `BUGS.md` together with the work during which the problem was found, or on their own.

## Keep render measurements short

Use high sample counts only when a result needs them, so that iterations stay fast. Long references (thousands of spp) are rarely needed:

- Render on the GPU (PATHOCL) by default. Use the CPU engines only when a test needs them: CPU/GPU parity, BiDir or the other CPU-only engines, or a machine without a working GPU backend.

- Noise can be measured without a reference: render the same spp twice with different seeds; the variance of one render is `mean((a - b)^2) / 2`. The G0 scripts (`tests/benchmark/g0_run.sh`, `g0_analyze.py`) work this way, with 64 and 256 spp by default.
- Render a reference only to check bias, or to compare the look of two images, and keep it moderate (around 1024 spp). Do not repeat a bias check that already passed on the same scene and engine unless the change can affect the image.
- Start with a small resolution or a few spp to check that a setup works, before any long run.
- Compare noise figures at matching spp: with the Sobol sampler, variance times spp falls as the spp grows.
