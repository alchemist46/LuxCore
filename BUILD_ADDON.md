# Building the BlendLuxCore addon zip

How to build the addon zip on the Windows machine and where it goes. The zip lands in `dist/` (git-ignored); load it in Blender with Preferences > Get Extensions > Install from Disk.

The addon repo is `C:\Users\Ricardo\Documents\GitHub\BlendLuxCore`. Its `scripts/package_bundle.py` builds the zips with Blender's own `extension build` command, on a clean copy of the committed files (`git archive HEAD`), so **commit the addon changes first** (`--worktree` packages the working tree instead, to test). Nothing in the repo is modified by a build, and no compiler or CMake is needed: only `python` and `blender`, found in the PATH, through `--blender <path>`, or through `BLENDER_ROOT` (the folder of the executable).

There are two kinds of zip:

- **Platform zips** (`BlendLuxCore-<version>-<platform>.zip`, for the users): `pyluxcore` is inside, listed in the `wheels` of the manifest, and Blender installs it by itself, offline, with "Install from Disk". Built by CI too (`addon-bundle.yml`, called by the wheel releaser, which adds the zips to the release).
- **The slim zip** (`BlendLuxCore-<version>.zip`, `--slim`, no platform in its name): no `pyluxcore`, the addon downloads it at the first start (see `luxloader/__init__.py` in the addon) or takes the wheel of `blc_settings.json`. This is what the install-into-Blender script here uses, with the settings file of this machine pointing at the local wheel.

When the zip holds a `pyluxcore` wheel, the loader skips the download, unless `blc_settings.json` exists (the developer override).

## Build the platform zips

```bash
cd /c/Users/Ricardo/Documents/GitHub/BlendLuxCore
BLENDER_ROOT="C:/Program Files (x86)/Steam/steamapps/common/Blender" python scripts/package_bundle.py --wheels <folder with the pyluxcore wheels> --platform windows-x64 --out /c/Users/Ricardo/Documents/GitHub/LuxCore/dist
```

It picks the wheels of the platform for Python 3.11 (Blender 4.5) and 3.13 (Blender 5.x), plus the CUDA NVRTC wheel pyluxcore requires (taken from the folder, or downloaded with pip). A platform with no wheel in the folder is skipped (`--python 13` for the Python of Blender 5 only). The Windows zip with both Pythons is about 250 MB (the pyluxcore wheel is 90 MB, NVRTC 76 MB).

Check a zip in a clean profile, offline, with no settings file: `BLENDER_USER_RESOURCES=<empty dir> blender --offline-mode --command extension install-file -r user_default --enable <zip>`, then enable the extension in `blender -b` and check `BlendLuxCore <version> registered (with pyluxcore <version>)`.

## Build the slim zip into `dist/`

```bash
cd /c/Users/Ricardo/Documents/GitHub/BlendLuxCore
rm -f /c/Users/Ricardo/Documents/GitHub/LuxCore/dist/BlendLuxCore-*.zip
BLENDER_ROOT="C:/Program Files (x86)/Steam/steamapps/common/Blender" python scripts/package_bundle.py --slim --out /c/Users/Ricardo/Documents/GitHub/LuxCore/dist
```

- `BLENDER_ROOT` is a Windows path (Python does not read the `/c/...` form), because the Steam Blender is not in the PATH.
- The zip is `dist/BlendLuxCore-<version>.zip`. The version is read from `blender_manifest.toml`; keep it equal to the core's version in `build-system/build-settings.json`.
- `--tag Latest` builds a nightly-style package named "Latest".

## Build and install into Blender 5.2

```bash
bash /c/Users/Ricardo/lc-work/addon-install.sh
```

It builds the slim zip as above, backs up the installed copy to `lc-work/addon-backup-<date>`, and runs `blender --command extension install-file -r user_default --enable <zip>`.

- **Close Blender first.** The script refuses to run while `blender.exe` is open: the install cannot replace pyluxcore's DLLs while they are locked, and leaves a broken `pyluxcore`. Never kill Blender; wait for it to close.
- Many `Failed to remove ... Access is denied` lines for old `.local` pyluxcore DLLs can appear even with Blender closed. The install still works.
- Check it with `blender -b`: the start must print `BlendLuxCore <version> registered (with pyluxcore <version>)`.
- Do not bump `PYLUXCORE_VERSION` in the addon's `luxloader/__init__.py` until that version is on PyPI.

## Order after a commit

1. Commit the addon (or core) change.
2. Check `version` in the addon's `blender_manifest.toml` equals the core's version; commit it if it changed.
3. Build the slim zip into `dist/` (above).
4. Install it into Blender (above), with Blender closed.
