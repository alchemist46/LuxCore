# Building the BlendLuxCore addon zip

How to build the addon zip on the Windows machine and where it goes. The zip lands in `dist/` (git-ignored); load it in Blender with Preferences > Get Extensions > Install from Disk.

The addon repo is `C:\Users\Ricardo\Documents\GitHub\BlendLuxCore`. The zip is built from a clean `git archive HEAD` copy, not from the working tree, so **commit the addon changes first**. The CMake configure wipes `wheels/` and rewrites the `wheels` line of the manifest, which must not happen in the repo.

There are two kinds of zip:

- **Platform zips** (`BlendLuxCore-<version>-<platform>.zip`, for the users): `pyluxcore` is inside, listed in the `wheels` of the manifest, and Blender installs it by itself, offline, with "Install from Disk". Built by `scripts/package_bundle.py` (below) and by CI (`addon-bundle.yml`, called by the wheel releaser, which adds the zips to the release).
- **The slim zip** (the CMake build below, no platform in its name): no `pyluxcore`, the addon downloads it at the first start (see `luxloader/__init__.py` in the addon) or takes the wheel of `blc_settings.json`. This is what the install-into-Blender scripts here use, with the settings file of this machine pointing at the local wheel.

When the zip holds a `pyluxcore` wheel, the loader skips the download, unless `blc_settings.json` exists (the developer override).

## Build the platform zips

```bash
python scripts/package_bundle.py --wheels <folder with the pyluxcore wheels> --platform windows-x64 --out dist
```

Run in the addon repo, with `blender` in the PATH. It picks the wheels of the platform for Python 3.11 (Blender 4.5) and 3.13 (Blender 5.x), plus the CUDA NVRTC wheel pyluxcore requires (taken from the folder, or downloaded with pip). A platform with no wheel in the folder is skipped. It packages `git archive HEAD`, so commit first; `--worktree` packages the working tree, to test. The Windows zip with both Pythons is about 250 MB (the pyluxcore wheel is 90 MB, NVRTC 76 MB).

Check a zip in a clean profile, offline, with no settings file: `BLENDER_USER_RESOURCES=<empty dir> blender --offline-mode --command extension install-file -r user_default --enable <zip>`, then enable the extension in `blender -b` and check `BlendLuxCore <version> registered (with pyluxcore <version>)`.

## Build the zip and copy it to `dist/`

Run from Git Bash:

```bash
W=/c/Users/Ricardo/lc-work/addon; rm -rf $W && mkdir -p $W/addon-src
(cd /c/Users/Ricardo/Documents/GitHub/BlendLuxCore && git archive HEAD | tar -x -C $W/addon-src)
cp /c/Users/Ricardo/lc-work/addon-build.bat $W/
cmd //c "$(cygpath -w $W/addon-build.bat)" | tail -3
rm -f /c/Users/Ricardo/Documents/GitHub/LuxCore/dist/BlendLuxCore-*.zip
cp $W/addon-src/build/out/BlendLuxCore-*.zip /c/Users/Ricardo/Documents/GitHub/LuxCore/dist/
```

`addon-build.bat` (in `C:\Users\Ricardo\lc-work`, outside Documents so the sync tool leaves it alone) does this inside the archive copy:

```bat
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set PATH=C:\Program Files (x86)\Steam\steamapps\common\Blender;%PATH%
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

- `cmake` is only in the PATH after `vcvars64.bat` (the VS copy), so a plain terminal cannot run the build.
- The addon's CMake looks for `blender`, hence the Steam Blender folder in the PATH.
- A `vswhere.exe is not recognized` line in the output is harmless.
- The zip is `build/out/BlendLuxCore-<version>.zip`. The version is read from `blender_manifest.toml`; keep it equal to the core's version in `build-system/build-settings.json`.
- `-DCMAKE_BUILD_TYPE=Latest` builds a nightly-style package whose version string is "Latest".

## Build and install into Blender 5.2

```bash
bash /c/Users/Ricardo/lc-work/addon-install.sh
```

It does the same build from a clean `git archive HEAD`, backs up the installed copy to `lc-work/addon-backup-<date>`, and runs `blender --command extension install-file -r user_default --enable <zip>`.

- **Close Blender first.** The script refuses to run while `blender.exe` is open: the install cannot replace pyluxcore's DLLs while they are locked, and leaves a broken `pyluxcore`. Never kill Blender; wait for it to close.
- Many `Failed to remove ... Access is denied` lines for old `.local` pyluxcore DLLs can appear even with Blender closed. The install still works.
- Check it with `blender -b`: the start must print `BlendLuxCore <version> registered (with pyluxcore <version>)`.
- Do not bump `PYLUXCORE_VERSION` in the addon's `luxloader/__init__.py` until that version is on PyPI.

## Order after a commit

1. Commit the addon (or core) change.
2. Check `version` in the addon's `blender_manifest.toml` equals the core's version; commit it if it changed.
3. Build the zip and copy it to `dist/` (above).
4. Install it into Blender (above), with Blender closed.
