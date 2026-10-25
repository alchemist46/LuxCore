# Building the BlendLuxCore addon zip

How to build the addon zip on the Windows machine and where it goes. The zip lands in `dist/` (git-ignored); load it in Blender with Preferences > Get Extensions > Install from Disk.

The addon repo is `C:\Users\Ricardo\Documents\GitHub\BlendLuxCore`. The zip is built from a clean `git archive HEAD` copy, not from the working tree, so **commit the addon changes first**. The CMake configure wipes `wheels/` and rewrites the `wheels` line of the manifest, which must not happen in the repo.

The zip does not contain `pyluxcore`: the addon installs it at runtime (see `luxloader/__init__.py` in the addon).

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
