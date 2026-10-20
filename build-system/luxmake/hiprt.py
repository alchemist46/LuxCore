# SPDX-FileCopyrightText: 2026 Authors (see AUTHORS.txt)
#
# SPDX-License-Identifier: Apache-2.0

"""HIPRT command.

HIPRT (https://github.com/GPUOpen-LibrariesAndSDKs/HIPRT), AMD's hardware ray
tracing library, is shipped with pyluxcore (see LUXRAYS_HIPRT_LIBRARY in
src/pyluxcore/CMakeLists.txt). It is built once per version, with its kernels
precompiled for the RDNA2+ architectures, so it needs neither the HIP SDK nor
hiprtc at run time, and published as a GitHub release: the "HIPRT" entry of
build-system/build-settings.json pins it, and the wheel builds download it
(see .github/workflows/wheel-builder.yml and hiprt-builder.yml).

    make hiprt
        Download the pinned HIPRT build into out/hiprt. The next 'make config'
        (and 'make wheel-test') ships it. With --archive <zip>, install a
        downloaded or locally built archive instead.

    python -m build-system.luxmake hiprt --build <HIPRT source tree> [--hip-path <dir>]
        Build HIPRT from a checkout of the pinned ref (patched with
        build-system/hiprt/*.patch) with the HIP SDK / ROCm of --hip-path
        (default: HIP_PATH or ROCM_PATH), then package it.

    python -m build-system.luxmake hiprt --package <HIPRT source tree>
        Package an existing build in out/hiprt-<ref>-<platform>.zip.

The archive keeps the layout of the HIPRT source tree, which the LuxCore
build expects: dist/bin/Release/<library> (with oro_compiled_kernels.hipfb,
which marks a HIPRT with precompiled kernels) and the hiprt/ headers 3 levels
above it.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZIP_DEFLATED, ZipFile

from .constants import PARAMS
from .deps import find_platform
from .utils import fail, logger, run_cmake

# The architectures whose HIPRT kernels are precompiled: the ones of the
# generic families the LuxCore HIP kernels are baked for (see
# GetHIPBakedKernelFamily() in include/luxrays/utils/hipkernel.h). HIPRT is
# used only on them (see IsHIPRTSupported() in hipdevice.cpp).
HIPRT_ARCHS = (
    # RDNA2 (gfx10-3-generic)
    "gfx1030", "gfx1031", "gfx1032", "gfx1033", "gfx1034", "gfx1035", "gfx1036",
    # RDNA3 and 3.5 (gfx11-generic)
    "gfx1100", "gfx1101", "gfx1102", "gfx1103",
    "gfx1150", "gfx1151", "gfx1152", "gfx1153",
    # RDNA4 (gfx12-generic)
    "gfx1200", "gfx1201",
)

# The platforms with HIP
HIPRT_PLATFORMS = ("Windows-X64", "Linux-X64")

LIBRARY_DIR = Path("dist", "bin", "Release")
PRECOMPILED_MARKER = "oro_compiled_kernels.hipfb"


def _settings():
    """Return the HIPRT entry of build-settings.json."""
    settings_file = PARAMS.SOURCE_DIR / "build-system" / "build-settings.json"
    with open(settings_file, encoding="utf-8") as f:
        return json.load(f)["HIPRT"]


def library_name(platform=None):
    """Return the HIPRT library file name hiprtew loads (HIPRTEW_LIBRARY_NAME)."""
    hiprtew_h = PARAMS.SOURCE_DIR / "deps" / "hipew" / "include" / "hiprtew.h"
    for line in hiprtew_h.read_text(encoding="utf-8").splitlines():
        if line.startswith("#define HIPRTEW_VERSION_STR "):
            version = line.split('"')[1]
            break
    else:
        fail("HIPRTEW_VERSION_STR not found in '%s'", hiprtew_h)

    platform = platform or find_platform()
    if platform.startswith("Windows"):
        return f"hiprt{version}64.dll"
    return f"libhiprt{version}64.so"


def archive_name(ref, platform):
    """Return the name of the HIPRT archive of a platform."""
    return f"hiprt-{ref}-{platform}.zip"


def hiprt_dir():
    """Return the directory where the hiprt command downloads HIPRT."""
    return PARAMS.BINARY_DIR / "hiprt"


def get_hiprt_library():
    """Return the HIPRT library downloaded by the hiprt command, or None if
    not available or not the pinned version."""
    directory = hiprt_dir()
    library = directory / LIBRARY_DIR / library_name()
    info_file = directory / "build-info.json"
    if not library.is_file() or not info_file.is_file():
        return None

    info = json.loads(info_file.read_text(encoding="utf-8"))
    if info.get("ref") != _settings()["ref"]:
        logger.warning(
            "Ignoring HIPRT in '%s': it is %s, not the pinned %s "
            "(re-run 'make hiprt')",
            str(directory),
            info.get("ref"),
            _settings()["ref"],
        )
        return None

    return library


def _sha256(path):
    """Return the SHA-256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(dest, local_archive=None):
    """Download the pinned HIPRT build of this platform into dest (or install
    local_archive, a downloaded or locally built archive)."""
    platform = find_platform()
    if platform not in HIPRT_PLATFORMS:
        logger.info("No HIP on %s: HIPRT is not needed", platform)
        return None

    settings = _settings()
    archive = archive_name(settings["ref"], platform)
    url = "/".join(
        (
            "https://github.com",
            settings["user"],
            settings["repo"],
            "releases",
            "download",
            settings["release"],
            archive,
        )
    )

    dest = Path(dest).absolute()
    if dest.exists() and any(dest.iterdir()):
        if not (dest / "build-info.json").is_file():
            fail(
                "'%s' exists and is not a HIPRT download: not replacing it",
                str(dest),
            )

    with tempfile.TemporaryDirectory() as tmpdir:
        if local_archive:
            local_archive = Path(local_archive)
            if local_archive.name != archive:
                fail("'%s' is not '%s'", local_archive.name, archive)
            logger.info("Installing HIPRT from '%s'", str(local_archive))
        else:
            local_archive = Path(tmpdir) / archive
            logger.info("Downloading HIPRT (url='%s')", url)
            try:
                urlretrieve(url, filename=local_archive)
            except OSError as err:
                fail("Unable to download '%s': %s", url, err)

        # The archives are binaries shipped in the wheels: their checksums
        # are pinned with the release
        expected = settings.get("sha256", {}).get(platform)
        actual = _sha256(local_archive)
        if expected:
            if actual != expected:
                fail(
                    "Checksum mismatch for '%s': %s, expected %s",
                    archive,
                    actual,
                    expected,
                )
            logger.info("'%s': checksum OK", archive)
        else:
            logger.warning(
                "No checksum pinned for '%s' in build-settings.json "
                "(HIPRT.sha256.%s): its SHA-256 is %s",
                archive,
                platform,
                actual,
            )

        extract_dir = Path(tmpdir) / "hiprt"
        with ZipFile(local_archive) as zipped:
            zipped.extractall(extract_dir)

        library = extract_dir / LIBRARY_DIR / library_name()
        if not library.is_file():
            fail("'%s' doesn't contain '%s'", archive, library.name)

        if dest.exists():
            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(extract_dir), str(dest))

    library = dest / LIBRARY_DIR / library_name()
    logger.info("HIPRT %s installed: '%s'", settings["ref"], str(library))
    return library


def _git(src, *args):
    """Run a git command in src and return its output."""
    return subprocess.run(
        ["git", "-C", str(src), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def _apply_patches(src):
    """Apply the LuxCore patches to the HIPRT source tree (once)."""
    for patch in sorted((PARAMS.SOURCE_DIR / "build-system" / "hiprt").glob("*.patch")):
        if _git(src, "apply", "--check", str(patch)).returncode == 0:
            res = _git(src, "apply", str(patch))
            if res.returncode:
                fail("Unable to apply '%s':\n%s", patch.name, res.stderr)
            logger.info("Applied '%s'", patch.name)
        elif _git(src, "apply", "--reverse", "--check", str(patch)).returncode == 0:
            logger.info("'%s' already applied", patch.name)
        else:
            fail("'%s' doesn't apply to '%s'", patch.name, str(src))


def build(src, hip_path):
    """Build HIPRT with its kernels precompiled for HIPRT_ARCHS."""
    src = Path(src).absolute()
    if not hip_path:
        hip_path = os.environ.get("HIP_PATH") or os.environ.get("ROCM_PATH")
    if not hip_path:
        fail("No HIP SDK / ROCm: pass --hip-path or set HIP_PATH or ROCM_PATH")

    _apply_patches(src)

    build_dir = src / "build-luxcore"
    env = dict(os.environ)
    env["HIPRT_PRECOMPILE_ARCHS"] = ",".join(HIPRT_ARCHS)
    # The Python of HIPRT's kernel source generation (tools/bakeKernel.*)
    env["PYTHON_BIN"] = sys.executable
    run_cmake(
        [
            "-S", str(src),
            "-B", str(build_dir),
            "-G", "Ninja",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DHIP_PATH={hip_path}",
            # The kernels precompiled and embedded in the library: HIPRT
            # never calls hiprtc
            "-DPRECOMPILE=ON",
            "-DBAKE_COMPILED_KERNEL=ON",
            "-DBAKE_KERNEL=ON",
            "-DFORCE_DISABLE_CUDA=ON",
            "-DNO_UNITTEST=ON",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
        ],
        env=env,
    )
    run_cmake(["--build", str(build_dir)], env=env)


def package(src, output=None):
    """Package a HIPRT build in a zip archive and return its path."""
    src = Path(src).absolute()
    platform = find_platform()
    library = src / LIBRARY_DIR / library_name(platform)
    marker = src / LIBRARY_DIR / PRECOMPILED_MARKER
    if not library.is_file():
        fail("HIPRT library not found: '%s'", str(library))
    if not marker.is_file():
        fail(
            "'%s' not found: the HIPRT build has no precompiled kernels "
            "(PRECOMPILE and BAKE_COMPILED_KERNEL)",
            str(marker),
        )

    ref = _settings()["ref"]
    describe = _git(src, "describe", "--tags").stdout.strip()
    if describe != ref:
        logger.warning(
            "The HIPRT source tree is '%s', not the pinned ref '%s'",
            describe,
            ref,
        )

    info = {
        "ref": ref,
        "describe": describe,
        "commit": _git(src, "rev-parse", "HEAD").stdout.strip(),
        "platform": platform,
        "archs": list(HIPRT_ARCHS),
        "patches": sorted(
            p.name for p in (PARAMS.SOURCE_DIR / "build-system" / "hiprt").glob("*.patch")
        ),
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    output = Path(output) if output else PARAMS.BINARY_DIR / archive_name(ref, platform)
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "w", ZIP_DEFLATED) as zipped:
        zipped.write(library, (LIBRARY_DIR / library.name).as_posix())
        zipped.write(marker, (LIBRARY_DIR / marker.name).as_posix())
        for header in sorted((src / "hiprt").rglob("*.h")):
            relative = header.relative_to(src)
            if "cache" in relative.parts:
                continue
            zipped.write(header, relative.as_posix())
        zipped.write(
            PARAMS.SOURCE_DIR / "deps" / "hipew" / "HIPRT-LICENSE.txt",
            "HIPRT-LICENSE.txt",
        )
        zipped.writestr("build-info.json", json.dumps(info, indent=2) + "\n")

    logger.info("HIPRT packaged in '%s'", str(output))
    logger.info("SHA-256: %s", _sha256(output))
    return output


def hiprt(args):
    """Download, or build and package, HIPRT."""
    if args.build:
        build(args.build, args.hip_path)
        package(args.build, args.output)
    elif args.package:
        package(args.package, args.output)
    else:
        download(args.dest or hiprt_dir(), args.archive)
        if not args.dest:
            logger.info(
                "It will be used by the next 'make config' "
                "(see the LUXRAYS_HIPRT_LIBRARY CMake variable)"
            )
