# SPDX-FileCopyrightText: 2025 Authors (see AUTHORS.txt)
#
# SPDX-License-Identifier: Apache-2.0

"""Wheel commands: 'make wheel' builds and repairs the pyluxcore wheel.

The wheel is built by scikit-build-core (pyproject.toml), the same backend
CI uses through cibuildwheel, in the configured build tree (so an up-to-date
tree only relinks pyluxcore), then repaired with the platform's tool
(delvewheel, auditwheel, delocate) by repair_wheel(), which CI also calls for
its wheels ('make repair-wheel'). The result is a wheel equal to a CI one.
"""

import importlib.util
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from .constants import PARAMS
from .utils import logger, fail
from .config import config

# The build backend and front end (installed in the Python running luxmake)
BUILD_MODULES = (("build", "build"), ("scikit_build_core", "scikit-build-core"))

# The wheel repair tool of each platform: (module, pip package, command)
REPAIR_TOOLS = {
    "Windows": ("delvewheel", "delvewheel", "delvewheel"),
    "Linux": ("auditwheel", "auditwheel", "auditwheel"),
    "Darwin": ("delocate", "delocate", "delocate-wheel"),
}


def _version():
    """The version of build-settings.json, as 'major.minor.patch[-pre]'."""
    settings_file = PARAMS.SOURCE_DIR / "build-system" / "build-settings.json"
    with open(settings_file, encoding="utf-8") as in_file:
        default_version = json.load(in_file)["DefaultVersion"]
    version = ".".join(default_version[i] for i in ("major", "minor", "patch"))
    if prerelease := default_version.get("prerelease"):
        version = f"{version}-{prerelease}"
    return version


def _check_module(module, package):
    """Fail with an install hint when a Python module is missing."""
    if importlib.util.find_spec(module) is None:
        fail(
            "Python module '%s' not found: run '%s -m pip install %s'",
            module,
            sys.executable,
            package,
        )


def _tool_command(module, package, command):
    """The command line of a tool: its executable, or 'python -m module'."""
    if executable := shutil.which(command):
        return [executable]
    _check_module(module, package)
    return [sys.executable, "-m", module]


def _run(cmd, **kwargs):
    """Run a command, failing on error."""
    cmd = [str(c) for c in cmd]
    logger.info("Running: %s", shlex.join(cmd))
    if subprocess.run(cmd, check=False, **kwargs).returncode:
        fail("Error while executing %s", cmd[0])


def _dependency_dirs(name):
    """The '<name>' directories of the deployed dependencies (bin, lib...)."""
    base = PARAMS.BINARY_DIR / "dependencies" / "full_deploy" / "host"
    return [str(p.absolute()) for p in base.rglob(name) if p.is_dir()]


def make_wheel(args):
    """Build the pyluxcore wheel into PARAMS.WHEELHOUSE_DIR."""
    for module, package in BUILD_MODULES:
        _check_module(module, package)
    system = platform.system()
    if system not in REPAIR_TOOLS:
        fail("Unknown platform: '%s'", system)
    repair_tool = _tool_command(*REPAIR_TOOLS[system])

    # Configure: checks that 'make deps' was run, and ships the pre-baked
    # SPIR-V modules and the HIPRT library when they are there
    config(args)

    # The version read by pyproject.toml
    version = _version()
    (PARAMS.SOURCE_DIR / "SKVERSION").write_text(
        version + "\n", encoding="utf-8"
    )
    logger.info("Building the pyluxcore wheel, version '%s'", version)

    # The same CMake arguments as the CI build (wheel-builder.yml): the conan
    # preset of 'make deps', its generator, and the version
    env = os.environ.copy()
    env["SKBUILD_CMAKE_ARGS"] = ";".join(
        (
            "--preset conan-default",
            f"-DLUXCORE_VERSION={version}",
            "-G Ninja Multi-Config",
        )
    )
    env["SKBUILD_CMAKE_BUILD_TYPE"] = PARAMS.BUILD_TYPE
    env["SKBUILD_BUILD_DIR"] = str(PARAMS.BUILD_DIR.absolute())

    with tempfile.TemporaryDirectory() as raw_dir:
        _run(
            [
                sys.executable,
                "-m",
                "build",
                "--wheel",
                "--no-isolation",
                "--outdir",
                raw_dir,
                PARAMS.SOURCE_DIR,
            ],
            env=env,
        )
        wheels = list(Path(raw_dir).glob("*.whl"))
        if len(wheels) != 1:
            fail("Expected one wheel in '%s', found %s", raw_dir, len(wheels))
        repair_wheel(wheels[0], PARAMS.WHEELHOUSE_DIR, repair_tool)

    # Finally, execute hook if exists
    if PARAMS.WHEEL_HOOK:
        logger.info("Executing hook: %s", PARAMS.WHEEL_HOOK)
        try:
            result = subprocess.check_output(
                shlex.split(PARAMS.WHEEL_HOOK), text=True
            )
        except subprocess.CalledProcessError as err:
            fail(err)
        logger.info(result)


def repair_wheel(wheel, dest_dir, tool=None):
    """Repair a raw pyluxcore wheel into dest_dir: vendor the libraries it
    needs, with the platform's tool."""
    wheel = Path(wheel).absolute()
    dest_dir = Path(dest_dir).absolute()
    dest_dir.mkdir(parents=True, exist_ok=True)
    system = platform.system()
    if system not in REPAIR_TOOLS:
        fail("Unknown platform: '%s'", system)
    if tool is None:
        tool = _tool_command(*REPAIR_TOOLS[system])
    logger.info("Repairing '%s' into '%s'", wheel, dest_dir)
    if system == "Windows":
        _repair_windows(wheel, dest_dir, tool)
    elif system == "Linux":
        _repair_linux(wheel, dest_dir, tool)
    else:
        _repair_macos(wheel, dest_dir, tool)
    logger.info("Wheel: '%s'", dest_dir / wheel.name)


def _repair_windows(wheel, dest_dir, tool):
    """delvewheel: the dependencies' DLLs and the MSVC runtime, from the
    deployed dependencies and the Visual C++ redistributable of the
    compiler (VCToolsRedistDir, set by vcvars)."""
    paths = _dependency_dirs("bin")
    redist = os.environ.get("VCToolsRedistDir", "") + os.environ.get(
        "Platform", ""
    )
    if redist and Path(redist).is_dir():
        paths.append(redist)
        paths += [str(p) for p in Path(redist).rglob("*") if p.is_dir()]
    else:
        logger.warning(
            "VCToolsRedistDir is not set (no vcvars?): the MSVC runtime DLLs "
            "will be taken from the PATH"
        )
    _run(
        tool
        + [
            "repair",
            "-v",
            "--add-path",
            os.pathsep.join(paths),
            # Also patch the imports of the libraries the install puts in
            # the wheel (the OIDN CPU device module, loaded by OIDN at run
            # time under its own name; the HIPRT library), without
            # renaming them
            "--analyze-existing",
            "-w",
            dest_dir,
            wheel,
        ]
    )


def _repair_linux(wheel, dest_dir, tool):
    """auditwheel: the dependencies' shared libraries, for the manylinux
    platform of GLIBC_VERSION (default 2_28, as CI)."""
    env = os.environ.copy()
    lib_dirs = _dependency_dirs("lib") + _dependency_dirs("lib64")
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        lib_dirs + [p for p in (env.get("LD_LIBRARY_PATH"),) if p]
    )
    plat = f"manylinux_{env.get('GLIBC_VERSION', '2_28')}_{platform.machine()}"
    cmd = tool + ["repair", "-w", dest_dir, "--plat", plat, "--only-plat"]
    if PARAMS.BUILD_TYPE == "Release":
        cmd.append("--strip")
    _run(cmd + [wheel], env=env)


def _repair_macos(wheel, dest_dir, tool):
    """delocate: the dependencies' dynamic libraries."""
    env = os.environ.copy()
    lib_dirs = _dependency_dirs("lib")
    env["DYLD_LIBRARY_PATH"] = os.pathsep.join(
        lib_dirs + [p for p in (env.get("DYLD_LIBRARY_PATH"),) if p]
    )
    _run(
        tool
        + ["--require-archs", platform.machine(), "-w", dest_dir, "-v", wheel],
        env=env,
    )


def repair(args):
    """'make repair-wheel <wheel> <dest_dir>'."""
    repair_wheel(args.wheel, args.dest_dir)
