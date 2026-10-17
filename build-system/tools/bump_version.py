#!/usr/bin/env python3
"""Bump the default LuxCore version in build-system/build-settings.json.

Usage: bump_version.py [patch|minor|major] [--set X.Y.Z[-pre]] [--commit]

Without arguments, the patch number is incremented. The prerelease tag is
cleared by a bump. With --commit, the change is committed ("Bump to vX.Y.Z").
"""

import argparse
import json
import re
import subprocess
from pathlib import Path

SETTINGS = Path(__file__).resolve().parents[1] / "build-settings.json"
SEMVER = re.compile(r"(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z.-]+))?")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("part", nargs="?", default="patch",
                        choices=("patch", "minor", "major"))
    parser.add_argument("--set", metavar="VERSION",
                        help="set an explicit version instead of bumping")
    parser.add_argument("--commit", action="store_true",
                        help="commit the change")
    args = parser.parse_args()

    data = json.loads(SETTINGS.read_text())
    version = data["DefaultVersion"]

    if args.set:
        match = SEMVER.fullmatch(args.set)
        if not match:
            parser.error(f"'{args.set}' is not major.minor.patch[-pre]")
        major, minor, patch, pre = match.groups()
        new = dict(major=major, minor=minor, patch=patch, prerelease=pre or "")
    else:
        numbers = {k: int(version[k]) for k in ("major", "minor", "patch")}
        numbers[args.part] += 1
        if args.part == "major":
            numbers["minor"] = numbers["patch"] = 0
        elif args.part == "minor":
            numbers["patch"] = 0
        new = {k: str(v) for k, v in numbers.items()}
        new["prerelease"] = ""

    version.update(new)
    SETTINGS.write_text(json.dumps(data, indent=2) + "\n")

    text = "{major}.{minor}.{patch}".format(**new)
    if new["prerelease"]:
        text += "-" + new["prerelease"]
    print(text)

    if args.commit:
        subprocess.run(["git", "add", str(SETTINGS)], check=True)
        subprocess.run(["git", "commit", "-m", f"Bump to v{text}"], check=True)


if __name__ == "__main__":
    main()
