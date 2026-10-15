#!/bin/bash
# SPDX-FileCopyrightText: 2026 Authors (see AUTHORS.txt)
#
# SPDX-License-Identifier: Apache-2.0
#
# G0 Cycles/LuxCore A/B on one scene prepared by g0_prepare.py: for each
# engine the spp series (two seeds per spp value), then g0_analyze.py. The
# noise comes from the seed pairs, so no reference is needed; the default
# (64 and 256 spp) takes a few minutes per scene. The steps run one after the
# other.
#
#   tests/benchmark/g0_run.sh <name> <prepared.blend> <venv> <percentage> [spp...]
#
# Environment:
#   STEPS    steps to run, in order (default "run"): "cold" times the first
#            render with an empty kernel cache, "ref" renders a reference of
#            REF spp for the bias check, "run" is the spp series
#   REF      spp of the reference (default 1024)
#   ENGINES  default "cycles luxcore"
#   ADDON    BlendLuxCore source, default ~/GitHub/BlendLuxCore
#   OUT      default out/g0/results
#   RENDER_ARGS  extra g0_render.py arguments, e.g.
#            "--luxcore-set path.russianroulette.depth=5"
#
# Examples:
#   tests/benchmark/g0_run.sh cornell out/g0/cornell/cornell-g0.blend out/venvs/g0 100
#   STEPS="cold ref run" tests/benchmark/g0_run.sh classroom \
#       out/g0/classroom/classroom/classroom-g0.blend out/venvs/g0 100 64 256
set -eo pipefail
NAME=$1; BLEND=$2; VENV=$3; PCT=$4; shift 4
SPP=${*:-64 256}
cd "$(dirname "$0")/../.."
ADDON=${ADDON:-$HOME/GitHub/BlendLuxCore}
ENGINES=${ENGINES:-cycles luxcore}
STEPS=${STEPS:-run}
REF=${REF:-1024}
OUT=${OUT:-out/g0/results}/$NAME
SITE=$(ls -d "$VENV"/lib*/python3*/site-packages | head -1)
mkdir -p out/tmp

for e in $ENGINES; do
    E=$(echo "$e" | tr a-z A-Z)
    D=$OUT/$e
    mkdir -p "$D"
    for step in $STEPS; do
        case $step in
            cold) extra=(--baseline-runs 0); cache=$(mktemp -d "$PWD/out/tmp/g0cache-XXXX") ;;
            ref) extra=(--baseline-runs 0 --repeats 1 --seed 100000 --spp "$REF"); cache= ;;
            run) extra=(--spp $SPP); cache= ;;
            *) echo "unknown step $step"; exit 1 ;;
        esac
        rm -f "$D/$step".json "$D/$step"-*.npy
        echo "[$(date +%T)] $NAME $e $step"
        env ${cache:+XDG_CACHE_HOME=$cache} blender -b --factory-startup --disable-autoexec "$BLEND" \
            --python-exit-code 1 --python tests/benchmark/g0_render.py -- \
            --engine "$E" --out "$D" --tag "$step" --percentage "$PCT" \
            --addon "$ADDON" --site-packages "$SITE" "${extra[@]}" $RENDER_ARGS > "$D/$step.log" 2>&1 \
            || { echo "failed, see $D/$step.log"; tail -20 "$D/$step.log"; exit 1; }
        grep "^\[G0\]" "$D/$step.log" | tail -3
        [ -n "$cache" ] && rm -rf "$cache"
    done
done
"$VENV/bin/python" tests/benchmark/g0_analyze.py "$OUT" | tee "$OUT/report.txt"
