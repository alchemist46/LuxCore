#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Authors (see AUTHORS.txt)
#
# SPDX-License-Identifier: Apache-2.0

# Run a command; on failure, also report the end of its output as an error
# annotation, which (unlike the job logs) can be read without logging in
# (e.g. through the check-runs API).
#
#   bash .github/scripts/run-annotated.sh <title> <command> [args...]

title="$1"
shift

log=$(mktemp)
"$@" 2>&1 | tee "$log"
status=${PIPESTATUS[0]}

if [ "$status" -ne 0 ]; then
  # Annotation messages are one line: %, CR and LF are escaped
  message=$(tail -n 40 "$log" | sed -e 's/%/%25/g' -e 's/\r/%0D/g' | awk '{ printf "%s%%0A", $0 }')
  echo "::error title=${title}::${message}"
fi

rm -f "$log"
exit "$status"
