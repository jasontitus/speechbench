#!/bin/bash
# Convenience wrapper for the most common speechbench launch.
#
# Defaults:
#   - run id "main" (additive — re-running fills in only missing jobs)
#   - all non-API models
#   - all 6 ungated datasets
#   - 8 spot T4 VMs across 8 regions
#
# Override with env vars:
#   RUN_ID, GPU, MAX_VMS
# Pass --quick / --dry-run / --rerun / --yes through as positional args.

set -euo pipefail
cd "$(dirname "$0")/.."

RUN_ID="${RUN_ID:-main}"
GPU="${GPU:-t4}"
MAX_VMS="${MAX_VMS:-8}"

speechbench launch \
    --run-id "$RUN_ID" \
    --gpu "$GPU" \
    --max-vms "$MAX_VMS" \
    "$@"
