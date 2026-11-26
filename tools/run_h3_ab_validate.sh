#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
#
# Parallel A/B runner for MiniMax-H3 changes.
#
# Stage 1 (always): CPU smoke. Runs the CPU-marked MiniMax-H3 unit tests on
#   both arms. Stops before any NPU time is spent if the candidate fails a
#   test the baseline passes.
# Stage 2 (optional): runs AB_CMD on both arms at the same time, baseline on
#   BASE_CARDS and candidate on CAND_CARDS (or on two hosts, by running this
#   script once per host with only one arm selected via ARMS).
#
# Usage:
#   BASE_COMMIT=<sha> CAND_COMMIT=<sha> tools/run_h3_ab_validate.sh
#   BASE_COMMIT=<sha> CAND_COMMIT=<sha> AB_CMD='python my_component_ab.py --out "$ARM_OUT"' \
#     BASE_CARDS=0,1,2,3 CAND_CARDS=4,5,6,7 tools/run_h3_ab_validate.sh
#
# AB_CMD runs inside each arm's worktree with ASCEND_RT_VISIBLE_DEVICES set to
# that arm's cards and ARM / ARM_OUT exported. Splitting cards between arms
# halves the cards per arm; only use it when the measured config fits.
#
# Env: ARMS=base,cand (default both), WORK_DIR (default ./h3_ab_<timestamp>).

set -euo pipefail

BASE_COMMIT="${BASE_COMMIT:?set BASE_COMMIT}"
CAND_COMMIT="${CAND_COMMIT:?set CAND_COMMIT}"
BASE_CARDS="${BASE_CARDS:-0,1,2,3}"
CAND_CARDS="${CAND_CARDS:-4,5,6,7}"
ARMS="${ARMS:-base,cand}"
AB_CMD="${AB_CMD:-}"
REPO_ROOT="$(git -C "$(dirname "$0")/.." rev-parse --show-toplevel)"
WORK_DIR="${WORK_DIR:-./h3_ab_$(date +%Y%m%d_%H%M%S)}"
SMOKE_TESTS="tests/diffusion/models/minimax_h3"

mkdir -p "$WORK_DIR" && WORK_DIR="$(cd "$WORK_DIR" && pwd)"

arm_commit() { [[ "$1" == base ]] && echo "$BASE_COMMIT" || echo "$CAND_COMMIT"; }
arm_cards() { [[ "$1" == base ]] && echo "$BASE_CARDS" || echo "$CAND_CARDS"; }

cleanup() {
    for arm in base cand; do
        [[ -d "$WORK_DIR/$arm/src" ]] && git -C "$REPO_ROOT" worktree remove --force "$WORK_DIR/$arm/src" || true
    done
}
trap cleanup EXIT

IFS=',' read -r -a arm_list <<< "$ARMS"
for arm in "${arm_list[@]}"; do
    mkdir -p "$WORK_DIR/$arm"
    git -C "$REPO_ROOT" worktree add --detach "$WORK_DIR/$arm/src" "$(arm_commit "$arm")" >/dev/null
    echo "$arm: $(git -C "$WORK_DIR/$arm/src" rev-parse HEAD)" | tee -a "$WORK_DIR/identity.txt"
done

# Stage 1: CPU smoke on both arms, in parallel.
pids=()
for arm in "${arm_list[@]}"; do
    (
        cd "$WORK_DIR/$arm/src"
        python -m pytest "$SMOKE_TESTS" -m cpu -q -p no:cacheprovider \
            --junitxml="$WORK_DIR/$arm/smoke.xml" >"$WORK_DIR/$arm/smoke.log" 2>&1
    ) &
    pids+=($!)
done
smoke_fail=0
for i in "${!arm_list[@]}"; do
    arm="${arm_list[$i]}"
    if wait "${pids[$i]}"; then
        echo "smoke $arm: pass"
    else
        echo "smoke $arm: FAIL (see $WORK_DIR/$arm/smoke.log)"
        [[ "$arm" == cand ]] && smoke_fail=1
    fi
done
if [[ "$smoke_fail" -eq 1 ]]; then
    echo "Candidate failed CPU smoke. Compare with the baseline result above before using NPU time."
    exit 1
fi

[[ -z "$AB_CMD" ]] && { echo "No AB_CMD; stopping after smoke."; exit 0; }

# Stage 2: hardware A/B, both arms at once on disjoint cards.
pids=()
for arm in "${arm_list[@]}"; do
    (
        cd "$WORK_DIR/$arm/src"
        export ARM="$arm" ARM_OUT="$WORK_DIR/$arm/out"
        export ASCEND_RT_VISIBLE_DEVICES="$(arm_cards "$arm")"
        mkdir -p "$ARM_OUT"
        bash -c "$AB_CMD" >"$WORK_DIR/$arm/ab.log" 2>&1
    ) &
    pids+=($!)
done
ab_fail=0
for i in "${!arm_list[@]}"; do
    arm="${arm_list[$i]}"
    if wait "${pids[$i]}"; then echo "ab $arm: exit 0"; else echo "ab $arm: FAIL"; ab_fail=1; fi
done
echo "Outputs: $WORK_DIR/{base,cand}/out, logs: $WORK_DIR/{base,cand}/ab.log"
exit "$ab_fail"
