#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/zt/project/myproject/G1DWAQ_Lab/TienKung-Lab"
PYTHON="/home/zt/miniconda3/envs/g1/bin/python"
EVALUATOR="$ROOT/tools/recovery/evaluate_cross_project_fixed_push.py"
OUTPUT_DIR="$ROOT/tools/recovery/generated/cross_project_fixed_push_b050_2026-09-01"

UNITREE_CKPT="/home/zt/project/g1_base/unitree_rl_lab/logs/rsl_rl/unitree_g1_29dof_velocity/2026-04-24_06-32-29/model_17800.pt"
OURS020_L3_CKPT="$ROOT/logs/g1_flat_symmetric/2026-08-31_00-20-09_stage2_ours_certonly020_async_from4999/model_9300.pt"
OURS020_CKPT="$ROOT/logs/g1_flat_symmetric/2026-08-31_21-43-19_stage2_ours_certonly020_resume_L5_from11700/model_14998.pt"
OURS025_CKPT="$ROOT/logs/our0.25_model_14998_no_sharereward.pt"

mkdir -p "$OUTPUT_DIR"

is_complete() {
  local output="$1"
  jq -e '.completed_episode_count == 256 and .pending_episode_count == 0' "$output" >/dev/null 2>&1
}

run_one() {
  local policy="$1"
  local label="$2"
  local checkpoint="$3"
  local seed="$4"
  local output="$OUTPUT_DIR/${label}_seed${seed}.json"
  local log="$OUTPUT_DIR/${label}_seed${seed}.log"

  if [[ -f "$output" ]] && is_complete "$output"; then
    echo "[fixed-push-b050] skip complete label=$label seed=$seed"
    return
  fi

  echo "[fixed-push-b050] start label=$label seed=$seed"
  set +e
  timeout 20m "$PYTHON" "$EVALUATOR" \
    --policy "$policy" \
    --checkpoint "$checkpoint" \
    --bound 0.5 \
    --episodes 256 \
    --num_envs 64 \
    --prepare_steps 50 \
    --max_recovery_time_s 10.0 \
    --max_steps 12000 \
    --seed "$seed" \
    --output "$output" \
    --device cuda:0 \
    --force_exit_after_report \
    --headless >"$log" 2>&1
  local status=$?
  set -e

  if [[ -f "$output" ]] && is_complete "$output"; then
    echo "[fixed-push-b050] complete label=$label seed=$seed status=$status"
    return
  fi

  echo "[fixed-push-b050] failed label=$label seed=$seed status=$status log=$log"
  tail -n 80 "$log"
  return 1
}

for seed in 42 123 2026; do
  # `timefix` keeps the old, invalid Unitree JSON separate.  The adapter now
  # converts Unitree policy steps to physics steps before computing time.
  run_one unitree unitree_latest_timefix "$UNITREE_CKPT" "$seed"
  run_one ours ours_cert020_L3_matched "$OURS020_L3_CKPT" "$seed"
  run_one ours ours_cert020_curriculum "$OURS020_CKPT" "$seed"
  run_one ours ours_cert025_curriculum "$OURS025_CKPT" "$seed"
done

echo "[fixed-push-b050] all evaluations complete: $OUTPUT_DIR"
