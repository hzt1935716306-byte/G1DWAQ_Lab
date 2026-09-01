#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/zt/project/myproject/G1DWAQ_Lab/TienKung-Lab"
PYTHON="/home/zt/miniconda3/envs/g1/bin/python"
EVALUATOR="$ROOT/tools/recovery/evaluate_policy_disturbance_suite.py"
OUTPUT_DIR="$ROOT/tools/recovery/generated/cross_project_survival_suite_2026-09-01"

UNITREE_CKPT="/home/zt/project/g1_base/unitree_rl_lab/logs/rsl_rl/unitree_g1_29dof_velocity/2026-04-24_06-32-29/model_17800.pt"
OURS020_L3_CKPT="$ROOT/logs/g1_flat_symmetric/2026-08-31_00-20-09_stage2_ours_certonly020_async_from4999/model_9300.pt"
OURS025_CKPT="$ROOT/logs/our0.25_model_14998_no_sharereward.pt"
BASELINE_SHARED020_L3_CKPT="$ROOT/logs/g1_flat_symmetric/2026-08-29_13-15-09_stage2_baseline_scale02_solverfix_resume/model_8200.pt"
BASELINE_ORIGINAL_NC_CKPT="$ROOT/logs/g1_flat_symmetric/2026-08-31_23-48-44_stage2_baseline_no_curriculum_from4999/model_9998.pt"

mkdir -p "$OUTPUT_DIR"

is_complete() {
  local output="$1"
  jq -e '.completed_episode_count == 2048 and .pending_episode_count == 0' "$output" >/dev/null 2>&1
}

run_one() {
  local policy="$1"
  local label="$2"
  local checkpoint="$3"
  local seed="$4"
  local output="$OUTPUT_DIR/${label}_seed${seed}.json"
  local log="$OUTPUT_DIR/${label}_seed${seed}.log"

  if [[ -f "$output" ]] && is_complete "$output"; then
    echo "[cross-survival] skip complete label=$label seed=$seed"
    return
  fi

  echo "[cross-survival] start label=$label seed=$seed"
  set +e
  timeout 30m "$PYTHON" "$EVALUATOR" \
    --policy "$policy" \
    --checkpoint "$checkpoint" \
    --family all \
    --episodes_per_condition 64 \
    --num_envs 256 \
    --prepare_steps 50 \
    --onset_jitter_steps 40 \
    --max_recovery_time_s 10.0 \
    --max_steps 30000 \
    --seed "$seed" \
    --cross_project_protocol \
    --fixed_survival_horizon \
    --force_exit_after_report \
    --output "$output" \
    --device cuda:0 \
    --headless >"$log" 2>&1
  local status=$?
  set -e

  if [[ -f "$output" ]] && is_complete "$output"; then
    echo "[cross-survival] complete label=$label seed=$seed status=$status"
    return
  fi

  echo "[cross-survival] failed label=$label seed=$seed status=$status log=$log"
  tail -n 100 "$log"
  return 1
}

for seed in 42 123 2026; do
  run_one unitree unitree_latest "$UNITREE_CKPT" "$seed"
  run_one ours ours_cert020_L3_matched "$OURS020_L3_CKPT" "$seed"
  run_one ours ours_cert025_final "$OURS025_CKPT" "$seed"
  run_one ours baseline_shared020_L3 "$BASELINE_SHARED020_L3_CKPT" "$seed"
  run_one ours baseline_original_no_curriculum "$BASELINE_ORIGINAL_NC_CKPT" "$seed"
done

echo "[cross-survival] all evaluations complete: $OUTPUT_DIR"
