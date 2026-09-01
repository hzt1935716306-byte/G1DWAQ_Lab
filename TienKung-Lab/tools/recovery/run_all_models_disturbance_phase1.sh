#!/usr/bin/env bash
set -uo pipefail

project_dir=/home/zt/project/myproject/G1DWAQ_Lab/TienKung-Lab
python_bin=/home/zt/miniconda3/envs/g1/bin/python
evaluator="$project_dir/tools/recovery/evaluate_policy_disturbance_suite.py"
output_dir="$project_dir/tools/recovery/generated/all_models_disturbance_phase1_2026-09-01"
max_parallel=5
expected_episodes=4480

labels=(
  baseline_original_curriculum
  baseline_shared020_curriculum
  baseline_original_no_curriculum
  ours_shared_cert020_curriculum
  ours_cert015_curriculum
  ours_cert020_curriculum
  ours_cert025_curriculum
  ours_cert050_curriculum
  ours_cert020_no_curriculum
  dwaq_flat_new
  dwaq_old
)
policies=(
  baseline baseline baseline ours ours ours ours ours ours dwaq dwaq
)
checkpoints=(
  logs/g1_flat_symmetric/2026-08-30_02-35-28_stage2_baseline_original_from4999/model_14998.pt
  logs/g1_flat_symmetric/2026-08-29_13-15-09_stage2_baseline_scale02_solverfix_resume/model_14999.pt
  logs/g1_flat_symmetric/2026-08-31_23-48-44_stage2_baseline_no_curriculum_from4999/model_9998.pt
  logs/g1_flat_symmetric/2026-08-29_13-12-16_stage2_ours_scale02_solverfix_resume/model_14999.pt
  logs/our0.15_model_14998_no_sharereware.pt
  logs/g1_flat_symmetric/2026-08-31_21-43-19_stage2_ours_certonly020_resume_L5_from11700/model_14998.pt
  logs/our0.25_model_14998_no_sharereward.pt
  logs/g1_flat_symmetric/2026-08-30_02-39-40_stage2_ours_cert050_async_from4999/model_14998.pt
  logs/g1_flat_symmetric/2026-08-31_23-44-06_stage2_ours_certonly020_no_curriculum_from4999/model_9998.pt
  logs/model_9999.pt
  logs/g1_dwaq/2026-01-16_00-46-00/model_9999.pt
)
seeds=(42 123 2026)

mkdir -p "$output_dir"
export PYTHONPATH="$project_dir/rsl_rl:$project_dir${PYTHONPATH:+:$PYTHONPATH}"

child_pids=()
cleanup_children() {
  local pid attempt
  for pid in "${child_pids[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
  for attempt in 1 2 3 4 5; do
    sleep 1
    for pid in "${child_pids[@]:-}"; do
      kill -0 "$pid" 2>/dev/null || continue
      (( attempt < 5 )) || kill -KILL "$pid" 2>/dev/null || true
    done
  done
}
handle_signal() {
  cleanup_children
  exit 130
}
trap handle_signal INT TERM
trap cleanup_children EXIT

report_complete() {
  local report=$1
  [[ -s "$report" ]] || return 1
  jq -e --argjson expected "$expected_episodes" '
    .planned_episode_count == $expected
    and .completed_episode_count == $expected
    and .pending_episode_count == 0
  ' "$report" >/dev/null 2>&1
}

report_finalized() {
  local report=$1
  [[ -s "$report" ]] || return 1
  jq -e '.schema_version == 1 and (.planned_episode_count | type == "number")' \
    "$report" >/dev/null 2>&1
}

monitor_batch() {
  local -n batch_pids_ref=$1
  local -n batch_reports_ref=$2
  local -n batch_names_ref=$3
  local active=${#batch_pids_ref[@]}
  local status=0
  local index pid report name
  local -a finished=()
  for ((index=0; index<active; index++)); do
    finished+=(0)
  done

  while (( active > 0 )); do
    for ((index=0; index<${#batch_pids_ref[@]}; index++)); do
      (( finished[index] == 0 )) || continue
      pid=${batch_pids_ref[index]}
      report=${batch_reports_ref[index]}
      name=${batch_names_ref[index]}
      if report_complete "$report"; then
        echo "[phase1] complete $name; reclaiming Isaac process $pid"
        kill -TERM "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
        finished[index]=1
        ((active--))
      elif report_finalized "$report"; then
        echo "[phase1] INCOMPLETE $name; reclaiming Isaac process $pid" >&2
        kill -TERM "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
        finished[index]=1
        ((active--))
        status=1
      elif ! kill -0 "$pid" 2>/dev/null; then
        wait "$pid" 2>/dev/null || true
        echo "[phase1] FAILED $name; inspect ${report%.json}.log" >&2
        finished[index]=1
        ((active--))
        status=1
      fi
    done
    (( active == 0 )) || sleep 5
  done
  return "$status"
}

status=0
for seed in "${seeds[@]}"; do
  echo "[phase1] seed=$seed"
  pending_indices=()
  for ((index=0; index<${#labels[@]}; index++)); do
    label=${labels[index]}
    checkpoint="$project_dir/${checkpoints[index]}"
    report="$output_dir/${label}_seed${seed}.json"
    if [[ ! -f "$checkpoint" ]]; then
      echo "[phase1] missing checkpoint: $checkpoint" >&2
      status=1
    elif report_complete "$report"; then
      echo "[phase1] skip complete ${label}/seed${seed}"
    else
      pending_indices+=("$index")
    fi
  done
  for ((batch_start=0; batch_start<${#pending_indices[@]}; batch_start+=max_parallel)); do
    batch_pids=()
    batch_reports=()
    batch_names=()
    batch_end=$((batch_start + max_parallel))
    (( batch_end > ${#pending_indices[@]} )) && batch_end=${#pending_indices[@]}
    for ((pending_position=batch_start; pending_position<batch_end; pending_position++)); do
      index=${pending_indices[pending_position]}
      label=${labels[index]}
      policy=${policies[index]}
      checkpoint="$project_dir/${checkpoints[index]}"
      report="$output_dir/${label}_seed${seed}.json"
      log="$output_dir/${label}_seed${seed}.log"
      "$python_bin" "$evaluator" \
        --policy "$policy" \
        --checkpoint "$checkpoint" \
        --family all \
        --num_envs 256 \
        --prepare_steps 50 \
        --onset_jitter_steps 40 \
        --max_recovery_time_s 10 \
        --max_steps 25000 \
        --seed "$seed" \
        --device cuda:0 \
        --output "$report" \
        --headless >"$log" 2>&1 &
      pid=$!
      child_pids+=("$pid")
      batch_pids+=("$pid")
      batch_reports+=("$report")
      batch_names+=("${label}/seed${seed}")
      echo "[phase1] started ${label}/seed${seed} pid=$pid"
    done
    if ((${#batch_pids[@]} > 0)); then
      child_pids=("${batch_pids[@]}")
      if ! monitor_batch batch_pids batch_reports batch_names; then
        status=1
      fi
      child_pids=()
    fi
  done
done

trap - INT TERM EXIT
if (( status == 0 )); then
  echo "[phase1] all 33 model/seed reports completed"
else
  echo "[phase1] finished with failures" >&2
fi
exit "$status"
