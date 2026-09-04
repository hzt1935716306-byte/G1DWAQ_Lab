#!/usr/bin/env bash
set -uo pipefail

project_dir=/home/zt/project/myproject/G1DWAQ_Lab/TienKung-Lab
python_bin=/home/zt/miniconda3/envs/g1/bin/python
evaluator="$project_dir/tools/recovery/evaluate_policy_level_sweep.py"
analyzer="$project_dir/tools/recovery/analyze_velocity_jump_limits.py"
output_dir="$project_dir/tools/recovery/generated/best_models_velocity_jump_limit_2026-09-03"
report="$project_dir/tools/recovery/BEST_MODELS_VELOCITY_JUMP_LIMITS.md"

labels=(baseline_original_nc dwaq_flat_new ours_025_final input_context_final stage1_symmetric_4999 stage1_flat_4999 slope_nosys_d_final dwaq_slope_d_final slope_sys_d_final)
policies=(baseline dwaq ours stage2_input baseline flat slope dwaq_slope slope_sys_d)
checkpoints=(
  "$project_dir/logs/g1_flat_symmetric/2026-08-31_23-48-44_stage2_baseline_no_curriculum_from4999/model_9998.pt"
  "$project_dir/logs/model_9999.pt"
  "$project_dir/logs/our0.25_model_14998_no_sharereward.pt"
  "$project_dir/logs/g1_flat_symmetric/2026-09-02_14-56-46_input_only_4096_resume_L3_to_10000/model_9998.pt"
  "$project_dir/logs/g1_flat_symmetric/2026-08-24_23-53-41/model_4999.pt"
  "$project_dir/logs/g1_flat/2026-08-24_16-46-35/model_4999.pt"
  "$project_dir/logs/g1_slope_nosys_d/2026-09-04_01-41-55/model_9999.pt"
  "$project_dir/logs/g1_dwaq_slope_d.pt"
  "$project_dir/logs/g1_slope_sys_d.pt"
)
magnitudes=(0 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00 2.25 2.50 2.75 3.00)

mkdir -p "$output_dir"
export PYTHONPATH="$project_dir/rsl_rl:$project_dir${PYTHONPATH:+:$PYTHONPATH}"

report_complete() {
  "$python_bin" - "$1" "$2" "$3" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    report = json.loads(path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
valid = (
    report.get("checkpoint") == str(Path(sys.argv[2]).resolve())
    and report.get("policy") == sys.argv[3]
    and report.get("planned_episode_count") == 1664
    and report.get("completed_episode_count") == 1664
    and report.get("pending_episode_count") == 0
    and report.get("common_protocol", {}).get("survival_horizon_s") == 10.0
)
raise SystemExit(0 if valid else 1)
PY
}

pids=()
running_labels=()
status=0
for index in "${!labels[@]}"; do
  label=${labels[$index]}
  policy=${policies[$index]}
  checkpoint=${checkpoints[$index]}
  output="$output_dir/$label.json"
  log="$output_dir/$label.log"
  if report_complete "$output" "$checkpoint" "$policy"; then
    echo "[velocity-limit] label=$label already complete; skipping"
    continue
  fi
  "$python_bin" "$evaluator" \
    --policy "$policy" \
    --checkpoint "$checkpoint" \
    --velocity_magnitudes "${magnitudes[@]}" \
    --direction_count 8 \
    --episodes_per_level 128 \
    --num_envs 256 \
    --prepare_steps 50 \
    --survival_horizon_s 10 \
    --max_steps 30000 \
    --seed 42 \
    --output "$output" \
    --headless \
    --device "${DEVICE:-cuda:0}" >"$log" 2>&1 &
  pids+=("$!")
  running_labels+=("$label")
  echo "[velocity-limit] started label=$label pid=$! log=$log"
done

for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    echo "[velocity-limit] failed label=${running_labels[$index]}"
    status=1
  else
    echo "[velocity-limit] completed label=${running_labels[$index]}"
  fi
done

if [[ $status -eq 0 ]]; then
  for index in "${!labels[@]}"; do
    if ! report_complete "$output_dir/${labels[$index]}.json" "${checkpoints[$index]}" "${policies[$index]}"; then
      echo "[velocity-limit] incomplete report label=${labels[$index]}"
      status=1
    fi
  done
fi

if [[ $status -eq 0 ]]; then
  "$python_bin" "$analyzer" --input_dir "$output_dir" --output "$report"
  echo "[velocity-limit] report written to $report"
fi
exit "$status"
