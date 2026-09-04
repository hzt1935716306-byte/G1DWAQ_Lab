#!/usr/bin/env bash
set -uo pipefail

project_dir=/home/zt/project/myproject/G1DWAQ_Lab/TienKung-Lab
python_bin=/home/zt/miniconda3/envs/g1/bin/python
evaluator="$project_dir/tools/recovery/evaluate_policy_level_sweep.py"
checkpoint="$project_dir/logs/g1_dwaq_slope_d.pt"
output_dir="$project_dir/tools/recovery/generated/dwaq_slope_d_stability_2026-09-04"
slopes=(-20 -10 0 10 20)
magnitudes=(0 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00 2.25 2.50 2.75 3.00)

mkdir -p "$output_dir"
export PYTHONPATH="$project_dir/rsl_rl:$project_dir${PYTHONPATH:+:$PYTHONPATH}"

# Add this checkpoint to the exact mixed-command flat benchmark. Existing
# policies are skipped via persisted-report checks.
bash "$project_dir/tools/recovery/run_best_models_velocity_jump_limit.sh" || exit $?

slug_for_slope() {
  local slope=$1
  if [[ $slope == -* ]]; then
    echo "minus${slope#-}"
  elif [[ $slope == 0 ]]; then
    echo "flat"
  else
    echo "plus${slope}"
  fi
}

report_complete() {
  "$python_bin" - "$1" "$2" <<'PY'
import json
import math
import sys
from pathlib import Path

path = Path(sys.argv[1])
slope = float(sys.argv[2])
try:
    report = json.loads(path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
protocol = report.get("common_protocol", {})
valid = (
    report.get("policy") == "dwaq_slope"
    and report.get("planned_episode_count") == 1664
    and report.get("completed_episode_count") == 1664
    and report.get("pending_episode_count") == 0
    and protocol.get("command_mode") == "slope_forward"
    and protocol.get("survival_horizon_s") == 10.0
    and math.isclose(float(protocol.get("slope_degrees", 999.0)), slope)
)
raise SystemExit(0 if valid else 1)
PY
}

for slope in "${slopes[@]}"; do
  slug=$(slug_for_slope "$slope")
  output="$output_dir/slope_${slug}.json"
  log="$output_dir/slope_${slug}.log"
  if report_complete "$output" "$slope"; then
    echo "[dwaq-slope-stability] slope=$slope already complete; skipping"
    continue
  fi
  echo "[dwaq-slope-stability] slope=$slope starting"
  "$python_bin" "$evaluator" \
    --policy dwaq_slope \
    --checkpoint "$checkpoint" \
    --velocity_magnitudes "${magnitudes[@]}" \
    --direction_count 8 \
    --command_mode slope_forward \
    --slope_degrees "$slope" \
    --episodes_per_level 128 \
    --num_envs 256 \
    --prepare_steps 50 \
    --survival_horizon_s 10 \
    --max_steps 30000 \
    --seed 42 \
    --output "$output" \
    --headless \
    --device "${DEVICE:-cuda:0}" >"$log" 2>&1
  status=$?
  if [[ $status -ne 0 ]] || ! report_complete "$output" "$slope"; then
    echo "[dwaq-slope-stability] slope=$slope failed; inspect $log"
    exit 1
  fi
  echo "[dwaq-slope-stability] slope=$slope completed"
done

echo "[dwaq-slope-stability] raw suite completed: $output_dir"
