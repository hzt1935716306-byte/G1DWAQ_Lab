#!/usr/bin/env bash
set -uo pipefail

project_dir=/home/zt/project/myproject/G1DWAQ_Lab/TienKung-Lab
python_bin=/home/zt/miniconda3/envs/g1/bin/python
evaluator="$project_dir/tools/recovery/evaluate_policy_level_sweep.py"
analyzer="$project_dir/tools/recovery/analyze_l2_boundary_comparison.py"
output_dir="$project_dir/tools/recovery/generated/l2_boundary_comprehensive_2026-09-02"
report="$project_dir/tools/recovery/L2_BOUNDARY_COMPREHENSIVE_COMPARISON.md"
seeds=(42 123 2026)

# Every checkpoint is the last saved checkpoint still in L2.  The following
# saved checkpoint is already after that run's L2 -> L3 transition.
labels=(
  input_context_l2
  baseline_original_l2
  baseline_shared020_l2
  ours_shared_cert020_l2
  ours_cert020_l2
  ours_cert050_l2
  baseline_original_nc
  ours_cert020_nc
  dwaq_new
  dwaq_old
)
policies=(stage2_input baseline baseline ours ours ours baseline ours dwaq dwaq)
checkpoints=(
  "$project_dir/logs/g1_flat_symmetric/2026-09-02_10-41-16_pilot_input_only_256_seed42/model_3500.pt"
  "$project_dir/logs/g1_flat_symmetric/2026-08-30_02-35-28_stage2_baseline_original_from4999/model_7700.pt"
  "$project_dir/logs/g1_flat_symmetric/2026-08-29_13-15-09_stage2_baseline_scale02_solverfix_resume/model_6400.pt"
  "$project_dir/logs/g1_flat_symmetric/2026-08-29_13-12-16_stage2_ours_scale02_solverfix_resume/model_8400.pt"
  "$project_dir/logs/g1_flat_symmetric/2026-08-31_00-20-09_stage2_ours_certonly020_async_from4999/model_7500.pt"
  "$project_dir/logs/g1_flat_symmetric/2026-08-30_02-39-40_stage2_ours_cert050_async_from4999/model_8500.pt"
  "$project_dir/logs/g1_flat_symmetric/2026-08-31_23-48-44_stage2_baseline_no_curriculum_from4999/model_9998.pt"
  "$project_dir/logs/g1_flat_symmetric/2026-08-31_23-44-06_stage2_ours_certonly020_no_curriculum_from4999/model_9998.pt"
  "$project_dir/logs/model_9999.pt"
  "$project_dir/logs/g1_dwaq/2026-01-16_00-46-00/model_9999.pt"
)

mkdir -p "$output_dir"
export PYTHONPATH="$project_dir/rsl_rl:$project_dir${PYTHONPATH:+:$PYTHONPATH}"

# These four reference policies have already completed this exact protocol.
# Reuse their audited JSON (same evaluator, plan hashes and three seeds) rather
# than consuming GPU while the new Input-context policy is training.
for seed in "${seeds[@]}"; do
  cp -f \
    "$project_dir/tools/recovery/generated/five_policy_no_shared_comprehensive_2026-09-01/baseline_no_curriculum_seed${seed}.json" \
    "$output_dir/baseline_original_nc_seed${seed}.json"
  cp -f \
    "$project_dir/tools/recovery/generated/five_policy_no_shared_comprehensive_2026-09-01/ours020_no_curriculum_seed${seed}.json" \
    "$output_dir/ours_cert020_nc_seed${seed}.json"
  cp -f \
    "$project_dir/tools/recovery/generated/three_policy_comprehensive_final/dwaq_flat_new_seed${seed}.json" \
    "$output_dir/dwaq_new_seed${seed}.json"
  cp -f \
    "$project_dir/tools/recovery/generated/three_policy_comprehensive_final/dwaq_seed${seed}.json" \
    "$output_dir/dwaq_old_seed${seed}.json"
done

report_complete() {
  "$python_bin" - "$1" "$2" "$3" "$4" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
checkpoint = str(Path(sys.argv[2]).resolve())
policy = sys.argv[3]
seed = int(sys.argv[4])
try:
    report = json.loads(path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
valid = (
    report.get("policy") == policy
    and report.get("checkpoint") == checkpoint
    and report.get("planned_episode_count") == 1536
    and report.get("completed_episode_count") == 1536
    and report.get("pending_episode_count") == 0
    and report.get("common_protocol", {}).get("seed") == seed
)
raise SystemExit(0 if valid else 1)
PY
}

overall_status=0
for seed in "${seeds[@]}"; do
  for index in "${!labels[@]}"; do
    label=${labels[$index]}
    policy=${policies[$index]}
    checkpoint=${checkpoints[$index]}
    output="$output_dir/${label}_seed${seed}.json"
    log="$output_dir/${label}_seed${seed}.log"
    if report_complete "$output" "$checkpoint" "$policy" "$seed"; then
      echo "[l2-boundary-sweep] seed=$seed label=$label already completed; skipping"
      continue
    fi
    echo "[l2-boundary-sweep] seed=$seed label=$label starting"
    "$python_bin" "$evaluator" \
      --policy "$policy" \
      --checkpoint "$checkpoint" \
      --levels 1 2 3 4 5 6 \
      --episodes_per_level 256 \
      --num_envs 64 \
      --prepare_steps 50 \
      --max_recovery_time_s 10 \
      --max_steps 15000 \
      --seed "$seed" \
      --output "$output" \
      --headless \
      --device "${DEVICE:-cuda:0}" >"$log" 2>&1
    status=$?
    if [[ $status -ne 0 ]] || ! report_complete "$output" "$checkpoint" "$policy" "$seed"; then
      echo "[l2-boundary-sweep] seed=$seed label=$label failed; inspect $log"
      overall_status=1
      break 2
    fi
    echo "[l2-boundary-sweep] seed=$seed label=$label completed"
  done
done

if [[ $overall_status -eq 0 ]]; then
  "$python_bin" "$analyzer" --input_dir "$output_dir" --output "$report"
  echo "[l2-boundary-sweep] report written to $report"
fi
exit "$overall_status"
