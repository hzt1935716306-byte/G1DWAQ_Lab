#!/usr/bin/env bash
set -uo pipefail

project_dir=/home/zt/project/myproject/G1DWAQ_Lab/TienKung-Lab
python_bin=/home/zt/miniconda3/envs/g1/bin/python
evaluator="$project_dir/tools/recovery/evaluate_policy_level_sweep.py"
output_dir="$project_dir/tools/recovery/generated/three_policy_level_sweep_ckpt14300"

mkdir -p "$output_dir"
export PYTHONPATH="$project_dir/rsl_rl:$project_dir${PYTHONPATH:+:$PYTHONPATH}"

common_args=(
  --levels 1 2 3 4 5
  --episodes_per_level 256
  --num_envs 64
  --prepare_steps 50
  --max_recovery_time_s 10
  --max_steps 12000
  --seed 42
  --headless
)

"$python_bin" "$evaluator" \
  --policy baseline \
  --checkpoint "$project_dir/logs/g1_flat_symmetric/2026-08-30_02-35-28_stage2_baseline_original_from4999/model_14300.pt" \
  --output "$output_dir/baseline.json" \
  "${common_args[@]}" >"$output_dir/baseline.log" 2>&1 &
baseline_pid=$!

"$python_bin" "$evaluator" \
  --policy ours \
  --checkpoint "$project_dir/logs/g1_flat_symmetric/2026-08-30_02-39-40_stage2_ours_cert050_async_from4999/model_14300.pt" \
  --output "$output_dir/ours.json" \
  "${common_args[@]}" >"$output_dir/ours.log" 2>&1 &
ours_pid=$!

"$python_bin" "$evaluator" \
  --policy dwaq \
  --checkpoint "$project_dir/logs/g1_dwaq/2026-01-16_00-46-00/model_9999.pt" \
  --output "$output_dir/dwaq.json" \
  "${common_args[@]}" >"$output_dir/dwaq.log" 2>&1 &
dwaq_pid=$!

echo "[three-policy-sweep] started baseline=$baseline_pid ours=$ours_pid dwaq=$dwaq_pid"
echo "[three-policy-sweep] logs: $output_dir"

status=0
for pid in "$baseline_pid" "$ours_pid" "$dwaq_pid"; do
  if ! wait "$pid"; then
    status=1
  fi
done

if [[ "$status" -eq 0 ]]; then
  echo "[three-policy-sweep] all evaluations completed"
else
  echo "[three-policy-sweep] at least one evaluation failed; inspect *.log"
fi
exit "$status"
