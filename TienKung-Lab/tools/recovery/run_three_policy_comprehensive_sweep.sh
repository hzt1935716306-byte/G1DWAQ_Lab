#!/usr/bin/env bash
set -uo pipefail

project_dir=/home/zt/project/myproject/G1DWAQ_Lab/TienKung-Lab
python_bin=/home/zt/miniconda3/envs/g1/bin/python
evaluator="$project_dir/tools/recovery/evaluate_policy_level_sweep.py"
output_dir="$project_dir/tools/recovery/generated/three_policy_comprehensive_final"
seeds=(42 123 2026)

mkdir -p "$output_dir"
export PYTHONPATH="$project_dir/rsl_rl:$project_dir${PYTHONPATH:+:$PYTHONPATH}"

seed_reports_complete() {
  "$python_bin" - "$output_dir" "$1" <<'PY'
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
seed = sys.argv[2]
for policy in ("baseline", "ours", "dwaq"):
    path = output_dir / f"{policy}_seed{seed}.json"
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise SystemExit(1)
    planned = report.get("planned_episode_count")
    completed = report.get("completed_episode_count")
    pending = report.get("pending_episode_count")
    if planned != 1536 or completed != planned or pending != 0:
        raise SystemExit(1)
raise SystemExit(0)
PY
}

overall_status=0
for seed in "${seeds[@]}"; do
  if seed_reports_complete "$seed"; then
    echo "[comprehensive-sweep] seed=$seed already completed; skipping"
    continue
  fi
  common_args=(
    --levels 1 2 3 4 5 6
    --episodes_per_level 256
    --num_envs 64
    --prepare_steps 50
    --max_recovery_time_s 10
    --max_steps 15000
    --seed "$seed"
    --headless
  )

  "$python_bin" "$evaluator" \
    --policy baseline \
    --checkpoint "$project_dir/logs/g1_flat_symmetric/2026-08-30_02-35-28_stage2_baseline_original_from4999/model_14998.pt" \
    --output "$output_dir/baseline_seed${seed}.json" \
    "${common_args[@]}" >"$output_dir/baseline_seed${seed}.log" 2>&1 &
  baseline_pid=$!

  "$python_bin" "$evaluator" \
    --policy ours \
    --checkpoint "$project_dir/logs/g1_flat_symmetric/2026-08-30_02-39-40_stage2_ours_cert050_async_from4999/model_14998.pt" \
    --output "$output_dir/ours_seed${seed}.json" \
    "${common_args[@]}" >"$output_dir/ours_seed${seed}.log" 2>&1 &
  ours_pid=$!

  "$python_bin" "$evaluator" \
    --policy dwaq \
    --checkpoint "$project_dir/logs/g1_dwaq/2026-01-16_00-46-00/model_9999.pt" \
    --output "$output_dir/dwaq_seed${seed}.json" \
    "${common_args[@]}" >"$output_dir/dwaq_seed${seed}.log" 2>&1 &
  dwaq_pid=$!

  echo "[comprehensive-sweep] seed=$seed started baseline=$baseline_pid ours=$ours_pid dwaq=$dwaq_pid"
  pids=("$baseline_pid" "$ours_pid" "$dwaq_pid")
  seed_status=0
  while true; do
    if seed_reports_complete "$seed"; then
      echo "[comprehensive-sweep] seed=$seed reports complete; allowing Isaac shutdown"
      for _ in {1..10}; do
        any_running=0
        for pid in "${pids[@]}"; do
          if kill -0 "$pid" 2>/dev/null; then
            any_running=1
          fi
        done
        [[ "$any_running" -eq 0 ]] && break
        sleep 1
      done
      for pid in "${pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
          kill "$pid" 2>/dev/null || true
        fi
      done
      sleep 2
      for pid in "${pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
          kill -KILL "$pid" 2>/dev/null || true
        fi
        wait "$pid" 2>/dev/null || true
      done
      break
    fi

    any_running=0
    for pid in "${pids[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        any_running=1
      fi
    done
    if [[ "$any_running" -eq 0 ]]; then
      for pid in "${pids[@]}"; do
        wait "$pid" 2>/dev/null || true
      done
      seed_status=1
      overall_status=1
      break
    fi

    for policy in baseline ours dwaq; do
      progress=$(grep -a '\[policy-level-sweep\].*step=' "$output_dir/${policy}_seed${seed}.log" 2>/dev/null | tail -n 1)
      if [[ -n "$progress" ]]; then
        echo "$progress"
      else
        echo "[comprehensive-sweep] seed=$seed policy=$policy initializing Isaac"
      fi
    done
    sleep 20
  done
  if [[ "$seed_status" -eq 0 ]]; then
    echo "[comprehensive-sweep] seed=$seed completed"
  else
    echo "[comprehensive-sweep] seed=$seed failed; inspect *_seed${seed}.log"
  fi
done

if [[ "$overall_status" -eq 0 ]]; then
  echo "[comprehensive-sweep] all final evaluations completed"
else
  echo "[comprehensive-sweep] at least one evaluation failed"
fi
exit "$overall_status"
