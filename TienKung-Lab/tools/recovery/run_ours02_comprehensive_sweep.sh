#!/usr/bin/env bash
set -uo pipefail

project_dir=/home/zt/project/myproject/G1DWAQ_Lab/TienKung-Lab
python_bin=/home/zt/miniconda3/envs/g1/bin/python
evaluator="$project_dir/tools/recovery/evaluate_policy_level_sweep.py"
checkpoint="$project_dir/logs/g1_flat_symmetric/2026-08-29_13-12-16_stage2_ours_scale02_solverfix_resume/model_14999.pt"
output_dir="$project_dir/tools/recovery/generated/three_policy_comprehensive_final"
seeds=(42 123 2026)

mkdir -p "$output_dir"
export PYTHONPATH="$project_dir/rsl_rl:$project_dir${PYTHONPATH:+:$PYTHONPATH}"

report_complete() {
  "$python_bin" - "$output_dir/ours02_seed$1.json" "$checkpoint" "$1" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
checkpoint = str(Path(sys.argv[2]).resolve())
seed = int(sys.argv[3])
try:
    report = json.loads(path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
valid = (
    report.get("policy") == "ours"
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
  if report_complete "$seed"; then
    echo "[ours02-sweep] seed=$seed already completed; skipping"
    continue
  fi

  "$python_bin" "$evaluator" \
    --policy ours \
    --checkpoint "$checkpoint" \
    --output "$output_dir/ours02_seed${seed}.json" \
    --levels 1 2 3 4 5 6 \
    --episodes_per_level 256 \
    --num_envs 64 \
    --prepare_steps 50 \
    --max_recovery_time_s 10 \
    --max_steps 15000 \
    --seed "$seed" \
    --headless >"$output_dir/ours02_seed${seed}.log" 2>&1 &
  evaluator_pid=$!
  echo "[ours02-sweep] seed=$seed started pid=$evaluator_pid"

  seed_status=0
  while true; do
    if report_complete "$seed"; then
      echo "[ours02-sweep] seed=$seed report complete; allowing Isaac shutdown"
      for _ in {1..10}; do
        if ! kill -0 "$evaluator_pid" 2>/dev/null; then
          break
        fi
        sleep 1
      done
      if kill -0 "$evaluator_pid" 2>/dev/null; then
        kill "$evaluator_pid" 2>/dev/null || true
        sleep 2
      fi
      if kill -0 "$evaluator_pid" 2>/dev/null; then
        kill -KILL "$evaluator_pid" 2>/dev/null || true
      fi
      wait "$evaluator_pid" 2>/dev/null || true
      break
    fi

    if ! kill -0 "$evaluator_pid" 2>/dev/null; then
      wait "$evaluator_pid" 2>/dev/null || true
      seed_status=1
      overall_status=1
      break
    fi

    progress=$(grep -a '\[policy-level-sweep\].*step=' "$output_dir/ours02_seed${seed}.log" 2>/dev/null | tail -n 1)
    if [[ -n "$progress" ]]; then
      echo "$progress"
    else
      echo "[ours02-sweep] seed=$seed initializing Isaac"
    fi
    sleep 20
  done

  if [[ "$seed_status" -eq 0 ]]; then
    echo "[ours02-sweep] seed=$seed completed"
  else
    echo "[ours02-sweep] seed=$seed failed; inspect ours02_seed${seed}.log"
  fi
done

if [[ "$overall_status" -eq 0 ]]; then
  echo "[ours02-sweep] all evaluations completed"
else
  echo "[ours02-sweep] at least one evaluation failed"
fi
exit "$overall_status"
