#!/usr/bin/env bash
set -uo pipefail

project_dir=/home/zt/project/myproject/G1DWAQ_Lab/TienKung-Lab
python_bin=/home/zt/miniconda3/envs/g1/bin/python
evaluator="$project_dir/tools/recovery/evaluate_policy_level_sweep.py"
output_dir="$project_dir/tools/recovery/generated/five_policy_no_shared_comprehensive_2026-09-01"
seeds=(42 123 2026)
labels=(
  ours015_curriculum
  ours020_curriculum
  ours025_curriculum
  ours020_no_curriculum
  baseline_no_curriculum
)
policies=(ours ours ours ours baseline)
checkpoints=(
  "$project_dir/logs/our0.15_model_14998_no_sharereware.pt"
  "$project_dir/logs/g1_flat_symmetric/2026-08-31_21-43-19_stage2_ours_certonly020_resume_L5_from11700/model_14998.pt"
  "$project_dir/logs/our0.25_model_14998_no_sharereward.pt"
  "$project_dir/logs/g1_flat_symmetric/2026-08-31_23-44-06_stage2_ours_certonly020_no_curriculum_from4999/model_9998.pt"
  "$project_dir/logs/g1_flat_symmetric/2026-08-31_23-48-44_stage2_baseline_no_curriculum_from4999/model_9998.pt"
)

mkdir -p "$output_dir"
export PYTHONPATH="$project_dir/rsl_rl:$project_dir${PYTHONPATH:+:$PYTHONPATH}"

report_complete() {
  "$python_bin" - "$output_dir/$1_seed$4.json" "$2" "$3" "$4" <<'PY'
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

seed_complete() {
  local seed=$1
  local index
  for index in "${!labels[@]}"; do
    report_complete "${labels[$index]}" "${checkpoints[$index]}" "${policies[$index]}" "$seed" || return 1
  done
  return 0
}

overall_status=0
for seed in "${seeds[@]}"; do
  if seed_complete "$seed"; then
    echo "[five-policy-sweep] seed=$seed already complete; skipping"
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
  pids=()
  for index in "${!labels[@]}"; do
    label=${labels[$index]}
    "$python_bin" "$evaluator" \
      --policy "${policies[$index]}" \
      --checkpoint "${checkpoints[$index]}" \
      --output "$output_dir/${label}_seed${seed}.json" \
      "${common_args[@]}" >"$output_dir/${label}_seed${seed}.log" 2>&1 &
    pids+=("$!")
    echo "[five-policy-sweep] seed=$seed label=$label started pid=$!"
  done

  seed_status=0
  while true; do
    if seed_complete "$seed"; then
      echo "[five-policy-sweep] seed=$seed reports complete; allowing Isaac shutdown"
      for _ in {1..10}; do
        any_running=0
        for pid in "${pids[@]}"; do
          kill -0 "$pid" 2>/dev/null && any_running=1
        done
        [[ "$any_running" -eq 0 ]] && break
        sleep 1
      done
      for pid in "${pids[@]}"; do
        kill -0 "$pid" 2>/dev/null && kill "$pid" 2>/dev/null || true
      done
      sleep 2
      for pid in "${pids[@]}"; do
        kill -0 "$pid" 2>/dev/null && kill -KILL "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
      done
      break
    fi

    any_running=0
    for pid in "${pids[@]}"; do
      kill -0 "$pid" 2>/dev/null && any_running=1
    done
    if [[ "$any_running" -eq 0 ]]; then
      seed_status=1
      overall_status=1
      break
    fi

    for label in "${labels[@]}"; do
      progress=$(grep -a '\[policy-level-sweep\].*step=' "$output_dir/${label}_seed${seed}.log" 2>/dev/null | tail -n 1)
      if [[ -n "$progress" ]]; then
        echo "$progress label=$label"
      else
        echo "[five-policy-sweep] seed=$seed label=$label initializing Isaac"
      fi
    done
    sleep 20
  done

  if [[ "$seed_status" -eq 0 ]]; then
    echo "[five-policy-sweep] seed=$seed completed"
  else
    echo "[five-policy-sweep] seed=$seed failed; inspect seed logs"
  fi
done

if [[ "$overall_status" -eq 0 ]]; then
  echo "[five-policy-sweep] all evaluations completed"
else
  echo "[five-policy-sweep] at least one evaluation failed"
fi
exit "$overall_status"
