#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/zt/project/myproject/G1DWAQ_Lab/TienKung-Lab"
PYTHON="/home/zt/miniconda3/envs/g1/bin/python"
EVALUATOR="$ROOT/tools/push_test/evaluate_external_push.py"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/tools/push_test/generated/all_models_push_limits_2026-09-02}"
TRIALS="${TRIALS:-20}"
DEVICE="${DEVICE:-cuda:0}"

CONTINUOUS_LEVELS=($(seq 0 10 200))
IMPULSE_LEVELS=(0 $(seq 100 50 1000))
CONTINUOUS_REFINEMENT_LEVELS=(2 4 6 8 $(seq 22 2 48))
IMPULSE_REFINEMENT_LEVELS=(20 40 60 80 120 140 160 180 220 240 260 280 320 340)

run_case() {
    local label="$1"
    local policy="$2"
    local checkpoint="$3"
    local mode="$4"
    shift 4
    local output="$OUTPUT_ROOT/$label/$mode"
    if [[ -f "$output/summary.json" ]]; then
        echo "[push-suite] skip completed $label/$mode"
        return
    fi
    mkdir -p "$output"
    echo "[push-suite] start $label/$mode"
    "$PYTHON" "$EVALUATOR" \
        --policy "$policy" \
        --checkpoint "$checkpoint" \
        --mode "$mode" \
        --force "$@" \
        --trials_per_force "$TRIALS" \
        --seed 42 \
        --application_link torso_link \
        --application_point body_com \
        --push_direction_body 1 0 0 \
        --step_displacement_threshold 0.03 \
        --output_dir "$output" \
        --device "$DEVICE" \
        --headless \
        --force_exit_after_report \
        >"$output/run.log" 2>&1
    echo "[push-suite] done $label/$mode"
}

run_model() {
    local label="$1"
    local policy="$2"
    local checkpoint="$3"
    run_case "$label" "$policy" "$checkpoint" continuous "${CONTINUOUS_LEVELS[@]}"
    run_case "$label" "$policy" "$checkpoint" impulse "${IMPULSE_LEVELS[@]}"
}

run_refinement() {
    local label="$1"
    local policy="$2"
    local checkpoint="$3"
    local mode="$4"
    shift 4
    local output="$OUTPUT_ROOT/$label/refinement/$mode"
    if [[ -f "$output/summary.json" ]]; then
        echo "[push-suite] skip completed $label/refinement/$mode"
        return
    fi
    mkdir -p "$output"
    echo "[push-suite] start $label/refinement/$mode"
    "$PYTHON" "$EVALUATOR" \
        --policy "$policy" \
        --checkpoint "$checkpoint" \
        --mode "$mode" \
        --force "$@" \
        --trials_per_force "$TRIALS" \
        --seed 4242 \
        --application_link torso_link \
        --application_point body_com \
        --push_direction_body 1 0 0 \
        --step_displacement_threshold 0.03 \
        --output_dir "$output" \
        --device "$DEVICE" \
        --headless \
        --force_exit_after_report \
        >"$output/run.log" 2>&1
    echo "[push-suite] done $label/refinement/$mode"
}

refine_model() {
    local label="$1"
    local policy="$2"
    local checkpoint="$3"
    run_refinement "$label" "$policy" "$checkpoint" continuous "${CONTINUOUS_REFINEMENT_LEVELS[@]}"
    run_refinement "$label" "$policy" "$checkpoint" impulse "${IMPULSE_REFINEMENT_LEVELS[@]}"
}

run_model \
    unitree_17800 unitree \
    /home/zt/project/g1_base/unitree_rl_lab/logs/rsl_rl/unitree_g1_29dof_velocity/2026-04-24_06-32-29/model_17800.pt
run_model \
    ours_020_l3 ours \
    "$ROOT/logs/g1_flat_symmetric/2026-08-31_00-20-09_stage2_ours_certonly020_async_from4999/model_9300.pt"
run_model \
    ours_025_final ours \
    "$ROOT/logs/our0.25_model_14998_no_sharereward.pt"
run_model \
    baseline_shared_020_l3 ours \
    "$ROOT/logs/g1_flat_symmetric/2026-08-29_13-15-09_stage2_baseline_scale02_solverfix_resume/model_8200.pt"
run_model \
    baseline_original_nc ours \
    "$ROOT/logs/g1_flat_symmetric/2026-08-31_23-48-44_stage2_baseline_no_curriculum_from4999/model_9998.pt"
run_model \
    input_context_final stage2_input \
    "$ROOT/logs/g1_flat_symmetric/2026-09-02_14-56-46_input_only_4096_resume_L3_to_10000/model_9998.pt"
run_model \
    dwaq_flat_new dwaq \
    "$ROOT/logs/model_9999.pt"

refine_model \
    unitree_17800 unitree \
    /home/zt/project/g1_base/unitree_rl_lab/logs/rsl_rl/unitree_g1_29dof_velocity/2026-04-24_06-32-29/model_17800.pt
refine_model \
    ours_020_l3 ours \
    "$ROOT/logs/g1_flat_symmetric/2026-08-31_00-20-09_stage2_ours_certonly020_async_from4999/model_9300.pt"
refine_model \
    ours_025_final ours \
    "$ROOT/logs/our0.25_model_14998_no_sharereward.pt"
refine_model \
    baseline_shared_020_l3 ours \
    "$ROOT/logs/g1_flat_symmetric/2026-08-29_13-15-09_stage2_baseline_scale02_solverfix_resume/model_8200.pt"
refine_model \
    baseline_original_nc ours \
    "$ROOT/logs/g1_flat_symmetric/2026-08-31_23-48-44_stage2_baseline_no_curriculum_from4999/model_9998.pt"
refine_model \
    input_context_final stage2_input \
    "$ROOT/logs/g1_flat_symmetric/2026-09-02_14-56-46_input_only_4096_resume_L3_to_10000/model_9998.pt"
refine_model \
    dwaq_flat_new dwaq \
    "$ROOT/logs/model_9999.pt"

"$PYTHON" "$ROOT/tools/push_test/summarize_push_test.py" \
    --input_root "$OUTPUT_ROOT" \
    --output "$ROOT/tools/push_test/ALL_MODELS_PUSH_LIMITS.md"
