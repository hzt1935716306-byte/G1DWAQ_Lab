# DWAQ V2 / V3 fixed-plan evaluation

Entry: `python -u -B tools/evaluation/g1_dwaq_ablation_execute.py`.

Only the specified final checkpoints are admitted. The original DWAQ native
inference family is retained, while full task names, training lineages and
checkpoint hashes remain distinct. No training or detector parameters change.

Each new checkpoint executes the original standard390 (8 environments) and
exactly the original reduced1960 (64 environments). All initial conditions,
trial IDs, direction/phase assignments and waveforms are retained. OOD slopes
remain separate. Total new trials:4700. Existing five-model results are read-only.

Outputs: `experiments/g1_dwaq_reward_ablation_eval_v1/`. New standard runs stay
inside this directory. A SHA-verified explicit standard run path lets the
unchanged robustness executor bind each new checkpoint to its own standard run.
No old standard index or sealed run is overwritten.

The exact runtime transition from8af6cdd is checked before execution: only new
DWAQ task aliases, hashing the existing ablation config, and explicit standard
source paths may differ. The physical loop, native inference, interventions,
Common detector, thresholds and W/H remain unchanged. Different runtime hashes
are retained and explained by this narrow source proof, never suppressed.

Physical and detector hashes must match the original DWAQ reference. Realized
reset states are checked before robustness actions. Legal falls/precondition
failures are retained and execution continues; evaluation errors stop the queue.
Completion uses the existing LIGHT+SAMPLED seal, with no per-run FULL replay.
The report revalidates all new seals with LIGHT. Old audited references are read
without rewriting their audit files. This does not claim new FULL replay evidence.

The user approved sharing the GPU with training. Only owned evaluator processes
can be terminated by the queue, after completion/shutdown or on an error. Training
processes are never stopped. Wall-clock throughput is not a model-quality metric.

Final MD/DOCX compare seven models, with original DWAQ vs no-idle vs no-swing
highlighted, standard and each disturbance family, envelope curves and paired
statistics. No success conclusion is generated from incomplete runs.
