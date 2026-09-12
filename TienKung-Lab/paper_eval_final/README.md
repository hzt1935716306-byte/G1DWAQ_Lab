# Final paper evaluation (protocol v1.2)

This directory is the only admissible implementation for new paper evaluation.
It does not import code, manifests, reports, or results from the retired
`tools/evaluation` system. Existing data under `experiments/g1_recovery_eval*`
is legacy/read-only and cannot pass the new identity gate.

## Environment and preparation

Run from `TienKung-Lab` with the `g1` Conda environment:

```bash
conda run -n g1 python paper_eval_final/run.py --prepare --experiments 1 2 3
```

Preparation writes immutable model-independent manifests before any model is
run. Screening, pilot, and formal use disjoint seed namespaces. A manifest's
`manifest_sha256` is SHA-256 of its canonical JSON payload before adding that
field; the loader recomputes it on every use.

## Simple execution CLI

Experiment 1 is bound to the registered `g1_slope_nosys_d_matched` seed-42
checkpoint. The pilot starts with 500 candidates and then schedules deterministic
targeted continuation batches until it has 120 certificate-valid, unrounded
binary64 margin observations for each of `Nmin=3,4,5`:

```bash
conda run -n g1 python paper_eval_final/run.py --stage pilot --experiment 1 --baseline slope_nosys_d_matched
conda run -n g1 python paper_eval_final/run.py --stage formal --experiment 1 --baseline slope_nosys_d_matched
```

The pilot pools the balanced 3x120 calibration margins, sorts all 360 values,
and freezes one shared pair, `q1=(m120+m121)/2` and
`q2=(m240+m241)/2`, in `configs/experiment1_margin_boundaries.yaml`. All three
Nmin levels use that same pair. The pilot then uses existing calibration rows
and outcome-blind targeted continuation to fill all nine Nmin-margin cells to
40 trials each. Formal Experiment 1 refuses to start without the independently
generated frozen artifact and fills those same nine cells to 160 trials each.
Recovery status, recovery time, touchdown counts, falls, survival, and success
rate are prohibited from pilot/formal admission decisions.
Boundary ties or evidence of clamp/precision-driven duplicate values produce
`MARGIN_DEGENERATE`, retain raw inputs/intermediates in the diagnostic artifact,
and leave formal execution disabled.

Raw terminal records remain immutable. Final dual-role membership (a calibration
row may also be a pilot-analysis row), `calibration_only`, shared q values, and
both calibration/evaluation manifest hashes are materialized in each completed
shard's `sampling_assignments.json` and `sampling_assignments.csv` sidecars.

Experiments 2 and 3 run serially in separate Isaac processes:

```bash
conda run -n g1 python paper_eval_final/run.py --stage pilot --model M1 --experiments 2 3
conda run -n g1 python paper_eval_final/run.py --stage formal --model M1 --experiments 2 3
```

Aggregate completed shards without changing raw records:

```bash
conda run -n g1 python paper_eval_final/run.py --aggregate --stage formal --experiments 2 3
```

Formal execution is fail-closed until all of the following are explicitly
frozen in `protocol/implementation_freeze.yaml`: pilot admission, the Exp1
baseline, Exp1 finite candidate caps, three independent training seeds per
method, the implementation configuration, and the selected environment count.
The currently registered seed-42 checkpoints are sufficient only for technical
screening/pilot work.

## Technical smoke and performance selection

The supplied full-context checkpoint may exercise all three physical paths as
a non-confirmatory smoke. Its Experiment 1 records are explicitly labelled
`TECHNICAL_SMOKE_NONCONFIRMATORY` because a certificate-context policy is not a
valid Experiment 1 baseline.

```bash
conda run -n g1 python paper_eval_final/run.py \
  --stage screening --model M5 --experiments 1 2 3 --smoke --num-envs 64

conda run -n g1 python paper_eval_final/run.py \
  --benchmark --model M5 --env-counts 64 128 256 512 1024 2048
```

The benchmark uses native inference and reports valid simulated seconds per
wall second after warmup. The winning count is then reviewed and frozen as
`parallel_environments.selected`.

## Storage and resume

Each protocol-hash/model/training-seed/experiment is an independent shard, so
the earlier fixed-margin pilot remains read-only and cannot collide with or be
aggregated into this revision. Each trial has an
atomic JSON terminal record, compressed trajectory, event log, and (on physical
failure) a pre-reset snapshot. Existing valid terminal records are never run
again. Aggregation checks protocol version, stage, experiment, manifest,
metrics, physics, code commit, asset and simulator version before merging;
checkpoint hashes may differ. Pilot/formal and legacy/new-final merges fail.

Normal run completion performs LIGHT validation (identity/hash, record and file
integrity, manifest coverage, and denominator closure). Use `--full-audit` for
an explicit later audit; full trace replay is not performed after every run.
