# Trace validation policy

Validation is independent of policy inference and detector mathematics. The default
`RunStore.validate()` and `RobustnessStore.validate()` level is `light`.

| Level | Work | Automatic use |
| --- | --- | --- |
| `light` | Protocol/manifest/runtime/resource/physics/environment identity; membership, required records and allowed statuses; trace checksums, NPZ array headers and time axes; completion file hashes and recomputed summary | Resume, register, report |
| `sampled` | LIGHT plus deterministic full replay of a selected subset | Before a new run can be sealed/registered |
| `full` | LIGHT plus the preserved common or robustness replay and field/event comparisons for every recorded trial | Explicit CPU audit only |

LIGHT never calls the detector or replay state machine. It reads trace headers and
timestamps; checksum verification still reads the archived bytes. Consequently it
is not constant-time. SAMPLED includes one LIGHT pass, not an extra FULL pass.

Sample size is `min(n, 64, max(16, ceil(0.01*n)))`. Trial scores are
`SHA256(evaluation_id + trial_id + "sampled_replay_v1")`. Selection first covers
all existing recovered/alive-unrecovered/fell/out-of-area/precondition-failed
classes and relapse cases. It then greedily adds missing family, magnitude, slope,
direction and available saved Context validity/failure categories; score order
breaks ties. Output IDs are sorted. There is no manual sample selection.

An unsealed run receives `sampled_replay_audit.json` before its completion seal.
It records assignment, runtime and audit-code hashes, record-content digest,
counts and mismatch details. A mismatch prevents completion and registration.
A passed audit can be reused only after LIGHT verifies its assignment and exact
record digest. Sealed historical runs are never modified to add an audit.
Missing historical samples are displayed as `SAMPLED_REPLAY_NOT_AVAILABLE`.

Explicit commands, CPU only, with deterministic result order:

```bash
python -B tools/evaluation/g1_recovery_eval.py audit \
  --output_root experiments/g1_complete_robustness_v2/cohorts/wrench_fix_96e85ee1 \
  --mode full --workers 16
```

`--output_root` may also name an individual sealed run. Audit outputs go outside
raw runs, under `audits/<evaluation_id>/<timestamp>/full_replay_audit.json`.
FULL records completion SHA, runtime SHA, audit-code SHA, replay counts, start/end
times and per-trial field/stored/replayed/trace-SHA failures. A mismatch produces
`AUDIT_FAILED` and a nonzero audit CLI exit; it does not alter raw records or
launch a replacement experiment. Reports exclude failed audit evidence.

During the remaining experiment queue, completion uses LIGHT+SAMPLED. The final
explicit FULL audit of all final usable runs is scheduled after physical execution
and before the final paper-oriented report. Report generation itself never starts
FULL, including when prior sampled/full evidence is missing.

## Audit-only code transition

Historical full-file runtime identities remain immutable and truthful. An explicit
`audit_policy_transition.json` records old/new full-file hashes and commits.
Non-audit definitions are compared structurally (AST); all remaining runtime
sources must be byte-identical. Actual physics, realized environment, manifest,
initial states, candidate detector parameters and native resources still require
strict equality. New runtime metadata is admitted separately without rewriting
old `runtime_commit.json`, paired-physics metadata or sealed identities.

The transition is confined to validation/storage completion, registry admission,
provenance metadata and audit CLI dispatch. It does not authorize a changed
simulation loop, force adapter, action history, native policy, certificate,
checkpoint, detector, threshold or W/H. Reports explicitly show the distinct
audit-only commits and hashes, rather than calling them identical runtimes.
