# Isolated batch-size performance and invariance study

Status: implementation prepared; no profiling simulation before DWAQ COMPLETE.

The running DWAQ cohort remains at 64 environments. Its orchestration parent is
paused before model three, while its simulator process continues uninterrupted.
A separate guard verifies the DWAQ seal and permits profiling only afterwards.

All performance outputs are under
`experiments/g1_batch_size_profiling_v2/PERFORMANCE_ONLY/`. They are never registered
in `completed_models`, never receive a formal robustness completion seal, and are
excluded from the five-model report and paired statistics. Raw profiling traces,
events, process failures and resource telemetry are retained.

The immutable workload selects 64 canonical 64-trial blocks: 4,096 unique
trial IDs spanning all eight suites. The first 128 IDs are predeclared validation
targets, 16 per suite. Candidates are 64, 128, 256, 512, 1,024, **2,046** (the
latest user-specified value), and 4,096. No padding or duplicate trial is used;
a short tail is never launched. A common workload does not imply that every
90-second screen completes the same number of trials.

Each screen has a 90-second execution budget (initialization/save/audit excluded).
Unfinished traces are retained as censored PERFORMANCE_ONLY data, never failures
or formal outcomes. Rank warmed environment physics steps/s across both models;
require 100 policy steps and nonzero Context queries. This is only screening.
Then run 64 and the fastest shared candidate until the same 128 target trials
finish. Other environments execute genuine additional workload. Final selection
requires >10% faster completion of that identical target subset for both models,
plus every invariance comparison passing. Completed-workload throughput and total
wall time are also reported; initialization can make short validation slower.

PPO and Context-only are measured separately. Throughput is completed trials per
minute of physical trial execution, excluding initialization and offline replay.
Total wall time including initialization/replay is also retained. Simulation
throughput records both vector physics steps/s and environment transitions/s;
physics dt is 0.005 s. GPU utilization and used VRAM, process-tree RSS, CPU core
utilization and sampled queued certificate chunks are measured once per second.
100% process CPU means one fully occupied core; aggregate use can exceed 100%.
Memory peaks are sampled peaks, not guaranteed instantaneous allocator maxima.

Context uses the existing explicit native profiling hook only within this
PERFORMANCE_ONLY study, preserving solver settings and policy mathematics.
Native query counts, failed queries, worker wait/solve timing, IPC timing and
sampled queue depth are retained. Timing distributions are explicitly chunk-level
where the native IPC interface batches queries. No missing timing is replaced by
zero or relabeled as per-query latency. Queue depth excludes in-flight chunks.

A grade stops on OOM, process failure, initialization timeout (900 s), or 180 s
without policy-step progress. A warmed screen over 20% slower in environment
steps/s than 64 stops further larger grades for that model. Audit/save timeout is
900 s. These criteria are fixed before any measurement.

Invariance then compares the exact trial set and outcome/event timelines for PPO
64 versus candidate, Context-only 64 versus candidate, and PPO against the sealed
64-env source. Recovery, readiness and terminal outcomes must match. Time tolerance
is 1e-7 s; native margin absolute tolerance is 1e-6; discrete touchdown events,
certificate N/validity/categories and detector counts must match exactly.
Any difference conservatively retains 64; this does not assert that a small
subset proves universal batch invariance. No formal batch size changes before
reviewing the comparison and identity contracts.

Commands (the controller refuses execution before the DWAQ seal gate):

```bash
python -B tools/evaluation/g1_batch_profile_control.py
```

Do not resume the paused formal orchestration parent until the profiling decision
has been reviewed. The authorized audit-policy transition must be recorded before resuming
formal execution, while preserving the physical computation. If a larger size passes, formal execution metadata support and identity
compatibility must first be reviewed; never silently alter the committed protocol.
