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

The immutable plan selects 32 complete canonical 64-trial batches evenly across
the fixed 252-batch formal manifest: 2,048 unique trials spanning all eight suites.
No seed is introduced and no padding or duplicate trial counts are used. Each of
64, 128, 256, 512, 1,024 and 2,048 uses this same manifest and unchanged detector,
physics primitives, native model resources and checkpoint.

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
without policy-step progress. Clear slowdown means over 1.20 times the 64-env
execution duration for an identical completed trial prefix of at least 256 trials,
or over 1.20 times total execution duration. No larger grade for that model is
launched afterwards. All grades use the same thresholds declared before execution.

The performance candidate maximizes geometric mean speed gain for PPO and
Context-only among jointly completed sizes. Less than 10% gain retains 64.
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
has been reviewed. If 64 is retained, the existing formal runtime can resume
unchanged. If a larger size passes, formal execution metadata support and identity
compatibility must first be reviewed; never silently alter the committed protocol.
