# Evaluation wrench frame repair and bounded validation

This repair is evaluation-only. The installed IsaacLab/Warp sources, native
policies, checkpoints, certificate and common detector configuration are unchanged.
The previous INVALID attempt remains historical evidence under its old runtime.

For WXYZ link-to-world quaternion with matrix R, link position l and declared
application point p:

- r = p - l (world, m)
- arm moment = r × F (world, N m)
- equivalent moment = free moment + arm moment (world, N m)
- F_link = Rᵀ F; equivalent moment_link = Rᵀ equivalent moment.

The NumPy adapter checks rotation covariance. Float32 input round trips and the
installed composer's local buffers are checked with 64 float32 eps times vector
scale. The actual call supplies `positions=None, is_global=False`. The permanent
composer persists until overwritten/reset, so the evaluation hook resets and
recomputes from current physics link tensors at **every 0.005 s substep**.
It never requests observations. Release, terminal, reset and exception paths
explicitly clear force and torque; reset cannot reapply force to reset envs.

The committed protocol and original documentation define the application point
as **whole-robot mass-weighted CoM plus an offset in world axes**, applied through
`torso_link`. Ordinary force families have zero offset from CoM, which is not
necessarily the link origin. `upper_pitch`: +world Z 0.20 m.
`lateral_yaw`: world XY perpendicular to force, 0.15 m, with declared sign.
These points follow CoM; they are neither a fixed world point nor a link-frame
fixed offset. The pure math utility separately supports both of those explicit
semantics. Free moment remains zero for existing families. The adapter preserves
resultant moment about CoM while translating it to the link origin.

Each archived physical sample stores the current quaternion/position, application
point, declared offset, force, arm moment, free moment, equivalent moment, local
inputs, local composer buffers, reconstructed world buffers and API frame flags.
Records include actual start/end/duration and number of applied substeps. At
0.005 s, 0.05/0.20/0.50/5.0 s span 10/40/100/1000 intervals. Terminal events may
truncate a pulse legitimately; recovery starts at disturbance release.

## Gate order

1. All prior 220 tests plus wrench regressions (CPU).
2. One real Isaac environment: upright and pitch 30 deg in free space, one known
   wrench substep per pose followed by a verified zero-wrench substep. No policy
   inference or actor history advancement.
3. PPO: four fixed force-pulse trials, Δv equivalent 0.5 m/s, duration 0.05 s.
4. DWAQ: the exact same four trials and physical identity.

The subset is declared before outcomes: full-plan repeat IDs 0, 33, 66, 99,
cardinal directions 0/90/180/270 deg, both reference feet and both phases.
The explicit `wrench_frame_fix_physical_smoke` role uses its own prepare, checks
the exact subset against the original full plan, and preserves original seeds,
IDs and initial-state parameters. It is not a complete robustness benchmark.
All runtime changes must be committed before physical gates. API receipts and
model seals are bound to the same runtime hash and commit. A model gate needs
observed pulse/release/post-disturbance evidence; a failed recovery or fall is a
valid outcome and never causes seed substitution. Full-suite execution is not
part of this repair task and is not automatically resumed.

Use `g1_wrench_validation.py kernel`, `api-smoke`, `prepare-smoke` and
`check-smoke`, plus the existing `g1_complete_robustness.py run` command for the
explicit prepared smoke root. Evidence goes to
`experiments/g1_wrench_frame_fix_validation/`, separate from the old INVALID run.

## Old-data disposition

The old adapter entered the global-position composer path even for zero forces.
Consequently all 12,416 non-error records are classified UNKNOWN under the
strict criterion requiring complete absence of this path for UNAFFECTED. They
preceded the first nonzero force request. This classification does not assert
that their zero-wrench dynamics were corrupted. The 64 errors remain errors,
and the run remains INVALID. No old record is merged into the new runtime.
`wrench_fix_data_reuse_audit.json` records counts and this distinction. All
37,449 old-run files, including its log, are hash-compared before and after work.
