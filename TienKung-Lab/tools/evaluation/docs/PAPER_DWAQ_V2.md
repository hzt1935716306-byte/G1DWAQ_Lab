# DWAQ V2 extension of the frozen paper experiment

Task: `g1_dwaq_slope_nosys_d_matched_v2` (idle_penalty disabled,
feet_swing_height retained). Fixed checkpoint:
`logs/g1_dwaq_slope_nosys_d_matched_v2/2026-09-09_01-12-29_dwaq_nosys_matched_no_idle/model_9999.pt`.
SHA256: `0214e48a669227d926770698675d647f891dc0c8235d25686f79400a283dacb8`.

The extension copies the frozen plan, protocol and physical profile from
`experiments/g1_paper_fixed_time_v1`. It verifies simulator/judge source hashes
against the repaired original B/C cohort; the launcher commit is recorded
separately. No new threshold/strength selection. Counts:600/2160/2160/1200/300
for A/B1/B2/C_force/C_load; total6420. The existing runs are read-only.

Simulation commit: `1a48653`. Run from TienKung-Lab in the existing g1 environment:

```bash
export PYTHONPATH="$PWD/rsl_rl:$PWD"
/home/zt/miniconda3/envs/g1/bin/python -B tools/evaluation/g1_paper_dwaq_v2.py
```

Frozen inputs reject altered execution identities. Each completed suite is
verified and skipped. Failed attempts stop instead of automatically retrying.
No training starts. To regenerate the four-model report without simulation:

```bash
/home/zt/miniconda3/envs/g1/bin/python -B tools/evaluation/g1_paper_dwaq_v2.py --report
```

Outputs: `experiments/g1_paper_dwaq_v2_fixed_time_v1/formal/report/`.
Existing A's earlier no-force runtime compatibility follows PAPER_FIXED_TIME_V1.
One V2 C_force episode fell before onset: designated1200, actually forced1199;
the main episode-success denominator remains1200. The raw pre_push_failure
field is B-only; use actual_onset/actually_perturbed to distinguish C-force onset.
