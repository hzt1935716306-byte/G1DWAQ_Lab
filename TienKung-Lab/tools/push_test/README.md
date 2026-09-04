# G1 external push limit evaluation

This directory contains an inference-only test suite. It does not modify policy,
reward, curriculum, or training configuration.

The evaluator applies a real external wrench to `torso_link` at its current
center of mass. The requested rear-to-front body-frame direction is converted
to a world-frame vector from the robot yaw at every control step. Continuous
tests use 10 s force plus 3 s observation; impulse tests use a 0.1 s pulse plus
5 s observation. Both no-step and no-fall outcomes are saved.

Run the complete strong-model sweep:

```bash
cd /home/zt/project/myproject/G1DWAQ_Lab/TienKung-Lab
conda activate g1
bash tools/push_test/run_all_models_push_test.sh
```

Override trial count, device, or output root if needed:

```bash
TRIALS=20 DEVICE=cuda:0 OUTPUT_ROOT=/absolute/output/path \
  bash tools/push_test/run_all_models_push_test.sh
```

Each model/mode directory contains `results.csv`, `traces.csv`, `summary.json`,
and `run.log`. The launcher is restartable and skips a case that already has a
`summary.json`.
