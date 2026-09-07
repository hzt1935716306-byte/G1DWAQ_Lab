"""Run in a fresh interpreter with the selected NumPy on PYTHONPATH; no Torch/Isaac."""
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from g1_recovery_metrics import trapezoid_integral, trajectory_metrics, quantiles, binomial_rate


def main():
    if len(sys.argv) > 1:
        assert np.__version__ == sys.argv[1]
    assert math.isclose(trapezoid_integral([1, 2, 4], [0, 1, 3]), 7.5)
    assert trapezoid_integral([4], [0]) == 0
    plan = {'command_vx': .4, 'command_vy': 0.}
    frames = [{'time': t, 'root_velocity': [.6, 0], 'com_velocity': [.6, 0],
               'roll_pitch': [0, 0], 'yaw': 0.} for t in np.arange(101)*.02]
    result = trajectory_metrics(frames, plan)
    assert math.isclose(result['integrated_velocity_error'], .4)
    assert math.isclose(result['com_xy_velocity_rmse'], .2)
    assert quantiles([1, 2, 3, 4, 5])['iqr'] == 2
    assert math.isclose(binomial_rate(5, 10, 'executed')['wilson_95'][0], .2365930905, abs_tol=1e-9)
    print(json.dumps({'numpy': np.__version__, 'integral': result['integrated_velocity_error'],
                      'trapz_available': hasattr(np, 'trapz'), 'trapezoid_available': hasattr(np, 'trapezoid'),
                      'status': 'PASS'}))


if __name__ == '__main__':
    main()
