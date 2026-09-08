"""Instantiate real configs and audit ablations; no environment or physics steps."""
import argparse
import json
from pathlib import Path
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser()
p.add_argument('--output',type=Path,required=True)
AppLauncher.add_app_launcher_args(p)
a=p.parse_args()
app=AppLauncher(a).app
try:
    import legged_lab.envs
    from legged_lab.utils import task_registry
    from reward_shaping_contract import audit_registry
    result=audit_registry(task_registry)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(result,f,indent=2,ensure_ascii=False,allow_nan=False)
    print('REWARD_SHAPING_AUDIT_PASS',flush=True)
finally:
    app.close()
