"""Run the complete pre-training config and checkpoint audit."""

import argparse
import json
from pathlib import Path
import sys

from isaaclab.app import AppLauncher


LAB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LAB / "rsl_rl"))
sys.path.insert(0, str(LAB))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app
try:
    from legged_lab.envs import *  # noqa: F401,F403,E402
    from legged_lab.utils import task_registry  # noqa: E402
    from reward_shaping_v3_v4_contract import audit_registry  # noqa: E402

    result = audit_registry(task_registry)
    if result["parent"] is not None:
        result["parent"]["path"] = str(result["parent"]["path"])
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            stream.write(rendered + "\n")
    print(rendered)
    print("REWARD_SHAPING_V3_V4_AUDIT_PASS", flush=True)
finally:
    app.close()
