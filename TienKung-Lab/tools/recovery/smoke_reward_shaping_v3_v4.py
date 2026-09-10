"""Two-iteration native smokes for Plane V3 resume and DWAQ V4 random init."""

from __future__ import annotations

import argparse
import faulthandler
import inspect
import json
from pathlib import Path
import runpy
import sys
import traceback


LAB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LAB / "rsl_rl"))
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from reward_shaping_v3_v4_contract import (  # noqa: E402
    DWAQ_V4,
    ESTIMATOR,
    ESTIMATOR_SHA256,
    PARENT_CHECKPOINT_ITERATION,
    PARENT_CHECKPOINT_SHA256,
    PLANE_V3,
    audit_registry,
    file_sha256,
    locate_formal_plane_v2_checkpoint,
)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", choices=(PLANE_V3, DWAQ_V4), required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)

parent = locate_formal_plane_v2_checkpoint() if args.task == PLANE_V3 else None
argv = [
    "legged_lab/scripts/train.py",
    "--task",
    args.task,
    "--num_envs",
    "32",
    "--max_iterations",
    "2",
    "--seed",
    "42",
    "--headless",
    "--device",
    "cuda:0",
    "--experiment_name",
    f"{args.task}_smoke",
    "--run_name",
    "SMOKE_ONLY_NOT_FORMAL",
]
if parent is not None:
    argv += [
        "--resume_checkpoint_path",
        str(parent["path"]),
        "--estimator_checkpoint_path",
        str(ESTIMATOR),
    ]
sys.argv = argv

app = runpy.run_path(str(LAB / "legged_lab/scripts/train.py"), run_name="v3_v4_smoke_train")
globals_ = app["train"].__globals__
import torch  # noqa: E402


def assert_same(left, right, path="root"):
    if torch.is_tensor(left):
        assert torch.equal(left.cpu(), right.cpu()), path
    elif isinstance(left, dict):
        assert left.keys() == right.keys(), path
        for key in left:
            assert_same(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right), path
        for index, (a, b) in enumerate(zip(left, right)):
            assert_same(a, b, f"{path}[{index}]")
    else:
        assert left == right, path


faulthandler.dump_traceback_later(120, repeat=True)
try:
    audit = audit_registry(globals_["task_registry"])
    env_cfg, agent_cfg = globals_["task_registry"].get_cfgs(args.task)
    runner_name = agent_cfg.runner_class_name
    NativeRunner = globals_[runner_name]
    assert Path(inspect.getfile(NativeRunner)).resolve().is_relative_to(LAB / "rsl_rl")

    class SmokeRunner(NativeRunner):
        load_verified = False

        def load(self, checkpoint_path, *load_args, **load_kwargs):
            assert args.task == PLANE_V3, "DWAQ V4 must never invoke load()"
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            result = super().load(checkpoint_path, *load_args, **load_kwargs)
            assert self.current_learning_iteration == PARENT_CHECKPOINT_ITERATION
            assert_same(self.alg.policy.state_dict(), checkpoint["model_state_dict"], "policy")
            assert_same(self.alg.optimizer.state_dict(), checkpoint["optimizer_state_dict"], "optimizer")
            assert self.alg.optimizer.state
            self.load_verified = True
            return result

        def learn(self, num_learning_iterations, **kwargs):
            assert num_learning_iterations == 2
            env = self.env
            if args.task == PLANE_V3:
                assert self.load_verified
                assert self.current_learning_iteration == PARENT_CHECKPOINT_ITERATION + 1
                assert env.reward_manager.get_term_cfg("idle_penalty").weight == -2.0
                assert env.reward_manager.get_term_cfg("feet_swing_height").weight == -0.2
                assert not env._plane_v1_reward_enabled
                assert not env._estimator.training
                assert all(not parameter.requires_grad for parameter in env._estimator.parameters())
            else:
                assert self.current_learning_iteration == 0
                assert not self.alg.optimizer.state
                assert "idle_penalty" not in env.reward_manager.active_terms
                assert "feet_swing_height" not in env.reward_manager.active_terms
                assert "gait_phase_contact" not in env.reward_manager.active_terms

            super().learn(num_learning_iterations=num_learning_iterations, **kwargs)
            observation, extras = env.get_observations()
            if args.task == PLANE_V3:
                critic_observation = extras["observations"]["critic"]
                with torch.inference_mode():
                    action = self.alg.policy.act_inference(observation)
            else:
                observation_history = extras
                critic_observation, _ = env.get_privileged_observations()
                with torch.inference_mode():
                    action = self.alg.policy.act_inference(observation, observation_history)
            assert torch.isfinite(observation).all()
            assert torch.isfinite(critic_observation).all()
            assert action.shape == (32, 29) and torch.isfinite(action).all()
            assert all(torch.isfinite(parameter).all() for parameter in self.alg.policy.parameters())

            result = {
                "status": "PASS",
                "role": "SMOKE_ONLY_NOT_FORMAL",
                "task": args.task,
                "run_directory": str(self.log_dir),
                "num_envs": env.num_envs,
                "iterations_executed": 2,
                "current_iteration": int(self.current_learning_iteration),
                "actor_observation_shape": list(observation.shape),
                "critic_observation_shape": list(critic_observation.shape),
                "action_shape": list(action.shape),
                "active_rewards": list(env.reward_manager.active_terms),
                "policy_class": type(self.alg.policy).__name__,
                "algorithm_class": type(self.alg).__name__,
                "runner_class": type(self).__mro__[1].__name__,
                "optimizer_state_nonempty": bool(self.alg.optimizer.state),
                "finite": True,
                "audit": audit,
            }
            if args.task == PLANE_V3:
                assert self.current_learning_iteration == PARENT_CHECKPOINT_ITERATION + 2
                assert env._estimator_forward_count > 0
                assert env._context_refresh_evaluations > 0
                assert env._recovery_context.shape == (32, 3)
                result.update(
                    {
                        "resume_checkpoint": str(parent["path"]),
                        "resume_checkpoint_sha256": file_sha256(parent["path"]),
                        "source_iteration": PARENT_CHECKPOINT_ITERATION,
                        "resume_start_iteration": PARENT_CHECKPOINT_ITERATION + 1,
                        "expected_final_iteration": PARENT_CHECKPOINT_ITERATION + 2,
                        "policy_and_optimizer_exactly_loaded": True,
                        "estimator_sha256": file_sha256(ESTIMATOR),
                        "estimator_forwards": env._estimator_forward_count,
                        "certificate_queries": env._context_refresh_evaluations,
                        "certificate_valid_queries": env._context_valid_evaluations,
                        "context_shape": list(env._recovery_context.shape),
                    }
                )
                assert result["resume_checkpoint_sha256"] == PARENT_CHECKPOINT_SHA256
                assert result["estimator_sha256"] == ESTIMATOR_SHA256
            else:
                assert self.current_learning_iteration == 1
                assert agent_cfg.resume is False
                assert type(self.alg.policy).__name__ == "ActorCritic_DWAQ"
                assert type(self.alg).__name__ == "DWAQPPO"
                assert self.alg.policy.actor[0].in_features == 115
                assert self.alg.policy.critic[0].in_features == 307
                result.update(
                    {
                        "random_initialization": True,
                        "actor_network_input": self.alg.policy.actor[0].in_features,
                        "critic_network_input": self.alg.policy.critic[0].in_features,
                        "encoder_input": self.alg.policy.encoder[0].in_features,
                        "vae_optimizer_updated": bool(self.alg.optimizer.state),
                    }
                )

            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x") as stream:
                json.dump(result, stream, indent=2, allow_nan=False, default=str)
            print(f"V3_V4_SMOKE_PASS {args.task}", flush=True)

    globals_[runner_name] = SmokeRunner
    app["train"]()
except BaseException:
    traceback.print_exc()
    raise
finally:
    faulthandler.cancel_dump_traceback_later()
    app["simulation_app"].close()
