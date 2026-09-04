"""Runtime integration for the final Plane V1 2x2 experiment."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict
import statistics
import time

import torch

from legged_lab.envs.base.base_env import BaseEnv
from legged_lab.envs.g1.g1_plane_recovery_env import G1PlaneRecoveryEnv
from legged_lab.estimation import (
    EstimatorFrameHistory,
    latest_actor_frame,
    load_com_velocity_estimator_for_inference,
)
from legged_lab.recovery.plane_certificate_runtime import (
    PlaneCalibratedG1CertificateEvaluator,
)
from legged_lab.recovery.plane_v1 import (
    plane_v1_allowed_type_indices,
    plane_v1_learning_iteration,
    plane_v1_terrain_level,
    replace_com_velocity_for_certificate,
)
from legged_lab.recovery.practical_metrics import practical_interval_means_from_sums
from legged_lab.recovery.recovery_context import normalize_recovery_context
from legged_lab.recovery.stage2_reward import (
    PlaneV1RewardParameters,
    plane_v1_touchdown_reward,
)


class G1PlaneV1Env(G1PlaneRecoveryEnv):
    """Final context environment with selectable CoM-velocity source.

    This subclass deliberately calls :class:`BaseEnv` for stepping.  It reuses
    all Plane extraction/evaluation machinery while leaving the legacy Stage2
    reward state machine and its practical-success termination untouched.
    """

    def _create_certificate_evaluator(self):
        cfg = self.cfg.stage2_reward
        plane_cfg = self.cfg.plane_recovery
        return PlaneCalibratedG1CertificateEvaluator(
            cfg.certificate_parameters_path,
            plane_cfg.nominal_parameters_path,
            workers=cfg.certificate_workers,
            executor_type=cfg.certificate_executor,
            failure_window_size=cfg.certificate_failure_window_size,
            failure_rate_threshold=cfg.certificate_failure_rate_threshold,
            z_sole=plane_cfg.z_sole,
            use_state_b=True,
        )

    def __init__(self, cfg, headless):
        source = str(cfg.com_velocity_source)
        estimator = None
        estimator_metadata = None
        if source == "estimator":
            estimator, estimator_metadata = load_com_velocity_estimator_for_inference(
                cfg.estimator_checkpoint_path,
                device="cpu",
            )
        elif source != "privileged":
            raise ValueError("com_velocity_source must be 'estimator' or 'privileged'")

        super().__init__(cfg, headless)
        self.com_velocity_source = source
        self._estimator = estimator.to(self.device) if estimator is not None else None
        self._estimator_metadata = estimator_metadata
        self._estimator_forward_count = 0
        self._estimator_history = None
        self._estimator_history_count = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._estimator_prediction = torch.zeros((self.num_envs, 2), device=self.device)
        if self._estimator is not None:
            assert estimator_metadata is not None
            checkpoint_scale = float(estimator_metadata["imu_acceleration_scale"])
            if checkpoint_scale != float(cfg.estimator_imu_acceleration_scale):
                raise RuntimeError(
                    "estimator IMU scale mismatch: "
                    f"checkpoint={checkpoint_scale}, config={cfg.estimator_imu_acceleration_scale}"
                )
            self._estimator.eval()
            self._estimator.requires_grad_(False)
            self._estimator_history = EstimatorFrameHistory(
                self.num_envs,
                history_length=5,
                actor_frame_dim=96,
                imu_dim=3,
                imu_acceleration_scale=checkpoint_scale,
                device=self.device,
            )

        reward_cfg = cfg.plane_v1_reward
        self._plane_v1_reward_enabled = bool(reward_cfg.enabled)
        self._plane_v1_reward_parameters = PlaneV1RewardParameters(
            **{key: value for key, value in asdict(reward_cfg).items() if key != "enabled"}
        )
        self._defer_plane_v1_reset_cleanup = False
        self._initialize_plane_v1_buffers()

    def _initialize_plane_v1_buffers(self) -> None:
        count = self.num_envs
        device = self.device
        self._v1_touchdown_index = torch.full((count,), -1, dtype=torch.long, device=device)
        self._v1_previous_phi = torch.zeros(count, device=device)
        self._v1_previous_phi_valid = torch.zeros(count, dtype=torch.bool, device=device)
        self._v1_last_n = torch.full((count,), -1, dtype=torch.long, device=device)
        self._v1_initial_n = torch.full((count,), -1, dtype=torch.long, device=device)
        self._v1_final_n = torch.full((count,), -1, dtype=torch.long, device=device)
        self._v1_minimum_n = torch.full((count,), -1, dtype=torch.long, device=device)
        self._v1_n_decrease = torch.zeros(count, dtype=torch.long, device=device)
        self._v1_n_same = torch.zeros(count, dtype=torch.long, device=device)
        self._v1_n_increase = torch.zeros(count, dtype=torch.long, device=device)
        self._v1_cumulative_reward = torch.zeros(count, device=device)
        self._v1_locomotion_reward = torch.zeros(count, device=device)
        self._v1_last_touchdown_time = torch.zeros(count, dtype=torch.float64, device=device)
        self._v1_event_progress = torch.zeros(count, device=device)
        self._v1_event_step_cost = torch.zeros(count, device=device)
        self._v1_event_td5 = torch.zeros(count, device=device)
        self._v1_event_total = torch.zeros(count, device=device)
        self._v1_last_geometry_valid = torch.zeros(count, dtype=torch.bool, device=device)
        self._v1_last_solver_valid = torch.zeros(count, dtype=torch.bool, device=device)
        self._v1_solver_failure_count = 0
        self._v1_completed_episodes: list[dict[str, float | str]] = []
        self._v1_reward_ratios: deque[float] = deque(maxlen=4096)
        self._v1_touchdown_intervals: deque[float] = deque(maxlen=16384)

    def set_num_steps_per_learning_iteration(self, steps: int) -> None:
        super().set_num_steps_per_learning_iteration(steps)

    def _learning_iteration(self) -> int:
        policy_step = self.sim_step_counter // self.cfg.sim.decimation
        steps = int(
            getattr(
                self,
                "_steps_per_learning_iteration",
                self.cfg.push_curriculum.num_steps_per_iteration,
            )
        )
        return plane_v1_learning_iteration(policy_step, steps)

    def _terrain_log(self, level: int) -> dict[str, float]:
        slopes = tuple(float(value) for value in self.cfg.plane_recovery.slopes_degrees)
        terrain_types = self.scene.terrain.terrain_types
        log = {
            "TerrainCurriculum/level": float(level),
            "TerrainCurriculum/max_abs_slope": float(
                max(abs(slopes[index]) for index in plane_v1_allowed_type_indices(level, slopes))
            ),
        }
        for index, slope in enumerate(slopes):
            label = f"minus{abs(int(slope))}" if slope < 0 else f"plus{int(slope)}"
            log[f"TerrainCurriculum/P_slope_{label}"] = float(
                (terrain_types == index).to(torch.float32).mean().item()
            )
        return log

    def update_terrain_levels(self, env_ids):
        """Uniformly select an existing plane tile from the global level support."""

        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        level = plane_v1_terrain_level(
            self._learning_iteration(),
            level_1_iteration=self.cfg.terrain_level_1_iteration,
            level_2_iteration=self.cfg.terrain_level_2_iteration,
        )
        slopes = tuple(float(value) for value in self.cfg.plane_recovery.slopes_degrees)
        allowed = torch.tensor(
            plane_v1_allowed_type_indices(level, slopes),
            dtype=torch.long,
            device=self.device,
        )
        sampled = allowed[torch.randint(0, allowed.numel(), (env_ids.numel(),), device=self.device)]
        terrain = self.scene.terrain
        terrain.terrain_levels[env_ids] = 0
        terrain.terrain_types[env_ids] = sampled
        origins = terrain.terrain_origins[0, sampled]
        terrain.env_origins[env_ids] = origins
        self.scene.env_origins[env_ids] = origins
        self._terrain_curriculum_level = level
        return self._terrain_log(level)

    def reset(self, env_ids):
        BaseEnv.reset(self, env_ids)
        if not hasattr(self, "_v1_touchdown_index"):
            return
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self._clear_recovery_context(ids)
        self._estimator_history_count[ids] = 0
        if self._estimator_history is not None:
            self._estimator_history.buffer[ids] = 0.0
        if not self._defer_plane_v1_reset_cleanup:
            self._clear_plane_v1_episode(ids)

    def _clear_plane_v1_episode(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        self._clear_recovery(env_ids, clear_event_buffers=True)
        self._v1_touchdown_index[env_ids] = -1
        self._v1_previous_phi[env_ids] = 0.0
        self._v1_previous_phi_valid[env_ids] = False
        self._v1_last_n[env_ids] = -1
        self._v1_initial_n[env_ids] = -1
        self._v1_final_n[env_ids] = -1
        self._v1_minimum_n[env_ids] = -1
        self._v1_n_decrease[env_ids] = 0
        self._v1_n_same[env_ids] = 0
        self._v1_n_increase[env_ids] = 0
        self._v1_cumulative_reward[env_ids] = 0.0
        self._v1_locomotion_reward[env_ids] = 0.0
        self._v1_last_touchdown_time[env_ids] = 0.0

    def _on_curriculum_push(
        self,
        env_ids: torch.Tensor,
        delta_v_xy: torch.Tensor,
        sampled_level_indices: torch.Tensor,
    ) -> None:
        super()._on_curriculum_push(env_ids, delta_v_xy, sampled_level_indices)
        self._v1_touchdown_index[env_ids] = -1
        self._v1_previous_phi_valid[env_ids] = False
        self._v1_last_n[env_ids] = -1
        self._v1_initial_n[env_ids] = -1
        self._v1_final_n[env_ids] = -1
        self._v1_minimum_n[env_ids] = -1
        self._v1_n_decrease[env_ids] = 0
        self._v1_n_same[env_ids] = 0
        self._v1_n_increase[env_ids] = 0
        self._v1_cumulative_reward[env_ids] = 0.0
        self._v1_locomotion_reward[env_ids] = 0.0
        self._v1_last_touchdown_time[env_ids] = 0.0

    def _update_estimator(
        self,
        actor_obs: torch.Tensor,
        reset_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self._estimator is None:
            return torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        assert self._estimator_history is not None
        actor_frame = latest_actor_frame(
            actor_obs,
            history_length=5,
            per_frame_obs_dim=96,
        )
        imu_acceleration = self.scene.sensors["imu"].data.lin_acc_b
        estimator_input = self._estimator_history.append(
            actor_frame,
            imu_acceleration,
            reset_mask,
        )
        self._estimator_history_count = torch.where(
            reset_mask,
            torch.ones_like(self._estimator_history_count),
            torch.clamp(self._estimator_history_count + 1, max=5),
        )
        with torch.inference_mode():
            self._estimator_prediction.copy_(self._estimator(estimator_input))
        self._estimator_forward_count += 1
        return self._estimator_history_count >= 5

    def _certificate_state(self, physical_state):
        if self.com_velocity_source == "privileged":
            return physical_state
        return replace_com_velocity_for_certificate(
            physical_state,
            self._estimator_prediction,
        )

    def _refresh_plane_v1_context(
        self,
        certificate_state,
        dones: torch.Tensor,
        estimator_ready: torch.Tensor,
    ) -> None:
        touchdown_mask = certificate_state.touchdown & ~dones
        self._context_touchdown_mask.copy_(touchdown_mask)
        self._context_refresh_mask.zero_()
        self._touchdown_certificate_cache_mask.zero_()
        env_ids = touchdown_mask.nonzero(as_tuple=False).flatten()
        if env_ids.numel() == 0:
            return

        # All touchdown rows receive an explicit cache result.  Estimator rows
        # that are not yet warm are invalid and never fall back to GT velocity.
        self._touchdown_certificate_cache_mask[env_ids] = True
        self._touchdown_certificate_cache_n[env_ids] = 6
        self._touchdown_certificate_cache_margin[env_ids] = -3.0
        self._touchdown_certificate_cache_valid[env_ids] = False
        self._recovery_context[env_ids] = 0.0
        self.current_n_min[env_ids] = -1
        self.current_margin[env_ids] = 0.0
        self.current_certificate_valid[env_ids] = False
        self._v1_last_geometry_valid[env_ids] = certificate_state.terrain_plane_valid[env_ids]
        self._v1_last_solver_valid[env_ids] = False

        solve_ids = env_ids[estimator_ready[env_ids]]
        if solve_ids.numel() == 0:
            return
        assert self._certificate_evaluator is not None
        started = time.perf_counter()
        n_min, margin, valid = self._certificate_evaluator.evaluate_with_validity(
            certificate_state, solve_ids
        )
        elapsed = time.perf_counter() - started
        self._context_refresh_mask[solve_ids] = True
        self._touchdown_certificate_cache_n[solve_ids] = n_min
        self._touchdown_certificate_cache_margin[solve_ids] = margin
        self._touchdown_certificate_cache_valid[solve_ids] = valid
        self._recovery_context[solve_ids] = normalize_recovery_context(n_min, margin, valid)
        self.current_n_min[solve_ids] = torch.where(valid, n_min, torch.full_like(n_min, -1))
        self.current_margin[solve_ids] = torch.where(valid, margin, torch.zeros_like(margin))
        self.current_certificate_valid[solve_ids] = valid
        self._v1_last_solver_valid[solve_ids] = valid

        numerical_failure = certificate_state.terrain_plane_valid[solve_ids] & ~valid
        self._v1_solver_failure_count += int(numerical_failure.sum().item())
        self._context_refresh_batches += 1
        self._context_refresh_evaluations += int(solve_ids.numel())
        valid_count = int(valid.sum().item())
        self._context_valid_evaluations += valid_count
        if valid_count:
            self._context_n_counts += torch.bincount(
                torch.clamp(n_min[valid], min=0, max=6), minlength=7
            )
        self._context_refresh_total_seconds += elapsed
        self._context_refresh_last_seconds = elapsed
        self._context_refresh_latencies_s.append(elapsed)
        self._context_refresh_batch_sizes.append(int(solve_ids.numel()))

    def _update_practical_diagnostic(self, state, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        sample_count = self._interval_sample_count[env_ids]
        complete = self._interval_started_after_touchdown[env_ids] & (sample_count > 0)
        mean_velocity_error, mean_abs_tilt = practical_interval_means_from_sums(
            self._interval_velocity_error_sum[env_ids],
            self._interval_abs_tilt_sum[env_ids],
            sample_count,
        )
        touchdown_foot = state.touchdown_foot[env_ids]
        alternating = (self._last_touchdown_foot[env_ids] < 0) | (
            touchdown_foot != self._last_touchdown_foot[env_ids]
        )
        velocity_threshold, roll_threshold, pitch_threshold = self._practical_thresholds(
            state, env_ids
        )
        good = (
            complete
            & alternating
            & (mean_velocity_error <= velocity_threshold)
            & (mean_abs_tilt[:, 0] <= roll_threshold)
            & (mean_abs_tilt[:, 1] <= pitch_threshold)
        )
        newly_entered = good & ~self._practical_entered[env_ids]
        entered_ids = env_ids[newly_entered]
        self._practical_entered[entered_ids] = True
        self._practical_enter_step[entered_ids] = torch.clamp(
            self._v1_touchdown_index[entered_ids], min=0
        )
        self._last_touchdown_foot[env_ids] = touchdown_foot
        self._interval_started_after_touchdown[env_ids] = True
        self._interval_sample_count[env_ids] = 0
        self._interval_velocity_error_sum[env_ids] = 0.0
        self._interval_abs_tilt_sum[env_ids] = 0.0

    def _track_n_transition(self, env_id: int, n_min: int) -> None:
        previous_n = int(self._v1_last_n[env_id].item())
        if previous_n >= 0:
            if n_min < previous_n:
                self._v1_n_decrease[env_id] += 1
            elif n_min == previous_n:
                self._v1_n_same[env_id] += 1
            else:
                self._v1_n_increase[env_id] += 1
        else:
            self._v1_initial_n[env_id] = n_min
            self._v1_minimum_n[env_id] = n_min
        self._v1_last_n[env_id] = n_min
        self._v1_final_n[env_id] = n_min
        current_min = int(self._v1_minimum_n[env_id].item())
        self._v1_minimum_n[env_id] = n_min if current_min < 0 else min(current_min, n_min)

    def _finish_plane_v1_episode(self, env_id: int, outcome: str) -> None:
        locomotion = float(self._v1_locomotion_reward[env_id].item())
        recovery = float(self._v1_cumulative_reward[env_id].item())
        ratio = abs(recovery) / (abs(locomotion) + 1.0e-8)
        self._v1_reward_ratios.append(ratio)
        self._v1_completed_episodes.append(
            {
                "outcome": outcome,
                "cumulative_recovery_reward": recovery,
                "recovery_touchdown_count": float(max(int(self._v1_touchdown_index[env_id]), 0)),
                "initial_N": float(self._v1_initial_n[env_id].item()),
                "final_N": float(self._v1_final_n[env_id].item()),
                "minimum_N": float(self._v1_minimum_n[env_id].item()),
                "N_decrease_count": float(self._v1_n_decrease[env_id].item()),
                "N_same_count": float(self._v1_n_same[env_id].item()),
                "N_increase_count": float(self._v1_n_increase[env_id].item()),
                "recovery_to_locomotion_abs_ratio": ratio,
            }
        )
        ids = torch.tensor([env_id], dtype=torch.long, device=self.device)
        self._clear_plane_v1_episode(ids)

    def _process_plane_v1_touchdowns(self, state, dones: torch.Tensor) -> None:
        mask = state.touchdown & self.recovery_active & ~dones & ~self._push_started_this_step
        env_ids = mask.nonzero(as_tuple=False).flatten()
        if env_ids.numel() == 0:
            return
        self._v1_touchdown_index[env_ids] += 1
        self._recovery_touchdowns[env_ids] = torch.clamp(
            self._v1_touchdown_index[env_ids], min=0
        )
        self._update_practical_diagnostic(state, env_ids)

        now = state.time[env_ids].to(torch.float64)
        previous_time = self._v1_last_touchdown_time[env_ids]
        has_previous = previous_time > 0.0
        for interval in (now[has_previous] - previous_time[has_previous]).detach().cpu().tolist():
            self._v1_touchdown_intervals.append(float(interval))
        self._v1_last_touchdown_time[env_ids] = now

        finish_ids: list[int] = []
        for env_id in env_ids.detach().cpu().tolist():
            index = int(self._v1_touchdown_index[env_id].item())
            if index > self._plane_v1_reward_parameters.certificate_horizon_touchdowns:
                raise RuntimeError("Plane V1 reward episode exceeded TD5")
            n_min = int(self._touchdown_certificate_cache_n[env_id].item())
            margin = float(self._touchdown_certificate_cache_margin[env_id].item())
            geometry_valid = bool(self._v1_last_geometry_valid[env_id].item())
            solver_valid = bool(self._v1_last_solver_valid[env_id].item())
            previous_phi = (
                float(self._v1_previous_phi[env_id].item())
                if bool(self._v1_previous_phi_valid[env_id].item())
                else None
            )
            result = plane_v1_touchdown_reward(
                previous_phi,
                n_min,
                margin,
                touchdown_index=index,
                terrain_plane_valid=geometry_valid,
                solver_valid=solver_valid,
                enabled=self._plane_v1_reward_enabled,
                parameters=self._plane_v1_reward_parameters,
            )
            if result.update_previous_phi:
                assert result.phi_current is not None
                self._v1_previous_phi[env_id] = result.phi_current
                self._v1_previous_phi_valid[env_id] = True
            if solver_valid:
                self._track_n_transition(env_id, n_min)
            self._v1_event_progress[env_id] = result.progress
            self._v1_event_step_cost[env_id] = result.step_cost
            self._v1_event_td5[env_id] = result.td5_penalty
            self._v1_event_total[env_id] = result.total
            self._v1_cumulative_reward[env_id] += result.total
            if index == self._plane_v1_reward_parameters.certificate_horizon_touchdowns:
                finish_ids.append(env_id)
        for env_id in finish_ids:
            self._finish_plane_v1_episode(env_id, "TD5")

    def _process_plane_v1_falls(self, dones: torch.Tensor) -> None:
        ids = (dones & self.recovery_active).nonzero(as_tuple=False).flatten()
        for env_id in ids.detach().cpu().tolist():
            outcome = "TRUNCATED" if bool(self.time_out_buf[env_id].item()) else "FALL"
            self._finish_plane_v1_episode(env_id, outcome)

    def _update_plane_v1_logs(self, extras: dict) -> None:
        log = extras.setdefault("log", {})
        level = plane_v1_terrain_level(
            self._learning_iteration(),
            level_1_iteration=self.cfg.terrain_level_1_iteration,
            level_2_iteration=self.cfg.terrain_level_2_iteration,
        )
        log.update(self._terrain_log(level))
        log.update(
            {
                "Push/fixed_full_range": 1.0,
                "Push/max_abs_delta_v_x": 1.0,
                "Push/max_abs_delta_v_y": 1.0,
                "RecoveryReward/progress_mean": float(self._v1_event_progress.mean().item()),
                "RecoveryReward/step_cost_mean": float(self._v1_event_step_cost.mean().item()),
                "RecoveryReward/td5_penalty_mean": float(self._v1_event_td5.mean().item()),
                "RecoveryReward/total_mean": float(self._v1_event_total.mean().item()),
                "RecoveryReward/nonzero_fraction": float(
                    (self._v1_event_total != 0.0).to(torch.float32).mean().item()
                ),
                "RecoveryReward/enabled": float(self._plane_v1_reward_enabled),
                "RecoveryReward/solver_failure_count": float(self._v1_solver_failure_count),
                "Recovery/active_fraction": float(self.recovery_active.float().mean().item()),
                "RecoveryContext/refresh_batches": float(self._context_refresh_batches),
                "RecoveryContext/evaluations": float(self._context_refresh_evaluations),
                "RecoveryContext/valid_fraction": float(
                    self.current_certificate_valid.float().mean().item()
                ),
                "RecoveryContext/estimator_source": float(self.com_velocity_source == "estimator"),
                "RecoveryContext/estimator_history_ready_fraction": float(
                    (self._estimator_history_count >= 5).to(torch.float32).mean().item()
                    if self.com_velocity_source == "estimator"
                    else 1.0
                ),
                "Push/observed_abs_max_delta_v_x": float(
                    torch.abs(self.last_push_delta_v_xy[:, 0]).max().item()
                ),
                "Push/observed_abs_max_delta_v_y": float(
                    torch.abs(self.last_push_delta_v_xy[:, 1]).max().item()
                ),
            }
        )
        if self._v1_reward_ratios:
            values = torch.tensor(tuple(self._v1_reward_ratios), dtype=torch.float32)
            log["RecoveryReward/abs_ratio_median"] = float(torch.quantile(values, 0.50))
            log["RecoveryReward/abs_ratio_P75"] = float(torch.quantile(values, 0.75))
            log["RecoveryReward/abs_ratio_P90"] = float(torch.quantile(values, 0.90))
        else:
            log["RecoveryReward/abs_ratio_median"] = 0.0
            log["RecoveryReward/abs_ratio_P75"] = 0.0
            log["RecoveryReward/abs_ratio_P90"] = 0.0
        if self._v1_touchdown_intervals:
            intervals = tuple(self._v1_touchdown_intervals)
            log["RecoveryReward/touchdown_interval_mean"] = float(statistics.mean(intervals))
            log["RecoveryReward/touchdown_interval_median"] = float(statistics.median(intervals))
        else:
            log["RecoveryReward/touchdown_interval_mean"] = 0.0
            log["RecoveryReward/touchdown_interval_median"] = 0.0
        if self._v1_completed_episodes:
            keys = (
                "cumulative_recovery_reward",
                "recovery_touchdown_count",
                "initial_N",
                "final_N",
                "minimum_N",
                "N_decrease_count",
                "N_same_count",
                "N_increase_count",
                "recovery_to_locomotion_abs_ratio",
            )
            for key in keys:
                log[f"RecoveryEpisode/{key}"] = sum(
                    float(item[key]) for item in self._v1_completed_episodes
                ) / len(self._v1_completed_episodes)
        extras["plane_v1_recovery_episodes"] = list(self._v1_completed_episodes)
        extras["push_mode"] = "fixed_full_range"

    def step(self, actions: torch.Tensor):
        context_before_step = self._recovery_context.clone()
        recovery_before_step = self.recovery_active.clone()
        self._v1_completed_episodes = []
        self._v1_event_progress.zero_()
        self._v1_event_step_cost.zero_()
        self._v1_event_td5.zero_()
        self._v1_event_total.zero_()
        self._defer_plane_v1_reset_cleanup = True
        try:
            actor_obs, locomotion_reward, dones, extras = BaseEnv.step(self, actions)
        finally:
            self._defer_plane_v1_reset_cleanup = False

        physical_state = self._state_extractor.extract()
        reset_mask = dones | physical_state.episode_reset
        estimator_ready = self._update_estimator(actor_obs, reset_mask)
        certificate_state = self._certificate_state(physical_state)
        self._last_recovery_state = physical_state
        self._last_certificate_state = certificate_state

        active_for_locomotion = (recovery_before_step | self.recovery_active) & ~dones
        self._v1_locomotion_reward[active_for_locomotion] += locomotion_reward[
            active_for_locomotion
        ]
        self._process_plane_v1_falls(dones)
        done_ids = dones.nonzero(as_tuple=False).flatten()
        self._clear_recovery_context(done_ids)
        self._refresh_plane_v1_context(certificate_state, dones, estimator_ready)
        held_mask = ~self._context_touchdown_mask & ~dones
        if not torch.equal(self._recovery_context[held_mask], context_before_step[held_mask]):
            raise RuntimeError("Plane V1 recovery context changed between touchdowns")
        self._process_plane_v1_touchdowns(physical_state, dones)
        self._accumulate_practical_metrics(physical_state)

        locomotion_reward += self._v1_event_total
        if not self._plane_v1_reward_enabled and torch.any(self._v1_event_total != 0.0):
            raise RuntimeError("reward-OFF Plane V1 task produced a recovery reward")
        self._last_push_started_mask = self._push_started_this_step.clone()
        self._push_started_this_step.zero_()
        self._update_plane_v1_logs(extras)
        actor_with_context = self._append_recovery_context(actor_obs)
        if (
            not torch.all(torch.isfinite(actor_with_context))
            or not torch.all(torch.isfinite(locomotion_reward))
        ):
            raise RuntimeError("Plane V1 step produced NaN or Inf")
        return actor_with_context, locomotion_reward, dones, extras


__all__ = ["G1PlaneV1Env"]
