"""Thin Stage2 recovery environment with a certificate-only A/B reward path."""

from __future__ import annotations

from pathlib import Path

import torch

from legged_lab.envs.base.base_env import BaseEnv
from legged_lab.recovery.g1_certificate_runtime import CalibratedG1CertificateEvaluator
from legged_lab.recovery.push_curriculum import (
    CurriculumRecoveryOutcome,
    CurriculumUpgradeReason,
    PushCurriculumController,
    record_episode_batch,
)
from legged_lab.recovery.stage2_reward import (
    CERTIFICATE_PROGRESS_SCALE,
    SUCCESS_MAX,
    TIMEOUT_PENALTY,
    TOUCHDOWN_COST,
    certificate_potential_tensor,
)
from legged_lab.recovery.state_extractor import G1PrivilegedStateExtractor, G1StateExtractorCfg


class G1RecoveryEnv(BaseEnv):
    """Keep the Stage1A task intact while adding Stage2-only bookkeeping.

    Recovery state is simulator-privileged and is never appended to actor or
    critic observations.  Baseline keeps the original RewardManager output;
    Ours enables the Stage2 recovery-event and certificate reward channel.
    """

    def __init__(self, cfg, headless):
        if float(cfg.push_curriculum.recovery_reward_weight) != 0.0:
            raise ValueError("use stage2_reward.event_scale instead of recovery_reward_weight")
        super().__init__(cfg, headless)

        self.push_curriculum = PushCurriculumController(cfg.push_curriculum)
        self._stage2_reward_enabled = bool(cfg.stage2_reward.enabled)
        self._shared_event_reward_enabled = bool(
            cfg.stage2_reward.enable_shared_event_reward
        )
        self._certificate_reward_enabled = bool(cfg.stage2_reward.enable_certificate_reward)
        self._soft_reward_scaling_enabled = bool(
            cfg.stage2_reward.enable_soft_reward_scaling
        )
        self._deferred_certificate_configured = bool(
            cfg.stage2_reward.defer_certificate_reward_to_rollout_end
        )
        if not self._stage2_reward_enabled and (
            self._shared_event_reward_enabled
            or self._certificate_reward_enabled
            or self._soft_reward_scaling_enabled
        ):
            raise ValueError("Stage2 subchannels require stage2_reward.enabled=True")
        if self._deferred_certificate_configured and not self._certificate_reward_enabled:
            raise ValueError("deferred certificate rewards require the certificate channel")
        self._steps_per_learning_iteration = int(cfg.push_curriculum.num_steps_per_iteration)
        self._state_extractor = G1PrivilegedStateExtractor(
            self,
            G1StateExtractorCfg(h_eff=0.6884990671277046),
        )
        self._certificate_evaluator = (
            CalibratedG1CertificateEvaluator(
                cfg.stage2_reward.certificate_parameters_path,
                workers=cfg.stage2_reward.certificate_workers,
                failure_window_size=cfg.stage2_reward.certificate_failure_window_size,
                failure_rate_threshold=cfg.stage2_reward.certificate_failure_rate_threshold,
            )
            if self._stage2_reward_enabled and self._certificate_reward_enabled
            else None
        )
        self._initialize_recovery_buffers()
        self._initialize_soft_reward_terms()

    def _initialize_recovery_buffers(self) -> None:
        device = self.device
        count = self.num_envs
        self.recovery_active = torch.zeros(count, dtype=torch.bool, device=device)
        self._recovery_touchdowns = torch.zeros(count, dtype=torch.long, device=device)
        self._recovery_level_indices = torch.zeros(count, dtype=torch.long, device=device)
        self._last_touchdown_foot = torch.full((count,), -1, dtype=torch.long, device=device)
        self._interval_started_after_touchdown = torch.zeros(count, dtype=torch.bool, device=device)
        self._interval_sample_count = torch.zeros(count, dtype=torch.long, device=device)
        self._interval_velocity_error_sum = torch.zeros(count, device=device)
        self._interval_abs_tilt_sum = torch.zeros((count, 2), device=device)
        self._practical_entered = torch.zeros(count, dtype=torch.bool, device=device)
        self._practical_enter_step = torch.zeros(count, dtype=torch.long, device=device)
        self._truncated_recovery_count = 0
        self._pending_initial_certificate = torch.zeros(count, dtype=torch.bool, device=device)
        self._push_started_this_step = torch.zeros(count, dtype=torch.bool, device=device)
        self._certificate_n = torch.full((count,), -1, dtype=torch.long, device=device)
        self._certificate_margin = torch.zeros(count, device=device)
        self._certificate_phi_previous = torch.zeros(count, device=device)

        self._event_touchdown_cost = torch.zeros(count, device=device)
        self._event_success = torch.zeros(count, device=device)
        self._event_timeout = torch.zeros(count, device=device)
        self._event_certificate = torch.zeros(count, device=device)
        self._episode_locomotion_reward = torch.zeros(count, device=device)
        self._episode_locomotion_reward_abs_sum = torch.zeros(count, device=device)
        self._episode_shared_reward = torch.zeros(count, device=device)
        self._episode_certificate_reward = torch.zeros(count, device=device)
        self._episode_total_reward = torch.zeros(count, device=device)
        self._episode_recovery_event_reward_abs_sum = torch.zeros(count, device=device)
        self._step_completed_reward_episodes: list[dict] = []
        self._event_dt_diagnostic_printed = False
        self._last_recovery_reward_mean = 0.0
        self._certificate_event_count = 0
        self._certificate_nonzero_event_count = 0
        self._defer_recovery_reset_cleanup = False
        self._last_event_rewards: dict[str, torch.Tensor] = {}
        self._last_soft_scaling_recovery_mask = torch.zeros(
            count, dtype=torch.bool, device=device
        )
        self._last_push_started_mask = torch.zeros(count, dtype=torch.bool, device=device)

        # The runner activates this only for complete PPO rollouts.  Play and
        # standalone smoke paths keep using the synchronous evaluator.
        self._deferred_rollout_active = False
        self._deferred_rollout_length = 0
        self._deferred_rollout_step = 0
        self._deferred_certificate_batches: list[dict] = []
        self._deferred_closed_tokens: set[tuple[int, int]] = set()
        self._deferred_phi_by_token: dict[tuple[int, int], float] = {}
        self._deferred_episode_certificate_sum: dict[tuple[int, int], float] = {}
        self._deferred_episode_certificate_abs_sum: dict[tuple[int, int], float] = {}
        self._deferred_pending_episode_metrics: dict[tuple[int, int], dict] = {}
        self._recovery_episode_generation = torch.zeros(
            count, dtype=torch.long, device=device
        )

        self.last_push_delta_v_xy = torch.zeros((count, 2), device=device)
        self.push_curriculum_level = self.push_curriculum.level
        self.push_curriculum_level_ratio = self.push_curriculum.level_ratio
        self.push_curriculum_max_xy = torch.tensor(
            self.push_curriculum.current_abs_delta_v_xy,
            dtype=torch.float32,
            device=device,
        )

    def _initialize_soft_reward_terms(self) -> None:
        self._soft_reward_term_indices: dict[str, tuple[int, float]] = {}
        self._last_soft_reward_multipliers: dict[str, torch.Tensor] = {}
        if not self._stage2_reward_enabled or not self._soft_reward_scaling_enabled:
            return
        active_terms = list(self.reward_manager.active_terms)
        for term_name, alpha_min in self.cfg.stage2_reward.soft_reward_min_multipliers.items():
            alpha_min = float(alpha_min)
            if not 0.0 <= alpha_min <= 1.0:
                raise ValueError(f"soft reward alpha_min for {term_name} must be in [0, 1]")
            if term_name not in active_terms:
                raise ValueError(f"soft reward term is not active in the shared task: {term_name}")
            self._soft_reward_term_indices[term_name] = (active_terms.index(term_name), alpha_min)
            self._last_soft_reward_multipliers[term_name] = torch.ones(
                self.num_envs, device=self.device
            )

    def reset(self, env_ids):
        """Clear recovery one-shot buffers on explicit resets.

        Resets performed inside ``BaseEnv.step`` are deferred until their FALL
        or horizon outcome has been recorded below.
        """

        super().reset(env_ids)
        if hasattr(self, "recovery_active") and not self._defer_recovery_reset_cleanup:
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
            self._clear_recovery(env_ids, clear_event_buffers=True)

    def set_num_steps_per_learning_iteration(self, steps: int) -> None:
        if steps <= 0:
            raise ValueError("num_steps_per_learning_iteration must be positive")
        self._steps_per_learning_iteration = int(steps)

    def configure_curriculum_logging(self, log_dir: str | Path) -> None:
        self.push_curriculum.configure_upgrade_log(log_dir)
        if self._certificate_evaluator is not None:
            self._certificate_evaluator.configure_diagnostics(log_dir)

    def begin_deferred_reward_rollout(self, num_steps: int) -> bool:
        """Enable asynchronous certificate jobs for one complete PPO rollout."""

        if not self._deferred_certificate_configured:
            return False
        if self._deferred_rollout_active:
            raise RuntimeError("a deferred certificate rollout is already active")
        if num_steps <= 0:
            raise ValueError("deferred rollout length must be positive")
        self._deferred_rollout_active = True
        self._deferred_rollout_length = int(num_steps)
        self._deferred_rollout_step = 0
        self._deferred_certificate_batches = []
        self._deferred_closed_tokens = set()
        return True

    def _tokens_for(self, env_ids: torch.Tensor) -> tuple[tuple[int, int], ...]:
        ids = env_ids.detach().cpu().tolist()
        generations = self._recovery_episode_generation[env_ids].detach().cpu().tolist()
        return tuple((int(env_id), int(generation)) for env_id, generation in zip(ids, generations))

    def _submit_deferred_certificate_batch(
        self,
        state,
        env_ids: torch.Tensor,
        *,
        kind: str,
    ) -> None:
        if not self._deferred_rollout_active:
            raise RuntimeError("deferred certificate submission occurred outside a PPO rollout")
        if kind not in ("push", "touchdown"):
            raise ValueError(f"unknown deferred certificate event kind: {kind}")
        assert self._certificate_evaluator is not None
        self._deferred_certificate_batches.append(
            {
                "kind": kind,
                "rollout_step": self._deferred_rollout_step,
                "env_ids": env_ids.detach().clone(),
                "tokens": self._tokens_for(env_ids),
                "pending": self._certificate_evaluator.submit(state, env_ids),
            }
        )

    @staticmethod
    def _episode_reward_metric(
        base: dict,
        certificate: float,
        certificate_abs_sum: float,
    ) -> dict:
        locomotion = float(base["episode_locomotion_reward_during_recovery"])
        locomotion_abs_sum = float(
            base["episode_locomotion_reward_abs_sum_during_recovery"]
        )
        shared = float(base["episode_shared_recovery_reward"])
        total = shared + certificate
        event_abs_sum = float(base["episode_shared_reward_abs_sum"]) + certificate_abs_sum
        return {
            "env_id": int(base["env_id"]),
            "outcome": base["outcome"],
            "episode_locomotion_reward_during_recovery": locomotion,
            "episode_shared_recovery_reward": shared,
            "episode_certificate_reward": certificate,
            "episode_total_recovery_reward": total,
            "recovery_to_locomotion_abs_ratio": abs(total) / (abs(locomotion) + 1.0e-8),
            "episode_recovery_event_reward_abs_sum": event_abs_sum,
            "episode_locomotion_reward_abs_sum_during_recovery": locomotion_abs_sum,
            "absolute_recovery_to_locomotion_ratio": event_abs_sum
            / (locomotion_abs_sum + 1.0e-8),
        }

    def resolve_deferred_reward_rollout(self) -> dict:
        """Wait for LP jobs and return rewards indexed by their original steps."""

        if not self._deferred_rollout_active:
            raise RuntimeError("no deferred certificate rollout is active")
        if self._deferred_rollout_step != self._deferred_rollout_length:
            raise RuntimeError(
                "deferred rollout resolved before every environment step was collected"
            )
        assert self._certificate_evaluator is not None
        corrections = torch.zeros(
            (self._deferred_rollout_length, self.num_envs),
            dtype=torch.float32,
            device=self.device,
        )
        scale = float(self.cfg.stage2_reward.event_scale)
        for event in self._deferred_certificate_batches:
            n_min, margin = self._certificate_evaluator.resolve(event["pending"])
            phi = certificate_potential_tensor(n_min, margin)
            env_ids = event["env_ids"]
            tokens = event["tokens"]
            if event["kind"] == "push":
                for token, value in zip(tokens, phi.detach().cpu().tolist()):
                    self._deferred_phi_by_token[token] = float(value)
                    self._deferred_episode_certificate_sum.setdefault(token, 0.0)
                    self._deferred_episode_certificate_abs_sum.setdefault(token, 0.0)
                continue

            missing = [token for token in tokens if token not in self._deferred_phi_by_token]
            if missing:
                raise RuntimeError(
                    f"touchdown certificate has no resolved push potential: {missing[:3]}"
                )
            previous = torch.tensor(
                [self._deferred_phi_by_token[token] for token in tokens],
                dtype=phi.dtype,
                device=phi.device,
            )
            reward = scale * CERTIFICATE_PROGRESS_SCALE * (phi - previous)
            corrections[event["rollout_step"], env_ids] += reward
            self._certificate_nonzero_event_count += int(
                torch.count_nonzero(torch.abs(reward) > 1.0e-8).item()
            )
            for token, phi_value, reward_value in zip(
                tokens,
                phi.detach().cpu().tolist(),
                reward.detach().cpu().tolist(),
            ):
                reward_value = float(reward_value)
                self._deferred_phi_by_token[token] = float(phi_value)
                self._deferred_episode_certificate_sum[token] = (
                    self._deferred_episode_certificate_sum.get(token, 0.0) + reward_value
                )
                self._deferred_episode_certificate_abs_sum[token] = (
                    self._deferred_episode_certificate_abs_sum.get(token, 0.0)
                    + abs(reward_value)
                )

            current_generations = self._recovery_episode_generation[env_ids]
            token_generations = torch.tensor(
                [token[1] for token in tokens],
                dtype=current_generations.dtype,
                device=self.device,
            )
            current = current_generations == token_generations
            current_ids = env_ids[current]
            self._certificate_n[current_ids] = n_min[current]
            self._certificate_margin[current_ids] = margin[current]
            self._certificate_phi_previous[current_ids] = phi[current]

        completed_episode_rewards = []
        for token in self._deferred_closed_tokens:
            base = self._deferred_pending_episode_metrics.pop(token, None)
            certificate = self._deferred_episode_certificate_sum.pop(token, 0.0)
            certificate_abs_sum = self._deferred_episode_certificate_abs_sum.pop(token, 0.0)
            self._deferred_phi_by_token.pop(token, None)
            if base is not None:
                completed_episode_rewards.append(
                    self._episode_reward_metric(base, certificate, certificate_abs_sum)
                )

        log = {
            "Reward/recovery_touchdown_cost": 0.0,
            "Reward/recovery_success": 0.0,
            "Reward/recovery_timeout": 0.0,
            "Reward/recovery_certificate": float(corrections.mean().item()),
            "Reward/recovery_shared_total": 0.0,
            "Reward/recovery_total": float(corrections.mean().item()),
            "Recovery/reward": float(corrections.mean().item()),
        }
        if completed_episode_rewards:
            metric_keys = (
                "episode_locomotion_reward_during_recovery",
                "episode_shared_recovery_reward",
                "episode_certificate_reward",
                "episode_total_recovery_reward",
                "recovery_to_locomotion_abs_ratio",
                "episode_recovery_event_reward_abs_sum",
                "episode_locomotion_reward_abs_sum_during_recovery",
                "absolute_recovery_to_locomotion_ratio",
            )
            for key in metric_keys:
                log[f"Recovery/{key}"] = sum(
                    item[key] for item in completed_episode_rewards
                ) / len(completed_episode_rewards)

        self._deferred_rollout_active = False
        self._deferred_certificate_batches = []
        self._deferred_closed_tokens = set()
        return {
            "reward_corrections": corrections,
            "episode_rewards": completed_episode_rewards,
            "log": log,
        }

    def eligible_recovery_push_env_ids(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Do not start a second recovery episode before the first one exits."""

        return env_ids[~self.recovery_active[env_ids]]

    def _on_curriculum_push(
        self,
        env_ids: torch.Tensor,
        delta_v_xy: torch.Tensor,
        sampled_level_indices: torch.Tensor,
    ) -> None:
        """Enter RECOVERY immediately after the known velocity perturbation."""

        if torch.any(self.recovery_active[env_ids]):
            raise RuntimeError("overlapping recovery episodes are not allowed")
        self.recovery_active[env_ids] = True
        self._recovery_episode_generation[env_ids] += 1
        self._recovery_touchdowns[env_ids] = 0
        self._recovery_level_indices[env_ids] = sampled_level_indices
        self._last_touchdown_foot[env_ids] = -1
        self._interval_started_after_touchdown[env_ids] = False
        self._interval_sample_count[env_ids] = 0
        self._interval_velocity_error_sum[env_ids] = 0.0
        self._interval_abs_tilt_sum[env_ids] = 0.0
        self._practical_entered[env_ids] = False
        self._practical_enter_step[env_ids] = 0
        self.last_push_delta_v_xy[env_ids] = delta_v_xy
        self._push_started_this_step[env_ids] = True
        if self._stage2_reward_enabled:
            self._clear_event_buffers(env_ids)
            self._pending_initial_certificate[env_ids] = self._certificate_reward_enabled

    def _clear_event_buffers(self, env_ids: torch.Tensor) -> None:
        self._event_touchdown_cost[env_ids] = 0.0
        self._event_success[env_ids] = 0.0
        self._event_timeout[env_ids] = 0.0
        self._event_certificate[env_ids] = 0.0

    def _clear_recovery(self, env_ids: torch.Tensor, *, clear_event_buffers: bool = False) -> None:
        if env_ids.numel() == 0:
            return
        if self._deferred_rollout_active and self._certificate_reward_enabled:
            active_ids = env_ids[self.recovery_active[env_ids]]
            self._deferred_closed_tokens.update(self._tokens_for(active_ids))
        self.recovery_active[env_ids] = False
        self._recovery_touchdowns[env_ids] = 0
        self._last_touchdown_foot[env_ids] = -1
        self._interval_started_after_touchdown[env_ids] = False
        self._interval_sample_count[env_ids] = 0
        self._interval_velocity_error_sum[env_ids] = 0.0
        self._interval_abs_tilt_sum[env_ids] = 0.0
        self._practical_entered[env_ids] = False
        self._practical_enter_step[env_ids] = 0
        self._pending_initial_certificate[env_ids] = False
        self._push_started_this_step[env_ids] = False
        self._certificate_n[env_ids] = -1
        self._certificate_margin[env_ids] = 0.0
        self._certificate_phi_previous[env_ids] = 0.0
        self._episode_locomotion_reward[env_ids] = 0.0
        self._episode_locomotion_reward_abs_sum[env_ids] = 0.0
        self._episode_shared_reward[env_ids] = 0.0
        self._episode_certificate_reward[env_ids] = 0.0
        self._episode_total_reward[env_ids] = 0.0
        self._episode_recovery_event_reward_abs_sum[env_ids] = 0.0
        if clear_event_buffers:
            self._clear_event_buffers(env_ids)

    def _record_episode_reward_metrics(
        self,
        env_ids: torch.Tensor,
        outcome: CurriculumRecoveryOutcome,
    ) -> None:
        if not self._stage2_reward_enabled:
            return
        locomotion = self._episode_locomotion_reward[env_ids].detach().cpu().tolist()
        locomotion_abs_sum = (
            self._episode_locomotion_reward_abs_sum[env_ids].detach().cpu().tolist()
        )
        shared = self._episode_shared_reward[env_ids].detach().cpu().tolist()
        certificate = self._episode_certificate_reward[env_ids].detach().cpu().tolist()
        total = self._episode_total_reward[env_ids].detach().cpu().tolist()
        event_abs_sum = (
            self._episode_recovery_event_reward_abs_sum[env_ids].detach().cpu().tolist()
        )
        if self._deferred_rollout_active:
            for token, env_id, locomotion_value, locomotion_abs_sum_value, shared_value, shared_abs in zip(
                self._tokens_for(env_ids),
                env_ids.detach().cpu().tolist(),
                locomotion,
                locomotion_abs_sum,
                shared,
                event_abs_sum,
            ):
                self._deferred_pending_episode_metrics[token] = {
                    "env_id": int(env_id),
                    "outcome": outcome.value,
                    "episode_locomotion_reward_during_recovery": float(locomotion_value),
                    "episode_locomotion_reward_abs_sum_during_recovery": float(
                        locomotion_abs_sum_value
                    ),
                    "episode_shared_recovery_reward": float(shared_value),
                    "episode_shared_reward_abs_sum": float(shared_abs),
                }
            return
        for (
            env_id,
            locomotion_value,
            locomotion_abs_sum_value,
            shared_value,
            certificate_value,
            total_value,
            event_abs_sum_value,
        ) in zip(
            env_ids.detach().cpu().tolist(),
            locomotion,
            locomotion_abs_sum,
            shared,
            certificate,
            total,
            event_abs_sum,
        ):
            ratio = abs(total_value) / (abs(locomotion_value) + 1.0e-8)
            absolute_sum_ratio = event_abs_sum_value / (locomotion_abs_sum_value + 1.0e-8)
            self._step_completed_reward_episodes.append(
                {
                    "env_id": int(env_id),
                    "outcome": outcome.value,
                    "episode_locomotion_reward_during_recovery": float(locomotion_value),
                    "episode_shared_recovery_reward": float(shared_value),
                    "episode_certificate_reward": float(certificate_value),
                    "episode_total_recovery_reward": float(total_value),
                    "recovery_to_locomotion_abs_ratio": float(ratio),
                    "episode_recovery_event_reward_abs_sum": float(event_abs_sum_value),
                    "episode_locomotion_reward_abs_sum_during_recovery": float(
                        locomotion_abs_sum_value
                    ),
                    "absolute_recovery_to_locomotion_ratio": float(absolute_sum_ratio),
                }
            )

    def _record_outcomes(
        self,
        env_ids: torch.Tensor,
        outcome: CurriculumRecoveryOutcome,
        practical_enter_steps: torch.Tensor | None = None,
        clear_event_buffers: bool = False,
    ) -> None:
        if env_ids.numel() == 0:
            return
        levels = self._recovery_level_indices[env_ids].detach().cpu().tolist()
        if practical_enter_steps is None:
            steps = [None] * len(levels)
        else:
            steps = practical_enter_steps.detach().cpu().tolist()
        record_episode_batch(
            self.push_curriculum,
            levels,
            [outcome] * len(levels),
            steps,
        )
        self._record_episode_reward_metrics(env_ids, outcome)
        self._clear_recovery(env_ids, clear_event_buffers=clear_event_buffers)

    def _process_resets(self, dones: torch.Tensor) -> None:
        reset_ids = (dones & self.recovery_active).nonzero(as_tuple=False).flatten()
        if reset_ids.numel() == 0:
            return
        real_fall_ids = reset_ids[~self.time_out_buf[reset_ids]]
        self._record_outcomes(
            real_fall_ids,
            CurriculumRecoveryOutcome.FALL,
            clear_event_buffers=True,
        )

        # An environment horizon is a rollout truncation, not the formal TD5
        # TIMEOUT and not a physical FALL.  Exclude it from performance windows.
        horizon_ids = reset_ids[self.time_out_buf[reset_ids]]
        self._truncated_recovery_count += int(horizon_ids.numel())
        self._clear_recovery(horizon_ids, clear_event_buffers=True)

    def _initialize_pending_certificates(self, state, dones: torch.Tensor) -> None:
        if not self._certificate_reward_enabled:
            return
        pending = self._pending_initial_certificate & self.recovery_active & ~dones
        env_ids = pending.nonzero(as_tuple=False).flatten()
        if env_ids.numel() == 0:
            return
        assert self._certificate_evaluator is not None
        if self._deferred_rollout_active:
            self._submit_deferred_certificate_batch(state, env_ids, kind="push")
            self._pending_initial_certificate[env_ids] = False
            return
        n_min, margin = self._certificate_evaluator.evaluate(state, env_ids)
        self._certificate_n[env_ids] = n_min
        self._certificate_margin[env_ids] = margin
        self._certificate_phi_previous[env_ids] = certificate_potential_tensor(n_min, margin)
        self._pending_initial_certificate[env_ids] = False

    def _queue_touchdown_rewards(
        self,
        state,
        env_ids: torch.Tensor,
        entered_ids: torch.Tensor,
        timeout_ids: torch.Tensor,
    ) -> None:
        if not self._stage2_reward_enabled or env_ids.numel() == 0:
            return
        if torch.any(self._pending_initial_certificate[env_ids]):
            raise RuntimeError("touchdown reward observed before the push certificate was initialized")
        if any(
            torch.any(buffer[env_ids] != 0.0)
            for buffer in (
                self._event_touchdown_cost,
                self._event_success,
                self._event_timeout,
                self._event_certificate,
            )
        ):
            raise RuntimeError("a recovery event buffer was not consumed exactly once")

        scale = float(self.cfg.stage2_reward.event_scale)
        if self._shared_event_reward_enabled:
            self._event_touchdown_cost[env_ids] = scale * TOUCHDOWN_COST
        if self._certificate_reward_enabled:
            assert self._certificate_evaluator is not None
            self._certificate_event_count += int(env_ids.numel())
            if self._deferred_rollout_active:
                self._submit_deferred_certificate_batch(state, env_ids, kind="touchdown")
            else:
                n_min, margin = self._certificate_evaluator.evaluate(state, env_ids)
                phi_current = certificate_potential_tensor(n_min, margin)
                raw_certificate = CERTIFICATE_PROGRESS_SCALE * (
                    phi_current - self._certificate_phi_previous[env_ids]
                )
                self._certificate_phi_previous[env_ids] = phi_current
                self._certificate_n[env_ids] = n_min
                self._certificate_margin[env_ids] = margin
                self._event_certificate[env_ids] = scale * raw_certificate
                self._certificate_nonzero_event_count += int(
                    torch.count_nonzero(torch.abs(raw_certificate) > 1.0e-8).item()
                )
        if self._shared_event_reward_enabled and entered_ids.numel() > 0:
            touchdown_count = self._recovery_touchdowns[entered_ids].to(torch.float32)
            self._event_success[entered_ids] = (
                scale * SUCCESS_MAX * (6.0 - touchdown_count) / 5.0
            )
        if self._shared_event_reward_enabled and timeout_ids.numel() > 0:
            self._event_timeout[timeout_ids] = scale * TIMEOUT_PENALTY

        shared = (
            self._event_touchdown_cost[env_ids]
            + self._event_success[env_ids]
            + self._event_timeout[env_ids]
        )
        certificate = self._event_certificate[env_ids]
        self._episode_shared_reward[env_ids] += shared
        self._episode_certificate_reward[env_ids] += certificate
        self._episode_total_reward[env_ids] += shared + certificate
        self._episode_recovery_event_reward_abs_sum[env_ids] += torch.abs(shared + certificate)

    def _process_touchdowns(self, state, dones: torch.Tensor) -> None:
        touchdown_mask = (
            state.touchdown
            & self.recovery_active
            & ~dones
            & ~self._push_started_this_step
        )
        env_ids = touchdown_mask.nonzero(as_tuple=False).flatten()
        if env_ids.numel() == 0:
            return

        self._recovery_touchdowns[env_ids] += 1
        sample_count = self._interval_sample_count[env_ids]
        has_complete_interval = self._interval_started_after_touchdown[env_ids] & (sample_count > 0)
        safe_count = torch.clamp(sample_count, min=1).to(torch.float32)
        mean_velocity_error = self._interval_velocity_error_sum[env_ids] / safe_count
        mean_abs_tilt = self._interval_abs_tilt_sum[env_ids] / safe_count.unsqueeze(-1)
        touchdown_foot = state.touchdown_foot[env_ids]
        previous_foot = self._last_touchdown_foot[env_ids]
        alternating = (previous_foot < 0) | (touchdown_foot != previous_foot)
        cfg = self.cfg.push_curriculum
        good_cycle = (
            has_complete_interval
            & alternating
            & (mean_velocity_error <= cfg.mean_velocity_error_threshold)
            & (mean_abs_tilt[:, 0] <= cfg.mean_abs_roll_threshold)
            & (mean_abs_tilt[:, 1] <= cfg.mean_abs_pitch_threshold)
        )
        newly_entered = good_cycle & ~self._practical_entered[env_ids]
        entered_ids = env_ids[newly_entered]
        self._practical_entered[entered_ids] = True
        self._practical_enter_step[entered_ids] = self._recovery_touchdowns[entered_ids]

        # Start the next complete-cycle interval at this touchdown before
        # completing any formal recovery episodes.
        self._last_touchdown_foot[env_ids] = touchdown_foot
        self._interval_started_after_touchdown[env_ids] = True
        self._interval_sample_count[env_ids] = 0
        self._interval_velocity_error_sum[env_ids] = 0.0
        self._interval_abs_tilt_sum[env_ids] = 0.0

        timeout_mask = (
            self.recovery_active[env_ids]
            & (self._recovery_touchdowns[env_ids] >= cfg.max_recovery_touchdowns)
            & ~self._practical_entered[env_ids]
        )
        timeout_ids = env_ids[timeout_mask]
        self._queue_touchdown_rewards(state, env_ids, entered_ids, timeout_ids)

        self._record_outcomes(
            entered_ids,
            CurriculumRecoveryOutcome.SUCCESS,
            self._practical_enter_step[entered_ids],
        )
        self._record_outcomes(timeout_ids, CurriculumRecoveryOutcome.TIMEOUT)

    def _apply_soft_reward_scaling(
        self,
        reward_buf: torch.Tensor,
        recovery_mask: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        if not self._stage2_reward_enabled or not self._soft_reward_scaling_enabled:
            return reward_buf
        self._last_soft_scaling_recovery_mask = recovery_mask.clone()
        progress = torch.clamp(
            self._recovery_touchdowns.to(torch.float32) / 5.0,
            min=0.0,
            max=1.0,
        )
        for term_name, (term_index, alpha_min) in self._soft_reward_term_indices.items():
            kappa = alpha_min + (1.0 - alpha_min) * progress
            kappa = torch.where(recovery_mask, kappa, torch.ones_like(kappa))
            original = self.reward_manager._step_reward[:, term_index] * float(self.step_dt)
            adjustment = (kappa - 1.0) * original
            reward_buf += adjustment
            # Preserve manager episode accounting when it has not already been
            # consumed by BaseEnv.reset for a terminal transition.
            continuing = ~dones
            self.reward_manager._episode_sums[term_name][continuing] += adjustment[continuing]
            self.reward_manager._step_reward[:, term_index] *= kappa
            self._last_soft_reward_multipliers[term_name] = kappa
        return reward_buf

    def _consume_event_rewards(self, reward_buf: torch.Tensor) -> dict[str, torch.Tensor]:
        event = {
            "recovery_touchdown_cost": self._event_touchdown_cost.clone(),
            "recovery_success": self._event_success.clone(),
            "recovery_timeout": self._event_timeout.clone(),
            "recovery_certificate": self._event_certificate.clone(),
        }
        event["recovery_shared_total"] = (
            event["recovery_touchdown_cost"]
            + event["recovery_success"]
            + event["recovery_timeout"]
        )
        event["recovery_total"] = event["recovery_shared_total"] + event["recovery_certificate"]
        reward_before = reward_buf.clone()
        reward_buf += event["recovery_total"]
        actual_delta = reward_buf - reward_before
        if not torch.allclose(actual_delta, event["recovery_total"], atol=1.0e-6, rtol=1.0e-6):
            raise RuntimeError("one-shot recovery reward changed while entering reward_buf")
        nonzero_ids = torch.nonzero(event["recovery_total"] != 0.0, as_tuple=False).flatten()
        if nonzero_ids.numel() > 0 and not self._event_dt_diagnostic_printed:
            env_id = int(nonzero_ids[0].item())
            print(
                "[Stage2Reward] direct one-shot injection (RewardManager dt bypass): "
                f"step_dt={self.step_dt}, designed_raw_event="
                f"{float(event['recovery_total'][env_id].item()):.6f}, "
                f"reward_buf_delta={float(actual_delta[env_id].item()):.6f}",
                flush=True,
            )
            self._event_dt_diagnostic_printed = True
        self._clear_event_buffers(torch.arange(self.num_envs, device=self.device))
        return event

    def _update_reward_logs(self, event: dict[str, torch.Tensor]) -> None:
        self._last_event_rewards = event
        if self._deferred_rollout_active:
            self.extras["recovery_episode_rewards"] = []
            return
        log = self.extras.setdefault("log", {})
        for name, values in event.items():
            log[f"Reward/{name}"] = float(values.mean().item())
        self._last_recovery_reward_mean = float(event["recovery_total"].mean().item())
        for term_name, multipliers in self._last_soft_reward_multipliers.items():
            log[f"Reward/soft_multiplier/{term_name}"] = float(multipliers.mean().item())
        if self._step_completed_reward_episodes:
            self.extras["recovery_episode_rewards"] = list(self._step_completed_reward_episodes)
            keys = (
                "episode_locomotion_reward_during_recovery",
                "episode_shared_recovery_reward",
                "episode_certificate_reward",
                "episode_total_recovery_reward",
                "recovery_to_locomotion_abs_ratio",
                "episode_recovery_event_reward_abs_sum",
                "episode_locomotion_reward_abs_sum_during_recovery",
                "absolute_recovery_to_locomotion_ratio",
            )
            for key in keys:
                log[f"Recovery/{key}"] = sum(
                    item[key] for item in self._step_completed_reward_episodes
                ) / len(self._step_completed_reward_episodes)
        else:
            self.extras["recovery_episode_rewards"] = []

    def _accumulate_practical_metrics(self, state) -> None:
        active = self.recovery_active
        if not torch.any(active):
            return
        velocity_error = torch.linalg.vector_norm(
            state.com_velocity[:, :2] - state.command_velocity[:, :2],
            dim=1,
        )
        self._interval_velocity_error_sum[active] += velocity_error[active]
        self._interval_abs_tilt_sum[active] += torch.abs(state.root_roll_pitch[active])
        self._interval_sample_count[active] += 1

    def _update_curriculum_logs(self) -> None:
        policy_step = self.sim_step_counter // self.cfg.sim.decimation
        learning_iteration = policy_step // self._steps_per_learning_iteration
        reason = self.push_curriculum.set_learning_iteration(learning_iteration)
        self.push_curriculum_level = self.push_curriculum.level
        self.push_curriculum_level_ratio = self.push_curriculum.level_ratio
        self.push_curriculum_max_xy = torch.tensor(
            self.push_curriculum.current_abs_delta_v_xy,
            dtype=torch.float32,
            device=self.device,
        )

        snapshot = self.push_curriculum.snapshot()
        current_stats = snapshot["current_level_statistics"]
        reason_code = {
            CurriculumUpgradeReason.NONE: 0.0,
            CurriculumUpgradeReason.PERFORMANCE: 1.0,
            CurriculumUpgradeReason.MAX_ITERATIONS: 2.0,
        }[reason]
        log = {
            "Curriculum/level": float(snapshot["curriculum_level"]),
            "Curriculum/level_ratio": snapshot["level_ratio"],
            "Curriculum/current_delta_v_max_x": snapshot["current_delta_v_max_x"],
            "Curriculum/current_delta_v_max_y": snapshot["current_delta_v_max_y"],
            "Curriculum/iterations_in_level": float(snapshot["iterations_in_current_level"]),
            "Curriculum/P5": snapshot["P5"] if snapshot["P5"] is not None else -1.0,
            "Curriculum/median_enter_step": (
                snapshot["median_enter_step"] if snapshot["median_enter_step"] is not None else -1.0
            ),
            "Curriculum/consecutive_pass_windows": float(snapshot["consecutive_pass_windows"]),
            "Curriculum/upgrade_reason_code": reason_code,
            "Curriculum/easy_sample_fraction": snapshot["easy_sample_fraction"],
            "Recovery/active_fraction": float(self.recovery_active.float().mean().item()),
            "Recovery/episodes_current_level": float(current_stats["recovery_episodes"]),
            "Recovery/success_current_level": float(current_stats["success"]),
            "Recovery/timeout_current_level": float(current_stats["timeout"]),
            "Recovery/fall_current_level": float(current_stats["fall"]),
            "Recovery/P5_current_level": current_stats["P5"],
            "Recovery/mean_enter_step_current_level": (
                current_stats["mean_practical_enter_step"]
                if current_stats["mean_practical_enter_step"] is not None
                else -1.0
            ),
            "Recovery/median_enter_step_current_level": (
                current_stats["median_practical_enter_step"]
                if current_stats["median_practical_enter_step"] is not None
                else -1.0
            ),
            "Recovery/truncated_by_env_horizon": float(self._truncated_recovery_count),
        }
        if not self._deferred_rollout_active:
            log["Recovery/reward"] = self._last_recovery_reward_mean
        if self._certificate_evaluator is not None:
            certificate_stats = self._certificate_evaluator.statistics
            for name, value in certificate_stats.items():
                log[f"Certificate/{name}"] = float(value)
        for index, level_stats in enumerate(snapshot["all_level_statistics"], start=1):
            prefix = f"RecoveryLevel{index}"
            log[f"{prefix}/episodes"] = float(level_stats["recovery_episodes"])
            log[f"{prefix}/SUCCESS"] = float(level_stats["success"])
            log[f"{prefix}/TIMEOUT"] = float(level_stats["timeout"])
            log[f"{prefix}/FALL"] = float(level_stats["fall"])
            log[f"{prefix}/P5"] = level_stats["P5"]
            log[f"{prefix}/mean_enter_step"] = (
                level_stats["mean_practical_enter_step"]
                if level_stats["mean_practical_enter_step"] is not None
                else -1.0
            )
            log[f"{prefix}/median_enter_step"] = (
                level_stats["median_practical_enter_step"]
                if level_stats["median_practical_enter_step"] is not None
                else -1.0
            )
        self.extras.setdefault("log", {}).update(log)

    def step(self, actions: torch.Tensor):
        if (
            self._deferred_rollout_active
            and self._deferred_rollout_step >= self._deferred_rollout_length
        ):
            raise RuntimeError("deferred rollout received more steps than configured")
        recovery_at_step_start = self.recovery_active.clone()
        self._step_completed_reward_episodes = []
        self._defer_recovery_reset_cleanup = True
        try:
            actor_obs, reward_buf, dones, extras = super().step(actions)
        finally:
            self._defer_recovery_reset_cleanup = False
        state = self._state_extractor.extract()
        reward_buf = self._apply_soft_reward_scaling(reward_buf, recovery_at_step_start, dones)
        if self._stage2_reward_enabled:
            recovery_locomotion_reward = reward_buf[recovery_at_step_start]
            self._episode_locomotion_reward[recovery_at_step_start] += recovery_locomotion_reward
            self._episode_locomotion_reward_abs_sum[recovery_at_step_start] += torch.abs(
                recovery_locomotion_reward
            )
        self._process_resets(dones)
        self._initialize_pending_certificates(state, dones)
        self._process_touchdowns(state, dones)
        self._accumulate_practical_metrics(state)
        event = self._consume_event_rewards(reward_buf) if self._stage2_reward_enabled else {
            name: torch.zeros(self.num_envs, device=self.device)
            for name in (
                "recovery_touchdown_cost",
                "recovery_success",
                "recovery_timeout",
                "recovery_certificate",
                "recovery_shared_total",
                "recovery_total",
            )
        }
        self._update_reward_logs(event)
        self._last_push_started_mask = self._push_started_this_step.clone()
        self._push_started_this_step[:] = False
        self._update_curriculum_logs()
        if self._deferred_rollout_active:
            self._deferred_rollout_step += 1
        return actor_obs, reward_buf, dones, extras
