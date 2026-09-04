"""Standalone whole-body CoM horizontal velocity estimator primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn


@dataclass
class ComVelocityEstimatorTrainCfg:
    """Compact configuration for online supervised estimator training."""

    teacher_task: str = "g1_slope_sys_d"
    estimator_task: str = "g1_com_velocity_estimator"
    teacher_checkpoint: str = "logs/g1_slope_sys_d.pt"
    num_envs: int = 4096
    train_envs: int = 3584
    validation_envs: int = 512
    train_policy_steps: int = 5000
    evaluation_policy_steps: int = 1000
    learning_rate: float = 1.0e-3
    transient_delta_v_threshold: float = 0.15
    transient_weight: float = 4.0
    teacher_history_length: int = 10
    estimator_history_length: int = 5
    per_frame_obs_dim: int = 96
    hidden_dims: list[int] = field(default_factory=lambda: [256, 128, 64])
    output_dim: int = 2
    reset_warmup_policy_steps: int = 5
    validation_interval_steps: int = 100
    log_interval_steps: int = 100
    seed: int = 42

    @property
    def input_dim(self) -> int:
        return self.estimator_history_length * self.per_frame_obs_dim

    def validate(self) -> None:
        if self.num_envs != self.train_envs + self.validation_envs:
            raise ValueError("num_envs must equal train_envs + validation_envs")
        if self.train_envs <= 0 or self.validation_envs <= 0:
            raise ValueError("both train and validation partitions must be non-empty")
        if not 0 < self.estimator_history_length <= self.teacher_history_length:
            raise ValueError("estimator history must be within the teacher history")
        if self.per_frame_obs_dim != 96:
            raise ValueError("the frozen G1 teacher requires a 96-D per-frame observation")
        if self.output_dim != 2:
            raise ValueError("the estimator output must be horizontal velocity [vx, vy]")
        if self.train_policy_steps <= 0 or self.evaluation_policy_steps <= 0:
            raise ValueError("training and evaluation step counts must be positive")
        if self.learning_rate <= 0.0 or self.transient_weight < 1.0:
            raise ValueError("invalid optimizer or transient weighting configuration")


@dataclass
class ComVelocityEstimatorV2TrainCfg(ComVelocityEstimatorTrainCfg):
    """V2 configuration with deployable IMU input and recovery-heavy sampling."""

    estimator_task: str = "g1_com_velocity_estimator_v2"
    imu_input_dim: int = 3
    imu_acceleration_scale: float = 0.05
    recovery_group_fraction: float = 0.5
    recovery_push_interval_s: tuple[float, float] = (0.6, 0.9)
    recovery_window_s: float = 0.5
    policy_dt: float = 0.02
    transient_weight: float = 1.0

    @property
    def estimator_frame_dim(self) -> int:
        return self.per_frame_obs_dim + self.imu_input_dim

    @property
    def input_dim(self) -> int:
        return self.estimator_history_length * self.estimator_frame_dim

    @property
    def recovery_push_interval_steps(self) -> tuple[int, int]:
        return tuple(max(1, round(value / self.policy_dt)) for value in self.recovery_push_interval_s)

    @property
    def recovery_window_steps(self) -> int:
        return max(1, round(self.recovery_window_s / self.policy_dt))

    def validate(self) -> None:
        super().validate()
        if self.imu_input_dim != 3 or self.estimator_frame_dim != 99 or self.input_dim != 495:
            raise ValueError("V2 estimator contract must be 5 x (96 + 3) = 495")
        if self.imu_acceleration_scale <= 0.0:
            raise ValueError("IMU acceleration scale must be positive")
        if not 0.0 < self.recovery_group_fraction < 1.0:
            raise ValueError("recovery group fraction must lie strictly between zero and one")
        lower, upper = self.recovery_push_interval_s
        if lower <= 0.0 or upper < lower or self.recovery_window_s <= 0.0:
            raise ValueError("invalid recovery-heavy timing configuration")


class ComVelocityEstimator(nn.Module):
    """MLP mapping five proprioceptive observation frames to heading-frame CoM velocity."""

    def __init__(
        self,
        input_dim: int = 480,
        hidden_dims: tuple[int, ...] | list[int] = (256, 128, 64),
        output_dim: int = 2,
    ) -> None:
        super().__init__()
        dimensions = [input_dim, *hidden_dims, output_dim]
        layers: list[nn.Module] = []
        for index, (in_features, out_features) in enumerate(zip(dimensions[:-1], dimensions[1:])):
            layers.append(nn.Linear(in_features, out_features))
            if index < len(dimensions) - 2:
                layers.append(nn.ELU())
        self.network = nn.Sequential(*layers)
        self.input_dim = int(input_dim)
        self.hidden_dims = tuple(int(value) for value in hidden_dims)
        self.output_dim = int(output_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.ndim != 2 or observations.shape[1] != self.input_dim:
            raise ValueError(
                f"expected estimator observations [N, {self.input_dim}], got {tuple(observations.shape)}"
            )
        return self.network(observations)


def extract_recent_actor_history(
    actor_observations: torch.Tensor,
    *,
    teacher_history_length: int = 10,
    estimator_history_length: int = 5,
    per_frame_obs_dim: int = 96,
) -> torch.Tensor:
    """Return newest frames from BaseEnv's oldest-to-newest flattened history."""

    expected_dim = teacher_history_length * per_frame_obs_dim
    if actor_observations.ndim != 2 or actor_observations.shape[1] != expected_dim:
        raise ValueError(
            f"expected actor history [N, {expected_dim}], got {tuple(actor_observations.shape)}"
        )
    if not 0 < estimator_history_length <= teacher_history_length:
        raise ValueError("estimator_history_length is outside the teacher history")
    history = actor_observations.reshape(
        actor_observations.shape[0], teacher_history_length, per_frame_obs_dim
    )
    return history[:, -estimator_history_length:, :].reshape(actor_observations.shape[0], -1)


class EstimatorFrameHistory:
    """Five-frame history combining noisy/scaled Actor frames with deployable IMU data."""

    def __init__(
        self,
        num_envs: int,
        *,
        history_length: int = 5,
        actor_frame_dim: int = 96,
        imu_dim: int = 3,
        imu_acceleration_scale: float = 0.05,
        device: str | torch.device,
    ) -> None:
        self.history_length = int(history_length)
        self.actor_frame_dim = int(actor_frame_dim)
        self.imu_dim = int(imu_dim)
        self.imu_acceleration_scale = float(imu_acceleration_scale)
        self.frame_dim = self.actor_frame_dim + self.imu_dim
        self.buffer = torch.zeros(
            num_envs,
            self.history_length,
            self.frame_dim,
            dtype=torch.float32,
            device=device,
        )

    def append(
        self,
        actor_frame: torch.Tensor,
        imu_linear_acceleration: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        if actor_frame.shape != (self.buffer.shape[0], self.actor_frame_dim):
            raise ValueError("actor frame shape does not match V2 history")
        if imu_linear_acceleration.shape != (self.buffer.shape[0], self.imu_dim):
            raise ValueError("IMU acceleration shape does not match V2 history")
        dones = dones.to(device=self.buffer.device, dtype=torch.bool)
        if dones.shape != self.buffer.shape[:1]:
            raise ValueError("dones shape does not match V2 history")
        self.buffer = torch.roll(self.buffer, shifts=-1, dims=1)
        self.buffer[dones] = 0.0
        frame = torch.cat(
            (actor_frame, imu_linear_acceleration * self.imu_acceleration_scale), dim=1
        )
        self.buffer[:, -1] = frame
        return self.buffer.reshape(self.buffer.shape[0], -1)


def latest_actor_frame(
    actor_observations: torch.Tensor,
    *,
    history_length: int = 10,
    per_frame_obs_dim: int = 96,
) -> torch.Tensor:
    """Return the newest noisy/scaled frame from the Actor history."""

    expected = history_length * per_frame_obs_dim
    if actor_observations.ndim != 2 or actor_observations.shape[1] != expected:
        raise ValueError(f"expected actor observation dimension {expected}")
    return actor_observations.reshape(-1, history_length, per_frame_obs_dim)[:, -1]


def partitioned_recovery_group_mask(
    num_envs: int,
    train_envs: int,
    fraction: float,
    device: str | torch.device,
) -> torch.Tensor:
    """Select the same recovery-heavy fraction inside train and validation partitions."""

    if not 0 < train_envs < num_envs or not 0.0 < fraction < 1.0:
        raise ValueError("invalid train/validation partition or recovery fraction")
    result = torch.zeros(num_envs, dtype=torch.bool, device=device)
    for start, stop in ((0, train_envs), (train_envs, num_envs)):
        count = max(1, round((stop - start) * fraction))
        result[stop - count : stop] = True
    return result


class TouchdownAfterTransientTracker:
    """Identify TD0 and TD1 following a GT horizontal-velocity transient."""

    def __init__(self, num_envs: int, device: str | torch.device) -> None:
        self._next_touchdown = torch.full((num_envs,), -1, dtype=torch.long, device=device)

    def update(
        self,
        transient: torch.Tensor,
        touchdown: torch.Tensor,
        dones: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        transient = transient.to(dtype=torch.bool, device=self._next_touchdown.device)
        touchdown = touchdown.to(dtype=torch.bool, device=self._next_touchdown.device)
        dones = dones.to(dtype=torch.bool, device=self._next_touchdown.device)
        self._next_touchdown[dones] = -1
        self._next_touchdown[transient & ~dones] = 0
        td0 = touchdown & (self._next_touchdown == 0) & ~dones
        self._next_touchdown[td0] = 1
        td1 = touchdown & (self._next_touchdown == 1) & ~td0 & ~dones
        self._next_touchdown[td1] = -1
        return td0, td1


class ResetWarmupMask:
    """Exclude the first complete policy transitions after every reset."""

    def __init__(self, num_envs: int, warmup_steps: int, device: str | torch.device) -> None:
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        self.warmup_steps = int(warmup_steps)
        self.remaining = torch.full(
            (num_envs,), self.warmup_steps, dtype=torch.long, device=device
        )

    def eligible_after_step(self, dones: torch.Tensor) -> torch.Tensor:
        """Mask the current sample, then advance/reset per-environment warm-up state."""

        dones = dones.to(device=self.remaining.device, dtype=torch.bool)
        if dones.shape != self.remaining.shape:
            raise ValueError("dones shape does not match warm-up state")
        eligible = (~dones) & (self.remaining == 0)
        active = (~dones) & (self.remaining > 0)
        self.remaining[active] -= 1
        self.remaining[dones] = self.warmup_steps
        return eligible


def extract_com_velocity_target(state: Any) -> torch.Tensor:
    """Select whole-body mass-weighted CoM XY velocity already expressed in heading frame."""

    velocity = state.com_velocity
    if velocity.ndim != 2 or velocity.shape[1] < 2:
        raise ValueError("state.com_velocity must have shape [N, >=2]")
    return velocity[:, :2]


def weighted_velocity_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    transient: torch.Tensor,
    transient_weight: float,
) -> torch.Tensor:
    """Mean per-sample XY MSE with additional transient-state weight."""

    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[1] != 2:
        raise ValueError("prediction and target must both have shape [N, 2]")
    if transient.shape != prediction.shape[:1]:
        raise ValueError("transient mask must have shape [N]")
    sample_mse = torch.mean(torch.square(prediction - target), dim=1)
    weights = torch.where(
        transient,
        torch.as_tensor(transient_weight, dtype=sample_mse.dtype, device=sample_mse.device),
        torch.ones((), dtype=sample_mse.dtype, device=sample_mse.device),
    )
    return torch.sum(sample_mse * weights) / torch.clamp(torch.sum(weights), min=1.0)


class ErrorMetricAccumulator:
    """Accumulate exact validation errors for a compact final metric report."""

    def __init__(self) -> None:
        self._errors: list[torch.Tensor] = []

    def add(self, prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> None:
        selected = (prediction - target)[mask]
        if selected.numel() > 0:
            self._errors.append(selected.detach().to(device="cpu", dtype=torch.float32))

    @property
    def count(self) -> int:
        return sum(chunk.shape[0] for chunk in self._errors)

    def summary(self) -> dict[str, float | int | None]:
        if not self._errors:
            return {
                "count": 0,
                "mae_x": None,
                "mae_y": None,
                "rmse_x": None,
                "rmse_y": None,
                "vector_rmse": None,
                "bias_x": None,
                "bias_y": None,
                "p95_vector_error": None,
            }
        error = torch.cat(self._errors, dim=0)
        absolute = torch.abs(error)
        squared = torch.square(error)
        vector_error = torch.linalg.vector_norm(error, dim=1)
        return {
            "count": int(error.shape[0]),
            "mae_x": float(absolute[:, 0].mean()),
            "mae_y": float(absolute[:, 1].mean()),
            "rmse_x": float(torch.sqrt(squared[:, 0].mean())),
            "rmse_y": float(torch.sqrt(squared[:, 1].mean())),
            "vector_rmse": float(torch.sqrt(squared.sum(dim=1).mean())),
            "bias_x": float(error[:, 0].mean()),
            "bias_y": float(error[:, 1].mean()),
            "p95_vector_error": float(torch.quantile(vector_error, 0.95)),
        }
