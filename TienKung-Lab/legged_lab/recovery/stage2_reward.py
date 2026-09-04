"""Pure Stage2 recovery reward bookkeeping.

This module deliberately has no Isaac Lab dependency.  The same implementation
is used by the online recovery environment and by saved-trajectory replay.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


MAX_RECOVERY_TOUCHDOWNS = 5
DEFAULT_EVENT_SCALE = 0.50
TOUCHDOWN_COST = -0.10
SUCCESS_MAX = 3.0
TIMEOUT_PENALTY = -3.0
CERTIFICATE_PROGRESS_SCALE = 0.50
CERTIFICATE_MARGIN_WEIGHT = 0.25


@dataclass(frozen=True)
class PlaneV1RewardParameters:
    """Final Plane V1 touchdown-event reward parameters."""

    certificate_progress_weight: float = 0.25
    certificate_delta_phi_clip: float = 2.0
    unrecovered_touchdown_cost: float = -0.05
    td5_unrecovered_penalty: float = -0.25
    certificate_target_n_max: int = 1
    certificate_horizon_touchdowns: int = 5

    def __post_init__(self) -> None:
        if self.certificate_progress_weight < 0.0:
            raise ValueError("certificate_progress_weight must be non-negative")
        if self.certificate_delta_phi_clip <= 0.0:
            raise ValueError("certificate_delta_phi_clip must be positive")
        if self.unrecovered_touchdown_cost > 0.0 or self.td5_unrecovered_penalty > 0.0:
            raise ValueError("Plane V1 costs and penalties must be non-positive")
        if self.certificate_target_n_max != 1:
            raise ValueError("Plane V1 recovery target is fixed at N <= 1")
        if self.certificate_horizon_touchdowns != 5:
            raise ValueError("Plane V1 recovery horizon is fixed at TD5")


@dataclass(frozen=True)
class PlaneV1RewardResult:
    """One final Plane V1 event and its potential-state transition."""

    progress: float = 0.0
    step_cost: float = 0.0
    td5_penalty: float = 0.0
    phi_current: float | None = None
    update_previous_phi: bool = False

    @property
    def total(self) -> float:
        return self.progress + self.step_cost + self.td5_penalty


def plane_v1_touchdown_reward(
    previous_phi: float | None,
    n_min: int,
    margin: float,
    *,
    touchdown_index: int,
    terrain_plane_valid: bool,
    solver_valid: bool,
    enabled: bool,
    parameters: PlaneV1RewardParameters | None = None,
) -> PlaneV1RewardResult:
    """Compute the final Plane V1 reward for TD0--TD5.

    TD0 initializes the potential without reward.  An invalid terrain geometry
    is treated as uncertified recovery, while a solver/transport failure on an
    otherwise valid geometry produces no reward and leaves the previous
    potential untouched.
    """

    cfg = parameters or PlaneV1RewardParameters()
    if not 0 <= touchdown_index <= cfg.certificate_horizon_touchdowns:
        raise ValueError("touchdown_index must be in [0, 5]")

    if not terrain_plane_valid:
        if touchdown_index == 0 or not enabled:
            return PlaneV1RewardResult()
        return PlaneV1RewardResult(
            step_cost=cfg.unrecovered_touchdown_cost,
            td5_penalty=(
                cfg.td5_unrecovered_penalty
                if touchdown_index == cfg.certificate_horizon_touchdowns
                else 0.0
            ),
        )

    # A numerical/transport failure must never become a policy penalty and
    # must not corrupt the last successfully resolved potential.
    if not solver_valid:
        return PlaneV1RewardResult()

    phi_current = certificate_potential(n_min, margin)
    if touchdown_index == 0:
        return PlaneV1RewardResult(
            phi_current=phi_current,
            update_previous_phi=True,
        )

    if not enabled:
        return PlaneV1RewardResult(
            phi_current=phi_current,
            update_previous_phi=True,
        )

    delta_phi = 0.0 if previous_phi is None else phi_current - previous_phi
    delta_phi = max(-cfg.certificate_delta_phi_clip, min(cfg.certificate_delta_phi_clip, delta_phi))
    unrecovered = n_min > cfg.certificate_target_n_max
    return PlaneV1RewardResult(
        progress=cfg.certificate_progress_weight * delta_phi,
        step_cost=cfg.unrecovered_touchdown_cost if unrecovered else 0.0,
        td5_penalty=(
            cfg.td5_unrecovered_penalty
            if touchdown_index == cfg.certificate_horizon_touchdowns and unrecovered
            else 0.0
        ),
        phi_current=phi_current,
        update_previous_phi=True,
    )


class PlaneV1RecoveryRewardChannel:
    """Reference TD0--TD5 state machine used by unit tests and diagnostics."""

    def __init__(
        self,
        *,
        enabled: bool,
        parameters: PlaneV1RewardParameters | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.parameters = parameters or PlaneV1RewardParameters()
        self.active = False
        self.touchdown_index = -1
        self.previous_phi: float | None = None

    def on_push(self) -> None:
        if self.active:
            raise RuntimeError("cannot push an active Plane V1 reward episode")
        self.active = True
        self.touchdown_index = -1
        self.previous_phi = None

    def on_touchdown(
        self,
        n_min: int,
        margin: float,
        *,
        terrain_plane_valid: bool = True,
        solver_valid: bool = True,
        practical_entered: bool = False,
    ) -> PlaneV1RewardResult:
        if not self.active:
            return PlaneV1RewardResult()
        # Practical recovery is diagnostic only and deliberately cannot close
        # the formal TD0--TD5 reward episode.
        del practical_entered
        self.touchdown_index += 1
        result = plane_v1_touchdown_reward(
            self.previous_phi,
            n_min,
            margin,
            touchdown_index=self.touchdown_index,
            terrain_plane_valid=terrain_plane_valid,
            solver_valid=solver_valid,
            enabled=self.enabled,
            parameters=self.parameters,
        )
        if result.update_previous_phi:
            self.previous_phi = result.phi_current
        if self.touchdown_index == self.parameters.certificate_horizon_touchdowns:
            self.active = False
        return result

    def on_fall(self) -> None:
        self.active = False
        self.touchdown_index = -1
        self.previous_phi = None


@dataclass(frozen=True)
class RecoveryEventReward:
    """One policy-step event reward, consumed exactly once by the caller."""

    touchdown_cost: float = 0.0
    success: float = 0.0
    timeout: float = 0.0
    certificate: float = 0.0

    @property
    def shared_total(self) -> float:
        return self.touchdown_cost + self.success + self.timeout

    @property
    def total(self) -> float:
        return self.shared_total + self.certificate


def certificate_level(n_min: int) -> int:
    """Return the requested hierarchy level, intentionally with L(0) == L(1)."""

    if isinstance(n_min, bool) or int(n_min) != n_min or n_min < 0:
        raise ValueError(f"N_min must be a non-negative integer, got {n_min!r}")
    n_min = int(n_min)
    if n_min > 5:
        return 0
    if n_min == 5:
        return 1
    if n_min == 4:
        return 2
    if n_min == 3:
        return 3
    if n_min == 2:
        return 4
    return 5


def normalized_certificate_margin(n_min: int, margin: float) -> float:
    """Normalize the existing signed certificate margin without changing it."""

    certificate_level(n_min)  # validates N_min
    margin = float(margin)
    if not math.isfinite(margin):
        raise ValueError(f"margin must be finite, got {margin!r}")
    if n_min > 5:
        return max(-1.0, min(0.0, margin / 2.0))
    if 2 <= n_min <= 5:
        return max(0.0, min(1.0, margin / 0.95))
    return 0.0


def certificate_potential(n_min: int, margin: float) -> float:
    """Phi(N, margin) = L(N) + 0.25 * mu(N, margin)."""

    return float(certificate_level(n_min)) + CERTIFICATE_MARGIN_WEIGHT * normalized_certificate_margin(
        n_min, margin
    )


def certificate_potential_tensor(n_min: torch.Tensor, margin: torch.Tensor) -> torch.Tensor:
    """Vectorized form of :func:`certificate_potential` for online environments."""

    if n_min.shape != margin.shape:
        raise ValueError("N_min and margin tensors must have the same shape")
    if torch.any(n_min < 0) or not torch.all(torch.isfinite(margin)):
        raise ValueError("N_min must be non-negative and margin must be finite")
    level = torch.where(
        n_min > 5,
        torch.zeros_like(margin),
        torch.where(n_min <= 1, torch.full_like(margin, 5.0), (6 - n_min).to(margin.dtype)),
    )
    mu_over_horizon = torch.clamp(margin / 2.0, min=-1.0, max=0.0)
    mu_finite = torch.clamp(margin / 0.95, min=0.0, max=1.0)
    mu = torch.where(
        n_min > 5,
        mu_over_horizon,
        torch.where((n_min >= 2) & (n_min <= 5), mu_finite, torch.zeros_like(margin)),
    )
    return level + CERTIFICATE_MARGIN_WEIGHT * mu


class Stage2RecoveryRewardChannel:
    """Single-environment one-shot reward channel used by offline replay.

    The online environment uses tensor buffers for throughput, but follows the
    same formulas and consumption rule implemented here.
    """

    def __init__(
        self,
        *,
        enable_certificate_reward: bool,
        enable_shared_event_reward: bool = False,
        event_scale: float = DEFAULT_EVENT_SCALE,
        max_touchdowns: int = MAX_RECOVERY_TOUCHDOWNS,
    ) -> None:
        if not math.isfinite(event_scale) or event_scale <= 0.0:
            raise ValueError("event_scale must be finite and positive")
        if max_touchdowns != MAX_RECOVERY_TOUCHDOWNS:
            raise ValueError("the Stage2 reward definition requires exactly five touchdowns")
        self.enable_certificate_reward = bool(enable_certificate_reward)
        self.enable_shared_event_reward = bool(enable_shared_event_reward)
        self.event_scale = float(event_scale)
        self.max_touchdowns = int(max_touchdowns)
        self.active = False
        self.touchdown_count = 0
        self.phi_previous: float | None = None
        self._pending = RecoveryEventReward()

    def on_push(self, n_min: int, margin: float) -> None:
        """Enter RECOVERY and initialize Phi without producing reward."""

        if self.active:
            raise RuntimeError("cannot push an already active recovery channel")
        self.active = True
        self.touchdown_count = 0
        self.phi_previous = certificate_potential(n_min, margin)
        self._pending = RecoveryEventReward()

    def on_touchdown(
        self,
        n_min: int,
        margin: float,
        *,
        practical_entered: bool,
    ) -> str | None:
        """Queue exactly one touchdown event and return its terminal outcome."""

        if not self.active:
            return None
        if self._pending != RecoveryEventReward():
            raise RuntimeError("the previous recovery event must be consumed before queuing another")
        if self.phi_previous is None:
            raise RuntimeError("certificate potential was not initialized at push")

        self.touchdown_count += 1
        phi_current = certificate_potential(n_min, margin)
        certificate = 0.0
        if self.enable_certificate_reward:
            certificate = CERTIFICATE_PROGRESS_SCALE * (phi_current - self.phi_previous)
        self.phi_previous = phi_current

        success = 0.0
        timeout = 0.0
        outcome = None
        if practical_entered:
            success = SUCCESS_MAX * (6 - self.touchdown_count) / MAX_RECOVERY_TOUCHDOWNS
            outcome = "SUCCESS"
        elif self.touchdown_count >= self.max_touchdowns:
            timeout = TIMEOUT_PENALTY
            outcome = "TIMEOUT"

        shared_scale = self.event_scale if self.enable_shared_event_reward else 0.0
        self._pending = RecoveryEventReward(
            touchdown_cost=shared_scale * TOUCHDOWN_COST,
            success=shared_scale * success,
            timeout=shared_scale * timeout,
            certificate=self.event_scale * certificate,
        )
        if outcome is not None:
            self.active = False
        return outcome

    def consume(self) -> RecoveryEventReward:
        """Return the queued event once and immediately clear it."""

        event = self._pending
        self._pending = RecoveryEventReward()
        return event

    def on_fall(self) -> None:
        """Close recovery without adding a second fall penalty."""

        self.active = False
        self.touchdown_count = 0
        self.phi_previous = None
        self._pending = RecoveryEventReward()

    def reset(self) -> None:
        """Clear state and every one-shot reward buffer."""

        self.on_fall()
