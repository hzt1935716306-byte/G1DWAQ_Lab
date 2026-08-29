"""Pure Stage2 recovery reward bookkeeping.

This module deliberately has no Isaac Lab dependency.  The same implementation
is used by the online recovery environment and by saved-trajectory replay.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


MAX_RECOVERY_TOUCHDOWNS = 5
DEFAULT_EVENT_SCALE = 0.2
TOUCHDOWN_COST = -0.10
SUCCESS_MAX = 3.0
TIMEOUT_PENALTY = -3.0
CERTIFICATE_PROGRESS_SCALE = 0.50
CERTIFICATE_MARGIN_WEIGHT = 0.25


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
        event_scale: float = DEFAULT_EVENT_SCALE,
        max_touchdowns: int = MAX_RECOVERY_TOUCHDOWNS,
    ) -> None:
        if not math.isfinite(event_scale) or event_scale <= 0.0:
            raise ValueError("event_scale must be finite and positive")
        if max_touchdowns != MAX_RECOVERY_TOUCHDOWNS:
            raise ValueError("the Stage2 reward definition requires exactly five touchdowns")
        self.enable_certificate_reward = bool(enable_certificate_reward)
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

        self._pending = RecoveryEventReward(
            touchdown_cost=self.event_scale * TOUCHDOWN_COST,
            success=self.event_scale * success,
            timeout=self.event_scale * timeout,
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
