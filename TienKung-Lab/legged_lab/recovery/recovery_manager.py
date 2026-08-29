"""Minimal touchdown-rate recovery state machine and transition logging.

This module is deliberately independent from rewards, observations, PPO, and
the certificate solver.  Callers provide already-computed ``N`` and ``margin``
plus independent practical-entered and practical-confirmed flags at touchdown.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import math


class RecoveryState(str, Enum):
    NORMAL = "NORMAL"
    RECOVERY = "RECOVERY"


class RecoveryExitReason(str, Enum):
    SUCCESS = "SUCCESS"
    TIMEOUT = "TIMEOUT"
    FALL = "FALL"


class RecoveryProgressKind(str, Enum):
    N_DECREASE = "N_DECREASE"
    SAME_N_MARGIN = "SAME_N_MARGIN"
    OVER_HORIZON_MARGIN = "OVER_HORIZON_MARGIN"
    N_INCREASE = "N_INCREASE"


@dataclass
class RecoveryTransition:
    touchdown_count: int
    n_previous: int
    margin_previous: float
    n_current: int
    margin_current: float
    delta_n: int
    delta_margin: float
    progress_kind: RecoveryProgressKind
    orbit_recovered: bool
    practical_entered: bool
    practical_confirmed: bool
    support_side: str | None = None
    support_alternating: bool | None = None
    policy_step: int | None = None
    time_after_push: float | None = None


@dataclass
class RecoveryEpisodeLog:
    push_delta_v_xy: tuple[float, float]
    push_magnitude: float
    push_direction_rad: float
    phase_at_push: float
    n0: int
    margin0: float
    recovery_enter_step: int | None
    transitions: list[RecoveryTransition] = field(default_factory=list)
    exit_reason: RecoveryExitReason | None = None
    recovery_success: bool = False
    touchdown_count: int = 0
    recovery_reward: float = 0.0
    orbit_recovered: bool = False
    practical_entered: bool = False
    practical_confirmed: bool = False
    orbit_recovery_step: int | None = None
    practical_enter_step: int | None = None
    practical_confirmed_step: int | None = None
    actual_recovery_enter_step: int | None = None
    actual_recovery_confirmed_step: int | None = None
    actual_recovery_time: float | None = None

    def to_dict(self) -> dict:
        """Return a serialization-friendly log without changing enum values."""
        result = asdict(self)
        if self.exit_reason is not None:
            result["exit_reason"] = self.exit_reason.value
        for item, transition in zip(result["transitions"], self.transitions):
            item["progress_kind"] = transition.progress_kind.value
        return result


@dataclass(frozen=True)
class RecoveryUpdate:
    transition: RecoveryTransition | None
    completed_episode: RecoveryEpisodeLog | None
    duplicate_touchdown: bool = False


class RecoveryManager:
    """Track one environment's NORMAL/RECOVERY state at touchdown rate."""

    def __init__(self, max_touchdowns: int = 5):
        if max_touchdowns <= 0:
            raise ValueError("max_touchdowns must be positive")
        self.max_touchdowns = int(max_touchdowns)
        self.state = RecoveryState.NORMAL
        self.current_episode: RecoveryEpisodeLog | None = None
        self.completed_episodes: list[RecoveryEpisodeLog] = []
        self.n_previous: int | None = None
        self.margin_previous: float | None = None
        self.touchdown_count = 0
        self._last_touchdown_token: object | None = None
        self._last_support_side: str | None = None

    @property
    def recovery_active(self) -> bool:
        return self.state is RecoveryState.RECOVERY

    @property
    def recovery_reward(self) -> float:
        """Recovery reward is intentionally frozen off in this stage."""
        return 0.0

    @staticmethod
    def _validated_certificate(n: int, margin: float) -> tuple[int, float]:
        if isinstance(n, bool) or int(n) != n or n < 0:
            raise ValueError(f"N must be a non-negative integer, got {n!r}")
        if not math.isfinite(float(margin)):
            raise ValueError(f"margin must be finite, got {margin!r}")
        return int(n), float(margin)

    def on_push(
        self,
        *,
        n: int,
        margin: float,
        delta_v_xy: tuple[float, float],
        phase_at_push: float,
        policy_step: int | None = None,
    ) -> RecoveryEpisodeLog:
        """Enter RECOVERY immediately after a known velocity perturbation."""
        if self.recovery_active:
            raise RuntimeError("Cannot start a new recovery episode while RECOVERY is active")
        n, margin = self._validated_certificate(n, margin)
        if not 0.0 <= phase_at_push < 1.0:
            raise ValueError("phase_at_push must be in [0, 1)")
        dx, dy = float(delta_v_xy[0]), float(delta_v_xy[1])
        episode = RecoveryEpisodeLog(
            push_delta_v_xy=(dx, dy),
            push_magnitude=math.hypot(dx, dy),
            push_direction_rad=math.atan2(dy, dx),
            phase_at_push=float(phase_at_push),
            n0=n,
            margin0=margin,
            recovery_enter_step=policy_step,
            orbit_recovered=n == 0,
            orbit_recovery_step=0 if n == 0 else None,
        )
        self.state = RecoveryState.RECOVERY
        self.current_episode = episode
        self.n_previous = n
        self.margin_previous = margin
        self.touchdown_count = 0
        self._last_touchdown_token = None
        self._last_support_side = None
        return episode

    def _finish(self, reason: RecoveryExitReason) -> RecoveryEpisodeLog:
        if not self.recovery_active or self.current_episode is None:
            raise RuntimeError("No active recovery episode to finish")
        episode = self.current_episode
        episode.exit_reason = reason
        episode.recovery_success = reason is RecoveryExitReason.SUCCESS
        episode.touchdown_count = self.touchdown_count
        self.completed_episodes.append(episode)
        self.state = RecoveryState.NORMAL
        self.current_episode = None
        self.n_previous = None
        self.margin_previous = None
        self.touchdown_count = 0
        self._last_touchdown_token = None
        self._last_support_side = None
        return episode

    def on_touchdown(
        self,
        *,
        n: int,
        margin: float,
        practical_entered: bool,
        practical_confirmed: bool,
        touchdown_token: object,
        support_side: str | None = None,
        policy_step: int | None = None,
        time_after_push: float | None = None,
    ) -> RecoveryUpdate:
        """Record one touchdown; entering practical gait controls SUCCESS."""
        if not self.recovery_active:
            return RecoveryUpdate(None, None)
        if touchdown_token == self._last_touchdown_token:
            return RecoveryUpdate(None, None, duplicate_touchdown=True)
        n, margin = self._validated_certificate(n, margin)
        if support_side not in (None, "left", "right"):
            raise ValueError("support_side must be 'left', 'right', or None")
        assert self.current_episode is not None
        assert self.n_previous is not None and self.margin_previous is not None

        self.touchdown_count += 1
        delta_n = n - self.n_previous
        delta_margin = margin - self.margin_previous
        orbit_recovered = n == 0
        practical_entered = bool(practical_entered)
        practical_confirmed = bool(practical_confirmed)
        if n < self.n_previous:
            progress_kind = RecoveryProgressKind.N_DECREASE
        elif n > self.n_previous:
            progress_kind = RecoveryProgressKind.N_INCREASE
        elif n > 5 and self.n_previous > 5:
            progress_kind = RecoveryProgressKind.OVER_HORIZON_MARGIN
        else:
            progress_kind = RecoveryProgressKind.SAME_N_MARGIN
        alternating = (
            None
            if support_side is None or self._last_support_side is None
            else support_side != self._last_support_side
        )
        transition = RecoveryTransition(
            touchdown_count=self.touchdown_count,
            n_previous=self.n_previous,
            margin_previous=self.margin_previous,
            n_current=n,
            margin_current=margin,
            delta_n=delta_n,
            delta_margin=delta_margin,
            progress_kind=progress_kind,
            orbit_recovered=orbit_recovered,
            practical_entered=practical_entered,
            practical_confirmed=practical_confirmed,
            support_side=support_side,
            support_alternating=alternating,
            policy_step=policy_step,
            time_after_push=time_after_push,
        )
        self.current_episode.transitions.append(transition)
        self.current_episode.touchdown_count = self.touchdown_count
        if orbit_recovered and not self.current_episode.orbit_recovered:
            self.current_episode.orbit_recovered = True
            self.current_episode.orbit_recovery_step = self.touchdown_count
        if practical_entered and not self.current_episode.practical_entered:
            self.current_episode.practical_entered = True
            self.current_episode.practical_enter_step = self.touchdown_count
        if practical_confirmed and not self.current_episode.practical_confirmed:
            self.current_episode.practical_confirmed = True
            self.current_episode.practical_confirmed_step = self.touchdown_count
        self.n_previous = n
        self.margin_previous = margin
        self._last_touchdown_token = touchdown_token
        if support_side is not None:
            self._last_support_side = support_side

        completed = None
        if practical_entered:
            completed = self._finish(RecoveryExitReason.SUCCESS)
        elif self.touchdown_count >= self.max_touchdowns:
            completed = self._finish(RecoveryExitReason.TIMEOUT)
        return RecoveryUpdate(transition, completed)

    def on_fall(self) -> RecoveryEpisodeLog | None:
        """Exit with FALL only when the environment's real termination says so."""
        if not self.recovery_active:
            return None
        return self._finish(RecoveryExitReason.FALL)

    def record_actual_recovery(
        self,
        *,
        enter_step: int | None,
        confirmed_step: int | None,
        recovery_time: float | None,
        practical_enter_step: int | None = None,
        episode: RecoveryEpisodeLog | None = None,
    ) -> None:
        """Attach detailed practical labels after its confirmed flag controls SUCCESS."""
        target = episode or self.current_episode
        if target is None and self.completed_episodes:
            target = self.completed_episodes[-1]
        if target is None:
            raise RuntimeError("No recovery episode is available for the actual recovery label")
        target.actual_recovery_enter_step = enter_step
        target.actual_recovery_confirmed_step = confirmed_step
        target.actual_recovery_time = recovery_time
        if practical_enter_step is not None:
            target.practical_entered = True
            target.practical_enter_step = practical_enter_step
        if confirmed_step is not None:
            target.practical_confirmed = True
            target.practical_confirmed_step = confirmed_step

    def reset(self) -> None:
        """Clear active state after an environment reset; preserve completed logs."""
        self.state = RecoveryState.NORMAL
        self.current_episode = None
        self.n_previous = None
        self.margin_previous = None
        self.touchdown_count = 0
        self._last_touchdown_token = None
        self._last_support_side = None
