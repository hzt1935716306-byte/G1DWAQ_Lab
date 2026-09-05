"""Lookup of policy-dependent nominal gait parameters on signed planes."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import yaml


DIRECTIONS = ("standing", "+x", "-x", "+y", "-y")
PRACTICAL_METRIC_LEGACY_V0 = "touchdown_instantaneous_legacy_v0"
PRACTICAL_METRIC_INTERVAL_MEAN_V1 = "interval_mean_v1"
PRACTICAL_METRIC_VERSIONS = (
    PRACTICAL_METRIC_LEGACY_V0,
    PRACTICAL_METRIC_INTERVAL_MEAN_V1,
)


@dataclass(frozen=True)
class PlaneNominalGait:
    alpha: float
    direction: str
    speed: float
    step_period: float
    h_eff: float
    step_width: float
    epsilon_b: tuple[float, float]
    epsilon_q: tuple[float, float]
    roll_star: float
    pitch_star: float
    mean_velocity_error_threshold: float
    mean_abs_roll_error_threshold: float
    mean_abs_pitch_error_threshold: float
    sample_count: int
    calibration_policy_id: str
    practical_metric_version: str = PRACTICAL_METRIC_LEGACY_V0

    @property
    def omega(self) -> float:
        return math.sqrt(9.81 / self.h_eff)


@dataclass(frozen=True)
class PlaneNominalLookup:
    value: PlaneNominalGait | None
    valid: bool
    reason: str = ""


def command_direction(
    vx_cmd: float,
    vy_cmd: float,
    yaw_cmd: float = 0.0,
    tolerance: float = 1.0e-6,
) -> tuple[str | None, float, str]:
    """Classify one heading-horizontal command without changing slope sign."""

    vx_cmd = float(vx_cmd)
    vy_cmd = float(vy_cmd)
    yaw_cmd = float(yaw_cmd)
    if not np.all(np.isfinite((vx_cmd, vy_cmd, yaw_cmd))):
        return None, 0.0, "command is not finite"
    if abs(yaw_cmd) > tolerance:
        return None, math.hypot(vx_cmd, vy_cmd), "yaw command is unsupported"
    x_active = abs(vx_cmd) > tolerance
    y_active = abs(vy_cmd) > tolerance
    if x_active == y_active:
        if x_active:
            return None, math.hypot(vx_cmd, vy_cmd), "diagonal command is unsupported"
        return "standing", 0.0, ""
    if x_active:
        return ("+x" if vx_cmd > 0.0 else "-x"), abs(vx_cmd), ""
    return ("+y" if vy_cmd > 0.0 else "-y"), abs(vy_cmd), ""


def _pair(value: object, name: str) -> tuple[float, float]:
    if isinstance(value, Mapping):
        result = (float(value["x"]), float(value["y"]))
    else:
        result = tuple(float(item) for item in value)  # type: ignore[arg-type]
    if len(result) != 2 or any(item <= 0.0 or not math.isfinite(item) for item in result):
        raise ValueError(f"{name} must contain two positive finite values")
    return result


def _node_from_mapping(node: Mapping[str, object]) -> PlaneNominalGait:
    direction = str(node["direction"])
    if direction not in DIRECTIONS:
        raise ValueError(f"unsupported nominal direction {direction!r}")
    value = PlaneNominalGait(
        alpha=math.radians(float(node["slope_degrees"])),
        direction=direction,
        speed=float(node["speed"]),
        step_period=float(node["T"]),
        h_eff=float(node["h_eff"]),
        step_width=float(node["w"]),
        epsilon_b=_pair(node["epsilon_b"], "epsilon_b"),
        epsilon_q=_pair(node["epsilon_q"], "epsilon_q"),
        roll_star=float(node["roll_star"]),
        pitch_star=float(node["pitch_star"]),
        mean_velocity_error_threshold=float(node["mean_velocity_error_threshold"]),
        mean_abs_roll_error_threshold=float(node["mean_abs_roll_error_threshold"]),
        mean_abs_pitch_error_threshold=float(node["mean_abs_pitch_error_threshold"]),
        sample_count=int(node["sample_count"]),
        calibration_policy_id=str(node["calibration_policy_id"]),
        practical_metric_version=str(
            node.get("practical_metric_version", PRACTICAL_METRIC_LEGACY_V0)
        ),
    )
    positive = (
        value.step_period,
        value.h_eff,
        value.step_width,
        value.mean_velocity_error_threshold,
        value.mean_abs_roll_error_threshold,
        value.mean_abs_pitch_error_threshold,
    )
    if any(item <= 0.0 or not math.isfinite(item) for item in positive):
        raise ValueError("nominal gait values and practical thresholds must be positive")
    if (
        not math.isfinite(value.speed)
        or value.speed < 0.0
        or (value.direction == "standing" and value.speed != 0.0)
    ):
        raise ValueError("standing nodes require speed 0 and all nominal speeds must be non-negative")
    if value.sample_count <= 0:
        raise ValueError("nominal sample_count must be positive")
    if value.practical_metric_version not in PRACTICAL_METRIC_VERSIONS:
        raise ValueError(
            "unsupported practical_metric_version "
            f"{value.practical_metric_version!r}"
        )
    return value


def _bracket(values: Iterable[float], query: float, tolerance: float) -> tuple[float, float] | None:
    ordered = sorted(set(float(value) for value in values))
    for value in ordered:
        if abs(query - value) <= tolerance:
            return value, value
    if not ordered or query < ordered[0] - tolerance or query > ordered[-1] + tolerance:
        return None
    lower = max(value for value in ordered if value < query)
    upper = min(value for value in ordered if value > query)
    return lower, upper


def _lerp(a: float, b: float, weight: float) -> float:
    return (1.0 - weight) * float(a) + weight * float(b)


def nominal_node_key(node: Mapping[str, object]) -> tuple[float, str, float]:
    """Return the unique stable key used by the nominal YAML collector."""

    return (
        float(node["slope_degrees"]),
        str(node["direction"]),
        float(node["speed"]),
    )


def upsert_nominal_nodes(
    existing_nodes: Sequence[Mapping[str, object]],
    new_nodes: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Preserve uncollected valid nodes and replace only successful matching keys."""

    by_key: dict[tuple[float, str, float], dict[str, object]] = {}
    for node in (*existing_nodes, *new_nodes):
        if not bool(node.get("valid", True)):
            continue
        by_key[nominal_node_key(node)] = dict(node)
    return [by_key[key] for key in sorted(by_key)]


class PlaneNominalParameterTable:
    """Bounded same-direction interpolation over slope and speed nodes."""

    def __init__(
        self,
        nodes: Iterable[PlaneNominalGait],
        tolerance: float = 1.0e-8,
        known_missing_slope_directions: Iterable[tuple[float, str]] = (),
    ):
        self.nodes = tuple(nodes)
        self.tolerance = float(tolerance)
        valid_slope_directions = {(node.alpha, node.direction) for node in self.nodes}
        self.known_missing_slope_directions = tuple(
            (float(alpha), direction)
            for alpha, direction in known_missing_slope_directions
            if not any(
                direction == valid_direction
                and abs(float(alpha) - valid_alpha) <= self.tolerance
                for valid_alpha, valid_direction in valid_slope_directions
            )
        )
        self._by_key: dict[tuple[str, float, float], PlaneNominalGait] = {}
        for node in self.nodes:
            key = (node.direction, node.alpha, node.speed)
            if key in self._by_key:
                raise ValueError(f"duplicate nominal calibration node {key}")
            self._by_key[key] = node

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PlaneNominalParameterTable":
        with Path(path).expanduser().resolve().open("r", encoding="utf-8") as stream:
            document = yaml.safe_load(stream)
        if int(document.get("schema_version", 0)) != 1:
            raise ValueError("unsupported plane nominal parameter schema")
        section = document.get("nominal_plane_gait", {})
        nodes = tuple(_node_from_mapping(node) for node in section.get("nodes", ()))
        # Collector reports retain explicit failed grid points.  If a sampled
        # slope/direction has no valid node at any speed, do not interpolate
        # across that known hole and silently resurrect it as valid.
        failed_pairs: set[tuple[float, str]] = set()
        reports = document.get("collection", {}).get("node_reports", {})
        for label, report in reports.items():
            if bool(report.get("valid", False)):
                continue
            try:
                fields = dict(item.split("=", 1) for item in str(label).split(","))
                pair = (math.radians(float(fields["alpha"])), fields["direction"])
            except (KeyError, TypeError, ValueError):
                continue
            failed_pairs.add(pair)
        valid_pairs = {(node.alpha, node.direction) for node in nodes}
        missing_pairs = {
            pair
            for pair in failed_pairs
            if not any(
                pair[1] == direction and abs(pair[0] - alpha) <= 1.0e-8
                for alpha, direction in valid_pairs
            )
        }
        return cls(nodes, known_missing_slope_directions=missing_pairs)

    def lookup_command(
        self,
        alpha: float,
        command: Iterable[float],
    ) -> PlaneNominalLookup:
        command = tuple(float(value) for value in command)
        if len(command) < 2:
            return PlaneNominalLookup(None, False, "command must contain vx and vy")
        yaw = command[2] if len(command) >= 3 else 0.0
        direction, speed, reason = command_direction(command[0], command[1], yaw)
        if direction is None:
            return PlaneNominalLookup(None, False, reason)
        return self.lookup(float(alpha), direction, speed)

    def lookup(self, alpha: float, direction: str, speed: float) -> PlaneNominalLookup:
        if direction not in DIRECTIONS:
            return PlaneNominalLookup(None, False, f"unsupported direction {direction!r}")
        candidates = [node for node in self.nodes if node.direction == direction]
        if not candidates:
            return PlaneNominalLookup(None, False, f"no calibration for direction {direction}")
        exact_slope_nodes = [
            node for node in candidates if abs(node.alpha - float(alpha)) <= self.tolerance
        ]
        known_missing_slope = any(
            missing_direction == direction
            and abs(missing_alpha - float(alpha)) <= self.tolerance
            for missing_alpha, missing_direction in self.known_missing_slope_directions
        )
        if known_missing_slope and not exact_slope_nodes:
            return PlaneNominalLookup(
                None,
                False,
                "nominal collection explicitly failed at this slope and direction",
            )
        alpha_bounds = _bracket((node.alpha for node in candidates), float(alpha), self.tolerance)
        if alpha_bounds is None:
            return PlaneNominalLookup(None, False, "slope is outside calibrated bounds")
        if alpha_bounds[0] != alpha_bounds[1] and any(
            missing_direction == direction
            and alpha_bounds[0] + self.tolerance
            < missing_alpha
            < alpha_bounds[1] - self.tolerance
            for missing_alpha, missing_direction in self.known_missing_slope_directions
        ):
            return PlaneNominalLookup(
                None,
                False,
                "interpolation would cross an explicitly failed calibration slope",
            )

        alpha_weight = 0.0 if alpha_bounds[0] == alpha_bounds[1] else (
            (alpha - alpha_bounds[0]) / (alpha_bounds[1] - alpha_bounds[0])
        )

        # Interpolate speed independently at each slope corner, then
        # interpolate those two values in slope.  This supports a bounded,
        # sparse calibration grid (for example the flat anchor may have
        # different speed nodes from the DWAQ slope bootstrap) without ever
        # borrowing a node from another direction or extrapolating.
        speed_corners: dict[float, tuple[PlaneNominalGait, PlaneNominalGait, float]] = {}
        for corner_alpha in set(alpha_bounds):
            at_slope = [
                node for node in candidates if abs(node.alpha - corner_alpha) <= self.tolerance
            ]
            speed_bounds = _bracket(
                (node.speed for node in at_slope), float(speed), self.tolerance
            )
            if speed_bounds is None:
                return PlaneNominalLookup(
                    None,
                    False,
                    "speed is outside calibrated bounds at a required slope corner",
                )
            low = self._by_key[(direction, corner_alpha, speed_bounds[0])]
            high = self._by_key[(direction, corner_alpha, speed_bounds[1])]
            weight = 0.0 if speed_bounds[0] == speed_bounds[1] else (
                (speed - speed_bounds[0]) / (speed_bounds[1] - speed_bounds[0])
            )
            speed_corners[corner_alpha] = (low, high, weight)

        def interpolate_scalar(attribute: str) -> float:
            along_alpha = []
            for corner_alpha in alpha_bounds:
                low_node, high_node, weight = speed_corners[corner_alpha]
                low = getattr(low_node, attribute)
                high = getattr(high_node, attribute)
                along_alpha.append(_lerp(low, high, weight))
            return _lerp(along_alpha[0], along_alpha[-1], alpha_weight)

        def interpolate_pair(attribute: str) -> tuple[float, float]:
            along_alpha = []
            for corner_alpha in alpha_bounds:
                low_node, high_node, weight = speed_corners[corner_alpha]
                along_alpha.append(
                    tuple(
                        _lerp(
                            getattr(low_node, attribute)[axis],
                            getattr(high_node, attribute)[axis],
                            weight,
                        )
                        for axis in range(2)
                    )
                )
            return tuple(
                _lerp(along_alpha[0][axis], along_alpha[-1][axis], alpha_weight)
                for axis in range(2)
            )

        source_nodes = {
            node
            for low, high, _ in speed_corners.values()
            for node in (low, high)
        }
        practical_metric_versions = {
            node.practical_metric_version for node in source_nodes
        }
        if len(practical_metric_versions) != 1:
            return PlaneNominalLookup(
                None,
                False,
                "cannot interpolate nominal nodes with different practical metric semantics",
            )
        practical_metric_version = next(iter(practical_metric_versions))
        policy_ids = sorted({node.calibration_policy_id for node in source_nodes})
        # The immutable flat anchor and slope bootstrap nodes can come from
        # different policies.  Bounded geometric interpolation is still
        # permitted by the runtime contract; keep every source identity in
        # the returned value so the mixed provenance is never hidden.
        policy_id = (
            policy_ids[0]
            if len(policy_ids) == 1
            else "interpolated[" + "|".join(policy_ids) + "]"
        )
        value = PlaneNominalGait(
            alpha=float(alpha),
            direction=direction,
            speed=float(speed),
            step_period=interpolate_scalar("step_period"),
            h_eff=interpolate_scalar("h_eff"),
            step_width=interpolate_scalar("step_width"),
            epsilon_b=interpolate_pair("epsilon_b"),
            epsilon_q=interpolate_pair("epsilon_q"),
            roll_star=interpolate_scalar("roll_star"),
            pitch_star=interpolate_scalar("pitch_star"),
            mean_velocity_error_threshold=interpolate_scalar("mean_velocity_error_threshold"),
            mean_abs_roll_error_threshold=interpolate_scalar("mean_abs_roll_error_threshold"),
            mean_abs_pitch_error_threshold=interpolate_scalar("mean_abs_pitch_error_threshold"),
            sample_count=min(node.sample_count for node in source_nodes),
            calibration_policy_id=policy_id,
            practical_metric_version=practical_metric_version,
        )
        return PlaneNominalLookup(value, True)


__all__ = [
    "DIRECTIONS",
    "PRACTICAL_METRIC_INTERVAL_MEAN_V1",
    "PRACTICAL_METRIC_LEGACY_V0",
    "PRACTICAL_METRIC_VERSIONS",
    "PlaneNominalGait",
    "PlaneNominalLookup",
    "PlaneNominalParameterTable",
    "command_direction",
    "nominal_node_key",
    "upsert_nominal_nodes",
]
