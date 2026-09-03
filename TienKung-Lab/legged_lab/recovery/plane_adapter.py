"""Geometry-only adapter from calibrated flat capability to a signed plane.

Certificate states remain in the yaw-only heading-horizontal frame.  This
module projects only the flat capability sets C/L and the flat swing-speed
limit; it must never be applied to measured b, q, landing, or CoP states.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class PlaneGeometry:
    alpha: float
    valid: bool
    reason: str = ""


@dataclass(frozen=True)
class Box2D:
    x: tuple[float, float]
    y: tuple[float, float]

    def __post_init__(self) -> None:
        if not self.x[0] < self.x[1] or not self.y[0] < self.y[1]:
            raise ValueError("box lower bounds must be strictly below upper bounds")

    def contains(self, point: Sequence[float], tolerance: float = 0.0) -> bool:
        x, y = (float(value) for value in point)
        return (
            self.x[0] - tolerance <= x <= self.x[1] + tolerance
            and self.y[0] - tolerance <= y <= self.y[1] + tolerance
        )


@dataclass(frozen=True)
class PlaneCapability:
    cop_left: Box2D
    cop_right: Box2D
    landing_left: Box2D
    landing_right: Box2D
    swing_velocity_limits: tuple[float, float]
    nominal_cop_valid: bool


def signed_slope_from_heading_normal(
    normal_heading: Sequence[float],
    slope_alignment_tolerance: float,
) -> PlaneGeometry:
    """Return ``alpha=atan2(-n_x,n_z)`` for a valid x-only heading plane."""

    normal = np.asarray(normal_heading, dtype=np.float64)
    if normal.shape != (3,) or not np.all(np.isfinite(normal)):
        return PlaneGeometry(0.0, False, "terrain normal is not a finite 3-vector")
    norm = float(np.linalg.norm(normal))
    if norm <= 0.0:
        return PlaneGeometry(0.0, False, "terrain normal has zero norm")
    normal = normal / norm
    if normal[2] <= 0.0:
        return PlaneGeometry(0.0, False, "terrain normal does not point upward")
    if abs(float(normal[1])) > float(slope_alignment_tolerance):
        return PlaneGeometry(0.0, False, "terrain has an unsupported cross-slope component")
    return PlaneGeometry(math.atan2(-float(normal[0]), float(normal[2])), True)


def vertical_height_above_plane(
    normal_world: Sequence[float],
    plane_point_world: Sequence[float],
    point_world: Sequence[float],
) -> float:
    """Return vertical point-to-plane height, not normal distance."""

    normal = np.asarray(normal_world, dtype=np.float64)
    plane_point = np.asarray(plane_point_world, dtype=np.float64)
    point = np.asarray(point_world, dtype=np.float64)
    if normal.shape != (3,) or plane_point.shape != (3,) or point.shape != (3,):
        raise ValueError("plane height inputs must be 3-vectors")
    if not np.all(np.isfinite(normal)) or not np.all(np.isfinite(plane_point)) or not np.all(np.isfinite(point)):
        raise ValueError("plane height inputs must be finite")
    if normal[2] <= 0.0:
        raise ValueError("plane normal z must be positive")
    return float(np.dot(normal, point - plane_point) / normal[2])


def projection_matrix(alpha: float) -> np.ndarray:
    """Return ``diag(cos(alpha), 1)`` in the heading-horizontal frame."""

    cosine = math.cos(float(alpha))
    if cosine <= 0.0:
        raise ValueError("plane projection requires abs(alpha) < pi/2")
    return np.diag((cosine, 1.0)).astype(np.float64)


def cop_translation(alpha: float, z_sole: float = -0.045) -> np.ndarray:
    """Affine ankle-origin correction ``[-z_sole*sin(alpha), 0]``."""

    return np.asarray((-float(z_sole) * math.sin(float(alpha)), 0.0), dtype=np.float64)


def _box(region: Mapping[str, Sequence[float]]) -> Box2D:
    return Box2D(tuple(float(value) for value in region["x"]), tuple(float(value) for value in region["y"]))


def _project_box(box: Box2D, alpha: float, translation_x: float = 0.0) -> Box2D:
    cosine = float(projection_matrix(alpha)[0, 0])
    return Box2D(
        (box.x[0] * cosine + translation_x, box.x[1] * cosine + translation_x),
        box.y,
    )


def adapt_flat_capability(
    flat_parameters: Mapping[str, object],
    alpha: float,
    z_sole: float = -0.045,
) -> PlaneCapability:
    """Project the existing flat capability basis to one signed plane.

    The L and v_max projections are reduced-order approximations, not strict
    mechanical reachability equalities.  No empirical slope capability is
    fitted here.
    """

    translation_x = float(cop_translation(alpha, z_sole)[0])
    cop_left = _project_box(_box(flat_parameters["C_left"]), alpha, translation_x)
    cop_right = _project_box(_box(flat_parameters["C_right"]), alpha, translation_x)
    landing_left = _project_box(_box(flat_parameters["L_left"]), alpha)
    landing_right = _project_box(_box(flat_parameters["L_right"]), alpha)
    cosine = float(projection_matrix(alpha)[0, 0])
    v_max = flat_parameters["v_max"]
    swing_velocity_limits = (float(v_max["x"]) * cosine, float(v_max["y"]))
    return PlaneCapability(
        cop_left=cop_left,
        cop_right=cop_right,
        landing_left=landing_left,
        landing_right=landing_right,
        swing_velocity_limits=swing_velocity_limits,
        nominal_cop_valid=cop_left.contains((0.0, 0.0)) and cop_right.contains((0.0, 0.0)),
    )


def inverse_project_horizontal(vector_heading: Sequence[float], alpha: float) -> np.ndarray:
    """Map a measured horizontal displacement to tangent-like diagnostics."""

    vector = np.asarray(vector_heading, dtype=np.float64)
    if vector.shape != (2,):
        raise ValueError("horizontal vector must have shape (2,)")
    return np.linalg.solve(projection_matrix(alpha), vector)


__all__ = [
    "Box2D",
    "PlaneCapability",
    "PlaneGeometry",
    "adapt_flat_capability",
    "cop_translation",
    "inverse_project_horizontal",
    "projection_matrix",
    "signed_slope_from_heading_normal",
    "vertical_height_above_plane",
]
