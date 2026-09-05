"""Continuous x-aligned planes used by plane-generalized recoverability.

Unlike Isaac Lab's pyramid slope, every tile generated here is one coplanar
surface whose height varies only along world x.  This is deliberate: the first
plane certificate does not cover cross-slope orientation or slope transitions.
"""

from __future__ import annotations

import math

import numpy as np
import trimesh
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from isaaclab.utils import configclass

from legged_lab.recovery.plane_terrain_math import (
    MATCHED_COLS,
    MATCHED_ROWS,
    MATCHED_SEED,
    MATCHED_SLOPE_RANGE,
    column_types as _column_types,
    slope_coefficient as _slope_coefficient,
    slope_table as _slope_table,
)


PLANE_RECOVERY_SLOPES_DEG = (-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0)
PLANE_RECOVERY_TILE_SIZE = (64.0, 32.0)
PLANE_BASELINE_MATCHED_TILE_SIZE = PLANE_RECOVERY_TILE_SIZE
PLANE_BASELINE_MATCHED_ROWS = MATCHED_ROWS
PLANE_BASELINE_MATCHED_COLS = MATCHED_COLS
PLANE_BASELINE_MATCHED_SEED = MATCHED_SEED
PLANE_BASELINE_MATCHED_SLOPE_RANGE = MATCHED_SLOPE_RANGE


def x_sloped_plane_terrain(
    difficulty: float,
    cfg: "MeshXSlopedPlaneTerrainCfg",
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate one plane ``z=tan(alpha)*(x-size_x/2)``.

    ``difficulty`` is intentionally ignored.  Each configured tile has one
    known signed slope, so runtime geometry never has to infer a randomized
    slope from a terrain row index.
    """

    del difficulty
    size_x, size_y = (float(value) for value in cfg.size)
    alpha = math.radians(float(cfg.slope_degrees))
    half_rise = math.tan(alpha) * size_x * 0.5
    vertices = np.asarray(
        (
            (0.0, 0.0, -half_rise),
            (size_x, 0.0, half_rise),
            (size_x, size_y, half_rise),
            (0.0, size_y, -half_rise),
        ),
        dtype=np.float64,
    )
    # This winding produces the upward normal [-sin(alpha), 0, cos(alpha)].
    faces = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    origin = np.asarray((size_x * 0.5, size_y * 0.5, 0.0), dtype=np.float64)
    return [mesh], origin


@configclass
class MeshXSlopedPlaneTerrainCfg(SubTerrainBaseCfg):
    """Configuration for one continuous plane aligned with world x."""

    function = x_sloped_plane_terrain
    slope_degrees: float = 0.0


def baseline_slope_coefficient(
    difficulty: float,
    slope_range: tuple[float, float] = PLANE_BASELINE_MATCHED_SLOPE_RANGE,
    *,
    inverted: bool = False,
) -> float:
    """Reproduce Isaac Lab's ``HfPyramidSlopedTerrain`` slope mapping exactly."""

    return _slope_coefficient(difficulty, slope_range, inverted=inverted)


def x_curriculum_sloped_plane_terrain(
    difficulty: float,
    cfg: "MeshXCurriculumSlopedPlaneTerrainCfg",
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate one coplanar tile using the baseline difficulty-to-slope rule."""

    coefficient = baseline_slope_coefficient(
        difficulty, cfg.slope_range, inverted=cfg.inverted
    )
    fixed = MeshXSlopedPlaneTerrainCfg(
        proportion=cfg.proportion,
        size=cfg.size,
        slope_degrees=math.degrees(math.atan(coefficient)),
    )
    return x_sloped_plane_terrain(difficulty, fixed)


@configclass
class MeshXCurriculumSlopedPlaneTerrainCfg(SubTerrainBaseCfg):
    """Continuous plane counterpart of Isaac Lab's pyramid-slope config."""

    function = x_curriculum_sloped_plane_terrain
    slope_range: tuple[float, float] = PLANE_BASELINE_MATCHED_SLOPE_RANGE
    inverted: bool = False


def baseline_matched_column_types(num_cols: int = PLANE_BASELINE_MATCHED_COLS) -> tuple[str, ...]:
    """Return Isaac Lab's deterministic curriculum column assignment."""

    return _column_types(num_cols)


def make_plane_baseline_matched_slope_table(
    seed: int = PLANE_BASELINE_MATCHED_SEED,
    *,
    num_rows: int = PLANE_BASELINE_MATCHED_ROWS,
    num_cols: int = PLANE_BASELINE_MATCHED_COLS,
    difficulty_range: tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    """Replay TerrainGenerator's RNG order and return exact signed alpha radians."""

    return _slope_table(
        seed,
        num_rows=num_rows,
        num_cols=num_cols,
        difficulty_range=difficulty_range,
    )


def make_plane_baseline_matched_terrain_cfg(
    seed: int = PLANE_BASELINE_MATCHED_SEED,
) -> TerrainGeneratorCfg:
    """Build the 8-flat/6-uphill/6-downhill continuous-plane curriculum."""

    return TerrainGeneratorCfg(
        seed=int(seed),
        curriculum=True,
        size=PLANE_BASELINE_MATCHED_TILE_SIZE,
        border_width=20.0,
        num_rows=PLANE_BASELINE_MATCHED_ROWS,
        num_cols=PLANE_BASELINE_MATCHED_COLS,
        use_cache=False,
        sub_terrains={
            "flat": MeshXSlopedPlaneTerrainCfg(proportion=0.4, slope_degrees=0.0),
            "uphill": MeshXCurriculumSlopedPlaneTerrainCfg(
                proportion=0.3,
                slope_range=PLANE_BASELINE_MATCHED_SLOPE_RANGE,
                inverted=False,
            ),
            "downhill": MeshXCurriculumSlopedPlaneTerrainCfg(
                proportion=0.3,
                slope_range=PLANE_BASELINE_MATCHED_SLOPE_RANGE,
                inverted=True,
            ),
        },
    )


def make_plane_recovery_terrain_cfg(
    slopes_degrees: tuple[float, ...] = PLANE_RECOVERY_SLOPES_DEG,
) -> TerrainGeneratorCfg:
    """Build deterministic flat/uphill/downhill plane tiles.

    One terrain column corresponds to one signed slope.  A single, long row
    avoids non-coplanar slope-top/slope-bottom transitions during an episode.
    """

    if not slopes_degrees:
        raise ValueError("at least one plane slope is required")
    proportion = 1.0 / len(slopes_degrees)
    sub_terrains = {
        f"plane_{index}_{slope:+g}deg": MeshXSlopedPlaneTerrainCfg(
            proportion=proportion,
            size=PLANE_RECOVERY_TILE_SIZE,
            slope_degrees=float(slope),
        )
        for index, slope in enumerate(slopes_degrees)
    }
    return TerrainGeneratorCfg(
        curriculum=True,
        size=PLANE_RECOVERY_TILE_SIZE,
        border_width=20.0,
        num_rows=1,
        num_cols=len(slopes_degrees),
        use_cache=False,
        sub_terrains=sub_terrains,
    )


PLANE_RECOVERY_TERRAINS_CFG = make_plane_recovery_terrain_cfg()
PLANE_BASELINE_MATCHED_TERRAINS_CFG = make_plane_baseline_matched_terrain_cfg()


__all__ = [
    "MeshXSlopedPlaneTerrainCfg",
    "MeshXCurriculumSlopedPlaneTerrainCfg",
    "PLANE_BASELINE_MATCHED_COLS",
    "PLANE_BASELINE_MATCHED_ROWS",
    "PLANE_BASELINE_MATCHED_SEED",
    "PLANE_BASELINE_MATCHED_SLOPE_RANGE",
    "PLANE_BASELINE_MATCHED_TERRAINS_CFG",
    "PLANE_BASELINE_MATCHED_TILE_SIZE",
    "PLANE_RECOVERY_SLOPES_DEG",
    "PLANE_RECOVERY_TILE_SIZE",
    "PLANE_RECOVERY_TERRAINS_CFG",
    "make_plane_recovery_terrain_cfg",
    "baseline_matched_column_types",
    "baseline_slope_coefficient",
    "make_plane_baseline_matched_slope_table",
    "make_plane_baseline_matched_terrain_cfg",
    "x_curriculum_sloped_plane_terrain",
    "x_sloped_plane_terrain",
]
