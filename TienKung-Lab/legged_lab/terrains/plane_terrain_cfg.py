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


PLANE_RECOVERY_SLOPES_DEG = (-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0)
PLANE_RECOVERY_TILE_SIZE = (64.0, 32.0)


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


__all__ = [
    "MeshXSlopedPlaneTerrainCfg",
    "PLANE_RECOVERY_SLOPES_DEG",
    "PLANE_RECOVERY_TILE_SIZE",
    "PLANE_RECOVERY_TERRAINS_CFG",
    "make_plane_recovery_terrain_cfg",
    "x_sloped_plane_terrain",
]
