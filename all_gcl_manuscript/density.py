"""
KDE utilities for 2D density estimation and normalized density maps.

These functions were extracted from dev-joesterle/density_analysis/cell_type_distribution.ipynb
so they can be reused across notebooks and scripts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Iterable, Optional
import numpy as np
from scipy.stats import gaussian_kde


@dataclass
class GridSpec:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    grid_size: int = 51

    def create(self) -> Tuple[np.ndarray, np.ndarray]:
        xx, yy = np.meshgrid(
            np.linspace(self.x_min, self.x_max, self.grid_size),
            np.linspace(self.y_min, self.y_max, self.grid_size),
        )
        return xx, yy


def create_evaluation_grid(x_min: float, x_max: float, y_min: float, y_max: float, grid_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create a 2D evaluation grid.

    Returns
    -------
    xx, yy : 2D ndarrays of shape (grid_size, grid_size)
    """
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_size),
                         np.linspace(y_min, y_max, grid_size))
    return xx, yy


def evaluate_kde_on_grid(kde: gaussian_kde, grid_points: np.ndarray, grid_shape: Tuple[int, int]) -> np.ndarray:
    """Evaluate a fitted gaussian_kde on provided grid points and reshape to grid shape."""
    return kde(grid_points).reshape(grid_shape)


def normalize_density(dens_group: np.ndarray, dens_all: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """Compute point-wise normalized density ratio dens_group / dens_all with stability epsilon."""
    return dens_group / (dens_all + epsilon)


def circle_mask(xx: np.ndarray, yy: np.ndarray, radius: float) -> np.ndarray:
    """Return a boolean mask for points outside a circle of given radius centered at 0."""
    distances = np.sqrt(xx ** 2 + yy ** 2)
    return distances > radius


def fit_kde(points_xy: np.ndarray, bw_method: Optional[float] = 1.0) -> gaussian_kde:
    """Fit a gaussian KDE for 2xN points array where rows are [x; y]."""
    return gaussian_kde(points_xy, bw_method=bw_method)


def grid_points_from_mesh(xx: np.ndarray, yy: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Return stacked grid points (2, M) and grid shape for evaluation/reshaping."""
    grid_points = np.vstack([xx.ravel(), yy.ravel()])
    return grid_points, xx.shape


def compute_normalized_kde_ratio(
    x: np.ndarray,
    y: np.ndarray,
    groups: Optional[Iterable] = None,
    r_max: float = 2000,
    grid_size: int = 51,
    bw_method: Optional[float] = 1.0,
):
    """
    Compute KDE of all points and per-group KDEs on a common grid, returning
    the normalized density ratio per group along with helpful grid artifacts.

    Parameters
    ----------
    x, y : 1D arrays of coordinates (same length)
    groups : iterable of group labels matching x/y. If None, returns only overall KDE.
    r_max : radius for circular mask and grid extents [-r_max, r_max]
    grid_size : grid resolution per axis
    bw_method : bandwidth method forwarded to gaussian_kde

    Returns
    -------
    result : dict with keys:
        - xx, yy: 2D grid
        - outside_circle: boolean mask of points outside the circle
        - dens_all: 2D array with overall KDE density
        - group_ids: np.ndarray of unique group ids (if groups provided)
        - dens_group: dict mapping group_id -> 2D density
        - norm_ratio: dict mapping group_id -> 2D normalized ratio dens_group/dens_all
    """
    x = np.asarray(x)
    y = np.asarray(y)
    assert x.shape == y.shape, "x and y must have the same shape"

    # Grid
    xx, yy = create_evaluation_grid(-r_max, +r_max, -r_max, +r_max, grid_size)
    grid_points, grid_shape = grid_points_from_mesh(xx, yy)

    # Overall KDE
    kde_all = fit_kde(np.vstack([x, y]), bw_method=bw_method)
    dens_all = evaluate_kde_on_grid(kde_all, grid_points, grid_shape)

    outside_circle = circle_mask(xx, yy, r_max)

    result = {
        "xx": xx,
        "yy": yy,
        "outside_circle": outside_circle,
        "dens_all": dens_all,
    }

    if groups is None:
        return result

    groups = np.asarray(groups)
    assert groups.shape == x.shape, "groups must have the same shape as x/y"

    group_ids = np.unique(groups)
    result["group_ids"] = group_ids

    dens_group_map = {}
    norm_ratio_map = {}
    for gid in group_ids:
        sel = groups == gid
        if np.count_nonzero(sel) < 2:
            # gaussian_kde needs at least 2 samples; fill with NaNs
            dens = np.full_like(dens_all, np.nan)
        else:
            kde_g = fit_kde(np.vstack([x[sel], y[sel]]), bw_method=bw_method)
            dens = evaluate_kde_on_grid(kde_g, grid_points, grid_shape)
        dens_group_map[gid] = dens
        norm = normalize_density(dens, dens_all)
        norm[outside_circle] = np.nan
        norm_ratio_map[gid] = norm

    result["dens_group"] = dens_group_map
    result["norm_ratio"] = norm_ratio_map
    return result

