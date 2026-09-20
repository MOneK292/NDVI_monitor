"""Common validity mask utilities for multi-year NDVI analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

from config import NDVI_MAX, NDVI_MIN, RASTER_NODATA
from modules.utils import (
    assert_same_grid,
    read_masked_raster,
    read_valid_mask,
    write_single_band_geotiff,
)


def create_common_valid_mask(
    ndvi_paths: Mapping[int, Path],
    output_path: Path,
) -> Path:
    """Create valid2015 AND valid2025 mask for all supplied NDVI rasters."""

    paths = list(ndvi_paths.values())
    if len(paths) < 2:
        raise ValueError("At least two NDVI rasters are required for a common mask.")

    reference_path = paths[0]
    common_mask: np.ndarray | None = None
    profile = None

    for path in paths:
        assert_same_grid(reference_path, path)
        values, current_profile = read_masked_raster(path)
        data = values.data
        valid = (
            ~np.ma.getmaskarray(values)
            & np.isfinite(data)
            & (data >= NDVI_MIN)
            & (data <= NDVI_MAX)
        )
        common_mask = valid if common_mask is None else common_mask & valid
        profile = current_profile if profile is None else profile

    if common_mask is None or profile is None:
        raise ValueError("Cannot build a common validity mask.")
    if not np.any(common_mask):
        raise ValueError("Common validity mask has no valid pixels.")

    mask_array = common_mask.astype("uint8")
    return write_single_band_geotiff(output_path, mask_array, profile, 0, "uint8")


def apply_common_mask_to_raster(
    raster_path: Path,
    mask_path: Path,
    output_path: Path,
    nodata: float | int = RASTER_NODATA,
    dtype: str = "float32",
) -> Path:
    """Apply a common valid mask to a raster and write a masked copy."""

    assert_same_grid(raster_path, mask_path)
    values, profile = read_masked_raster(raster_path)
    common_mask = read_valid_mask(mask_path)

    data = np.full(values.shape, nodata, dtype=dtype)
    valid = (
        common_mask
        & ~np.ma.getmaskarray(values)
        & np.isfinite(values.data)
    )
    data[valid] = values.data[valid].astype(dtype, copy=False)
    return write_single_band_geotiff(output_path, data, profile, nodata, dtype)
