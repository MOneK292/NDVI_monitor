"""Delta NDVI calculation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from config import RASTER_NODATA
from modules.utils import (
    assert_same_grid,
    read_masked_raster,
    write_single_band_geotiff,
)


def calculate_delta_ndvi(
    ndvi_base_path: Path,
    ndvi_target_path: Path,
    output_path: Path,
) -> Path:
    """Calculate Delta NDVI as target year NDVI minus base year NDVI."""

    assert_same_grid(ndvi_base_path, ndvi_target_path)
    base, profile = read_masked_raster(ndvi_base_path)
    target, _ = read_masked_raster(ndvi_target_path)

    valid = (
        ~np.ma.getmaskarray(base)
        & ~np.ma.getmaskarray(target)
        & np.isfinite(base.data)
        & np.isfinite(target.data)
    )
    delta = np.full(base.shape, RASTER_NODATA, dtype="float32")
    delta[valid] = (target.data[valid] - base.data[valid]).astype("float32", copy=False)
    return write_single_band_geotiff(output_path, delta, profile, RASTER_NODATA, "float32")
