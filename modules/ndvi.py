"""NDVI calculation and normalization."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import rasterio

from config import (
    BASE_YEAR,
    NDVI_DENOMINATOR_EPSILON,
    NDVI_FORMAT_TOLERANCE,
    NDVI_MAX,
    NDVI_MIN,
    NDVI_SCALE_FACTOR,
    NDVI_SCALED_ABS_LIMIT,
    RASTER_NODATA,
    TARGET_YEAR,
)
from modules.loader import InputDataError, InputVariant
from modules.utils import (
    assert_same_grid,
    read_masked_raster,
    write_single_band_geotiff,
)


def build_ndvi_products(
    aligned_rasters: Mapping[str, Path],
    variant: InputVariant,
    output_dir: Path,
) -> dict[int, Path]:
    """Create canonical NDVI rasters for all required years."""

    output_dir.mkdir(parents=True, exist_ok=True)
    if variant == InputVariant.RAW_BANDS:
        return {
            BASE_YEAR: calculate_ndvi(
                aligned_rasters["red_2015"],
                aligned_rasters["nir_2015"],
                output_dir / f"ndvi_{BASE_YEAR}.tif",
            ),
            TARGET_YEAR: calculate_ndvi(
                aligned_rasters["red_2025"],
                aligned_rasters["nir_2025"],
                output_dir / f"ndvi_{TARGET_YEAR}.tif",
            ),
        }

    return {
        BASE_YEAR: normalize_ndvi_raster(
            aligned_rasters["ndvi_2015"],
            output_dir / f"ndvi_{BASE_YEAR}.tif",
        ),
        TARGET_YEAR: normalize_ndvi_raster(
            aligned_rasters["ndvi_2025"],
            output_dir / f"ndvi_{TARGET_YEAR}.tif",
        ),
    }


def calculate_ndvi(red_path: Path, nir_path: Path, output_path: Path) -> Path:
    """Calculate NDVI from red and near-infrared Sentinel-2 bands."""

    assert_same_grid(red_path, nir_path)
    red, profile = read_masked_raster(red_path)
    nir, _ = read_masked_raster(nir_path)

    denominator = nir.data + red.data
    valid = (
        ~np.ma.getmaskarray(red)
        & ~np.ma.getmaskarray(nir)
        & np.isfinite(denominator)
        & (np.abs(denominator) > NDVI_DENOMINATOR_EPSILON)
    )

    ndvi = np.full(red.shape, RASTER_NODATA, dtype="float32")
    ndvi_values = (nir.data[valid] - red.data[valid]) / denominator[valid]
    ndvi[valid] = np.clip(ndvi_values, NDVI_MIN, NDVI_MAX).astype("float32", copy=False)
    return write_single_band_geotiff(output_path, ndvi, profile, RASTER_NODATA, "float32")


def normalize_ndvi_raster(source_path: Path, output_path: Path) -> Path:
    """Copy a ready NDVI raster to the canonical output with valid NDVI range."""

    ndvi, profile = _read_ready_ndvi(source_path)
    output = np.full(ndvi.shape, RASTER_NODATA, dtype="float32")
    valid = ~np.ma.getmaskarray(ndvi) & np.isfinite(ndvi.data)
    output[valid] = ndvi.data[valid].astype("float32", copy=False)
    return write_single_band_geotiff(output_path, output, profile, RASTER_NODATA, "float32")


def _read_ready_ndvi(source_path: Path) -> tuple[np.ma.MaskedArray, dict[str, object]]:
    """Read ready NDVI and detect whether values are float or scaled int16."""

    with rasterio.open(source_path) as dataset:
        band = dataset.read(1, masked=True)
        raw_data = np.asarray(band.data, dtype="float32")
        mask = np.ma.getmaskarray(band) | ~np.isfinite(raw_data)
        if dataset.nodata is not None:
            mask |= np.isclose(raw_data, float(dataset.nodata))

        profile = dataset.profile.copy()
        valid_values = raw_data[~mask]
        if valid_values.size == 0:
            raise InputDataError(f"Ready NDVI raster has no valid pixels: {source_path}")

        scale = _band_value(dataset.scales, 0, 1.0)
        offset = _band_value(dataset.offsets, 0, 0.0)
        metadata_values = raw_data * scale + offset

        if (scale != 1.0 or offset != 0.0) and _is_ndvi_range(metadata_values, mask):
            return np.ma.array(metadata_values.astype("float32"), mask=mask), profile

        if _is_ndvi_range(raw_data, mask):
            return np.ma.array(raw_data.astype("float32"), mask=mask), profile

        raw_min = float(np.min(valid_values))
        raw_max = float(np.max(valid_values))
        raw_abs = max(abs(raw_min), abs(raw_max))
        if raw_abs <= NDVI_SCALED_ABS_LIMIT:
            scaled = raw_data / NDVI_SCALE_FACTOR
            if _is_ndvi_range(scaled, mask):
                return np.ma.array(scaled.astype("float32"), mask=mask), profile

        raise InputDataError(
            "Cannot determine ready NDVI format for "
            f"{source_path}. Valid value range is {raw_min:.6g}..{raw_max:.6g}. "
            "Expected Float32 NDVI in [-1; 1] or Int16 NDVI scaled by 10000."
        )


def _is_ndvi_range(data: np.ndarray, mask: np.ndarray) -> bool:
    valid = data[~mask]
    if valid.size == 0:
        return False
    return (
        np.nanmin(valid) >= NDVI_MIN - NDVI_FORMAT_TOLERANCE
        and np.nanmax(valid) <= NDVI_MAX + NDVI_FORMAT_TOLERANCE
    )


def _band_value(
    values: tuple[float | None, ...] | None,
    index: int,
    default: float,
) -> float:
    if not values or index >= len(values) or values[index] is None:
        return default
    return float(values[index])
