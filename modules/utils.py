"""Shared utilities for raster IO, logging and validation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

from config import RASTER_NODATA


LOGGER_NAME = "ndvi_monitor"


def setup_logging(log_file: Path, level: int = logging.INFO) -> logging.Logger:
    """Configure file and console logging for the application."""

    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def ensure_parent(path: Path) -> None:
    """Create a file parent directory before writing."""

    path.parent.mkdir(parents=True, exist_ok=True)


def read_masked_raster(
    path: Path,
    apply_scale_offset: bool = True,
) -> tuple[np.ma.MaskedArray, dict[str, Any]]:
    """Read the first raster band as a floating masked array.

    Rasterio masks, explicit NoData, NaN and infinite values are all treated as
    invalid. Band scale and offset are applied by default because many remote
    sensing products store physical values through metadata.
    """

    with rasterio.open(path) as dataset:
        band = dataset.read(1, masked=True)
        data = np.asarray(band.data, dtype="float32")
        mask = np.ma.getmaskarray(band) | ~np.isfinite(data)
        if dataset.nodata is not None:
            mask |= np.isclose(data, float(dataset.nodata))

        if apply_scale_offset:
            scale = _band_value(dataset.scales, 0, 1.0)
            offset = _band_value(dataset.offsets, 0, 0.0)
            data = data * scale + offset

        masked = np.ma.array(data, mask=mask)
        return masked, dataset.profile.copy()


def write_single_band_geotiff(
    path: Path,
    array: np.ndarray,
    profile: dict[str, Any],
    nodata: float | int = RASTER_NODATA,
    dtype: str | None = None,
) -> Path:
    """Write a single-band GeoTIFF using a reference raster profile."""

    ensure_parent(path)
    output_dtype = dtype or str(array.dtype)
    output_profile = profile.copy()
    output_profile.update(
        driver="GTiff",
        count=1,
        dtype=output_dtype,
        nodata=nodata,
        compress="deflate",
    )
    output_profile.pop("blockxsize", None)
    output_profile.pop("blockysize", None)
    output_profile.pop("tiled", None)

    masked_array = np.ma.array(array, copy=False)
    mask = np.ma.getmaskarray(masked_array)
    data = np.asarray(masked_array.filled(nodata))
    if np.issubdtype(data.dtype, np.floating):
        mask |= np.isclose(data, float(nodata))
    else:
        mask |= data == nodata
    with rasterio.open(path, "w", **output_profile) as dataset:
        dataset.write(data.astype(output_dtype, copy=False), 1)
        dataset.write_mask(np.where(mask, 0, 255).astype("uint8"))
    return path


def assert_same_grid(first_path: Path, second_path: Path) -> None:
    """Ensure two rasters have identical CRS, transform and dimensions."""

    with rasterio.open(first_path) as first, rasterio.open(second_path) as second:
        same_grid = (
            first.crs == second.crs
            and first.transform == second.transform
            and first.width == second.width
            and first.height == second.height
        )
        if not same_grid:
            raise ValueError(
                "Rasters are not aligned to the same grid: "
                f"{first_path} and {second_path}"
            )


def read_valid_mask(mask_path: Path) -> np.ndarray:
    """Read a uint8 validity mask where 1 means valid and 0 means invalid."""

    with rasterio.open(mask_path) as dataset:
        mask = dataset.read(1)
    return mask == 1


def _band_value(
    values: tuple[float | None, ...] | None,
    index: int,
    default: float,
) -> float:
    if not values or index >= len(values) or values[index] is None:
        return default
    return float(values[index])
