"""Raster reprojection, grid alignment and clipping."""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.mask import mask
from rasterio.transform import from_origin
from rasterio.warp import reproject
from shapely.geometry import mapping
from tqdm import tqdm

from config import (
    ALIGNMENT_ORIGIN_X,
    ALIGNMENT_ORIGIN_Y,
    ALIGNMENT_RESOLUTION,
    ALIGNMENT_TARGET_CRS,
    RASTER_NODATA,
)
from modules.utils import ensure_parent, read_masked_raster


@dataclass(frozen=True)
class TargetGrid:
    """Explicit raster grid used by all aligned project outputs."""

    crs: rasterio.crs.CRS
    transform: rasterio.Affine
    width: int
    height: int
    resolution: float
    origin_x: float
    origin_y: float


def align_and_clip_rasters(
    raster_paths: Mapping[str, Path],
    boundary_path: Path,
    output_dir: Path,
    logger: logging.Logger,
) -> dict[str, Path]:
    """Reproject all rasters to one grid and clip them by the study boundary."""

    output_dir.mkdir(parents=True, exist_ok=True)
    target_grid = build_target_grid(boundary_path)
    logger.info(
        "Target grid: CRS=%s, resolution=%s, size=%sx%s, transform=%s",
        target_grid.crs,
        target_grid.resolution,
        target_grid.width,
        target_grid.height,
        target_grid.transform,
    )

    clipped_paths: dict[str, Path] = {}

    for label, source_path in tqdm(
        raster_paths.items(),
        desc="Aligning rasters",
        unit="raster",
    ):
        aligned_path = output_dir / f"{label}_aligned.tif"
        clipped_path = output_dir / f"{label}_aligned_clipped.tif"
        align_raster_to_grid(source_path, target_grid, aligned_path)
        clip_raster_to_boundary(aligned_path, boundary_path, clipped_path)
        clipped_paths[label] = clipped_path
        logger.info("Aligned and clipped raster %s -> %s", label, clipped_path)

    return clipped_paths


def build_target_grid(boundary_path: Path) -> TargetGrid:
    """Build an explicit grid from the study boundary and project settings."""

    boundary = gpd.read_file(boundary_path)
    target_crs = rasterio.crs.CRS.from_user_input(ALIGNMENT_TARGET_CRS)
    boundary = boundary.to_crs(target_crs)
    min_x, min_y, max_x, max_y = boundary.total_bounds

    resolution = float(ALIGNMENT_RESOLUTION)
    left = _snap_down(min_x, ALIGNMENT_ORIGIN_X, resolution)
    right = _snap_up(max_x, ALIGNMENT_ORIGIN_X, resolution)
    bottom = _snap_down(min_y, ALIGNMENT_ORIGIN_Y, resolution)
    top = _snap_up(max_y, ALIGNMENT_ORIGIN_Y, resolution)

    width = int(math.ceil((right - left) / resolution))
    height = int(math.ceil((top - bottom) / resolution))
    if width <= 0 or height <= 0:
        raise ValueError(f"Boundary produces an invalid target grid: {boundary_path}")

    return TargetGrid(
        crs=target_crs,
        transform=from_origin(left, top, resolution, resolution),
        width=width,
        height=height,
        resolution=resolution,
        origin_x=float(ALIGNMENT_ORIGIN_X),
        origin_y=float(ALIGNMENT_ORIGIN_Y),
    )


def align_raster_to_grid(
    source_path: Path,
    target_grid: TargetGrid,
    output_path: Path,
    resampling: Resampling = Resampling.bilinear,
) -> Path:
    """Reproject one raster to the explicit project grid."""

    ensure_parent(output_path)
    source_data, source_profile = read_masked_raster(source_path)
    source_filled = source_data.filled(RASTER_NODATA).astype("float32", copy=False)
    with rasterio.open(source_path) as source:
        destination = np.full(
            (target_grid.height, target_grid.width),
            RASTER_NODATA,
            dtype="float32",
        )
        reproject(
            source=source_filled,
            destination=destination,
            src_transform=source_profile["transform"],
            src_crs=source_profile["crs"],
            src_nodata=RASTER_NODATA,
            dst_transform=target_grid.transform,
            dst_crs=target_grid.crs,
            dst_nodata=RASTER_NODATA,
            resampling=resampling,
        )

        profile = source.profile.copy()
        profile.update(
            driver="GTiff",
            crs=target_grid.crs,
            transform=target_grid.transform,
            width=target_grid.width,
            height=target_grid.height,
            count=1,
            dtype="float32",
            nodata=RASTER_NODATA,
            compress="deflate",
        )
        profile.pop("blockxsize", None)
        profile.pop("blockysize", None)
        profile.pop("tiled", None)

        with rasterio.open(output_path, "w", **profile) as dataset:
            dataset.write(destination, 1)
            dataset.write_mask(
                np.where(np.isclose(destination, RASTER_NODATA), 0, 255).astype("uint8")
            )

    return output_path


def clip_raster_to_boundary(
    raster_path: Path,
    boundary_path: Path,
    output_path: Path,
) -> Path:
    """Clip a raster by the study area boundary while preserving georeferencing."""

    ensure_parent(output_path)
    with rasterio.open(raster_path) as raster:
        boundary = gpd.read_file(boundary_path)
        boundary = boundary[boundary.geometry.notna() & ~boundary.geometry.is_empty]
        if boundary.empty:
            raise ValueError(f"Boundary has no usable geometries: {boundary_path}")
        if boundary.crs != raster.crs:
            boundary = boundary.to_crs(raster.crs)

        geometries = [mapping(geometry) for geometry in boundary.geometry]
        clipped, transform = mask(
            raster,
            geometries,
            crop=True,
            nodata=RASTER_NODATA,
            filled=True,
        )
        profile = raster.profile.copy()
        profile.update(
            height=clipped.shape[1],
            width=clipped.shape[2],
            transform=transform,
            nodata=RASTER_NODATA,
            compress="deflate",
        )
        profile.pop("blockxsize", None)
        profile.pop("blockysize", None)
        profile.pop("tiled", None)

        with rasterio.open(output_path, "w", **profile) as dataset:
            dataset.write(clipped)
            valid_mask = ~np.isclose(clipped[0], RASTER_NODATA)
            dataset.write_mask(np.where(valid_mask, 255, 0).astype("uint8"))

    return output_path


def _snap_down(value: float, origin: float, resolution: float) -> float:
    return math.floor((value - origin) / resolution) * resolution + origin


def _snap_up(value: float, origin: float, resolution: float) -> float:
    return math.ceil((value - origin) / resolution) * resolution + origin
