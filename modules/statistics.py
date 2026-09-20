"""Statistical summaries for raster values and classified areas."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from pyproj import CRS, Geod, Transformer

from config import CLASS_NODATA
from modules.classify import ClassRule
from modules.utils import assert_same_grid, ensure_parent, read_masked_raster


AreaModel = float | np.ndarray


def compute_value_statistics(raster_path: Path, name: str) -> pd.DataFrame:
    """Calculate descriptive statistics for a continuous raster."""

    values, _ = read_masked_raster(raster_path)
    valid = ~np.ma.getmaskarray(values) & np.isfinite(values.data)
    if not valid.any():
        raise ValueError(f"Raster has no valid pixels for statistics: {raster_path}")

    data = values.data[valid].astype("float64", copy=False)
    return pd.DataFrame(
        [
            {
                "metric": name,
                "pixel_count": int(data.size),
                "mean": float(np.mean(data)),
                "minimum": float(np.min(data)),
                "maximum": float(np.max(data)),
                "median": float(np.median(data)),
                "std": float(np.std(data)),
            }
        ]
    )


def compute_class_areas(
    class_raster_path: Path,
    rules: Sequence[ClassRule],
) -> pd.DataFrame:
    """Calculate pixel counts, areas and territory share by class."""

    with rasterio.open(class_raster_path) as dataset:
        classes = dataset.read(1)
        area_model = build_area_model(
            dataset.transform,
            dataset.width,
            dataset.height,
            dataset.crs,
        )

    valid = classes != CLASS_NODATA
    total_area_m2 = _sum_area(valid, area_model)
    rows = []

    for rule in rules:
        mask = classes == rule.class_id
        area_m2 = _sum_area(mask, area_model)
        percent = (area_m2 / total_area_m2 * 100.0) if total_area_m2 else 0.0
        rows.append(
            {
                "class_id": rule.class_id,
                "label": rule.label,
                "pixel_count": int(np.count_nonzero(mask)),
                "area_m2": float(area_m2),
                "area_ha": float(area_m2 / 10_000.0),
                "percent": float(percent),
            }
        )

    return pd.DataFrame(rows)


def compute_transition_matrix(
    class_start_path: Path,
    class_end_path: Path,
    rules: Sequence[ClassRule],
) -> pd.DataFrame:
    """Calculate NDVI class transition matrix between two classified rasters."""

    assert_same_grid(class_start_path, class_end_path)

    with rasterio.open(class_start_path) as start_dataset:
        start_classes = start_dataset.read(1)
        transform = start_dataset.transform
        width = start_dataset.width
        height = start_dataset.height
        crs = start_dataset.crs

    with rasterio.open(class_end_path) as end_dataset:
        end_classes = end_dataset.read(1)

    if start_classes.shape != end_classes.shape:
        raise ValueError("Class rasters must have identical dimensions.")

    area_model = build_area_model(transform, width, height, crs)
    valid = (start_classes != CLASS_NODATA) & (end_classes != CLASS_NODATA)
    total_area_m2 = _sum_area(valid, area_model)
    total_pixels = int(np.count_nonzero(valid))
    rows = []

    for start_rule in rules:
        for end_rule in rules:
            transition_mask = (
                valid
                & (start_classes == start_rule.class_id)
                & (end_classes == end_rule.class_id)
            )
            pixel_count = int(np.count_nonzero(transition_mask))
            area_m2 = _sum_area(transition_mask, area_model)
            rows.append(
                {
                    "from_class": start_rule.class_id,
                    "from_label": start_rule.label,
                    "to_class": end_rule.class_id,
                    "to_label": end_rule.label,
                    "transition": f"{start_rule.class_id} -> {end_rule.class_id}",
                    "pixel_count": pixel_count,
                    "area_m2": float(area_m2),
                    "area_ha": float(area_m2 / 10_000.0),
                    "percent": (
                        float(area_m2 / total_area_m2 * 100.0)
                        if total_area_m2
                        else 0.0
                    ),
                    "pixel_percent": (
                        float(pixel_count / total_pixels * 100.0)
                        if total_pixels
                        else 0.0
                    ),
                }
            )

    return pd.DataFrame(rows)


def export_transition_matrix(table: pd.DataFrame, output_path: Path) -> Path:
    """Save transition matrix to a standalone Excel workbook."""

    ensure_parent(output_path)
    table.to_excel(output_path, index=False, engine="openpyxl")
    return output_path


def build_area_model(
    transform: Affine,
    width: int,
    height: int,
    crs: rasterio.crs.CRS,
) -> AreaModel:
    """Build a scalar, row-wise or full-grid pixel area model in square meters."""

    pyproj_crs = CRS.from_user_input(crs)
    if pyproj_crs.is_projected:
        linear_factor = _linear_unit_factor(pyproj_crs)
        return abs(transform.a * transform.e - transform.b * transform.d) * (
            linear_factor**2
        )

    if pyproj_crs.is_geographic:
        if transform.b == 0 and transform.d == 0:
            return _geographic_row_areas(transform, height, pyproj_crs)
        return _geographic_cell_areas(transform, width, height, pyproj_crs)

    raise ValueError(f"Unsupported CRS for area calculation: {crs}")


def _linear_unit_factor(crs: CRS) -> float:
    axis_info = crs.axis_info
    if axis_info:
        return float(axis_info[0].unit_conversion_factor)
    return 1.0


def _geographic_row_areas(
    transform: Affine,
    height: int,
    crs: CRS,
) -> np.ndarray:
    geod = Geod(ellps="WGS84")
    transformer = Transformer.from_crs(crs, CRS.from_epsg(4326), always_xy=True)
    areas = np.zeros(height, dtype="float64")

    for row in range(height):
        xs, ys = zip(
            transform * (0, row),
            transform * (1, row),
            transform * (1, row + 1),
            transform * (0, row + 1),
        )
        lon, lat = transformer.transform(xs, ys)
        area, _ = geod.polygon_area_perimeter(lon, lat)
        areas[row] = abs(area)

    return areas


def _geographic_cell_areas(
    transform: Affine,
    width: int,
    height: int,
    crs: CRS,
) -> np.ndarray:
    geod = Geod(ellps="WGS84")
    transformer = Transformer.from_crs(crs, CRS.from_epsg(4326), always_xy=True)
    areas = np.zeros((height, width), dtype="float64")

    for row in range(height):
        for col in range(width):
            xs, ys = zip(
                transform * (col, row),
                transform * (col + 1, row),
                transform * (col + 1, row + 1),
                transform * (col, row + 1),
            )
            lon, lat = transformer.transform(xs, ys)
            area, _ = geod.polygon_area_perimeter(lon, lat)
            areas[row, col] = abs(area)

    return areas


def _sum_area(mask: np.ndarray, area_model: AreaModel) -> float:
    if isinstance(area_model, float):
        return float(np.count_nonzero(mask) * area_model)

    if area_model.ndim == 1:
        counts_by_row = np.count_nonzero(mask, axis=1)
        return float(np.sum(counts_by_row * area_model))

    return float(np.sum(area_model[mask]))
