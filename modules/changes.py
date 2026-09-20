"""Detection and export of the largest NDVI change polygons."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape as shapely_shape

from config import (
    DEGRADATION_CLASSES,
    IMPROVEMENT_CLASSES,
    MAX_CHANGE_POLYGONS_PER_TYPE,
)
from modules.utils import ensure_parent


def extract_largest_change_polygons(
    delta_class_path: Path,
    output_vector_path: Path,
    output_excel_path: Path,
    max_per_type: int = MAX_CHANGE_POLYGONS_PER_TYPE,
) -> tuple[Path, Path, pd.DataFrame]:
    """Polygonize and export the largest degradation and improvement areas."""

    rows = _polygonize_change_regions(delta_class_path)
    if rows:
        table = pd.DataFrame(rows)
        table = table.sort_values(["change_type", "area_m2"], ascending=[True, False])
        table["rank"] = table.groupby("change_type").cumcount() + 1
        table = table[table["rank"] <= max_per_type].reset_index(drop=True)
    else:
        table = _empty_change_table()

    _write_change_vector(table, output_vector_path, delta_class_path)
    _write_change_excel(table.drop(columns=["geometry"], errors="ignore"), output_excel_path)
    return output_vector_path, output_excel_path, table.drop(columns=["geometry"], errors="ignore")


def _polygonize_change_regions(delta_class_path: Path) -> list[dict[str, object]]:
    with rasterio.open(delta_class_path) as dataset:
        class_array = dataset.read(1)
        selected = np.isin(
            class_array,
            tuple(DEGRADATION_CLASSES) + tuple(IMPROVEMENT_CLASSES),
        )
        rows = []
        for geometry_mapping, value in shapes(
            class_array.astype("uint8", copy=False),
            mask=selected,
            transform=dataset.transform,
        ):
            class_id = int(value)
            geometry = shapely_shape(geometry_mapping)
            if geometry.is_empty or geometry.area <= 0:
                continue
            change_type = (
                "degradation"
                if class_id in DEGRADATION_CLASSES
                else "improvement"
            )
            min_x, min_y, max_x, max_y = geometry.bounds
            rows.append(
                {
                    "change_type": change_type,
                    "delta_class": class_id,
                    "area_m2": float(geometry.area),
                    "area_ha": float(geometry.area / 10_000.0),
                    "bbox_min_x": float(min_x),
                    "bbox_min_y": float(min_y),
                    "bbox_max_x": float(max_x),
                    "bbox_max_y": float(max_y),
                    "geometry": geometry,
                    "crs": dataset.crs,
                }
            )
    return _add_lonlat_coordinates(rows)


def _add_lonlat_coordinates(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    if not rows:
        return rows
    crs = rows[0]["crs"]
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=crs).to_crs("EPSG:4326")
    for index, row in enumerate(gdf.itertuples()):
        centroid = row.geometry.centroid
        rows[index]["centroid_lon"] = float(centroid.x)
        rows[index]["centroid_lat"] = float(centroid.y)
        rows[index].pop("crs", None)
    return rows


def _write_change_vector(
    table: pd.DataFrame,
    output_vector_path: Path,
    delta_class_path: Path,
) -> None:
    ensure_parent(output_vector_path)
    with rasterio.open(delta_class_path) as dataset:
        crs = dataset.crs
    if table.empty:
        gdf = gpd.GeoDataFrame(_empty_change_table(), geometry="geometry", crs=crs)
    else:
        gdf = gpd.GeoDataFrame(table.copy(), geometry="geometry", crs=crs)
    if output_vector_path.suffix.lower() == ".gpkg":
        gdf.to_file(output_vector_path, layer="change_polygons", driver="GPKG")
    else:
        gdf.to_file(output_vector_path)


def _write_change_excel(table: pd.DataFrame, output_excel_path: Path) -> None:
    ensure_parent(output_excel_path)
    table.to_excel(output_excel_path, index=False, engine="openpyxl")


def _empty_change_table() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "change_type",
            "delta_class",
            "area_m2",
            "area_ha",
            "bbox_min_x",
            "bbox_min_y",
            "bbox_max_x",
            "bbox_max_y",
            "centroid_lon",
            "centroid_lat",
            "rank",
            "geometry",
        ]
    )
