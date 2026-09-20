"""Input discovery and validation for Sentinel-2 monitoring data."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import re
from typing import Mapping

import geopandas as gpd
import numpy as np
import rasterio
from shapely.geometry import box

from config import (
    BASE_YEAR,
    ALIGNMENT_TARGET_CRS,
    RAW_BAND_FILES,
    READY_NDVI_FILES,
    TARGET_YEAR,
    VECTOR_BOUNDARY_EXTENSIONS,
)


class InputVariant(str, Enum):
    """Supported input data variants."""

    RAW_BANDS = "raw_bands"
    READY_NDVI = "ready_ndvi"


class InputDataError(RuntimeError):
    """Raised when required input data is missing or invalid."""


@dataclass(frozen=True)
class RasterMetadata:
    """Validated technical metadata of a single-band raster."""

    path: Path
    crs: str
    width: int
    height: int
    pixel_size_x: float
    pixel_size_y: float
    dtype: str
    nodata: float | int | None
    scale: float
    offset: float
    band_count: int
    transform: tuple[float, float, float, float, float, float]
    bounds: tuple[float, float, float, float]
    valid_min: float
    valid_max: float
    intersects_boundary: bool
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class BoundaryMetadata:
    """Validated metadata of the study boundary vector layer."""

    path: Path
    crs: str
    feature_count: int
    bounds: tuple[float, float, float, float]
    geometry_type: str
    area: float


@dataclass(frozen=True)
class InputBundle:
    """Complete validated input selection for one pipeline run."""

    variant: InputVariant
    rasters: Mapping[str, Path]
    boundary: Path
    raster_metadata: Mapping[str, RasterMetadata]
    boundary_metadata: BoundaryMetadata


def validate_inputs(input_dir: Path) -> InputBundle:
    """Detect the available input variant and validate all required files."""

    raster_paths, variant = discover_input_rasters(input_dir)
    boundary_path = discover_boundary_file(input_dir)
    boundary = _read_boundary(boundary_path)
    boundary_metadata = validate_boundary(boundary_path, boundary)
    metadata = {
        label: validate_raster(path, boundary)
        for label, path in raster_paths.items()
    }
    return InputBundle(
        variant=variant,
        rasters=raster_paths,
        boundary=boundary_path,
        raster_metadata=metadata,
        boundary_metadata=boundary_metadata,
    )


def discover_boundary_file(input_dir: Path) -> Path:
    """Find exactly one supported vector boundary file in the input directory."""

    candidates = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VECTOR_BOUNDARY_EXTENSIONS
    )
    if not candidates:
        expected = ", ".join(f"*{suffix}" for suffix in VECTOR_BOUNDARY_EXTENSIONS)
        raise InputDataError(
            "No study boundary vector file was found in the input directory. "
            f"Expected exactly one of: {expected}."
        )
    if len(candidates) > 1:
        raise InputDataError(
            "Multiple study boundary vector files were found. "
            "Leave exactly one boundary file in input/: "
            + ", ".join(str(path) for path in candidates)
        )
    return candidates[0]


def discover_input_rasters(input_dir: Path) -> tuple[dict[str, Path], InputVariant]:
    """Find one complete input variant in the input directory."""

    raw_paths = _existing_paths(input_dir, RAW_BAND_FILES)
    ndvi_paths = _existing_paths(input_dir, READY_NDVI_FILES)

    if _is_complete(raw_paths, RAW_BAND_FILES):
        return raw_paths, InputVariant.RAW_BANDS

    if _is_complete(ndvi_paths, READY_NDVI_FILES):
        return ndvi_paths, InputVariant.READY_NDVI

    flexible_ndvi_paths = _discover_flexible_ndvi_pair(input_dir)
    if flexible_ndvi_paths is not None:
        return flexible_ndvi_paths, InputVariant.READY_NDVI

    missing_raw = _missing_files(input_dir, RAW_BAND_FILES)
    missing_ndvi = _missing_files(input_dir, READY_NDVI_FILES)
    raise InputDataError(
        "No complete input data variant was found. "
        f"Missing files for raw Sentinel-2 bands: {', '.join(missing_raw)}. "
        f"Missing files for ready NDVI rasters: {', '.join(missing_ndvi)}."
    )


def _discover_flexible_ndvi_pair(input_dir: Path) -> dict[str, Path] | None:
    """Detect exactly two ready NDVI rasters named ndvi*.tif."""

    candidates = sorted(input_dir.glob("ndvi*.tif"), key=lambda path: path.name.lower())
    if not candidates:
        return None
    if len(candidates) != 2:
        raise InputDataError(
            "Ready NDVI input must contain exactly two ndvi*.tif files when "
            "standard names ndvi2015.tif and ndvi2025.tif are not used. Found: "
            + ", ".join(str(path) for path in candidates)
        )

    by_year = _map_ndvi_candidates_by_year(candidates)
    if by_year is not None:
        return by_year

    return {
        f"ndvi_{BASE_YEAR}": candidates[0],
        f"ndvi_{TARGET_YEAR}": candidates[1],
    }


def _map_ndvi_candidates_by_year(candidates: list[Path]) -> dict[str, Path] | None:
    year_map: dict[int, Path] = {}
    for path in candidates:
        match = re.search(r"(20\d{2}|19\d{2})", path.stem)
        if match:
            year_map[int(match.group(1))] = path
    if BASE_YEAR in year_map and TARGET_YEAR in year_map:
        return {
            f"ndvi_{BASE_YEAR}": year_map[BASE_YEAR],
            f"ndvi_{TARGET_YEAR}": year_map[TARGET_YEAR],
        }
    return None


def validate_raster(path: Path, boundary: gpd.GeoDataFrame) -> RasterMetadata:
    """Validate CRS, pixel size, transform, range, masks and boundary overlap."""

    if not path.exists():
        raise InputDataError(f"Raster file is missing: {path}")

    try:
        with rasterio.open(path) as dataset:
            if dataset.count != 1:
                raise InputDataError(
                    f"Raster must contain exactly one band: {path}. "
                    f"Found bands: {dataset.count}."
                )
            if dataset.crs is None:
                raise InputDataError(f"Raster has no CRS: {path}")
            if dataset.width <= 0 or dataset.height <= 0:
                raise InputDataError(f"Raster has invalid dimensions: {path}")

            pixel_size_x = float(dataset.transform.a)
            pixel_size_y = float(dataset.transform.e)
            warnings = []
            if dataset.nodata is None:
                warnings.append("NoData value is not set")
            if abs(pixel_size_x) == 0 or abs(pixel_size_y) == 0:
                raise InputDataError(f"Raster has invalid pixel size: {path}")
            _validate_transform(path, dataset.transform)

            scale = _band_value(dataset.scales, 0, 1.0)
            offset = _band_value(dataset.offsets, 0, 0.0)
            if scale == 0 or not np.isfinite(scale) or not np.isfinite(offset):
                raise InputDataError(
                    f"Raster has invalid scale/offset metadata: {path}. "
                    f"scale={scale}, offset={offset}."
                )

            band = dataset.read(1, masked=True)
            data = np.asarray(band.data, dtype="float64")
            mask = np.ma.getmaskarray(band) | ~np.isfinite(data)
            if dataset.nodata is not None:
                mask |= np.isclose(data, float(dataset.nodata))
            valid_values = data[~mask]
            if valid_values.size == 0:
                raise InputDataError(f"Raster has no valid pixels: {path}")

            intersects_boundary = _raster_intersects_boundary(dataset, boundary)
            if not intersects_boundary:
                raise InputDataError(
                    f"Raster does not intersect the study boundary: {path}"
                )

            return RasterMetadata(
                path=path,
                crs=str(dataset.crs),
                width=dataset.width,
                height=dataset.height,
                pixel_size_x=pixel_size_x,
                pixel_size_y=pixel_size_y,
                dtype=dataset.dtypes[0],
                nodata=dataset.nodata,
                scale=scale,
                offset=offset,
                band_count=dataset.count,
                transform=tuple(float(value) for value in dataset.transform[:6]),
                bounds=tuple(float(value) for value in dataset.bounds),
                valid_min=float(np.min(valid_values)),
                valid_max=float(np.max(valid_values)),
                intersects_boundary=intersects_boundary,
                warnings=tuple(warnings),
            )
    except rasterio.errors.RasterioIOError as exc:
        raise InputDataError(f"Cannot open raster file {path}: {exc}") from exc


def validate_boundary(
    path: Path,
    boundary: gpd.GeoDataFrame | None = None,
) -> BoundaryMetadata:
    """Validate the vector boundary used for clipping rasters."""

    boundary = boundary if boundary is not None else _read_boundary(path)

    if boundary.empty:
        raise InputDataError(f"Boundary file has no features: {path}")
    if boundary.crs is None:
        raise InputDataError(f"Boundary file has no CRS: {path}")

    valid_geometry = boundary.geometry.notna() & ~boundary.geometry.is_empty
    if not valid_geometry.any():
        raise InputDataError(f"Boundary file has no valid geometries: {path}")

    invalid_count = int((~boundary.loc[valid_geometry].geometry.is_valid).sum())
    if invalid_count:
        raise InputDataError(
            f"Boundary file contains invalid geometries: {path}. "
            f"Invalid feature count: {invalid_count}."
        )

    geometry_types = set(boundary.loc[valid_geometry].geometry.geom_type)
    allowed_types = {"Polygon", "MultiPolygon"}
    if not geometry_types.issubset(allowed_types):
        raise InputDataError(
            f"Boundary must contain only polygon geometries: {path}. "
            f"Found geometry types: {', '.join(sorted(geometry_types))}."
        )

    bounds = boundary.total_bounds
    if not np.isfinite(bounds).all():
        raise InputDataError(f"Boundary has invalid bounds: {path}")

    area_boundary = boundary.loc[valid_geometry]
    if area_boundary.crs and not area_boundary.crs.is_projected:
        area_boundary = area_boundary.to_crs(ALIGNMENT_TARGET_CRS)
    area = float(area_boundary.geometry.area.sum())
    if area <= 0:
        raise InputDataError(f"Boundary has zero area: {path}")

    return BoundaryMetadata(
        path=path,
        crs=str(boundary.crs),
        feature_count=int(valid_geometry.sum()),
        bounds=tuple(float(value) for value in bounds),
        geometry_type=", ".join(sorted(geometry_types)),
        area=area,
    )


def _read_boundary(path: Path) -> gpd.GeoDataFrame:
    if not path.exists():
        raise InputDataError(
            f"Boundary file is missing: {path}."
        )
    _validate_shapefile_parts(path)
    try:
        return gpd.read_file(path)
    except Exception as exc:
        raise InputDataError(f"Cannot read boundary file {path}: {exc}") from exc


def _validate_shapefile_parts(path: Path) -> None:
    if path.suffix.lower() != ".shp":
        return
    required_suffixes = (".shp", ".shx", ".dbf", ".prj")
    missing = [
        str(path.with_suffix(suffix))
        for suffix in required_suffixes
        if not path.with_suffix(suffix).exists()
    ]
    if missing:
        raise InputDataError(
            "Boundary shapefile is incomplete. Missing required files: "
            + ", ".join(missing)
        )


def _validate_transform(path: Path, transform: rasterio.Affine) -> None:
    coefficients = tuple(float(value) for value in transform[:6])
    if not np.isfinite(coefficients).all():
        raise InputDataError(f"Raster has non-finite affine transform: {path}")
    determinant = transform.a * transform.e - transform.b * transform.d
    if determinant == 0:
        raise InputDataError(f"Raster has non-invertible affine transform: {path}")


def _raster_intersects_boundary(
    dataset: rasterio.io.DatasetReader,
    boundary: gpd.GeoDataFrame,
) -> bool:
    boundary_in_raster_crs = boundary
    if boundary.crs != dataset.crs:
        boundary_in_raster_crs = boundary.to_crs(dataset.crs)
    raster_box = box(*dataset.bounds)
    return bool(boundary_in_raster_crs.intersects(raster_box).any())


def _band_value(
    values: tuple[float | None, ...] | None,
    index: int,
    default: float,
) -> float:
    if not values or index >= len(values) or values[index] is None:
        return default
    return float(values[index])


def _existing_paths(
    input_dir: Path,
    expected_files: Mapping[str, str],
) -> dict[str, Path]:
    return {
        label: input_dir / filename
        for label, filename in expected_files.items()
        if (input_dir / filename).exists()
    }


def _is_complete(
    discovered: Mapping[str, Path],
    expected_files: Mapping[str, str],
) -> bool:
    return set(discovered) == set(expected_files)


def _missing_files(input_dir: Path, expected_files: Mapping[str, str]) -> list[str]:
    return [
        str(input_dir / filename)
        for filename in expected_files.values()
        if not (input_dir / filename).exists()
    ]
