"""Data-driven Markov and spatial NDVI forecast for urban environments."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio import features
from scipy import ndimage

from config import (
    CLASS_NODATA,
    FORECAST_CLIP_MAX,
    FORECAST_CLIP_MIN,
    NDVI_CLASS_THRESHOLDS,
    RASTER_NODATA,
)
from modules.utils import (
    assert_same_grid,
    read_masked_raster,
    read_valid_mask,
    write_single_band_geotiff,
)


LOGGER = logging.getLogger("ndvi_monitor")

SCENARIO_NAMES = (
    "new_development",
    "stable_urban",
    "stable_green",
    "degradation",
    "other",
)
KMEANS_MAX_ITERATIONS = 100
KMEANS_TOLERANCE = 1.0e-6
PIXEL_CLUSTER_SAMPLE_LIMIT = 1200
PREDICT_CHUNK_SIZE = 200_000
EPSILON = np.finfo("float32").eps
INFLUENCE_AREA_WEIGHT = 0.35
INFLUENCE_DELTA_WEIGHT = 0.30
INFLUENCE_NEIGHBOUR_WEIGHT = 0.15
INFLUENCE_COMPACTNESS_WEIGHT = 0.10
INFLUENCE_CONTEXT_DELTA_WEIGHT = 0.10
SCORE_INFLUENCE_WEIGHT = 0.45
SCORE_REMAINING_CAPACITY_WEIGHT = 0.25
SCORE_NEIGHBOURHOOD_LOSS_WEIGHT = 0.20
SCORE_LOCAL_VARIABILITY_WEIGHT = 0.10
LOCAL_ANALYSIS_WINDOW_SIZE = 5
EXHAUSTED_DELTA_LIMIT = 0.35
EXHAUSTED_NDVI_LIMIT = 0.18
URBAN_GREENING_MIN_GAIN = 0.03
URBAN_GREENING_MAX_GAIN = 0.08


@dataclass(frozen=True)
class RasterGrid:
    """Minimal raster grid description required for spatial modelling."""

    shape: tuple[int, int]
    transform: Any
    crs: Any

    @property
    def pixel_area(self) -> float:
        return float(
            abs(
                self.transform.a * self.transform.e
                - self.transform.b * self.transform.d
            )
        )

    @property
    def pixel_size(self) -> float:
        return float((abs(self.transform.a) + abs(self.transform.e)) / 2.0)


@dataclass(frozen=True)
class ClassInterval:
    """One NDVI class interval used by the Markov forecast."""

    class_id: int
    lower: float | None
    upper: float | None
    lower_inclusive: bool
    upper_inclusive: bool

    def contains(self, values: np.ndarray) -> np.ndarray:
        """Return values belonging to this interval."""

        mask = np.ones(values.shape, dtype=bool)
        if self.lower is not None:
            lower_mask = values >= self.lower if self.lower_inclusive else values > self.lower
            mask &= lower_mask
        if self.upper is not None:
            upper_mask = values <= self.upper if self.upper_inclusive else values < self.upper
            mask &= upper_mask
        return mask


@dataclass(frozen=True)
class MarkovTransition:
    """A Markov-derived quota for one NDVI class transition."""

    from_class: int
    to_class: int
    probability: float
    requested_pixels: int


@dataclass(frozen=True)
class KMeansResult:
    """Deterministic KMeans output with an approximate silhouette score."""

    labels: np.ndarray
    centroids: np.ndarray
    feature_names: tuple[str, ...]
    k: int
    silhouette: float | None


@dataclass(frozen=True)
class PolygonForecastModel:
    """Change polygons enriched with features, clusters and scenarios."""

    polygons: gpd.GeoDataFrame
    features: pd.DataFrame
    cluster_stats: pd.DataFrame
    clustering: KMeansResult | None


@dataclass(frozen=True)
class ScenarioReferences:
    """Data-derived NDVI references used for scenario-specific changes."""

    stable_urban_ndvi: float
    stable_green_ndvi: float
    recovery_ceiling: float


@dataclass(frozen=True)
class DynamicsScenarios:
    """Mutually exclusive scenario masks for valid NDVI pixels."""

    new_development: np.ndarray
    stable_urban: np.ndarray
    stable_green: np.ndarray
    degradation: np.ndarray
    other: np.ndarray

    def items(self) -> tuple[tuple[str, np.ndarray], ...]:
        """Return scenario names with their masks in reporting order."""

        return (
            ("new_development", self.new_development),
            ("stable_urban", self.stable_urban),
            ("stable_green", self.stable_green),
            ("degradation", self.degradation),
            ("other", self.other),
        )


def forecast_ndvi_linear(
    ndvi_start_path: Path,
    ndvi_end_path: Path,
    start_year: int,
    end_year: int,
    forecast_year: int,
    output_path: Path,
) -> Path:
    """Create a scenario forecast while preserving the historical API name.

    The function no longer performs linear extrapolation. It builds an
    automatic model from observed NDVI dynamics, clusters change polygons and
    stable pixels, interprets those clusters as urban-development scenarios,
    and uses the Markov transition matrix only as a statistical limit on class
    transitions.
    """

    assert_same_grid(ndvi_start_path, ndvi_end_path)
    raster_dir = output_path.parent
    tables_dir = raster_dir.parent / "tables"
    transition_path = tables_dir / "transition_matrix.xlsx"
    polygons_path = raster_dir / "change_polygons.gpkg"
    delta_path = raster_dir / "delta_ndvi.tif"
    valid_mask_path = raster_dir / "common_valid_mask.tif"
    class_start_path = raster_dir / f"ndvi_classes_{start_year}.tif"
    class_end_path = raster_dir / f"ndvi_classes_{end_year}.tif"

    _validate_forecast_inputs(
        ndvi_end_path,
        delta_path,
        valid_mask_path,
        class_start_path,
        class_end_path,
        polygons_path,
    )
    _assert_forecast_grid(
        ndvi_end_path,
        delta_path,
        valid_mask_path,
        class_start_path,
        class_end_path,
    )

    ndvi_2015, _ = read_masked_raster(ndvi_start_path)
    ndvi_2025, profile = read_masked_raster(ndvi_end_path)
    delta_ndvi, _ = read_masked_raster(delta_path)
    valid_mask = read_valid_mask(valid_mask_path)
    classes_2015 = _read_class_raster(class_start_path)
    classes_2025 = _read_class_raster(class_end_path)
    grid = _read_raster_grid(ndvi_end_path)

    polygons = load_and_filter_polygons(polygons_path)
    polygon_model = analyze_change_polygons(
        polygons,
        ndvi_2015,
        ndvi_2025,
        delta_ndvi,
        classes_2025,
        valid_mask,
        grid,
    )

    with rasterio.open(ndvi_end_path) as dataset:
        polygon_scenario_masks = rasterize_polygon_scenarios(
            polygon_model.polygons,
            dataset,
        )
        # Используем все полигоны изменений; каждый полигон уже получил
        # собственный вес влияния на этапе анализа.
        forecast_polygons = polygon_model.polygons.copy()
        influence_surface = create_spatial_influence(
            forecast_polygons,
            dataset,
            delta_ndvi,
            valid_mask,
        )
        influence_mask = create_buffer_mask(forecast_polygons, dataset)

    scenarios = classify_land_dynamics(
        ndvi_2015,
        ndvi_2025,
        delta_ndvi,
        classes_2015,
        classes_2025,
        valid_mask,
        polygon_scenario_masks,
    )
    _log_scenario_statistics(scenarios, profile)

    transition_counts = load_transition_counts(
        transition_path,
        class_start_path,
        class_end_path,
        valid_mask_path,
    )
    transition_probabilities = calculate_transition_probabilities(transition_counts)
    _log_transition_probabilities(transition_probabilities, start_year, end_year, forecast_year)

    references = _calculate_scenario_references(ndvi_2025, scenarios, valid_mask)
    mean_loss = calculate_mean_loss(
        forecast_polygons,
        delta_path=delta_path,
        valid_mask=valid_mask,
    )
    degradation_base_loss = _degradation_loss_surface(
        delta_ndvi,
        influence_surface,
        valid_mask,
    )
    (
        degradation_selection,
        degradation_transitions,
        degradation_applied_loss,
    ) = select_markov_degradation_pixels(
        classes_2025,
        ndvi_2025,
        delta_ndvi,
        valid_mask,
        influence_mask,
        influence_surface,
        scenarios.degradation,
        transition_probabilities,
        degradation_base_loss,
    )
    forecast_after_degradation = apply_spatial_change(
        ndvi_2025,
        influence_mask,
        degradation_applied_loss,
        selection_mask=degradation_selection,
        valid_mask=valid_mask,
        direction="decrease",
    )

    recovery_capacity = _development_recovery_surface(
        ndvi_2015,
        ndvi_2025,
        delta_ndvi,
        references,
    )
    (
        recovery_selection,
        recovery_transitions,
        recovery_applied_gain,
    ) = select_markov_recovery_pixels(
        classes_2025,
        ndvi_2025,
        scenarios.new_development,
        valid_mask,
        delta_ndvi,
        recovery_capacity,
        transition_probabilities,
    )
    forecast_masked = np.ma.array(
        forecast_after_degradation,
        mask=forecast_after_degradation == RASTER_NODATA,
    )
    forecast_after_recovery = apply_spatial_change(
        forecast_masked,
        scenarios.new_development.astype("uint8"),
        recovery_applied_gain,
        selection_mask=recovery_selection,
        valid_mask=valid_mask,
        direction="increase",
    )
    stable_greening_mask, stable_greening_gain = _stable_urban_greening_surface(
        forecast_after_recovery,
        ndvi_2025,
        scenarios,
        references,
        valid_mask,
    )
    forecast_recovery_masked = np.ma.array(
        forecast_after_recovery,
        mask=forecast_after_recovery == RASTER_NODATA,
    )
    final_forecast = apply_spatial_change(
        forecast_recovery_masked,
        stable_greening_mask.astype("uint8"),
        stable_greening_gain,
        selection_mask=stable_greening_mask,
        valid_mask=valid_mask,
        direction="increase",
    )
    _log_stable_urban_greening(stable_greening_mask, stable_greening_gain, profile)

    _log_forecast_summary(
        polygon_model,
        degradation_transitions,
        recovery_transitions,
        degradation_selection,
        recovery_selection,
        stable_greening_mask,
        influence_surface,
        profile,
        references,
        mean_loss,
        ndvi_2025,
        final_forecast,
        valid_mask,
    )
    return write_single_band_geotiff(
        output_path,
        final_forecast,
        profile,
        RASTER_NODATA,
        "float32",
    )


def classify_land_dynamics(
    ndvi_start: np.ma.MaskedArray,
    ndvi_end: np.ma.MaskedArray,
    delta_ndvi: np.ma.MaskedArray,
    classes_start: np.ndarray,
    classes_end: np.ndarray,
    valid_mask: np.ndarray,
    polygon_scenarios: Mapping[str, np.ndarray] | np.ndarray | None = None,
) -> DynamicsScenarios:
    """Classify valid pixels by automatically interpreted NDVI dynamics.

    Change-polygon clusters provide masks for new development and observed
    degradation. Remaining valid pixels are clustered from NDVI, Delta NDVI
    and class-change features; stable urban and stable green clusters are
    identified by their relative position in the data distribution.
    """

    valid = (
        valid_mask
        & ~np.ma.getmaskarray(ndvi_start)
        & ~np.ma.getmaskarray(ndvi_end)
        & ~np.ma.getmaskarray(delta_ndvi)
        & np.isfinite(ndvi_start.data)
        & np.isfinite(ndvi_end.data)
        & np.isfinite(delta_ndvi.data)
        & (classes_start != CLASS_NODATA)
        & (classes_end != CLASS_NODATA)
    )

    polygon_masks = _normalise_polygon_scenario_masks(polygon_scenarios, valid.shape)
    new_development = polygon_masks["new_development"] & valid
    degradation = polygon_masks["degradation"] & valid
    assigned = new_development | degradation
    remaining = valid & ~assigned

    pixel_labels, pixel_stats, pixel_model = _cluster_remaining_pixels(
        ndvi_start,
        ndvi_end,
        delta_ndvi,
        classes_start,
        classes_end,
        remaining,
    )
    pixel_interpretation = _interpret_pixel_clusters(pixel_stats)
    _log_pixel_clusters(pixel_stats, pixel_model, pixel_interpretation)

    stable_urban = np.zeros(valid.shape, dtype=bool)
    stable_green = np.zeros(valid.shape, dtype=bool)
    other = np.zeros(valid.shape, dtype=bool)

    for cluster_id, scenario in pixel_interpretation.items():
        mask = remaining & (pixel_labels == cluster_id)
        if scenario == "stable_urban":
            stable_urban |= mask
        elif scenario == "stable_green":
            stable_green |= mask
        elif scenario == "degradation":
            degradation |= mask
        else:
            other |= mask

    assigned = new_development | stable_urban | stable_green | degradation | other
    other |= valid & ~assigned
    return DynamicsScenarios(
        new_development=new_development,
        stable_urban=stable_urban,
        stable_green=stable_green,
        degradation=degradation,
        other=other,
    )


def load_and_filter_polygons(path: Path) -> gpd.GeoDataFrame:
    """Load change polygons and keep only geometrically usable features."""

    if not path.exists():
        raise FileNotFoundError(f"Change polygon file is missing: {path}")
    polygons = gpd.read_file(path)
    if polygons.empty:
        return polygons
    polygons = polygons[polygons.geometry.notna() & ~polygons.geometry.is_empty].copy()
    polygons = polygons[polygons.geometry.is_valid].copy()
    if polygons.empty:
        LOGGER.warning("Change polygon file contains no valid geometries: %s", path)
    return polygons.reset_index(drop=True)


def create_buffer_mask(gdf: gpd.GeoDataFrame, src: rasterio.DatasetReader) -> np.ndarray:
    """Return a data-derived spatial influence footprint for compatibility.

    The mask is not based on fixed buffer distances. Each polygon receives a
    neighbourhood radius derived from its geometry and the spacing of all
    observed change polygons.
    """

    if gdf.empty:
        return np.zeros((src.height, src.width), dtype="uint8")
    polygons = _to_raster_crs(gdf, src.crs)
    geometries = _adaptive_neighbourhood_geometries(polygons)
    return features.rasterize(
        [(geometry, 1) for geometry in geometries if not geometry.is_empty],
        out_shape=(src.height, src.width),
        transform=src.transform,
        fill=0,
        dtype="uint8",
    )


def calculate_mean_loss(
    gdf: gpd.GeoDataFrame,
    delta_path: Path | None = None,
    valid_mask: np.ndarray | None = None,
) -> float:
    """Calculate mean observed negative Delta NDVI inside change polygons."""

    if gdf.empty:
        return 0.0
    if delta_path is None:
        if "mean_delta_ndvi" in gdf.columns:
            losses = gdf["mean_delta_ndvi"].dropna().to_numpy(dtype="float64")
        elif "delta_ndvi" in gdf.columns:
            losses = gdf["delta_ndvi"].dropna().to_numpy(dtype="float64")
        else:
            return 0.0
        negatives = losses[losses < 0]
        return float(abs(np.mean(negatives))) if negatives.size else 0.0

    delta_values, _ = read_masked_raster(delta_path)
    with rasterio.open(delta_path) as dataset:
        polygon_mask = create_polygon_mask(gdf, dataset) == 1
    valid = ~np.ma.getmaskarray(delta_values) & np.isfinite(delta_values.data)
    if valid_mask is not None:
        valid &= valid_mask
    losses = delta_values.data[polygon_mask & valid & (delta_values.data < 0)]
    return float(abs(np.mean(losses))) if losses.size else 0.0


def apply_spatial_change(
    ndvi_data: np.ma.MaskedArray,
    mask: np.ndarray,
    loss: float | np.ndarray,
    selection_mask: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
    direction: str = "decrease",
) -> np.ndarray:
    """Apply a scenario-specific NDVI change only to selected valid pixels."""

    if direction not in {"decrease", "increase"}:
        raise ValueError(f"Unsupported spatial change direction: {direction}")
    forecast = np.full(ndvi_data.shape, RASTER_NODATA, dtype="float32")
    valid = ~np.ma.getmaskarray(ndvi_data) & np.isfinite(ndvi_data.data)
    if valid_mask is not None:
        valid &= valid_mask
    forecast[valid] = ndvi_data.data[valid].astype("float32", copy=False)

    target = (mask == 1) & valid
    if selection_mask is not None:
        target &= selection_mask
    change_surface = _change_surface(loss, ndvi_data.shape)
    if direction == "decrease":
        forecast[target] -= change_surface[target]
    else:
        forecast[target] += change_surface[target]
    forecast[valid] = np.clip(forecast[valid], FORECAST_CLIP_MIN, FORECAST_CLIP_MAX)
    return forecast


def analyze_change_polygons(
    polygons: gpd.GeoDataFrame,
    ndvi_start: np.ma.MaskedArray,
    ndvi_end: np.ma.MaskedArray,
    delta_ndvi: np.ma.MaskedArray,
    classes_end: np.ndarray,
    valid_mask: np.ndarray,
    grid: RasterGrid,
) -> PolygonForecastModel:
    """Build polygon features, run automatic clustering and assign scenarios."""

    if polygons.empty:
        empty = polygons.copy()
        empty["forecast_cluster"] = pd.Series(dtype="int64")
        empty["forecast_scenario"] = pd.Series(dtype="object")
        empty["forecast_weight"] = pd.Series(dtype="float64")
        return PolygonForecastModel(empty, pd.DataFrame(), pd.DataFrame(), None)

    polygons = _to_grid_crs(polygons, grid).reset_index(drop=True)
    label_raster = _rasterize_polygon_labels(polygons, grid)
    geometry_features = _geometry_features(polygons)
    raster_features = _polygon_raster_features(
        label_raster,
        ndvi_start,
        ndvi_end,
        delta_ndvi,
        classes_end,
        valid_mask,
        len(polygons),
    )
    context_features = _polygon_context_features(
        polygons,
        ndvi_end,
        delta_ndvi,
        valid_mask,
        grid,
    )
    feature_table = pd.concat(
        [geometry_features, raster_features, context_features],
        axis=1,
    )
    feature_table = feature_table.loc[:, ~feature_table.columns.duplicated()]
    feature_table = _fill_feature_table(feature_table)
    clustering = _cluster_feature_table(feature_table)
    feature_table["forecast_cluster"] = clustering.labels
    feature_table["forecast_weight"] = _calculate_forecast_weights(feature_table)
    cluster_stats = _polygon_cluster_stats(feature_table)
    scenario_by_cluster = _interpret_polygon_clusters(cluster_stats)
    feature_table["forecast_scenario"] = feature_table["forecast_cluster"].map(
        scenario_by_cluster
    ).fillna("other")

    polygons = polygons.copy()
    polygons["forecast_cluster"] = feature_table["forecast_cluster"].to_numpy()
    polygons["forecast_scenario"] = feature_table["forecast_scenario"].to_numpy()
    polygons["forecast_weight"] = feature_table["forecast_weight"].to_numpy()
    for column in (
        "mean_ndvi_2015",
        "mean_ndvi_2025",
        "mean_delta_ndvi",
        "area_m2",
        "compactness",
        "neighbour_polygon_count",
        "surrounding_mean_delta_ndvi",
    ):
        if column in feature_table:
            polygons[column] = feature_table[column].to_numpy()

    _log_polygon_model(cluster_stats, clustering, scenario_by_cluster)
    return PolygonForecastModel(polygons, feature_table, cluster_stats, clustering)


def rasterize_polygon_scenarios(
    polygons: gpd.GeoDataFrame,
    src: rasterio.DatasetReader,
) -> dict[str, np.ndarray]:
    """Rasterize automatically interpreted polygon scenarios."""

    masks = {
        name: np.zeros((src.height, src.width), dtype=bool)
        for name in SCENARIO_NAMES
    }
    if polygons.empty or "forecast_scenario" not in polygons.columns:
        return masks

    polygons = _to_raster_crs(polygons, src.crs)
    for scenario in SCENARIO_NAMES:
        geometries = polygons.loc[
            polygons["forecast_scenario"] == scenario,
            "geometry",
        ]
        if geometries.empty:
            continue
        masks[scenario] = features.rasterize(
            [(geometry, 1) for geometry in geometries],
            out_shape=(src.height, src.width),
            transform=src.transform,
            fill=0,
            dtype="uint8",
        ).astype(bool)
    return masks


def create_spatial_influence(
    gdf: gpd.GeoDataFrame,
    src: rasterio.DatasetReader,
    delta_ndvi: np.ma.MaskedArray | None = None,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Create a weighted potential surface from all observed change polygons.

    Pixel influence is calculated independently for every polygon as
    ``polygon_weight * distance_decay`` and overlapping polygons are combined
    by maximum value. The optional raster arguments are kept for API
    compatibility with the previous spatial algorithm.
    """

    if gdf.empty:
        return np.zeros((src.height, src.width), dtype="float32")

    polygons = _to_raster_crs(gdf, src.crs).reset_index(drop=True)
    if "forecast_weight" not in polygons.columns:
        polygons = polygons.copy()
        polygons["forecast_weight"] = _calculate_forecast_weights(polygons)
    pixel_size = float((abs(src.transform.a) + abs(src.transform.e)) / 2.0)
    context_distance = _adaptive_context_distance(polygons)
    influence = np.zeros((src.height, src.width), dtype="float32")

    for polygon in polygons.itertuples():
        geometry = polygon.geometry
        if geometry.is_empty:
            continue
        weight = float(getattr(polygon, "forecast_weight", 0.0))
        if weight <= EPSILON:
            continue
        max_distance = _adaptive_neighbourhood_distance(
            geometry,
            context_distance,
        )
        max_pixels = max(max_distance / max(pixel_size, EPSILON), 1.0)
        seed_mask = features.rasterize(
            [(geometry, 1)],
            out_shape=(src.height, src.width),
            transform=src.transform,
            fill=0,
            dtype="uint8",
        ).astype(bool)
        if not np.any(seed_mask):
            continue
        buffer_geometry = geometry.buffer(max_distance)
        buffer_mask = features.rasterize(
            [(buffer_geometry, 1)],
            out_shape=(src.height, src.width),
            transform=src.transform,
            fill=0,
            dtype="uint8",
        ).astype(bool)
        if not np.any(buffer_mask):
            continue
        distance = ndimage.distance_transform_edt(~seed_mask)
        distance_decay = np.clip(1.0 - distance / max_pixels, 0.0, 1.0)
        pixel_influence = (weight * distance_decay).astype("float32")
        pixel_influence[~buffer_mask] = 0.0
        influence = np.maximum(influence, pixel_influence)
    if valid_mask is not None:
        influence[~valid_mask] = 0.0

    LOGGER.info(
        "Spatial influence surface: polygons=%s, active_pixels=%s, max=%.6f, mean_active=%.6f",
        len(polygons),
        int(np.count_nonzero(influence > 0)),
        float(np.max(influence)) if influence.size else 0.0,
        float(np.mean(influence[influence > 0])) if np.any(influence > 0) else 0.0,
    )
    return influence.astype("float32")


def load_transition_counts(
    transition_path: Path,
    class_start_path: Path,
    class_end_path: Path,
    valid_mask_path: Path,
) -> pd.DataFrame:
    """Load transition counts from Excel or calculate them from class rasters."""

    required_columns = {"from_class", "to_class", "pixel_count"}
    if transition_path.exists():
        table = pd.read_excel(transition_path)
        if required_columns.issubset(table.columns):
            LOGGER.info("Loaded Markov transition counts from %s", transition_path)
            return table.loc[:, ["from_class", "to_class", "pixel_count"]].copy()
        LOGGER.warning(
            "Transition matrix %s lacks required columns; recalculating it.",
            transition_path,
        )

    assert_same_grid(class_start_path, class_end_path)
    assert_same_grid(class_start_path, valid_mask_path)
    start_classes = _read_class_raster(class_start_path)
    end_classes = _read_class_raster(class_end_path)
    valid = read_valid_mask(valid_mask_path)
    valid &= (start_classes != CLASS_NODATA) & (end_classes != CLASS_NODATA)
    rows = []
    for from_class in _class_ids():
        for to_class in _class_ids():
            rows.append(
                {
                    "from_class": from_class,
                    "to_class": to_class,
                    "pixel_count": int(
                        np.count_nonzero(
                            valid
                            & (start_classes == from_class)
                            & (end_classes == to_class)
                        )
                    ),
                }
            )
    LOGGER.info("Calculated Markov transition counts from NDVI class rasters.")
    return pd.DataFrame(rows)


def calculate_transition_probabilities(counts: pd.DataFrame) -> pd.DataFrame:
    """Normalize transition counts into a row-stochastic Markov matrix."""

    required_columns = {"from_class", "to_class", "pixel_count"}
    if not required_columns.issubset(counts.columns):
        raise ValueError("Transition counts lack required columns.")
    class_ids = _class_ids()
    matrix = pd.DataFrame(0.0, index=class_ids, columns=class_ids)
    for row in counts.itertuples(index=False):
        from_class = int(row.from_class)
        to_class = int(row.to_class)
        if from_class in matrix.index and to_class in matrix.columns:
            matrix.loc[from_class, to_class] += float(row.pixel_count)

    probabilities = pd.DataFrame(0.0, index=class_ids, columns=class_ids)
    for class_id in class_ids:
        total = float(matrix.loc[class_id].sum())
        if total > 0:
            probabilities.loc[class_id] = matrix.loc[class_id] / total
        else:
            probabilities.loc[class_id, class_id] = 1.0
    probabilities.index.name = "from_class"
    probabilities.columns.name = "to_class"
    return probabilities


def select_markov_degradation_pixels(
    current_classes: np.ndarray,
    ndvi_values: np.ma.MaskedArray,
    delta_values: np.ma.MaskedArray,
    valid_mask: np.ndarray,
    influence_mask: np.ndarray,
    influence_surface: np.ndarray,
    degradation_mask: np.ndarray,
    probabilities: pd.DataFrame,
    base_loss_surface: np.ndarray,
) -> tuple[np.ndarray, list[MarkovTransition], np.ndarray]:
    """Select localized degradation pixels under Markov downward quotas."""

    valid = (
        valid_mask
        & degradation_mask
        & (influence_mask == 1)
        & ~np.ma.getmaskarray(ndvi_values)
        & ~np.ma.getmaskarray(delta_values)
        & np.isfinite(ndvi_values.data)
        & np.isfinite(delta_values.data)
        & (influence_surface > 0)
    )
    selection = np.zeros(current_classes.shape, dtype=bool)
    applied_loss = np.zeros(current_classes.shape, dtype="float32")
    transitions: list[MarkovTransition] = []
    abs_delta = np.abs(delta_values.data).astype("float32")
    remaining_capacity = 1.0 - _normalise_positive(abs_delta)
    local_valid = (
        valid_mask
        & ~np.ma.getmaskarray(ndvi_values)
        & ~np.ma.getmaskarray(delta_values)
        & np.isfinite(ndvi_values.data)
        & np.isfinite(delta_values.data)
    )
    neighbourhood_loss = _local_mean_surface(
        np.abs(np.minimum(delta_values.data, 0.0)).astype("float32"),
        local_valid,
        LOCAL_ANALYSIS_WINDOW_SIZE,
    )
    local_variability = _local_std_surface(
        ndvi_values.data.astype("float32"),
        local_valid,
        LOCAL_ANALYSIS_WINDOW_SIZE,
    )
    score_surface = (
        SCORE_INFLUENCE_WEIGHT * _normalise_positive(influence_surface)
        + SCORE_REMAINING_CAPACITY_WEIGHT * remaining_capacity
        + SCORE_NEIGHBOURHOOD_LOSS_WEIGHT * _normalise_positive(neighbourhood_loss)
        + SCORE_LOCAL_VARIABILITY_WEIGHT * _normalise_positive(local_variability)
    )
    exhausted_urban = (
        (abs_delta > EXHAUSTED_DELTA_LIMIT)
        & (ndvi_values.data < EXHAUSTED_NDVI_LIMIT)
    )
    score_surface[~local_valid] = 0.0

    for from_class in _class_ids():
        current_pixels = int(np.count_nonzero(valid & (current_classes == from_class)))
        if current_pixels == 0:
            continue
        for to_class in _class_ids():
            if to_class >= from_class:
                continue
            probability = float(probabilities.loc[from_class, to_class])
            requested = int(np.rint(current_pixels * probability))
            if requested == 0:
                continue
            transitions.append(MarkovTransition(from_class, to_class, probability, requested))
            candidates = valid & ~selection & (current_classes == from_class)
            candidates &= ~exhausted_urban
            selected_indices = _select_by_score(selection, candidates, score_surface, requested)
            if selected_indices.size:
                target_value = _class_target_value(to_class, "decrease")
                current_ndvi = ndvi_values.data.ravel()[selected_indices]
                loss_needed = np.maximum(current_ndvi - target_value, 0.0)
                observed_loss = base_loss_surface.ravel()[selected_indices]
                applied_loss.ravel()[selected_indices] = np.maximum(
                    observed_loss,
                    loss_needed,
                )
            LOGGER.info(
                "Degradation Markov %s->%s: probability=%.6f, requested=%s, selected=%s",
                from_class,
                to_class,
                probability,
                requested,
                int(selected_indices.size),
            )
    return selection, transitions, applied_loss


def select_markov_recovery_pixels(
    current_classes: np.ndarray,
    ndvi_values: np.ma.MaskedArray,
    development_mask: np.ndarray,
    valid_mask: np.ndarray,
    delta_values: np.ma.MaskedArray,
    recovery_capacity: np.ndarray,
    probabilities: pd.DataFrame,
) -> tuple[np.ndarray, list[MarkovTransition], np.ndarray]:
    """Select limited greening within automatically inferred development."""

    valid = (
        valid_mask
        & development_mask
        & ~np.ma.getmaskarray(ndvi_values)
        & ~np.ma.getmaskarray(delta_values)
        & np.isfinite(ndvi_values.data)
        & np.isfinite(delta_values.data)
        & (recovery_capacity > 0)
    )
    selection = np.zeros(current_classes.shape, dtype=bool)
    applied_gain = np.zeros(current_classes.shape, dtype="float32")
    transitions: list[MarkovTransition] = []
    score_surface = (
        _normalise_positive(recovery_capacity)
        + _normalise_positive(np.abs(np.minimum(delta_values.data, 0.0)))
    )

    for from_class in _class_ids():
        available = int(np.count_nonzero(valid & (current_classes == from_class)))
        if available == 0:
            continue
        for to_class in _class_ids():
            if to_class <= from_class:
                continue
            probability = float(probabilities.loc[from_class, to_class])
            requested = int(np.rint(available * probability))
            if requested == 0:
                continue
            transitions.append(MarkovTransition(from_class, to_class, probability, requested))
            candidates = valid & ~selection & (current_classes == from_class)
            selected_indices = _select_by_score(selection, candidates, score_surface, requested)
            if selected_indices.size:
                target_value = _class_target_value(to_class, "increase")
                current_ndvi = ndvi_values.data.ravel()[selected_indices]
                gain_needed = np.maximum(target_value - current_ndvi, 0.0)
                capacity = recovery_capacity.ravel()[selected_indices]
                applied_gain.ravel()[selected_indices] = np.minimum(
                    capacity,
                    np.maximum(gain_needed, EPSILON),
                )
            LOGGER.info(
                "Development recovery Markov %s->%s: probability=%.6f, requested=%s, selected=%s",
                from_class,
                to_class,
                probability,
                requested,
                int(selected_indices.size),
            )
    return selection, transitions, applied_gain


def create_polygon_mask(gdf: gpd.GeoDataFrame, src: rasterio.DatasetReader) -> np.ndarray:
    """Rasterize polygon interiors onto the current forecast grid."""

    if gdf.empty:
        return np.zeros((src.height, src.width), dtype="uint8")
    polygons = _to_raster_crs(gdf, src.crs)
    return features.rasterize(
        [(geometry, 1) for geometry in polygons.geometry],
        out_shape=(src.height, src.width),
        transform=src.transform,
        fill=0,
        dtype="uint8",
    )


def _read_raster_grid(path: Path) -> RasterGrid:
    with rasterio.open(path) as dataset:
        return RasterGrid(
            shape=(dataset.height, dataset.width),
            transform=dataset.transform,
            crs=dataset.crs,
        )


def _to_grid_crs(gdf: gpd.GeoDataFrame, grid: RasterGrid) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf.copy()
    if gdf.crs is None:
        raise ValueError("Change polygons do not define CRS.")
    return gdf if gdf.crs == grid.crs else gdf.to_crs(grid.crs)


def _to_raster_crs(gdf: gpd.GeoDataFrame, crs: Any) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf.copy()
    if gdf.crs is None:
        raise ValueError("Change polygons do not define CRS.")
    return gdf if gdf.crs == crs else gdf.to_crs(crs)


def _rasterize_polygon_labels(
    polygons: gpd.GeoDataFrame,
    grid: RasterGrid,
) -> np.ndarray:
    return features.rasterize(
        [(geometry, index + 1) for index, geometry in enumerate(polygons.geometry)],
        out_shape=grid.shape,
        transform=grid.transform,
        fill=0,
        dtype="int32",
    )


def _geometry_features(polygons: gpd.GeoDataFrame) -> pd.DataFrame:
    areas = polygons.geometry.area.to_numpy(dtype="float64")
    perimeters = polygons.geometry.length.to_numpy(dtype="float64")
    compactness = np.divide(
        4.0 * np.pi * areas,
        np.maximum(perimeters**2, EPSILON),
    )
    nearest_distances = _nearest_polygon_distances(polygons)
    context_distance = _adaptive_context_distance(polygons)
    neighbour_counts = _neighbour_counts(polygons, context_distance)
    return pd.DataFrame(
        {
            "area_m2": areas,
            "perimeter_m": perimeters,
            "compactness": compactness,
            "eccentricity": [_geometry_eccentricity(geometry) for geometry in polygons.geometry],
            "area_perimeter_ratio": np.divide(
                areas,
                np.maximum(perimeters, EPSILON),
            ),
            "distance_to_nearest_polygon": nearest_distances,
            "neighbour_polygon_count": neighbour_counts,
            "equivalent_radius": np.sqrt(np.maximum(areas, 0.0) / np.pi),
        },
        index=polygons.index,
    )


def _polygon_raster_features(
    label_raster: np.ndarray,
    ndvi_start: np.ma.MaskedArray,
    ndvi_end: np.ma.MaskedArray,
    delta_ndvi: np.ma.MaskedArray,
    classes_end: np.ndarray,
    valid_mask: np.ndarray,
    polygon_count: int,
) -> pd.DataFrame:
    valid = (
        valid_mask
        & (label_raster > 0)
        & ~np.ma.getmaskarray(ndvi_start)
        & ~np.ma.getmaskarray(ndvi_end)
        & ~np.ma.getmaskarray(delta_ndvi)
        & np.isfinite(ndvi_start.data)
        & np.isfinite(ndvi_end.data)
        & np.isfinite(delta_ndvi.data)
    )
    base_index = pd.RangeIndex(polygon_count)
    if not np.any(valid):
        return pd.DataFrame(index=base_index)

    table = pd.DataFrame(
        {
            "polygon_index": label_raster[valid].astype("int64") - 1,
            "ndvi_2015": ndvi_start.data[valid],
            "ndvi_2025": ndvi_end.data[valid],
            "delta_ndvi": delta_ndvi.data[valid],
            "class_2025": classes_end[valid],
        }
    )
    grouped = table.groupby("polygon_index", sort=True)
    stats = pd.DataFrame(index=base_index)
    stats["valid_pixel_count"] = grouped.size().reindex(base_index, fill_value=0)
    for source, prefix in (
        ("ndvi_2015", "ndvi_2015"),
        ("ndvi_2025", "ndvi_2025"),
        ("delta_ndvi", "delta_ndvi"),
    ):
        stats[f"mean_{prefix}"] = grouped[source].mean().reindex(base_index)
        stats[f"median_{prefix}"] = grouped[source].median().reindex(base_index)
        stats[f"std_{prefix}"] = grouped[source].std().reindex(base_index)
        stats[f"min_{prefix}"] = grouped[source].min().reindex(base_index)
        stats[f"max_{prefix}"] = grouped[source].max().reindex(base_index)

    class_counts = (
        table.groupby(["polygon_index", "class_2025"]).size().unstack(fill_value=0)
    )
    for class_id in _class_ids():
        if class_id in class_counts:
            counts = class_counts[class_id]
        else:
            counts = pd.Series(0, index=base_index)
        stats[f"class_{class_id}_share"] = (
            counts.reindex(base_index, fill_value=0) / stats["valid_pixel_count"].replace(0, np.nan)
        )
    return stats


def _polygon_context_features(
    polygons: gpd.GeoDataFrame,
    ndvi_end: np.ma.MaskedArray,
    delta_ndvi: np.ma.MaskedArray,
    valid_mask: np.ndarray,
    grid: RasterGrid,
) -> pd.DataFrame:
    global_valid = (
        valid_mask
        & ~np.ma.getmaskarray(ndvi_end)
        & ~np.ma.getmaskarray(delta_ndvi)
        & np.isfinite(ndvi_end.data)
        & np.isfinite(delta_ndvi.data)
    )
    valid_ndvi = ndvi_end.data[global_valid]
    green_reference = float(np.nanmedian(valid_ndvi)) if valid_ndvi.size else 0.0
    context_distance = _adaptive_context_distance(polygons)
    rows: list[dict[str, float]] = []

    for geometry in polygons.geometry:
        distance = _adaptive_neighbourhood_distance(geometry, context_distance)
        ring = geometry.buffer(distance).difference(geometry)
        if ring.is_empty:
            rows.append(_empty_context_row())
            continue
        ring_mask = features.rasterize(
            [(ring, 1)],
            out_shape=grid.shape,
            transform=grid.transform,
            fill=0,
            dtype="uint8",
        ).astype(bool)
        context_valid = global_valid & ring_mask
        if not np.any(context_valid):
            rows.append(_empty_context_row())
            continue
        green_pixels = int(np.count_nonzero(context_valid & (ndvi_end.data >= green_reference)))
        green_area = green_pixels * grid.pixel_area
        rows.append(
            {
                "surrounding_mean_ndvi": float(np.mean(ndvi_end.data[context_valid])),
                "surrounding_mean_delta_ndvi": float(np.mean(delta_ndvi.data[context_valid])),
                "surrounding_green_area_ratio": float(
                    geometry.area / max(green_area, grid.pixel_area)
                ),
            }
        )
    return pd.DataFrame(rows, index=polygons.index)


def _empty_context_row() -> dict[str, float]:
    return {
        "surrounding_mean_ndvi": np.nan,
        "surrounding_mean_delta_ndvi": np.nan,
        "surrounding_green_area_ratio": np.nan,
    }


def _fill_feature_table(table: pd.DataFrame) -> pd.DataFrame:
    table = table.replace([np.inf, -np.inf], np.nan)
    for column in table.columns:
        if not pd.api.types.is_numeric_dtype(table[column]):
            continue
        median = table[column].median(skipna=True)
        fill_value = 0.0 if pd.isna(median) else float(median)
        table[column] = table[column].fillna(fill_value)
    return table.fillna(0.0)


def _cluster_feature_table(feature_table: pd.DataFrame) -> KMeansResult:
    numeric = feature_table.select_dtypes(include=[np.number])
    values = numeric.to_numpy(dtype="float64", copy=True)
    return _automatic_kmeans(values, tuple(numeric.columns), sample_limit=len(values))


def _polygon_cluster_stats(feature_table: pd.DataFrame) -> pd.DataFrame:
    grouped = feature_table.groupby("forecast_cluster", sort=True)
    rows = []
    for cluster_id, group in grouped:
        rows.append(
            {
                "cluster": int(cluster_id),
                "polygon_count": int(len(group)),
                "area_mean": float(group["area_m2"].mean()),
                "area_sum": float(group["area_m2"].sum()),
                "compactness_mean": float(group["compactness"].mean()),
                "eccentricity_mean": float(group["eccentricity"].mean()),
                "ndvi_2015_mean": float(group["mean_ndvi_2015"].mean()),
                "ndvi_2025_mean": float(group["mean_ndvi_2025"].mean()),
                "delta_mean": float(group["mean_delta_ndvi"].mean()),
                "delta_min_mean": float(group["min_delta_ndvi"].mean()),
                "neighbour_count_mean": float(group["neighbour_polygon_count"].mean()),
                "surrounding_delta_mean": float(group["surrounding_mean_delta_ndvi"].mean()),
                "surrounding_ndvi_mean": float(group["surrounding_mean_ndvi"].mean()),
                "forecast_weight_mean": float(group["forecast_weight"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _calculate_forecast_weights(table: pd.DataFrame | gpd.GeoDataFrame) -> np.ndarray:
    area_norm = _normalise_positive(_numeric_column(table, "area_m2"))
    delta_norm = _normalise_positive(np.abs(_numeric_column(table, "mean_delta_ndvi")))
    neighbour_norm = _normalise_positive(_numeric_column(table, "neighbour_polygon_count"))
    compactness_norm = _normalise_positive(_numeric_column(table, "compactness"))
    surrounding_delta_norm = _normalise_positive(
        np.abs(_numeric_column(table, "surrounding_mean_delta_ndvi"))
    )
    return (
        INFLUENCE_AREA_WEIGHT * area_norm
        + INFLUENCE_DELTA_WEIGHT * delta_norm
        + INFLUENCE_NEIGHBOUR_WEIGHT * neighbour_norm
        + INFLUENCE_COMPACTNESS_WEIGHT * compactness_norm
        + INFLUENCE_CONTEXT_DELTA_WEIGHT * surrounding_delta_norm
    ).astype("float32")


def _numeric_column(table: pd.DataFrame | gpd.GeoDataFrame, column: str) -> np.ndarray:
    if column not in table.columns:
        return np.zeros(len(table), dtype="float32")
    values = pd.to_numeric(table[column], errors="coerce").to_numpy(dtype="float32")
    return np.where(np.isfinite(values), values, 0.0).astype("float32")


def _interpret_polygon_clusters(cluster_stats: pd.DataFrame) -> dict[int, str]:
    if cluster_stats.empty:
        return {}
    interpretation = {
        int(row.cluster): "other"
        for row in cluster_stats.itertuples(index=False)
    }
    negative = cluster_stats[cluster_stats["delta_mean"] < 0].copy()
    if negative.empty:
        return interpretation

    negative = negative.set_index("cluster", drop=False)
    scaled = _zscore_frame(
        negative,
        (
            "area_mean",
            "compactness_mean",
            "ndvi_2015_mean",
            "ndvi_2025_mean",
            "delta_mean",
            "neighbour_count_mean",
            "surrounding_delta_mean",
        ),
    )
    loss_score = -scaled["delta_mean"]
    development_score = (
        loss_score
        + scaled["area_mean"]
        + scaled["compactness_mean"]
        + scaled["ndvi_2015_mean"]
        - scaled["ndvi_2025_mean"]
    )
    degradation_score = (
        loss_score
        - scaled["surrounding_delta_mean"]
        + scaled["neighbour_count_mean"]
        + scaled["ndvi_2025_mean"]
        - scaled["area_mean"]
    )

    development_cluster = int(development_score.idxmax())
    interpretation[development_cluster] = "new_development"
    remaining = degradation_score.drop(index=development_cluster, errors="ignore")
    if not remaining.empty:
        interpretation[int(remaining.idxmax())] = "degradation"
    elif degradation_score.loc[development_cluster] > development_score.loc[development_cluster]:
        interpretation[development_cluster] = "degradation"
    return interpretation


def _cluster_remaining_pixels(
    ndvi_start: np.ma.MaskedArray,
    ndvi_end: np.ma.MaskedArray,
    delta_ndvi: np.ma.MaskedArray,
    classes_start: np.ndarray,
    classes_end: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame, KMeansResult | None]:
    labels = np.full(mask.shape, -1, dtype="int16")
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return labels, pd.DataFrame(), None

    class_delta = classes_start.astype("int16") - classes_end.astype("int16")
    feature_arrays = (
        ndvi_start.data,
        ndvi_end.data,
        delta_ndvi.data,
        np.abs(delta_ndvi.data),
        classes_end.astype("float32"),
        class_delta.astype("float32"),
    )
    feature_names = (
        "ndvi_2015",
        "ndvi_2025",
        "delta_ndvi",
        "abs_delta_ndvi",
        "class_2025",
        "class_delta",
    )
    matrix = np.column_stack([array.ravel()[indices] for array in feature_arrays])
    finite = np.all(np.isfinite(matrix), axis=1)
    indices = indices[finite]
    matrix = matrix[finite]
    if matrix.size == 0:
        return labels, pd.DataFrame(), None

    clustering = _automatic_kmeans(
        matrix,
        feature_names,
        sample_limit=min(PIXEL_CLUSTER_SAMPLE_LIMIT, len(matrix)),
    )
    labels.ravel()[indices] = clustering.labels.astype("int16")
    stats = _pixel_cluster_stats(clustering.labels, matrix, feature_names)
    return labels, stats, clustering


def _pixel_cluster_stats(
    labels: np.ndarray,
    matrix: np.ndarray,
    feature_names: tuple[str, ...],
) -> pd.DataFrame:
    table = pd.DataFrame(matrix, columns=feature_names)
    table["cluster"] = labels
    rows = []
    for cluster_id, group in table.groupby("cluster", sort=True):
        rows.append(
            {
                "cluster": int(cluster_id),
                "pixel_count": int(len(group)),
                "mean_ndvi_2015": float(group["ndvi_2015"].mean()),
                "mean_ndvi_2025": float(group["ndvi_2025"].mean()),
                "median_ndvi_2025": float(group["ndvi_2025"].median()),
                "mean_delta": float(group["delta_ndvi"].mean()),
                "mean_abs_delta": float(group["abs_delta_ndvi"].mean()),
                "mean_class_2025": float(group["class_2025"].mean()),
                "mean_class_delta": float(group["class_delta"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _interpret_pixel_clusters(pixel_stats: pd.DataFrame) -> dict[int, str]:
    if pixel_stats.empty:
        return {}

    interpretation = {
        int(row.cluster): "other"
        for row in pixel_stats.itertuples(index=False)
    }
    stable_candidates = pixel_stats[
        pixel_stats["mean_abs_delta"] <= pixel_stats["mean_abs_delta"].median()
    ]
    if stable_candidates.empty:
        stable_candidates = pixel_stats

    urban_cluster = int(stable_candidates.loc[stable_candidates["median_ndvi_2025"].idxmin(), "cluster"])
    green_candidates = stable_candidates[stable_candidates["cluster"] != urban_cluster]
    if not green_candidates.empty:
        green_cluster = int(green_candidates.loc[green_candidates["median_ndvi_2025"].idxmax(), "cluster"])
    else:
        green_cluster = None

    interpretation[urban_cluster] = "stable_urban"
    if green_cluster is not None:
        interpretation[green_cluster] = "stable_green"

    used = {urban_cluster}
    if green_cluster is not None:
        used.add(green_cluster)
    negative_candidates = pixel_stats[
        (pixel_stats["mean_delta"] <= pixel_stats["mean_delta"].median())
        & ~pixel_stats["cluster"].isin(used)
    ]
    if not negative_candidates.empty:
        degradation_cluster = int(
            negative_candidates.loc[negative_candidates["mean_delta"].idxmin(), "cluster"]
        )
        interpretation[degradation_cluster] = "degradation"
    return interpretation


def _automatic_kmeans(
    values: np.ndarray,
    feature_names: tuple[str, ...],
    sample_limit: int,
) -> KMeansResult:
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("KMeans requires a non-empty 2D feature matrix.")
    clean = np.asarray(values, dtype="float64")
    clean = _impute_matrix(clean)
    standardized, mean, scale = _standardize_matrix(clean)
    sample_indices = _deterministic_sample_indices(len(standardized), sample_limit)
    sample = standardized[sample_indices]

    if len(sample) < 2 or np.all(np.nanstd(sample, axis=0) <= EPSILON):
        labels = np.zeros(len(clean), dtype="int16")
        centroid = np.mean(standardized, axis=0, keepdims=True)
        return KMeansResult(labels, centroid, feature_names, 1, None)

    max_k = min(len(sample) - 1, max(2, int(np.log2(len(sample))) + 1))
    best_score = -np.inf
    best_centroids = None
    best_k = 1
    best_sample_labels = None
    for k in range(2, max_k + 1):
        centroids, sample_labels = _fit_kmeans(sample, k)
        if len(np.unique(sample_labels)) < 2:
            continue
        score = _silhouette_score(sample, sample_labels)
        if score > best_score:
            best_score = score
            best_centroids = centroids
            best_k = k
            best_sample_labels = sample_labels

    if best_centroids is None or best_sample_labels is None:
        labels = np.zeros(len(clean), dtype="int16")
        centroid = np.mean(standardized, axis=0, keepdims=True)
        return KMeansResult(labels, centroid, feature_names, 1, None)

    centroids, _ = _fit_kmeans(sample, best_k, initial_centroids=best_centroids)
    labels = _predict_kmeans(standardized, centroids)
    return KMeansResult(
        labels.astype("int16"),
        centroids * scale + mean,
        feature_names,
        best_k,
        float(best_score),
    )


def _fit_kmeans(
    values: np.ndarray,
    k: int,
    initial_centroids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if initial_centroids is None:
        centroids = _initial_centroids(values, k)
    else:
        centroids = initial_centroids.astype("float64", copy=True)
    labels = np.zeros(len(values), dtype="int16")
    for _ in range(KMEANS_MAX_ITERATIONS):
        new_labels = _predict_kmeans(values, centroids)
        new_centroids = centroids.copy()
        for cluster_id in range(k):
            cluster_values = values[new_labels == cluster_id]
            if cluster_values.size:
                new_centroids[cluster_id] = np.mean(cluster_values, axis=0)
            else:
                distances = _distance_to_nearest_centroid(values, centroids)
                new_centroids[cluster_id] = values[int(np.argmax(distances))]
        shift = float(np.linalg.norm(new_centroids - centroids))
        labels = new_labels
        centroids = new_centroids
        if shift <= KMEANS_TOLERANCE:
            break
    return centroids, labels


def _initial_centroids(values: np.ndarray, k: int) -> np.ndarray:
    if k == 1:
        return np.mean(values, axis=0, keepdims=True)
    projection = values @ np.linspace(1.0, 2.0, values.shape[1])
    order = np.argsort(projection)
    positions = np.linspace(0, len(order) - 1, k).round().astype(int)
    return values[order[positions]].astype("float64", copy=True)


def _predict_kmeans(values: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    labels = np.empty(len(values), dtype="int16")
    for start in range(0, len(values), PREDICT_CHUNK_SIZE):
        end = min(start + PREDICT_CHUNK_SIZE, len(values))
        chunk = values[start:end]
        distances = np.sum((chunk[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        labels[start:end] = np.argmin(distances, axis=1).astype("int16")
    return labels


def _distance_to_nearest_centroid(values: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    distances = np.sum((values[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    return np.min(distances, axis=1)


def _silhouette_score(values: np.ndarray, labels: np.ndarray) -> float:
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return -1.0
    distances = np.sqrt(
        np.maximum(
            np.sum((values[:, None, :] - values[None, :, :]) ** 2, axis=2),
            0.0,
        )
    )
    scores = []
    for index, label in enumerate(labels):
        own = labels == label
        own_count = int(np.count_nonzero(own))
        if own_count > 1:
            a_value = float(np.sum(distances[index, own]) / (own_count - 1))
        else:
            a_value = 0.0
        b_value = np.inf
        for other_label in unique_labels:
            if other_label == label:
                continue
            other = labels == other_label
            if np.any(other):
                b_value = min(b_value, float(np.mean(distances[index, other])))
        denominator = max(a_value, b_value)
        scores.append(0.0 if denominator <= EPSILON else (b_value - a_value) / denominator)
    return float(np.mean(scores))


def _impute_matrix(values: np.ndarray) -> np.ndarray:
    clean = values.astype("float64", copy=True)
    clean[~np.isfinite(clean)] = np.nan
    medians = np.nanmedian(clean, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    rows, columns = np.where(~np.isfinite(clean))
    clean[rows, columns] = medians[columns]
    return clean


def _standardize_matrix(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(values, axis=0)
    scale = np.std(values, axis=0)
    scale = np.where(scale <= EPSILON, 1.0, scale)
    return (values - mean) / scale, mean, scale


def _deterministic_sample_indices(length: int, limit: int) -> np.ndarray:
    if length <= limit:
        return np.arange(length)
    return np.linspace(0, length - 1, limit).round().astype(int)


def _zscore_frame(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    scaled = pd.DataFrame(index=frame.index)
    for column in columns:
        values = frame[column].astype("float64")
        std = values.std(ddof=0)
        if std <= EPSILON or not np.isfinite(std):
            scaled[column] = 0.0
        else:
            scaled[column] = (values - values.mean()) / std
    return scaled


def _normalise_polygon_scenario_masks(
    polygon_scenarios: Mapping[str, np.ndarray] | np.ndarray | None,
    shape: tuple[int, int],
) -> dict[str, np.ndarray]:
    masks = {name: np.zeros(shape, dtype=bool) for name in SCENARIO_NAMES}
    if polygon_scenarios is None:
        return masks
    if isinstance(polygon_scenarios, Mapping):
        for name in SCENARIO_NAMES:
            if name in polygon_scenarios:
                masks[name] = np.asarray(polygon_scenarios[name], dtype=bool)
        return masks
    masks["degradation"] = np.asarray(polygon_scenarios, dtype=bool)
    return masks


def _calculate_scenario_references(
    ndvi_end: np.ma.MaskedArray,
    scenarios: DynamicsScenarios,
    valid_mask: np.ndarray,
) -> ScenarioReferences:
    valid = valid_mask & ~np.ma.getmaskarray(ndvi_end) & np.isfinite(ndvi_end.data)
    urban_values = ndvi_end.data[valid & scenarios.stable_urban]
    green_values = ndvi_end.data[valid & scenarios.stable_green]
    global_values = ndvi_end.data[valid]
    global_median = float(np.median(global_values)) if global_values.size else 0.0
    urban_reference = float(np.median(urban_values)) if urban_values.size else global_median
    green_reference = float(np.median(green_values)) if green_values.size else global_median
    lower, upper = sorted((urban_reference, green_reference))
    recovery_ceiling = float((lower + upper) / 2.0)
    LOGGER.info(
        "Scenario NDVI references: stable_urban=%.6f, stable_green=%.6f, recovery_ceiling=%.6f",
        urban_reference,
        green_reference,
        recovery_ceiling,
    )
    return ScenarioReferences(urban_reference, green_reference, recovery_ceiling)


def _development_recovery_surface(
    ndvi_start: np.ma.MaskedArray,
    ndvi_current: np.ma.MaskedArray,
    delta_values: np.ma.MaskedArray,
    references: ScenarioReferences,
) -> np.ndarray:
    observed_loss = np.abs(np.minimum(delta_values.data, 0.0)).astype("float32")
    target = np.minimum(ndvi_start.data, references.recovery_ceiling)
    capacity = np.maximum(target - ndvi_current.data, 0.0)
    return np.minimum(capacity, observed_loss).astype("float32")


def _stable_urban_greening_surface(
    forecast: np.ndarray,
    ndvi_current: np.ma.MaskedArray,
    scenarios: DynamicsScenarios,
    references: ScenarioReferences,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    valid = (
        valid_mask
        & (forecast != RASTER_NODATA)
        & np.isfinite(forecast)
        & ~np.ma.getmaskarray(ndvi_current)
        & np.isfinite(ndvi_current.data)
    )
    green_source = valid & (
        scenarios.stable_green
        | (ndvi_current.data >= references.stable_green_ndvi)
    )
    green_density = _local_mean_surface(
        green_source.astype("float32"),
        valid,
        LOCAL_ANALYSIS_WINDOW_SIZE,
    )
    density_norm = _normalise_positive(green_density)
    candidate = (
        valid
        & scenarios.stable_urban
        & (forecast <= references.stable_urban_ndvi)
        & (green_density > 0)
    )
    gain = (
        URBAN_GREENING_MIN_GAIN
        + (URBAN_GREENING_MAX_GAIN - URBAN_GREENING_MIN_GAIN) * density_norm
    ).astype("float32")
    ceiling_room = np.maximum(references.recovery_ceiling - forecast, 0.0)
    gain = np.minimum(gain, ceiling_room).astype("float32")
    selection = candidate & (gain > 0)
    gain[~selection] = 0.0
    return selection, gain


def _degradation_loss_surface(
    delta_values: np.ma.MaskedArray,
    influence_surface: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    observed_loss = np.abs(np.minimum(delta_values.data, 0.0)).astype("float32")
    valid = (
        valid_mask
        & ~np.ma.getmaskarray(delta_values)
        & np.isfinite(delta_values.data)
    )
    neighbourhood_loss = _local_mean_surface(
        observed_loss,
        valid,
        LOCAL_ANALYSIS_WINDOW_SIZE,
    )
    weighted_loss = neighbourhood_loss * _normalise_positive(influence_surface)
    return weighted_loss.astype("float32")


def _local_mean_surface(
    values: np.ndarray,
    valid_mask: np.ndarray,
    window_size: int,
) -> np.ndarray:
    valid = valid_mask & np.isfinite(values)
    numerator = ndimage.uniform_filter(
        np.where(valid, values, 0.0).astype("float32"),
        size=window_size,
        mode="nearest",
    )
    denominator = ndimage.uniform_filter(
        valid.astype("float32"),
        size=window_size,
        mode="nearest",
    )
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype="float32"),
        where=denominator > 0,
    ).astype("float32")


def _local_std_surface(
    values: np.ndarray,
    valid_mask: np.ndarray,
    window_size: int,
) -> np.ndarray:
    valid = valid_mask & np.isfinite(values)
    mean = _local_mean_surface(values, valid, window_size)
    mean_square = _local_mean_surface(values**2, valid, window_size)
    variance = np.maximum(mean_square - mean**2, 0.0)
    return np.sqrt(variance).astype("float32")


def _select_by_score(
    selection: np.ndarray,
    candidates: np.ndarray,
    score_surface: np.ndarray,
    requested: int,
) -> np.ndarray:
    candidate_indices = np.flatnonzero(candidates)
    if candidate_indices.size == 0:
        return np.empty(0, dtype=np.int64)
    scores = score_surface.ravel()[candidate_indices]
    order = np.argsort(scores)[::-1]
    selected_indices = candidate_indices[order[:requested]]
    selection.ravel()[selected_indices] = True
    return selected_indices


def _change_surface(change: float | np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if np.isscalar(change):
        return np.full(shape, float(change), dtype="float32")
    if change.shape != shape:
        raise ValueError("Scenario change surface does not match NDVI raster shape.")
    return change.astype("float32", copy=False)


def _normalise_positive(values: np.ndarray) -> np.ndarray:
    clean = np.where(np.isfinite(values), values, 0.0).astype("float32", copy=False)
    maximum = float(np.max(clean)) if clean.size else 0.0
    if maximum <= EPSILON:
        return np.zeros_like(clean, dtype="float32")
    return (clean / maximum).astype("float32")


def _validate_forecast_inputs(*paths: Path) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Forecast input files are missing: " + ", ".join(missing))


def _assert_forecast_grid(reference_path: Path, *paths: Path) -> None:
    for path in paths:
        assert_same_grid(reference_path, path)


def _read_class_raster(path: Path) -> np.ndarray:
    with rasterio.open(path) as dataset:
        return dataset.read(1)


def _class_intervals() -> Mapping[int, ClassInterval]:
    return {
        int(class_id): ClassInterval(
            int(class_id),
            lower,
            upper,
            lower_inclusive,
            upper_inclusive,
        )
        for class_id, _, lower, upper, _, lower_inclusive, upper_inclusive
        in NDVI_CLASS_THRESHOLDS
    }


def _class_ids() -> tuple[int, ...]:
    return tuple(int(threshold[0]) for threshold in NDVI_CLASS_THRESHOLDS)


def _class_target_value(class_id: int, direction: str) -> float:
    interval = _class_intervals()[class_id]
    if interval.lower is not None and interval.upper is not None:
        return float((interval.lower + interval.upper) / 2.0)
    if direction == "decrease":
        if interval.upper is None:
            return float(interval.lower if interval.lower is not None else FORECAST_CLIP_MIN)
        return float(np.nextafter(interval.upper, FORECAST_CLIP_MIN))
    if interval.lower is None:
        return float(interval.upper if interval.upper is not None else FORECAST_CLIP_MAX)
    return float(np.nextafter(interval.lower, FORECAST_CLIP_MAX))


def _nearest_polygon_distances(polygons: gpd.GeoDataFrame) -> np.ndarray:
    count = len(polygons)
    if count <= 1:
        return np.zeros(count, dtype="float64")
    centroids = polygons.geometry.centroid
    distances = np.full(count, np.inf, dtype="float64")
    for index, geometry in enumerate(centroids):
        current = centroids.distance(geometry).to_numpy(dtype="float64")
        current[index] = np.inf
        distances[index] = float(np.nanmin(current))
    return np.where(np.isfinite(distances), distances, 0.0)


def _neighbour_counts(polygons: gpd.GeoDataFrame, context_distance: float) -> np.ndarray:
    count = len(polygons)
    if count <= 1:
        return np.zeros(count, dtype="float64")
    centroids = polygons.geometry.centroid
    counts = np.zeros(count, dtype="float64")
    for index, geometry in enumerate(centroids):
        current = centroids.distance(geometry).to_numpy(dtype="float64")
        current[index] = np.inf
        counts[index] = float(np.count_nonzero(current <= context_distance))
    return counts


def _adaptive_context_distance(polygons: gpd.GeoDataFrame) -> float:
    if polygons.empty:
        return 0.0
    areas = polygons.geometry.area.to_numpy(dtype="float64")
    equivalent_radius = np.sqrt(np.maximum(areas, 0.0) / np.pi)
    nearest = _nearest_polygon_distances(polygons)
    values = np.concatenate(
        [
            equivalent_radius[np.isfinite(equivalent_radius)],
            nearest[np.isfinite(nearest) & (nearest > 0)],
        ]
    )
    if values.size == 0:
        return float(np.nanmedian(equivalent_radius)) if equivalent_radius.size else 0.0
    return float(np.nanmedian(values))


def _adaptive_neighbourhood_distance(geometry: Any, context_distance: float) -> float:
    area = max(float(geometry.area), 0.0)
    perimeter = max(float(geometry.length), EPSILON)
    equivalent_radius = float(np.sqrt(area / np.pi))
    compactness = float((4.0 * np.pi * area) / max(perimeter**2, EPSILON))
    shape_factor = 1.0 / max(np.sqrt(max(compactness, EPSILON)), EPSILON)
    return max(equivalent_radius * shape_factor, context_distance)


def _adaptive_neighbourhood_geometries(polygons: gpd.GeoDataFrame) -> list[Any]:
    context_distance = _adaptive_context_distance(polygons)
    geometries = []
    for geometry in polygons.geometry:
        distance = _adaptive_neighbourhood_distance(geometry, context_distance)
        geometries.append(geometry.buffer(distance))
    return geometries


def _geometry_eccentricity(geometry: Any) -> float:
    rectangle = geometry.minimum_rotated_rectangle
    try:
        coordinates = np.asarray(rectangle.exterior.coords, dtype="float64")
    except AttributeError:
        return 0.0
    if len(coordinates) < 4:
        return 0.0
    side_lengths = np.sqrt(np.sum(np.diff(coordinates[:5], axis=0) ** 2, axis=1))
    long_side = float(np.max(side_lengths))
    short_side = float(np.min(side_lengths[side_lengths > EPSILON])) if np.any(side_lengths > EPSILON) else 0.0
    if short_side <= EPSILON:
        return 0.0
    return float(long_side / short_side)


def _log_polygon_model(
    cluster_stats: pd.DataFrame,
    clustering: KMeansResult,
    interpretation: dict[int, str],
) -> None:
    LOGGER.info(
        "Polygon clustering: method=deterministic_kmeans, clusters=%s, silhouette=%s",
        clustering.k,
        "n/a" if clustering.silhouette is None else f"{clustering.silhouette:.6f}",
    )
    if cluster_stats.empty:
        return
    logged = cluster_stats.copy()
    logged["scenario"] = logged["cluster"].map(interpretation).fillna("other")
    LOGGER.info("Polygon cluster statistics:\n%s", logged.to_string(index=False))


def _log_pixel_clusters(
    pixel_stats: pd.DataFrame,
    clustering: KMeansResult | None,
    interpretation: dict[int, str],
) -> None:
    if clustering is None:
        LOGGER.info("Pixel clustering skipped: no remaining valid pixels.")
        return
    LOGGER.info(
        "Pixel clustering: method=deterministic_kmeans, clusters=%s, silhouette=%s",
        clustering.k,
        "n/a" if clustering.silhouette is None else f"{clustering.silhouette:.6f}",
    )
    if pixel_stats.empty:
        return
    logged = pixel_stats.copy()
    logged["scenario"] = logged["cluster"].map(interpretation).fillna("other")
    LOGGER.info("Pixel cluster statistics:\n%s", logged.to_string(index=False))


def _log_transition_probabilities(
    probabilities: pd.DataFrame,
    start_year: int,
    end_year: int,
    forecast_year: int,
) -> None:
    LOGGER.info(
        "Scenario Markov probabilities %s->%s applied to %s:\n%s",
        start_year,
        end_year,
        forecast_year,
        probabilities.to_string(float_format=lambda value: f"{value:.6f}"),
    )
    for class_id in _class_ids():
        lower = float(
            probabilities.loc[class_id, [item for item in _class_ids() if item < class_id]].sum()
        )
        higher = float(
            probabilities.loc[class_id, [item for item in _class_ids() if item > class_id]].sum()
        )
        LOGGER.info(
            "Markov class %s: stay=%.6f, lower=%.6f, higher=%.6f",
            class_id,
            float(probabilities.loc[class_id, class_id]),
            lower,
            higher,
        )


def _log_scenario_statistics(scenarios: DynamicsScenarios, profile: Mapping[str, object]) -> None:
    transform = profile["transform"]
    pixel_area = abs(transform.a * transform.e - transform.b * transform.d)
    for name, mask in scenarios.items():
        pixel_count = int(np.count_nonzero(mask))
        LOGGER.info(
            "Scenario %s: pixels=%s, area_m2=%.2f, area_ha=%.4f",
            name,
            pixel_count,
            pixel_count * pixel_area,
            pixel_count * pixel_area / 10_000.0,
        )


def _log_stable_urban_greening(
    greening_mask: np.ndarray,
    greening_gain: np.ndarray,
    profile: Mapping[str, object],
) -> None:
    transform = profile["transform"]
    pixel_area = abs(transform.a * transform.e - transform.b * transform.d)
    selected = int(np.count_nonzero(greening_mask))
    mean_gain = float(np.mean(greening_gain[greening_mask])) if selected else 0.0
    LOGGER.info(
        "Stable urban greening: pixels=%s, area_m2=%.2f, mean_gain=%.6f",
        selected,
        selected * pixel_area,
        mean_gain,
    )


def _log_forecast_summary(
    polygon_model: PolygonForecastModel,
    degradation_transitions: list[MarkovTransition],
    recovery_transitions: list[MarkovTransition],
    degradation_selection: np.ndarray,
    recovery_selection: np.ndarray,
    stable_greening_mask: np.ndarray,
    influence_surface: np.ndarray,
    profile: Mapping[str, object],
    references: ScenarioReferences,
    mean_loss: float,
    source: np.ma.MaskedArray,
    forecast: np.ndarray,
    valid_mask: np.ndarray,
) -> None:
    requested_degradation = sum(item.requested_pixels for item in degradation_transitions)
    requested_recovery = sum(item.requested_pixels for item in recovery_transitions)
    LOGGER.info(
        "Automatic scenario forecast summary: polygon_clusters=%s, "
        "degradation_requested=%s, degradation_selected=%s, "
        "recovery_requested=%s, recovery_selected=%s, stable_greening=%s, changed=%s, "
        "influence_area_m2=%.2f, mean_degradation_loss=%.6f, recovery_ceiling=%.6f",
        0 if polygon_model.clustering is None else polygon_model.clustering.k,
        requested_degradation,
        int(np.count_nonzero(degradation_selection)),
        requested_recovery,
        int(np.count_nonzero(recovery_selection)),
        int(np.count_nonzero(stable_greening_mask)),
        _count_changed_pixels(source, forecast, valid_mask),
        _calculate_influence_area(influence_surface, profile),
        mean_loss,
        references.recovery_ceiling,
    )


def _count_changed_pixels(
    source: np.ma.MaskedArray,
    forecast: np.ndarray,
    valid_mask: np.ndarray,
) -> int:
    valid = valid_mask & ~np.ma.getmaskarray(source)
    return int(np.count_nonzero(valid & ~np.isclose(source.data, forecast)))


def _calculate_influence_area(influence_surface: np.ndarray, profile: Mapping[str, object]) -> float:
    transform = profile["transform"]
    pixel_area = abs(transform.a * transform.e - transform.b * transform.d)
    return float(np.count_nonzero(influence_surface > 0) * pixel_area)
