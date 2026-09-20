"""Project configuration for NDVI Monitor."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

BASE_YEAR = 2015
TARGET_YEAR = 2025
FORECAST_YEAR = 2035

RASTER_NODATA = -9999.0
CLASS_NODATA = 0

NDVI_MIN = -1.0
NDVI_MAX = 1.0
NDVI_FORMAT_TOLERANCE = 0.02
NDVI_SCALE_FACTOR = 10_000.0
NDVI_SCALED_ABS_LIMIT = 10_000.0
NDVI_DENOMINATOR_EPSILON = 1e-10

ALIGNMENT_TARGET_CRS = "EPSG:32636"
ALIGNMENT_RESOLUTION = 10.0
ALIGNMENT_ORIGIN_X = 0.0
ALIGNMENT_ORIGIN_Y = 0.0

NDVI_CLASS_THRESHOLDS = (
    (1, "менее 0.10", None, 0.10, "#8c510a", True, False),
    (2, "0.10-0.25", 0.10, 0.25, "#d8b365", True, False),
    (3, "0.25-0.45", 0.25, 0.45, "#f6e8c3", True, False),
    (4, "0.45-0.65", 0.45, 0.65, "#80cdc1", True, True),
    (5, "более 0.65", 0.65, None, "#01665e", False, True),
)

DELTA_CLASS_THRESHOLDS = (
    (1, "Значительное ухудшение", None, -0.20, "#a50026", True, False),
    (2, "Умеренное ухудшение", -0.20, -0.05, "#f46d43", True, False),
    (3, "Без изменений", -0.05, 0.05, "#f7f7f7", True, False),
    (4, "Умеренное улучшение", 0.05, 0.20, "#66bd63", True, False),
    (5, "Значительное улучшение", 0.20, None, "#006837", True, True),
)

DELTA_DISPLAY_MIN_ABS_RANGE = 0.20
DELTA_DISPLAY_MAX_ABS_RANGE = 1.00

FORECAST_CLIP_MIN = NDVI_MIN
FORECAST_CLIP_MAX = NDVI_MAX

DEGRADATION_CLASSES = (1, 2)
IMPROVEMENT_CLASSES = (4, 5)
MAX_CHANGE_POLYGONS_PER_TYPE = 20

DATA_SOURCE_LABEL = "Sentinel-2"
PIXEL_AREA_METHOD = (
    "Площадь вычисляется как площадь пикселя целевой сетки, умноженная "
    "на количество валидных пикселей. На границе полигона используется "
    "пиксельная аппроксимация после маскирования по границе района."
)


@dataclass(frozen=True)
class ProjectPaths:
    """Filesystem layout used by the monitoring pipeline."""

    base_dir: Path
    input_dir: Path
    output_dir: Path
    rasters_dir: Path
    tables_dir: Path
    figures_dir: Path
    reports_dir: Path
    logs_dir: Path
    log_file: Path

    @classmethod
    def from_base(cls, base_dir: Path = BASE_DIR) -> "ProjectPaths":
        """Build all project paths from a single base directory."""

        output_dir = base_dir / "output"
        logs_dir = base_dir / "logs"
        return cls(
            base_dir=base_dir,
            input_dir=base_dir / "input",
            output_dir=output_dir,
            rasters_dir=output_dir / "rasters",
            tables_dir=output_dir / "tables",
            figures_dir=output_dir / "figures",
            reports_dir=output_dir / "reports",
            logs_dir=logs_dir,
            log_file=logs_dir / "monitor.log",
        )

    def ensure_directories(self) -> None:
        """Create the runtime directory tree if it does not exist."""

        for directory in (
            self.input_dir,
            self.output_dir,
            self.rasters_dir,
            self.tables_dir,
            self.figures_dir,
            self.reports_dir,
            self.logs_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


RAW_BAND_FILES = {
    "red_2015": "B04_2015.tif",
    "nir_2015": "B08_2015.tif",
    "red_2025": "B04_2025.tif",
    "nir_2025": "B08_2025.tif",
}

READY_NDVI_FILES = {
    "ndvi_2015": "ndvi2015.tif",
    "ndvi_2025": "ndvi2025.tif",
}

VECTOR_BOUNDARY_EXTENSIONS = (".gpkg", ".shp", ".geojson")


FORECAST_MIN_AREA = 1000.0
BUFFER_RULES = ((2000.0, 10.0), (10000.0, 20.0), (float('inf'), 40.0))