"""End-to-end NDVI monitoring pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from config import (
    ALIGNMENT_RESOLUTION,
    ALIGNMENT_TARGET_CRS,
    BASE_YEAR,
    FORECAST_YEAR,
    PIXEL_AREA_METHOD,
    ProjectPaths,
    TARGET_YEAR,
)
from modules.align import align_and_clip_rasters
from modules.changes import extract_largest_change_polygons
from modules.classify import DELTA_CLASS_RULES, NDVI_CLASS_RULES, classify_raster
from modules.delta import calculate_delta_ndvi
from modules.export_excel import export_excel_report
from modules.forecast import forecast_ndvi_linear
from modules.loader import InputBundle, validate_inputs
from modules.mask import apply_common_mask_to_raster, create_common_valid_mask
from modules.ndvi import build_ndvi_products
from modules.report import generate_word_report
from modules.statistics import (
    compute_class_areas,
    compute_transition_matrix,
    compute_value_statistics,
    export_transition_matrix,
)
from modules.utils import setup_logging
from modules.visualization import (
    NDVI_DISPLAY_PERCENTILES,
    calculate_display_range,
    plot_class_area_bar,
    plot_classification_map,
    plot_delta_map,
    plot_ndvi_map,
)


@dataclass(frozen=True)
class PipelineResult:
    """Output paths and report tables produced by one pipeline run."""

    generated_files: dict[str, Path]
    tables: dict[str, pd.DataFrame]
    summary: dict[str, object]


class MonitorPipeline:
    """Coordinate all specification stages from input validation to exports."""

    def __init__(
        self,
        paths: ProjectPaths | None = None,
        log_level: int = logging.INFO,
    ) -> None:
        self.paths = paths or ProjectPaths.from_base()
        self.paths.ensure_directories()
        self.logger = setup_logging(self.paths.log_file, log_level)
        self._boundary_path: Path | None = None
        self._ndvi_display_range: tuple[float, float] | None = None

    def run(self, input_dir: Path | None = None) -> PipelineResult:
        """Run every monitoring stage and return generated artifacts."""

        selected_input_dir = input_dir or self.paths.input_dir
        self.logger.info("NDVI Monitor pipeline started.")

        bundle = self._stage_1_validate(selected_input_dir)
        self._boundary_path = bundle.boundary
        aligned_rasters = self._stage_2_align(bundle)
        ndvi_paths = self._stage_3_ndvi(bundle, aligned_rasters)

        generated_files: dict[str, Path] = {}
        tables: dict[str, pd.DataFrame] = {}
        summary: dict[str, object] = self._build_summary(bundle)
        common_mask_path = self._stage_3_common_mask(ndvi_paths)
        generated_files["common_valid_mask_geotiff"] = common_mask_path
        ndvi_paths = self._stage_3_apply_common_mask(ndvi_paths, common_mask_path)

        generated_files.update(self._stage_4_ndvi_maps(ndvi_paths))
        ndvi_classes = self._stage_5_classify_ndvi(ndvi_paths)
        generated_files.update(class_paths_to_outputs(ndvi_classes, "ndvi_classes"))
        generated_files.update(class_figure_paths_to_outputs(ndvi_classes, self.paths))
        tables.update(self._stage_6_ndvi_areas(ndvi_classes))
        transition_table, transition_path = self._stage_6_transition_matrix(ndvi_classes)
        tables["transition_matrix"] = transition_table
        generated_files["transition_matrix_excel"] = transition_path

        delta_path = self._stage_7_delta(ndvi_paths)
        generated_files["delta_geotiff"] = delta_path
        generated_files.update(self._stage_8_delta_map(delta_path))

        delta_class_path, delta_class_png = self._stage_9_classify_delta(delta_path)
        generated_files["delta_class_geotiff"] = delta_class_path
        generated_files["delta_class_png"] = delta_class_png
        tables.update(self._stage_10_statistics(ndvi_paths, delta_path, delta_class_path))
        change_files, change_table = self._stage_10_change_polygons(delta_class_path)
        generated_files.update(change_files)
        tables["largest_changes"] = change_table

        forecast_path = self._stage_11_forecast(ndvi_paths)
        generated_files["forecast_geotiff"] = forecast_path
        generated_files.update(self._stage_11_forecast_maps(forecast_path))

        generated_files.update(ndvi_paths_to_outputs(ndvi_paths))

        export_files = self._stage_12_exports(summary, tables, generated_files)
        generated_files.update(export_files)

        self.logger.info("NDVI Monitor pipeline finished successfully.")
        return PipelineResult(
            generated_files=generated_files,
            tables=tables,
            summary=summary,
        )

    def _stage_1_validate(self, input_dir: Path) -> InputBundle:
        self.logger.info("Stage 1: input validation.")
        bundle = validate_inputs(input_dir)
        self.logger.info("Input variant: %s", bundle.variant.value)
        for label, metadata in bundle.raster_metadata.items():
            self.logger.info(
                "Raster %s: CRS=%s, size=%sx%s, pixel=%s/%s, "
                "NoData=%s, dtype=%s",
                label,
                metadata.crs,
                metadata.width,
                metadata.height,
                metadata.pixel_size_x,
                metadata.pixel_size_y,
                metadata.nodata,
                metadata.dtype,
            )
            self.logger.info(
                "Raster %s metadata: scale=%s, offset=%s, "
                "range=%s..%s, transform=%s, intersects_boundary=%s",
                label,
                metadata.scale,
                metadata.offset,
                metadata.valid_min,
                metadata.valid_max,
                metadata.transform,
                metadata.intersects_boundary,
            )
            for warning in metadata.warnings:
                self.logger.warning("Raster %s: %s", label, warning)
        self.logger.info(
            "Boundary: CRS=%s, features=%s, geometry=%s, area=%s, bounds=%s",
            bundle.boundary_metadata.crs,
            bundle.boundary_metadata.feature_count,
            bundle.boundary_metadata.geometry_type,
            bundle.boundary_metadata.area,
            bundle.boundary_metadata.bounds,
        )
        return bundle

    def _stage_2_align(self, bundle: InputBundle) -> dict[str, Path]:
        self.logger.info("Stage 2: raster alignment and clipping.")
        return align_and_clip_rasters(
            bundle.rasters,
            bundle.boundary,
            self.paths.rasters_dir / "aligned",
            self.logger,
        )

    def _stage_3_ndvi(
        self,
        bundle: InputBundle,
        aligned_rasters: dict[str, Path],
    ) -> dict[int, Path]:
        self.logger.info("Stage 3: NDVI preparation.")
        return build_ndvi_products(aligned_rasters, bundle.variant, self.paths.rasters_dir)

    def _stage_3_common_mask(self, ndvi_paths: dict[int, Path]) -> Path:
        self.logger.info("Stage 3: common valid pixel mask.")
        return create_common_valid_mask(
            ndvi_paths,
            self.paths.rasters_dir / "common_valid_mask.tif",
        )

    def _stage_3_apply_common_mask(
        self,
        ndvi_paths: dict[int, Path],
        common_mask_path: Path,
    ) -> dict[int, Path]:
        self.logger.info("Stage 3: applying common valid mask to NDVI rasters.")
        masked_paths = {}
        for year, path in ndvi_paths.items():
            masked_paths[year] = apply_common_mask_to_raster(
                path,
                common_mask_path,
                self.paths.rasters_dir / f"ndvi_{year}_valid.tif",
            )
        return masked_paths

    def _stage_4_ndvi_maps(self, ndvi_paths: dict[int, Path]) -> dict[str, Path]:
        self.logger.info("Stage 4: NDVI map rendering.")
        self._ndvi_display_range = calculate_display_range(
            tuple(ndvi_paths.values()),
            percentiles=NDVI_DISPLAY_PERCENTILES,
        )
        self.logger.info(
            "Common NDVI display range: %.6f..%.6f",
            *self._ndvi_display_range,
        )
        outputs = {}
        for year, path in ndvi_paths.items():
            output = self.paths.figures_dir / f"ndvi_{year}.png"
            outputs[f"ndvi_{year}_png"] = plot_ndvi_map(
                path,
                output,
                f"NDVI {year}",
                self._boundary_path,
                display_range=self._ndvi_display_range,
            )
        return outputs

    def _stage_5_classify_ndvi(self, ndvi_paths: dict[int, Path]) -> dict[int, Path]:
        self.logger.info("Stage 5: NDVI classification.")
        class_paths = {}
        for year, path in ndvi_paths.items():
            output = self.paths.rasters_dir / f"ndvi_classes_{year}.tif"
            class_paths[year] = classify_raster(path, output, NDVI_CLASS_RULES)
            figure = self.paths.figures_dir / f"ndvi_classes_{year}.png"
            plot_classification_map(
                class_paths[year],
                figure,
                f"Классы NDVI {year}",
                NDVI_CLASS_RULES,
                self._boundary_path,
            )
        return class_paths

    def _stage_6_ndvi_areas(
        self,
        class_paths: dict[int, Path],
    ) -> dict[str, pd.DataFrame]:
        self.logger.info("Stage 6: NDVI class area calculation.")
        tables = {}
        for year, path in class_paths.items():
            table = compute_class_areas(path, NDVI_CLASS_RULES)
            tables[f"ndvi_class_areas_{year}"] = table
            plot_class_area_bar(
                table,
                self.paths.figures_dir / f"ndvi_class_areas_{year}.png",
                f"Площади классов NDVI {year}",
            )
        return tables

    def _stage_6_transition_matrix(
        self,
        class_paths: dict[int, Path],
    ) -> tuple[pd.DataFrame, Path]:
        self.logger.info("Stage 6: NDVI class transition matrix.")
        table = compute_transition_matrix(
            class_paths[BASE_YEAR],
            class_paths[TARGET_YEAR],
            NDVI_CLASS_RULES,
        )
        output_path = self.paths.tables_dir / "transition_matrix.xlsx"
        export_transition_matrix(table, output_path)
        return table, output_path

    def _stage_7_delta(self, ndvi_paths: dict[int, Path]) -> Path:
        self.logger.info("Stage 7: Delta NDVI calculation.")
        return calculate_delta_ndvi(
            ndvi_paths[BASE_YEAR],
            ndvi_paths[TARGET_YEAR],
            self.paths.rasters_dir / "delta_ndvi.tif",
        )

    def _stage_8_delta_map(self, delta_path: Path) -> dict[str, Path]:
        self.logger.info("Stage 8: Delta NDVI map rendering.")
        output = self.paths.figures_dir / "delta_ndvi.png"
        return {
            "delta_png": plot_delta_map(
                delta_path,
                output,
                f"Delta NDVI {TARGET_YEAR}-{BASE_YEAR}",
                self._boundary_path,
            )
        }

    def _stage_9_classify_delta(self, delta_path: Path) -> tuple[Path, Path]:
        self.logger.info("Stage 9: Delta NDVI classification.")
        class_path = classify_raster(
            delta_path,
            self.paths.rasters_dir / "delta_class.tif",
            DELTA_CLASS_RULES,
        )
        figure_path = self.paths.figures_dir / "delta_class.png"
        plot_classification_map(
            class_path,
            figure_path,
            "Классы изменений NDVI",
            DELTA_CLASS_RULES,
            self._boundary_path,
        )
        return class_path, figure_path

    def _stage_10_statistics(
        self,
        ndvi_paths: dict[int, Path],
        delta_path: Path,
        delta_class_path: Path,
    ) -> dict[str, pd.DataFrame]:
        self.logger.info("Stage 10: statistical summaries.")
        tables = {
            f"ndvi_statistics_{BASE_YEAR}": compute_value_statistics(
                ndvi_paths[BASE_YEAR],
                f"NDVI {BASE_YEAR}",
            ),
            f"ndvi_statistics_{TARGET_YEAR}": compute_value_statistics(
                ndvi_paths[TARGET_YEAR],
                f"NDVI {TARGET_YEAR}",
            ),
            "delta_statistics": compute_value_statistics(delta_path, "Delta NDVI"),
            "delta_class_areas": compute_class_areas(delta_class_path, DELTA_CLASS_RULES),
        }
        plot_class_area_bar(
            tables["delta_class_areas"],
            self.paths.figures_dir / "delta_class_areas.png",
            "Площади классов изменений",
        )
        return tables

    def _stage_10_change_polygons(
        self,
        delta_class_path: Path,
    ) -> tuple[dict[str, Path], pd.DataFrame]:
        self.logger.info("Stage 10: largest degradation and improvement polygons.")
        vector_path = self.paths.rasters_dir / "change_polygons.gpkg"
        excel_path = self.paths.tables_dir / "largest_changes.xlsx"
        vector_path, excel_path, table = extract_largest_change_polygons(
            delta_class_path,
            vector_path,
            excel_path,
        )
        return {
            "change_polygons_vector": vector_path,
            "largest_changes_excel": excel_path,
        }, table

    def _stage_11_forecast(self, ndvi_paths: dict[int, Path]) -> Path:
        self.logger.info("Stage 11: linear NDVI forecast.")
        return forecast_ndvi_linear(
            ndvi_paths[BASE_YEAR],
            ndvi_paths[TARGET_YEAR],
            BASE_YEAR,
            TARGET_YEAR,
            FORECAST_YEAR,
            self.paths.rasters_dir / f"ndvi_forecast_{FORECAST_YEAR}.tif",
        )

    def _stage_11_forecast_maps(self, forecast_path: Path) -> dict[str, Path]:
        output = self.paths.figures_dir / f"ndvi_forecast_{FORECAST_YEAR}.png"
        return {
            "forecast_png": plot_ndvi_map(
                forecast_path,
                output,
                f"Прогноз NDVI {FORECAST_YEAR}",
                self._boundary_path,
                display_range=self._ndvi_display_range,
            )
        }

    def _stage_12_exports(
        self,
        summary: dict[str, object],
        tables: dict[str, pd.DataFrame],
        generated_files: dict[str, Path],
    ) -> dict[str, Path]:
        self.logger.info("Stage 12: Excel and Word export.")
        excel_path = self.paths.tables_dir / "ndvi_monitor_results.xlsx"
        report_path = self.paths.reports_dir / "ndvi_monitor_report.docx"
        export_excel_report(excel_path, summary, tables)
        files_for_report = {
            **generated_files,
            "excel_report": excel_path,
            "word_report": report_path,
        }
        generate_word_report(
            report_path,
            "NDVI Monitor: отчет по зеленым насаждениям",
            summary,
            tables,
            files_for_report,
        )
        return {"excel_report": excel_path, "word_report": report_path}

    def _build_summary(self, bundle: InputBundle) -> dict[str, object]:
        return {
            "input_variant": bundle.variant.value,
            "base_year": BASE_YEAR,
            "target_year": TARGET_YEAR,
            "forecast_year": FORECAST_YEAR,
            "target_crs": ALIGNMENT_TARGET_CRS,
            "target_resolution_m": ALIGNMENT_RESOLUTION,
            "boundary_crs": bundle.boundary_metadata.crs,
            "boundary_feature_count": bundle.boundary_metadata.feature_count,
            "area_method": PIXEL_AREA_METHOD,
            "boundary_file": str(bundle.boundary),
            "log_file": str(self.paths.log_file),
        }


def ndvi_paths_to_outputs(ndvi_paths: dict[int, Path]) -> dict[str, Path]:
    """Convert NDVI output paths to report labels."""

    return {f"ndvi_{year}_geotiff": path for year, path in ndvi_paths.items()}


def class_paths_to_outputs(
    class_paths: dict[int, Path],
    prefix: str,
) -> dict[str, Path]:
    """Convert class raster paths to report labels."""

    return {f"{prefix}_{year}_geotiff": path for year, path in class_paths.items()}


def class_figure_paths_to_outputs(
    class_paths: dict[int, Path],
    paths: ProjectPaths,
) -> dict[str, Path]:
    """Convert class figure paths to report labels."""

    return {
        f"ndvi_classes_{year}_png": paths.figures_dir / f"ndvi_classes_{year}.png"
        for year in class_paths
    }
