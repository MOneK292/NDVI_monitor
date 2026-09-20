"""NDVI and change classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from config import CLASS_NODATA, DELTA_CLASS_THRESHOLDS, NDVI_CLASS_THRESHOLDS
from modules.utils import read_masked_raster, write_single_band_geotiff


@dataclass(frozen=True)
class ClassRule:
    """A numeric interval mapped to a class identifier."""

    class_id: int
    label: str
    lower: float | None
    upper: float | None
    color: str
    lower_inclusive: bool = True
    upper_inclusive: bool = False

    def contains(self, values: np.ndarray) -> np.ndarray:
        """Return a boolean mask of values belonging to this class."""

        mask = np.ones(values.shape, dtype=bool)
        if self.lower is not None:
            if self.lower_inclusive:
                mask &= values >= self.lower
            else:
                mask &= values > self.lower
        if self.upper is not None:
            if self.upper_inclusive:
                mask &= values <= self.upper
            else:
                mask &= values < self.upper
        return mask


def build_class_rules(
    thresholds: Sequence[tuple[int, str, float | None, float | None, str, bool, bool]],
) -> tuple[ClassRule, ...]:
    """Build class rules from configuration tuples."""

    return tuple(ClassRule(*threshold) for threshold in thresholds)


NDVI_CLASS_RULES = build_class_rules(NDVI_CLASS_THRESHOLDS)
DELTA_CLASS_RULES = build_class_rules(DELTA_CLASS_THRESHOLDS)


def classify_raster(
    raster_path: Path,
    output_path: Path,
    rules: Sequence[ClassRule],
) -> Path:
    """Classify a continuous raster according to numeric class rules."""

    values, profile = read_masked_raster(raster_path)
    classes = classify_array(values, rules)
    return write_single_band_geotiff(output_path, classes, profile, CLASS_NODATA, "uint8")


def classify_array(
    values: np.ma.MaskedArray,
    rules: Sequence[ClassRule],
) -> np.ndarray:
    """Classify a masked numeric array into integer class IDs."""

    data = values.data
    valid = ~np.ma.getmaskarray(values) & np.isfinite(data)
    classes = np.full(data.shape, CLASS_NODATA, dtype="uint8")

    for rule in rules:
        rule_mask = valid & rule.contains(data)
        classes[rule_mask] = rule.class_id

    return classes
