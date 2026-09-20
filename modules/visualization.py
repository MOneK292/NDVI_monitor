"""Map and chart rendering for NDVI Monitor outputs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import geopandas as gpd

from config import BASE_DIR

os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / "logs" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib import colormaps
from matplotlib.colors import (
    BoundaryNorm,
    LinearSegmentedColormap,
    ListedColormap,
    TwoSlopeNorm,
)
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MaxNLocator

from config import (
    CLASS_NODATA,
    DATA_SOURCE_LABEL,
)
from modules.classify import ClassRule
from modules.utils import ensure_parent, read_masked_raster


FIGURE_DPI = 300
MAP_FIGSIZE = (10, 10)
BAR_FIGSIZE = (12, 6.8)
FONT_FAMILY = "DejaVu Sans"
TITLE_SIZE = 19
AXIS_LABEL_SIZE = 15
TICK_LABEL_SIZE = 11
LEGEND_SIZE = 12
COLORBAR_LABEL_SIZE = 13
MAP_GRID_COLOR = "#9a9a9a"
BOUNDARY_LINEWIDTH = 0.8
NDVI_DISPLAY_PERCENTILES = (5.0, 99.0)
DELTA_DISPLAY_PERCENTILES = (2.0, 98.0)
NDVI_COLORS = ("#f4ecd8", "#e8e9a3", "#b8d66b", "#63ad48", "#176b35", "#063d24")
NDVI_CLASS_COLORS = ("#8c510a", "#d9a441", "#f4e184", "#7fbc41", "#1a9850")
DELTA_CLASS_COLORS = ("#b2182b", "#ef8a62", "#f7f7f7", "#67a9cf", "#2166ac")


def _configure_article_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": FONT_FAMILY,
            "axes.titlesize": TITLE_SIZE,
            "axes.titleweight": "semibold",
            "axes.labelsize": AXIS_LABEL_SIZE,
            "xtick.labelsize": TICK_LABEL_SIZE,
            "ytick.labelsize": TICK_LABEL_SIZE,
            "legend.fontsize": LEGEND_SIZE,
            "axes.edgecolor": "#222222",
            "axes.linewidth": 0.8,
        }
    )


_configure_article_style()


def plot_ndvi_map(
    raster_path: Path,
    output_path: Path,
    title: str,
    boundary_path: Path | None = None,
    source_label: str = DATA_SOURCE_LABEL,
    display_range: tuple[float, float] | None = None,
) -> Path:
    """Render a continuous NDVI map as PNG."""

    vmin, vmax = display_range or calculate_display_range(
        (raster_path,),
        percentiles=NDVI_DISPLAY_PERCENTILES,
    )
    return _plot_continuous_map(
        raster_path,
        output_path,
        title,
        cmap=_ndvi_colormap(),
        vmin=vmin,
        vmax=vmax,
        colorbar_label="NDVI",
        boundary_path=boundary_path,
        source_label=source_label,
    )


def plot_delta_map(
    raster_path: Path,
    output_path: Path,
    title: str,
    boundary_path: Path | None = None,
    source_label: str = DATA_SOURCE_LABEL,
) -> Path:
    """Render a Delta NDVI map as PNG."""

    vmin, vmax = calculate_display_range(
        (raster_path,),
        include_zero=True,
        percentiles=DELTA_DISPLAY_PERCENTILES,
    )
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    return _plot_continuous_map(
        raster_path,
        output_path,
        title,
        cmap="RdYlGn",
        vmin=None,
        vmax=None,
        norm=norm,
        colorbar_label="ΔNDVI",
        boundary_path=boundary_path,
        source_label=source_label,
    )


def calculate_display_range(
    raster_paths: Sequence[Path],
    include_zero: bool = False,
    percentiles: tuple[float, float] = NDVI_DISPLAY_PERCENTILES,
) -> tuple[float, float]:
    """Calculate a robust common display range from valid raster values."""

    valid_values = []
    for raster_path in raster_paths:
        values, _ = read_masked_raster(raster_path)
        valid = ~np.ma.getmaskarray(values) & np.isfinite(values.data)
        if np.any(valid):
            valid_values.append(values.data[valid].astype("float32", copy=False))
    if not valid_values:
        raise ValueError("Cannot calculate display range: rasters have no valid pixels.")

    combined = np.concatenate(valid_values)
    vmin, vmax = np.percentile(
        combined,
        percentiles,
    )
    if include_zero:
        vmin = float(vmin)
        vmax = float(vmax)
        if vmin >= 0.0:
            vmin = -max(vmax * 0.05, 0.01)
        if vmax <= 0.0:
            vmax = max(abs(vmin) * 0.05, 0.01)
    vmin, vmax = _expand_degenerate_range(float(vmin), float(vmax))
    return vmin, vmax


def plot_classification_map(
    class_raster_path: Path,
    output_path: Path,
    title: str,
    rules: Sequence[ClassRule],
    boundary_path: Path | None = None,
    source_label: str = DATA_SOURCE_LABEL,
) -> Path:
    """Render an integer classification map as PNG."""

    ensure_parent(output_path)
    with rasterio.open(class_raster_path) as dataset:
        classes = dataset.read(1)
        extent = _dataset_extent(dataset)
        crs_text = str(dataset.crs)

    masked_classes = np.ma.masked_where(classes == CLASS_NODATA, classes)
    sorted_rules = tuple(sorted(rules, key=lambda item: item.class_id))
    colors = _classification_colors(sorted_rules, title)
    cmap = ListedColormap(colors)
    cmap.set_bad((1, 1, 1, 0))
    boundaries = np.arange(0.5, len(colors) + 1.5, 1)
    norm = BoundaryNorm(boundaries, cmap.N)

    fig, ax = _create_map_figure()
    ax.imshow(masked_classes, cmap=cmap, norm=norm, extent=extent)
    _style_map_title(ax, title)
    legend_items = [
        Patch(facecolor=color, edgecolor="#333333", linewidth=0.4, label=f"{rule.class_id}: {rule.label}")
        for rule, color in zip(sorted_rules, colors, strict=False)
    ]
    _decorate_map(ax, extent, crs_text, boundary_path, class_raster_path, source_label)
    _add_class_legend(
        ax,
        legend_items,
        compact=_is_delta_classification(sorted_rules, title),
    )
    _save_figure(fig, output_path)
    plt.close(fig)
    return output_path


def plot_class_area_bar(
    class_statistics: pd.DataFrame,
    output_path: Path,
    title: str,
) -> Path:
    """Render a class area bar chart as PNG."""

    ensure_parent(output_path)
    fig, ax = plt.subplots(figsize=BAR_FIGSIZE)
    fig.patch.set_facecolor("white")
    values = class_statistics["area_ha"].astype(float)
    colors = _bar_colors(len(values), title)
    bars = ax.bar(
        class_statistics["class_id"].astype(str),
        values,
        color=colors,
        edgecolor="#222222",
        linewidth=0.6,
    )
    _style_bar_axes(ax, title)
    ax.set_xlabel("Класс")
    ax.set_ylabel("Площадь, га")
    _add_bar_labels(ax, bars, values, class_statistics.get("percent"))
    fig.tight_layout()
    _save_figure(fig, output_path)
    plt.close(fig)
    return output_path


def _plot_continuous_map(
    raster_path: Path,
    output_path: Path,
    title: str,
    cmap: str | matplotlib.colors.Colormap,
    vmin: float | None,
    vmax: float | None,
    colorbar_label: str,
    boundary_path: Path | None,
    source_label: str,
    norm: TwoSlopeNorm | None = None,
) -> Path:
    ensure_parent(output_path)
    values, _ = read_masked_raster(raster_path)
    with rasterio.open(raster_path) as dataset:
        extent = _dataset_extent(dataset)
        crs_text = str(dataset.crs)

    fig, ax = _create_map_figure()
    colormap = _continuous_colormap(cmap)
    image = ax.imshow(
        values,
        cmap=colormap,
        vmin=vmin,
        vmax=vmax,
        norm=norm,
        extent=extent,
        interpolation="nearest",
    )
    _style_map_title(ax, title)
    _decorate_map(ax, extent, crs_text, boundary_path, raster_path, source_label)
    _add_colorbar(fig, ax, image, colorbar_label)
    _save_figure(fig, output_path)
    plt.close(fig)
    return output_path


def _create_map_figure() -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=MAP_FIGSIZE)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    return fig, ax


def _continuous_colormap(
    source: str | matplotlib.colors.Colormap,
) -> matplotlib.colors.Colormap:
    cmap = (
        colormaps[source].resampled(256).copy()
        if isinstance(source, str)
        else source.resampled(256).copy()
    )
    cmap.set_bad((1, 1, 1, 0))
    return cmap


def _ndvi_colormap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("ndvi_article", NDVI_COLORS, N=256)


def _style_map_title(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight="semibold", pad=14)


def _add_colorbar(
    fig: plt.Figure,
    ax: plt.Axes,
    image: matplotlib.image.AxesImage,
    label: str,
) -> None:
    colorbar = fig.colorbar(
        image,
        ax=ax,
        fraction=0.018,
        pad=0.012,
        shrink=0.80,
        aspect=42,
    )
    colorbar.set_label(label, fontsize=12, fontweight="semibold", labelpad=8)
    colorbar.ax.tick_params(labelsize=TICK_LABEL_SIZE, width=0.7, length=3)
    colorbar.locator = MaxNLocator(nbins=6)
    colorbar.update_ticks()
    colorbar.outline.set_linewidth(0.6)


def _add_class_legend(
    ax: plt.Axes,
    legend_items: list[Patch],
    compact: bool = False,
) -> None:
    legend = ax.legend(
        handles=legend_items,
        loc="upper left",
        bbox_to_anchor=(0.1, 1),
        frameon=True,
        title="Классы",
        borderpad=0.3,
        labelspacing=0.20,
        handlelength=0.4,
        handleheight=0.7,
        fontsize=11 if compact else LEGEND_SIZE,
    )
    frame = legend.get_frame()
    frame.set_facecolor("white")
    frame.set_alpha(0.83)
    frame.set_edgecolor("#bdbdbd")
    frame.set_linewidth(0.6)
    legend.get_title().set_fontsize(LEGEND_SIZE)
    legend.get_title().set_fontweight("semibold")


def _classification_colors(rules: Sequence[ClassRule], title: str) -> list[str]:
    if _is_delta_classification(rules, title):
        palette = DELTA_CLASS_COLORS
    else:
        palette = NDVI_CLASS_COLORS
    if len(rules) <= len(palette):
        return list(palette[: len(rules)])
    cmap = colormaps["Set2"].resampled(len(rules))
    return [matplotlib.colors.to_hex(cmap(index)) for index in range(len(rules))]


def _is_delta_classification(rules: Sequence[ClassRule], title: str) -> bool:
    lowered_title = title.lower()
    if "измен" in lowered_title or "delta" in lowered_title:
        return True
    labels = " ".join(rule.label.lower() for rule in rules)
    return "ухудш" in labels or "улучш" in labels


def _bar_colors(count: int, title: str) -> list[str]:
    palette = DELTA_CLASS_COLORS if "измен" in title.lower() else NDVI_CLASS_COLORS
    return [palette[index % len(palette)] for index in range(count)]


def _style_bar_axes(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight="semibold", pad=14)
    ax.set_facecolor("white")
    ax.grid(axis="y", color="#bdbdbd", alpha=0.45, linewidth=0.55)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE, width=0.7)
    ax.set_axisbelow(True)


def _add_bar_labels(
    ax: plt.Axes,
    bars: Sequence[Rectangle],
    values: pd.Series,
    percentages: pd.Series | None,
) -> None:
    max_value = float(values.max()) if len(values) else 0.0
    offset = max(max_value * 0.015, 0.05)
    ax.set_ylim(0, max_value * 1.12 if max_value > 0 else 1.0)
    if percentages is None:
        percentages = values / values.sum() * 100.0 if values.sum() else values * 0.0
    for bar, value, percent in zip(bars, values, percentages, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + offset,
            f"{_format_bar_value(float(value))} га ({float(percent):.1f} %)".replace(".", ","),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="semibold",
            color="#222222",
        )


def _format_bar_value(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:,.0f}".replace(",", " ")
    return f"{value:,.1f}".replace(",", " ")


def _save_figure(fig: plt.Figure, output_path: Path) -> None:
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")


def _legend_box(alpha: float = 0.82, pad: float = 0.25) -> dict[str, object]:
    return {
        "boxstyle": f"round,pad={pad}",
        "facecolor": "white",
        "alpha": alpha,
        "edgecolor": "#bdbdbd",
        "linewidth": 0.5,
    }


def _decorate_map(
    ax: plt.Axes,
    extent: tuple[float, float, float, float],
    crs_text: str,
    boundary_path: Path | None,
    raster_path: Path,
    source_label: str,
) -> None:
    ax.set_xlabel("X, м")
    ax.set_ylabel("Y, м")
    ax.grid(color=MAP_GRID_COLOR, alpha=0.20, linewidth=0.35)
    ax.ticklabel_format(style="plain", useOffset=False)
    ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_SIZE, width=0.7)
    _plot_boundary(ax, boundary_path, raster_path)
    _add_north_arrow(ax)
    _add_scale_bar(ax, extent)
    ax.text(
        0.2,
        0.02,
        f"CRS: {crs_text}\nИсточник: {source_label}",
        transform=ax.transAxes,
        fontsize=10,
        va="bottom",
        ha="right",
        bbox=_legend_box(alpha=0.78),
    )


def _plot_boundary(
    ax: plt.Axes,
    boundary_path: Path | None,
    raster_path: Path,
) -> None:
    if boundary_path is None:
        return
    with rasterio.open(raster_path) as dataset:
        raster_crs = dataset.crs
    boundary = gpd.read_file(boundary_path)
    if boundary.crs != raster_crs:
        boundary = boundary.to_crs(raster_crs)
    boundary.boundary.plot(ax=ax, color="black", linewidth=BOUNDARY_LINEWIDTH)


def _add_north_arrow(ax: plt.Axes) -> None:
    ax.annotate(
        "N",
        xy=(0.965, 0.885),
        xytext=(0.965, 0.795),
        xycoords="axes fraction",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        bbox=_legend_box(alpha=0.70, pad=0.18),
        arrowprops={
            "arrowstyle": "-|>",
            "color": "black",
            "lw": 1.15,
            "mutation_scale": 13,
            "shrinkA": 0,
            "shrinkB": 2,
        },
    )


def _add_scale_bar(
    ax: plt.Axes,
    extent: tuple[float, float, float, float],
) -> None:
    min_x, max_x, min_y, max_y = extent
    map_width = max_x - min_x
    map_height = max_y - min_y
    scale_length = _nice_scale_length(map_width / 5.5)
    x_start = min_x + map_width * 0.21
    y_start = min_y + map_height * 0.02
    bar_height = map_height * 0.008
    segment = scale_length / 2.0
    ax.add_patch(
        Rectangle(
            (x_start, y_start),
            segment,
            bar_height,
            facecolor="black",
            edgecolor="black",
            linewidth=0.8,
            zorder=6,
        )
    )
    ax.add_patch(
        Rectangle(
            (x_start + segment, y_start),
            segment,
            bar_height,
            facecolor="white",
            edgecolor="black",
            linewidth=0.8,
            zorder=6,
        )
    )
    ax.plot(
        [x_start, x_start, x_start + scale_length, x_start + scale_length],
        [y_start, y_start + bar_height * 1.8, y_start + bar_height * 1.8, y_start],
        color="black",
        linewidth=0.8,
        zorder=7,
    )
    ax.text(
        x_start + scale_length / 2.0,
        y_start + bar_height * 2.35,
        _format_scale_length(scale_length),
        ha="center",
        va="bottom",
        fontsize=11,
        bbox=_legend_box(alpha=0.78, pad=0.12),
    )


def _nice_scale_length(length: float) -> float:
    if length <= 0:
        return 1.0
    exponent = np.floor(np.log10(length))
    base = length / (10 ** exponent)
    if base >= 5:
        nice = 5
    elif base >= 2:
        nice = 2
    else:
        nice = 1
    return float(nice * (10 ** exponent))


def _format_scale_length(length: float) -> str:
    return f"{length / 1000:g} км"


def _expand_degenerate_range(vmin: float, vmax: float) -> tuple[float, float]:
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        raise ValueError("Display range contains non-finite values.")
    if vmax > vmin:
        return vmin, vmax
    margin = max(abs(vmin) * 0.05, 0.01)
    return vmin - margin, vmax + margin


def _dataset_extent(dataset: rasterio.io.DatasetReader) -> tuple[float, float, float, float]:
    bounds = dataset.bounds
    return (float(bounds.left), float(bounds.right), float(bounds.bottom), float(bounds.top))
