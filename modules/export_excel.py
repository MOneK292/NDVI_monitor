"""Excel export for NDVI Monitor tables."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd

from modules.utils import ensure_parent


def export_excel_report(
    output_path: Path,
    summary: Mapping[str, object],
    tables: Mapping[str, pd.DataFrame],
) -> Path:
    """Write monitoring results to an Excel workbook."""

    ensure_parent(output_path)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary_table = pd.DataFrame(
            [{"parameter": key, "value": value} for key, value in summary.items()]
        )
        summary_table.to_excel(writer, sheet_name="summary", index=False)
        _fit_columns(writer, "summary", summary_table)

        for sheet_name, table in tables.items():
            safe_name = _safe_sheet_name(sheet_name)
            table.to_excel(writer, sheet_name=safe_name, index=False)
            _fit_columns(writer, safe_name, table)

    return output_path


def _safe_sheet_name(name: str) -> str:
    replacements = {
        "\\": "_",
        "/": "_",
        "*": "_",
        "[": "(",
        "]": ")",
        ":": "_",
        "?": "_",
    }
    result = name
    for old, new in replacements.items():
        result = result.replace(old, new)
    return result[:31]


def _fit_columns(
    writer: pd.ExcelWriter,
    sheet_name: str,
    table: pd.DataFrame,
) -> None:
    worksheet = writer.sheets[sheet_name]
    for column_index, column_name in enumerate(table.columns, start=1):
        values = table[column_name].astype(str).tolist()
        width = min(max([len(str(column_name)), *map(len, values)]) + 2, 48)
        column_letter = worksheet.cell(row=1, column=column_index).column_letter
        worksheet.column_dimensions[column_letter].width = width
