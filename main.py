"""Command line entry point for NDVI Monitor."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from config import ProjectPaths


def build_parser() -> argparse.ArgumentParser:
    """Build the command line argument parser."""

    parser = argparse.ArgumentParser(
        description="Automated NDVI monitoring for Primorsky District green areas."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input directory with Sentinel-2 rasters and one vector boundary.",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Open the tkinter graphical interface.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity.",
    )
    return parser


def main() -> int:
    """Run the CLI pipeline or start the graphical interface."""

    _configure_console_encoding()
    args = build_parser().parse_args()
    if args.gui:
        try:
            from modules.gui import launch_gui
        except ModuleNotFoundError as exc:
            print(_dependency_error_message(exc))
            return 1

        launch_gui()
        return 0

    try:
        from modules.loader import InputDataError
        from modules.pipeline import MonitorPipeline
    except ModuleNotFoundError as exc:
        print(_dependency_error_message(exc))
        return 1

    paths = ProjectPaths.from_base()
    level = getattr(logging, args.log_level)
    pipeline = MonitorPipeline(paths, level)

    try:
        result = pipeline.run(input_dir=args.input)
    except (InputDataError, ValueError, RuntimeError) as exc:
        pipeline.logger.error("Processing failed: %s", exc)
        print(f"Ошибка обработки: {exc}")
        return 1

    print("Обработка завершена.")
    print(f"Excel: {result.generated_files['excel_report']}")
    print(f"Word: {result.generated_files['word_report']}")
    print(f"Лог: {paths.log_file}")
    return 0


def _dependency_error_message(error: ModuleNotFoundError) -> str:
    """Build a user-facing message for missing Python dependencies."""

    missing = error.name or str(error)
    return (
        f"Не установлена зависимость Python: {missing}. "
        "Выполните: pip install -r requirements.txt"
    )


def _configure_console_encoding() -> None:
    """Use UTF-8 console streams when Python supports reconfiguration."""

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


if __name__ == "__main__":
    raise SystemExit(main())
