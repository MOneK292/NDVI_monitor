"""Tkinter graphical interface for NDVI Monitor."""

from __future__ import annotations

import logging
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

from config import ProjectPaths
from modules.pipeline import MonitorPipeline


class MonitorGUI:
    """Desktop GUI wrapper around the monitoring pipeline."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.paths = ProjectPaths.from_base()
        self.message_queue: queue.Queue[str] = queue.Queue()
        self.worker: threading.Thread | None = None

        self.root.title("NDVI Monitor")
        self.root.geometry("840x560")
        self.root.minsize(720, 460)

        self.input_dir_var = tk.StringVar(value=str(self.paths.input_dir))
        self.status_var = tk.StringVar(value="Готово")

        self._build_ui()
        self._poll_messages()

    def _build_ui(self) -> None:
        frame = ttk.Frame(self.root, padding=16)
        frame.pack(fill=tk.BOTH, expand=True)

        input_frame = ttk.Frame(frame)
        input_frame.pack(fill=tk.X)

        ttk.Label(input_frame, text="Папка входных данных").pack(anchor=tk.W)
        path_row = ttk.Frame(input_frame)
        path_row.pack(fill=tk.X, pady=(4, 12))

        input_entry = ttk.Entry(path_row, textvariable=self.input_dir_var)
        input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        ttk.Button(path_row, text="Выбрать", command=self._choose_input_dir).pack(
            side=tk.LEFT,
            padx=(8, 0),
        )

        action_row = ttk.Frame(frame)
        action_row.pack(fill=tk.X, pady=(0, 12))

        self.run_button = ttk.Button(
            action_row,
            text="Запустить обработку",
            command=self._start_pipeline,
        )
        self.run_button.pack(side=tk.LEFT)

        self.progress = ttk.Progressbar(action_row, mode="indeterminate")
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(12, 0))

        ttk.Label(frame, textvariable=self.status_var).pack(anchor=tk.W)

        self.log_text = scrolledtext.ScrolledText(frame, wrap=tk.WORD, height=18)
        self.log_text.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        self.log_text.configure(state=tk.DISABLED)

    def _choose_input_dir(self) -> None:
        selected = filedialog.askdirectory(
            initialdir=self.input_dir_var.get(),
            title="Выберите папку с входными данными",
        )
        if selected:
            self.input_dir_var.set(selected)

    def _start_pipeline(self) -> None:
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("NDVI Monitor", "Обработка уже выполняется.")
            return

        input_dir = Path(self.input_dir_var.get())
        self.status_var.set("Выполняется обработка")
        self.run_button.configure(state=tk.DISABLED)
        self.progress.start(12)
        self._append_log("Запуск конвейера обработки.")

        self.worker = threading.Thread(
            target=self._run_pipeline_worker,
            args=(input_dir,),
            daemon=True,
        )
        self.worker.start()

    def _run_pipeline_worker(self, input_dir: Path) -> None:
        try:
            pipeline = MonitorPipeline(self.paths, logging.INFO)
            result = pipeline.run(input_dir=input_dir)
            self.message_queue.put(
                "Обработка завершена. "
                f"Сформировано файлов: {len(result.generated_files)}."
            )
            self.message_queue.put("__DONE__")
        except Exception as exc:
            self.message_queue.put(f"Ошибка: {exc}")
            self.message_queue.put("__ERROR__")

    def _poll_messages(self) -> None:
        while not self.message_queue.empty():
            message = self.message_queue.get()
            if message == "__DONE__":
                self._finish_run("Готово")
            elif message == "__ERROR__":
                self._finish_run("Ошибка обработки")
            else:
                self._append_log(message)
        self.root.after(200, self._poll_messages)

    def _finish_run(self, status: str) -> None:
        self.status_var.set(status)
        self.progress.stop()
        self.run_button.configure(state=tk.NORMAL)

    def _append_log(self, message: str) -> None:
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)


def launch_gui() -> None:
    """Start the tkinter user interface."""

    root = tk.Tk()
    MonitorGUI(root)
    root.mainloop()
