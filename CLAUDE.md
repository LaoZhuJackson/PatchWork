# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**PatchWork** — a PySide6 + QFluentWidgets desktop GUI that bundles several PV (photovoltaic) defect-detection utilities into one tool. Target users are the developer and colleagues; distributed as a single `.exe` via PyInstaller. Full background: `PROJECT_BOOTSTRAP.md` (if present) and `README.md`. UI strings and code comments are in Chinese.

## Tech Stack

- **Python ≥ 3.10** (conda env, typically named `patchwork`)
- **GUI**: PySide6 + [QFluentWidgets](https://qfluentwidgets.com/) (`qfluentwidgets`, package `PySide6-Fluent-Widgets`) — Fluent Design widgets and `FluentWindow` navigation
- **Image preview**: `QGraphicsView` (see `app/widgets/image_viewer.py`)
- **ML / inference**: `ultralytics` (YOLO) + `supervision` (`sv.Detections.from_ultralytics`); `opencv` (transitively via ultralytics) is imported directly in `inference.py`
- **Packaging**: PyInstaller → single `.exe`
- **SSH** (planned features): `paramiko`

## Commands

```bash
conda activate patchwork          # activate env
pip install -e .                  # install app + deps (editable); use .[dev] for pyinstaller/semantic-release
python main.py                    # run the app during development
pyinstaller --onefile --windowed main.py   # package as a single windowed exe
```

There are no automated tests. Verify changes by running `python main.py` and exercising the relevant panel.

## Architecture

Three layers, strictly one-directional: **widgets (UI) → services (logic, no Qt-UI) → utils (infra)**.

```
main.py                     # entry: builds QApplication, wires logging→popup bridge, shows MainWindow
app/
├── main_window.py          # FluentWindow: registers each panel as a nav sub-interface
├── widgets/                # one QWidget panel per feature; panels are mutually independent
│   ├── image_viewer.py     # shared QGraphicsView: zoom (wheel), pan (drag), bbox/polygon/text overlays
│   ├── dataset_split.py    # 数据集划分 — pairing check + train/val/test split (copy or move)
│   ├── model_infer.py      # 模型推理 — model load + click-to-infer + overlay preview
│   ├── label_preview.py    # Label 预览 — draw YOLO labels over images
│   └── export_onnx.py      # 导出 ONNX — YOLO .pt → ONNX
├── services/               # pure logic, no widget imports; safe to call from a Worker thread
│   ├── splitter.py         # find_pairs() + split_dataset()
│   ├── inference.py        # InferenceEngine: load once, infer() → annotation dicts (det + seg)
│   ├── label_reader.py     # parse_yolo_label() denormalize; CLASS_COLORS + get_color()
│   └── exporter.py         # ONNXExporter wrapping ultralytics model.export
└── utils/
    ├── worker.py           # Worker(QThread) base — override do_work(), emits finished/error/progress
    ├── logger.py           # logging setup + Qt signal bridge (WARNING+ → popup)
    ├── message.py          # MessageBox helpers: info/warning/error/confirm
    └── config.py           # QSettings("PatchWork","PatchWork") get/set str/int/bool
```

`GPU 监控` and `X-AnyLabeling` appear in the nav but are currently bare `QLabel` placeholders in `main_window.py` — their services/widgets do not exist yet. `README.md` also lists 视频抽帧 and 最优置信度 as planned (⏳).

## Key Patterns & Conventions

**Async via `Worker` (`app/utils/worker.py`).** Any blocking work (model load, inference, file copy/move, future SSH) runs off the UI thread. Subclass `Worker`, put the work in `do_work()` (its return value is emitted via `finished`), and connect `finished`/`error`/`progress` before `.start()`. Services accept an optional `progress_callback` that the Worker wires to `self.progress.emit`. Store the worker on `self._worker` so it isn't garbage-collected mid-run. See `SplitWorker`, `LoadModelWorker`, `InferWorker`.

**Annotation dict — the contract between services and `ImageViewer`.** Both `label_reader.parse_yolo_label()` and `InferenceEngine.infer()` return `list[dict]` with the *same* shape, and every panel renders them by looping and dispatching on `type`:
```python
{"type": "bbox",    "rect": QRectF, "class_id": int, "color": QColor, "label": str}
{"type": "polygon", "points": [QPointF, ...], "class_id": int, "color": QColor, "label": str}
```
Consume with `viewer.add_bbox(ann["rect"], ann["color"], ann["label"])` / `viewer.add_polygon(...)`. Keep this shape stable when adding formats or detectors.

**Per-class colors.** `label_reader.get_color(class_id)` cycles a fixed 20-color `CLASS_COLORS` list; both label preview and inference use it so a class keeps its color across panels.

**Logging doubles as user alerts.** `setup_logging()` (called once in `main.py`) installs a `QtLogHandler` that emits any **WARNING/ERROR/CRITICAL** log record over a Qt signal, which `main.py` turns into a `MessageBox` popup. So `get_logger(__name__).warning(...)` both writes to `logs/patchwork_<date>.log` and pops a dialog — don't log at WARNING+ for routine/expected conditions. Use `app.utils.message` (`info/warning/error/confirm`) for direct, intentional dialogs.

**Config persistence.** Every path/host/option the user picks is saved via `app/utils/config.py` (QSettings) and restored in each panel's `_load_settings()`. Keys are plain strings (e.g. `img_dir`, `infer_model_path`). **Never persist SSH passwords** — prompt per session or require key auth.

**Adding a panel.** Create `app/widgets/<name>.py` as a self-contained `QWidget` with a unique `setObjectName(...)` (QFluentWidgets requires it for navigation), then register it in `MainWindow._register_navigation()` via `addSubInterface(widget, FIF.<icon>, "<label>", position=...)`.

## Gotchas

- **`formats/` was removed.** Earlier docs described a `BaseFormat` plugin system in `app/formats/`; that module was deleted (and dropped from `pyproject.toml` to fix `pip install`). Do not reintroduce references to it. YOLO-format logic now lives directly in `splitter.py` / `label_reader.py`.
- **`exporter.py` `chdir`.** ultralytics `model.export()` writes ONNX to the current working directory, so `ONNXExporter.export` temporarily `os.chdir` into the target dir and restores it in a `finally`.
- **YOLOv12 weight/version mismatch.** `InferenceEngine.load_model` catches the `AttributeError` (`qkv`/`AAttn`) and re-raises a `RuntimeError` with an upgrade hint — preserve that when touching model loading.
- **Thumbnails are loaded eagerly.** `model_infer.py` / `label_preview.py` build a full `QPixmap` thumbnail for every image up front (not viewport-lazy). Keep this in mind for very large folders.
- **Commits use Conventional Commits** (`fix:`, `feat:`, `init:` …); `python-semantic-release` is a dev dependency for changelog/release, though no `[tool.semantic_release]` config is committed yet.
