# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**PatchWork** — a PySide6 + QFluentWidgets desktop GUI that bundles PV defect-detection utilities into one tool. Target: developer + colleagues. Distributed as a single `.exe` via PyInstaller. UI strings and code comments are in Chinese.

## Tech Stack

- **Python >= 3.10** (conda env `patchwork`)
- **GUI**: PySide6 + QFluentWidgets (`PySide6-Fluent-Widgets`), FluentWindow navigation, light/dark theme toggle
- **ML**: ultralytics (YOLO) + supervision; also SAHI (`sahi`) for sliced inference
- **Video**: opencv-python (video extract, video tracking)
- **SSH**: paramiko (GPU monitor)
- **Packaging**: PyInstaller
- **Logging**: stdlib `logging` + Qt signal bridge (WARNING+ → popup) + `faulthandler` (segfault trace)

## Commands

```bash
conda activate patchwork
pip install -e .
python main.py
```

No automated tests. Verify changes by running `python main.py` and exercising the panel.

## Architecture

```
main.py                     # entry: QApplication, logging, crash handler, theme, MainWindow
app/
├── main_window.py          # FluentWindow: wraps each panel in ScrollArea, nav + theme toggle
├── adapters/               # unified inference interface for benchmark
│   ├── base.py             # InferenceAdapter ABC
│   ├── normal_adapter.py   # wraps InferenceEngine
│   ├── sahi_adapter.py     # wraps SahiInferenceService
│   └── tracking_adapter.py # wraps YOLO model.track()
├── widgets/                # one QWidget panel per feature; mutually independent
│   ├── image_viewer.py     # QGraphicsView: zoom, pan, bbox/polygon/text overlays
│   ├── thumbnail_list.py   # horizontal lazy-loading thumbnail strip (background thread)
│   ├── image_browser.py    # composable: ThumbnailList + ImageViewer + nav buttons
│   ├── dataset_split.py    # F1: pairing check + train/val/test split (copy/move)
│   ├── model_infer.py      # F2: model load + click-to-infer + conf/iou controls
│   ├── label_preview.py    # F4: YOLO label overlay preview
│   ├── export_onnx.py      # F5: .pt → ONNX, imgsz/simplify/dynamic options
│   ├── gpu_monitor.py      # F6: remote GPU (nvidia-smi / gpustat / HTTP)
│   ├── xanylabeling.py     # F7: subprocess launch X-AnyLabeling
│   ├── video_extract.py    # F8: extract frames by time/frame interval
│   ├── sahi_infer.py       # SAHI sliced inference panel
│   ├── video_track.py      # video tracking panel (BoT-SORT / ByteTrack)
│   └── benchmark.py        # multi-adapter evaluation + comparison table
├── services/               # pure logic, no Qt imports; safe from Worker threads
│   ├── splitter.py         # find_pairs() + split_dataset()
│   ├── inference.py        # InferenceEngine: detect + seg, conf/iou params
│   ├── label_reader.py     # parse_yolo_label(), CLASS_COLORS, get_color()
│   ├── exporter.py         # ONNXExporter wrapping model.export()
│   ├── gpu_client.py       # fetch_via_nvidia_smi / fetch_via_gpustat / fetch_via_http
│   ├── xanylabeling.py     # subprocess.Popen wrapper
│   ├── video_extractor.py  # cv2 frame extraction
│   ├── sahi_inference.py   # SahiInferenceService (AutoDetectionModel)
│   ├── video_tracking.py   # VideoTrackingService (model.track)
│   ├── benchmark.py        # BenchmarkRunner: iterate adapters × images → per-class metrics
│   └── metrics.py          # hand-written IoU matching per class → P/R/F1/AP50
└── utils/
    ├── worker.py           # Worker(QThread): override do_work(), signals finished/error/progress
    ├── logger.py           # logging setup + QtLogHandler (WARNING+ → popup)
    ├── message.py          # MessageBox wrappers: info/warning/error/confirm
    ├── config.py           # QSettings("PatchWork","PatchWork") get/set for str/int/float/bool
    └── crash_handler.py    # sys.excepthook + Qt message handler + faulthandler → log files
```

## Key Patterns

- **Async via Worker.** Subclass `Worker`, put work in `do_work()`, connect `finished`/`error`/`progress` before `.start()`. Store on `self._worker` so it isn't GC'd. Use `progress_callback` parameter on services.
- **Annotation dict contract.**
  ```python
  {"type": "bbox",    "rect": QRectF, "class_id": int, "color": QColor, "label": str}
  {"type": "polygon", "points": [QPointF, ...], "class_id": int, "color": QColor, "label": str}
  ```
- **Colors.** `label_reader.get_color(class_id)` returns consistent per-class QColor from a 20-color list.
- **Logging as user alerts.** WARNING+ log records pop up as MessageDialogs via `QtLogHandler`. For intentional dialogs use `app.utils.message`.
- **Config persistence.** Every user setting saved via `app/utils/config.py` (QSettings). Restored in `_load_settings()`. Never persist SSH passwords.
- **Panels.** Each is a self-contained QWidget with unique `setObjectName()`. Registered in `MainWindow._register_navigation()` via `addSubInterface(self._wrap(widget), ...)`. All panels wrapped in `ScrollArea` — content overflows scroll instead of growing the main window.
- **QLabel / QPushButton / QListWidget** can stay native — Fluent theme engine auto-styles them. Only use qfluentwidgets-specific controls (`CardWidget`, `BodyLabel`, `CheckBox`, `RadioButton`, `ComboBox`, `Slider`, `SpinBox`, `DoubleSpinBox`, `ScrollArea`, `FluentWindow`) when you need Fluent behavior.
- **ImageBrowser** (`thumbnail_list.py` + `image_browser.py`) is the shared component for image list + viewer + prev/next buttons. Use it in any panel that browses images.

## Gotchas

- **No `app/formats/`.** That package was deleted. Format logic lives in `splitter.py` / `label_reader.py`.
- **exporter.py chdir.** ultralytics `model.export()` writes to CWD; ONNXExporter temporarily `os.chdir` and restores in `finally`.
- **YOLO version mismatch.** `InferenceEngine.load_model` catches `AttributeError` (qkv/AAttn) and re-raises RuntimeError with upgrade hint.
- **Benchmark do_work must return.** The method ends with `return results` — missing this line silently returns `None`.
- **Commits use Conventional Commits** (`fix:`, `feat:`, `refactor:`, etc.).
- **Theme toggle** uses `addItem(selectable=False, onClick=...)` at NavigationItemPosition.BOTTOM.
- **GPU monitoring** supports 3 modes selected by RadioButton: nvidia-smi (SSH + CSV), gpustat (SSH + `conda run`), HTTP (Flask API).
