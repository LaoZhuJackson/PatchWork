# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**PatchWork** — a PySide6 desktop GUI application that integrates multiple PV defect-detection utility scripts into a single visual tool. Target users are the developer and colleagues; distributed as a single `.exe` via PyInstaller.

Full background and rationale: `PROJECT_BOOTSTRAP.md`.

## Tech Stack

- **Python 3.10** (conda environment)
- **GUI**: PySide6 (LGPL, QGraphicsView for image preview + annotation overlays)
- **Packaging**: PyInstaller → single `.exe`
- **SSH**: paramiko (pure Python, no system ssh dependency)
- **ML**: ultralytics (YOLO), numpy, Pillow
- **Changelog**: python-semantic-release (auto from commits)
- **CI/CD**: GitHub Actions (push tag → build + package + release)

## Commands

```bash
# Activate conda environment
conda activate <env-name>

# Run the app (during development)
python main.py

# Package as single exe
pyinstaller --onefile --windowed main.py

# Semantic release (changelog + GitHub release)
semantic-release version
```

## Architecture

```
laozhu-gui/
├── main.py                     # Entry point
├── pyproject.toml              # Project metadata + dependencies
├── .github/
│   └── workflows/
│       └── release.yml         # CI: changelog + build + release
├── app/
│   ├── main_window.py          # QMainWindow: sidebar nav (QListWidget) + QStackedWidget
│   ├── widgets/                # One panel per feature (F1–F7), all independent
│   │   ├── image_viewer.py     # Shared QGraphicsView component: zoom, pan, overlay
│   │   ├── dataset_split.py    # F1
│   │   ├── model_infer.py      # F2 (most complex: thumbnail list + inference + preview)
│   │   ├── check_pair.py       # F3
│   │   ├── label_preview.py    # F4
│   │   ├── export_onnx.py      # F5
│   │   ├── gpu_monitor.py      # F6
│   │   └── xanylabeling.py     # F7
│   ├── services/               # Business logic, no UI
│   │   ├── splitter.py         # Dataset splitting (ported from devide.py)
│   │   ├── checker.py          # Image/label pairing (ported from check_image_label.py)
│   │   ├── label_reader.py     # YOLO label parsing + coordinate denormalization
│   │   ├── exporter.py         # ONNX export wrapper around ultralytics
│   │   ├── inference.py        # Inference engine (QThread)
│   │   ├── gpu_client.py       # SSH + nvidia-smi parsing (paramiko)
│   │   └── xanylabeling.py     # subprocess wrapper
│   ├── formats/
│   │   ├── base.py             # BaseFormat abstract base class
│   │   ├── yolo_detect.py      # YOLO detection format
│   │   ├── yolo_segment.py     # YOLO segmentation (reserved)
│   │   └── coco.py             # COCO format (reserved)
│   └── utils/
│       ├── config.py           # QSettings persistence
│       └── worker.py           # QThread Worker base class
└── resources/
    └── icon.png
```

## Key Design Principles

1. **Format plugin system** — `BaseFormat` abstract base class defines `find_pairs()`, `get_label_extension()`, etc. F1/F3/F4 depend on the base class, not concrete implementations. Add new formats by subclassing.
2. **Async via QThread** — model loading, inference, SSH connections, and batch file operations all run in `QThread` workers. Results and progress communicated via signals/slots. Never block the UI thread.
3. **Lazy loading** — image lists load only viewport-visible thumbnails. Large images are not held in memory.
4. **QSettings for all config** — every file path input, SSH host/user, and external exe path is persisted. **Do not store SSH passwords** — prompt each session or require key auth.
5. **Panels are independent** — each widget in `app/widgets/` has no dependencies on other panels. They can be developed and tested in isolation.
6. **Click-to-infer** — F2 model inference runs on-demand when a thumbnail is clicked, not pre-computed across the whole folder. Saves resources.

## Existing Scripts to Port

The original utility scripts are in the current directory and will be ported into `app/services/`:

| Script | Ported to |
|--------|-----------|
| `devide.py` | `app/services/splitter.py` (F1) |
| `check_image_label.py` | `app/services/checker.py` (F3) |

## Development Order

1. Project skeleton + main window (sidebar nav + QStackedWidget with blank panels)
2. F1 — Dataset split (simplest, validates the architecture pattern)
3. F4 — Label preview (builds `ImageViewer`, needed by F2)
4. F2 — Model inference + image preview (core, most complex)
5. F3 — Image/label pairing check (quick)
6. F5 — ONNX export
7. F6 — GPU monitor (SSH-dependent, independent)
8. F7 — X-AnyLabeling launcher (trivial)
9. Packaging + CI/CD
