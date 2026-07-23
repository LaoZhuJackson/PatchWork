"""NDJSON → YOLO 格式转换面板"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFormLayout,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    PrimaryPushButton, ProgressBar,
    BodyLabel, StrongBodyLabel, SubtitleLabel,
    CardWidget,
)

from app.services.ndjson_converter import convert
from app.utils.config import get_str
from app.utils.message import info, error
from app.utils.worker import Worker
from app.widgets.path_browser import PathBrowser


class ConvertWorker(Worker):
    """后台转换 NDJSON → YOLO"""

    def __init__(self, ndjson_path: str, output_dir: str) -> None:
        super().__init__()
        self.ndjson_path = ndjson_path
        self.output_dir = output_dir

    def do_work(self) -> str:
        yaml_path = convert(self.ndjson_path, self.output_dir)
        return str(yaml_path)


class NDJSONConvertPanel(QWidget):
    """NDJSON 转 YOLO 面板"""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("ndjson_convert_panel")
        self._worker: ConvertWorker | None = None

        self._setup_ui()
        self._load_settings()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(24, 24, 24, 24)

        # ---- 标题 ----
        layout.addWidget(SubtitleLabel("NDJSON → YOLO"))

        # ---- 路径设置 ----
        layout.addWidget(StrongBodyLabel("路径设置"))
        path_card = CardWidget()
        path_form = QFormLayout(path_card)

        self.ndjson_browser = PathBrowser(
            label="", mode="file",
            file_filter="NDJSON Files (*.ndjson *.jsonl);;All Files (*)",
            placeholder="选择 NDJSON 数据集文件...",
            config_key="ndjson_input_path",
        )
        path_form.addRow(BodyLabel("NDJSON 文件:"), self.ndjson_browser)

        self.out_browser = PathBrowser(
            label="", mode="dir",
            placeholder="选择输出目录...",
            config_key="ndjson_output_dir",
        )
        path_form.addRow(BodyLabel("输出目录:"), self.out_browser)

        layout.addWidget(path_card)

        # ---- 执行 ----
        self.convert_btn = PrimaryPushButton("开始转换")
        self.convert_btn.clicked.connect(self._on_convert)
        layout.addWidget(self.convert_btn, alignment=Qt.AlignmentFlag.AlignRight)

        self.status_label = BodyLabel("")
        layout.addWidget(self.status_label)

        self.progress = ProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        layout.addStretch()

    # ---- 转换 ----
    def _on_convert(self) -> None:
        ndjson_path = self.ndjson_browser.path
        if not ndjson_path or not Path(ndjson_path).is_file():
            error("错误", "请选择有效的 NDJSON 文件", self)
            return

        output_dir = self.out_browser.path
        if not output_dir:
            error("错误", "请选择输出目录", self)
            return
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        self.convert_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.status_label.setText("正在转换...")

        self._worker = ConvertWorker(ndjson_path, output_dir)
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_done(self, yaml_path: str) -> None:
        self.progress.setValue(100)
        self.status_label.setText(f"✅ 转换完成")
        self.convert_btn.setEnabled(True)
        info(
            "转换完成",
            f"data.yaml 已生成:\n{yaml_path}\n\n"
            f"可直接用于 YOLO 训练:\n"
            f"  model.train(data='{yaml_path}')",
            self,
        )
    def _on_error(self, err:str) -> None:
        self.status_label.setText("❌ 转换失败")
        self.progress.setVisible(False)
        self.convert_btn.setEnabled(True)
        error("转换失败", err,self)

    # ---- 持久化 ----

    def _load_settings(self) -> None:
        self.ndjson_browser.path = get_str("ndjson_input_path")
        self.out_browser.path = get_str("ndjson_output_dir")
