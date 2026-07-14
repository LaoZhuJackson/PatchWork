"""F8: 视频抽帧面板"""
from __future__ import annotations

import math
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
    QButtonGroup,
    QRadioButton, QDoubleSpinBox,
)
from qfluentwidgets import (
    PushButton,
    LineEdit,
    ProgressBar,
    BodyLabel,
    SubtitleLabel,
    DoubleSpinBox, ComboBox, PrimaryPushButton,
)

from app.services.video_extractor import extract_frames
from app.utils.config import get_str, set_str, get_float, set_float, get_int, set_int
from app.utils.message import info, warning, error
from app.utils.worker import Worker

from app.utils.logger import get_logger

logger = get_logger(__name__)


class ExtractWorker(Worker):
    """后台抽帧"""

    def __init__(self, video_path: str, output_dir: str, mode: str, interval: float, fmt: str) -> None:
        super().__init__()
        self.video_path = video_path
        self.output_dir = output_dir
        self.mode = mode
        self.interval = interval
        self.fmt = fmt

    def do_work(self) -> dict:
        return extract_frames(
            self.video_path,
            self.output_dir,
            self.mode,
            self.interval,
            self.fmt,
            progress_callback=lambda p: self.progress.emit(p),
        )


class VideoExtractPanel(QWidget):
    """视频抽帧面板"""
    status_message = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("video_extract_panel")
        self._worker: ExtractWorker | None = None

        # 视频信息缓存（用于预估产出）
        self._video_total: int = 0
        self._video_fps: float = 0.0

        self._setup_ui()
        self._load_settings()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(24, 24, 24, 24)

        # ---- 标题 ----
        layout.addWidget(SubtitleLabel("视频抽帧"))

        # ---- 路径选择 ----
        path_group = QGroupBox("路径设置")
        path_form = QFormLayout(path_group)

        video_row = QHBoxLayout()
        self.video_edit = LineEdit()
        self.video_edit.setPlaceholderText("选择视频文件...")
        self.video_edit.setReadOnly(True)
        video_btn = PushButton("浏览...")
        video_btn.clicked.connect(self._browse_video)
        video_row.addWidget(self.video_edit, 1)
        video_row.addWidget(video_btn)
        path_form.addRow("视频文件:", video_row)

        out_row = QHBoxLayout()
        self.out_edit = LineEdit()
        self.out_edit.setPlaceholderText("选择输出目录...")
        out_btn = PushButton("浏览...")
        out_btn.clicked.connect(self._browse_out)
        out_row.addWidget(self.out_edit, 1)
        out_row.addWidget(out_btn)
        path_form.addRow("输出目录:", out_row)

        layout.addWidget(path_group)

        # ---- 视频信息 ----
        self.video_info = BodyLabel("")
        layout.addWidget(self.video_info)

        # ---- 抽帧模式 ----
        mode_group = QGroupBox("抽帧设置")
        mode_layout = QVBoxLayout(mode_group)

        # 模式选择
        mode_row = QHBoxLayout()
        self.time_radio = QRadioButton("按时间间隔")
        self.frame_radio = QRadioButton("按帧间隔")
        self.time_radio.setChecked(True)
        self._mode_group = QButtonGroup(self)
        self._mode_group.addButton(self.time_radio, 0)
        self._mode_group.addButton(self.frame_radio, 1)
        self._mode_group.buttonClicked.connect(self._on_interval_changed)
        mode_row.addWidget(self.time_radio)
        mode_row.addWidget(self.frame_radio)
        mode_row.addStretch()
        mode_layout.addLayout(mode_row)

        # 间隔设置
        interval_row = QHBoxLayout()
        interval_row.addWidget(BodyLabel("间隔:"))
        self.interval_spin = DoubleSpinBox()
        self.interval_spin.setRange(0.1, 9999.0)
        self.interval_spin.setSingleStep(0.1)
        self.interval_spin.setDecimals(1)
        self.interval_spin.setValue(1.0)
        self.interval_spin.valueChanged.connect(self._on_interval_changed)
        interval_row.addWidget(self.interval_spin)
        self.interval_unit = BodyLabel("秒")
        interval_row.addWidget(self.interval_unit)
        interval_row.addStretch()
        mode_layout.addLayout(interval_row)

        # 输出格式
        fmt_row = QHBoxLayout()
        fmt_row.addWidget(BodyLabel("输出格式:"))
        self.fmt_combo = ComboBox()
        self.fmt_combo.addItems(["jpg", "png"])
        self.fmt_combo.currentTextChanged.connect(
            lambda v: (set_str("extract_format", v), self._update_estimate())
        )
        fmt_row.addWidget(self.fmt_combo)
        fmt_row.addStretch()
        mode_layout.addLayout(fmt_row)

        # 预估产出
        self.estimate_label = BodyLabel("")
        mode_layout.addWidget(self.estimate_label)

        layout.addWidget(mode_group)

        # ---- 执行 ----
        self.extract_btn = PrimaryPushButton("开始抽帧")
        self.extract_btn.clicked.connect(self._on_extract)
        layout.addWidget(self.extract_btn, alignment=Qt.AlignmentFlag.AlignRight)

        self.status_label = BodyLabel("")
        layout.addWidget(self.status_label)

        self.progress = ProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        layout.addStretch()

    # ---- 预览 ----

    def _update_estimate(self) -> None:
        """根据视频信息和抽帧设置更新预估产出"""
        logger.info(f"调用 _update_estimate: total={self._video_total}, fps={self._video_fps}")
        if self._video_total <= 0 or self._video_fps <= 0:
            self.estimate_label.setText("")
            return

        mode = "frame" if self._mode_group.checkedId() == 1 else "time"
        interval = self.interval_spin.value()

        if mode == "time":
            step = max(1, int(interval * self._video_fps))
        else:
            step = max(1, int(interval))

        estimated = math.ceil(self._video_total / step)
        self.estimate_label.setText(
            f"预计产出: {estimated} 张 ({self.fmt_combo.currentText().upper()})"
        )

    def _on_settings_changed(self) -> None:
        """模式或间隔变更时更新 UI"""
        mode = "frame" if self._mode_group.checkedId() == 1 else "time"
        set_str("extract_mode", mode)

        if mode == "time":
            self.interval_unit.setText("秒")
            self.interval_spin.setRange(0.1, 9999.0)
            self.interval_spin.setDecimals(1)
            self.interval_spin.setSingleStep(0.1)
            self.interval_spin.setValue(get_float("extract_time_interval", 1.0))
        else:
            self.interval_unit.setText("帧")
            self.interval_spin.setRange(1.0, 9999.0)
            self.interval_spin.setDecimals(0)
            self.interval_spin.setSingleStep(1.0)
            self.interval_spin.setValue(get_float("extract_frame_interval", 5.0))

        self._update_estimate()

    # ---- 事件 ----
    def _browse_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", self.video_edit.text(),
            "Video Files (*.mp4 *.avi *.mov *.mkv *.flv *.wmv);;All Files (*)"
        )
        if not path:
            return
        self.video_edit.setText(path)
        set_str("extract_video_path", path)

        # 自动读取视频信息
        self._load_video(path)

    def _browse_out(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "选择输出目录", self.out_edit.text()
        )
        if path:
            self.out_edit.setText(path)
            set_str("extract_output_dir", path)

    def _load_video(self, path: str):
        if not path:
            return
        import cv2
        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            self._video_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self._video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            duration = self._video_total / self._video_fps if self._video_fps > 0 else 0
            self.video_info.setText(
                f"总帧数: {self._video_total}  |  FPS: {self._video_fps:.1f}  |  时长: {duration:.1f}s"
            )
            cap.release()
        else:
            self.video_info.setText("⚠ 无法读取视频信息")
            self._video_total = 0
            self._video_fps = 0.0
        self._update_estimate()

    def _on_mode_changed(self) -> None:
        mode = "frame" if self._mode_group.checkedId() == 1 else "time"
        set_str("extract_mode", mode)

        # 临时断开，避免 setValue 触发保存
        self.interval_spin.blockSignals(True)

        if mode == "time":
            self.interval_unit.setText("秒")
            self.interval_spin.setRange(0.1, 99.0)
            self.interval_spin.setSingleStep(0.1)
            self.interval_spin.setDecimals(1)
            self.interval_spin.setValue(get_float("extract_time_interval", 1.0))
        else:
            self.interval_unit.setText("帧")
            self.interval_spin.setRange(1.0, 999.0)
            self.interval_spin.setSingleStep(1.0)
            self.interval_spin.setDecimals(0)
            self.interval_spin.setValue(get_float("extract_frame_interval", 5.0))

        self.interval_spin.blockSignals(False)
        self._update_estimate()

    def _on_interval_changed(self, value: float) -> None:
        """仅值变化时触发：保存 + 更新预估"""
        mode = "frame" if self._mode_group.checkedId() == 1 else "time"
        if mode == "time":
            set_float("extract_time_interval", value)
        else:
            set_float("extract_frame_interval", value)
        self._update_estimate()

    def _on_extract(self) -> None:
        video = self.video_edit.text().strip()
        out = self.out_edit.text().strip()

        if not video or not Path(video).is_file():
            warning("错误", "请选择有效的视频文件", self)
            return
        if not out:
            warning("错误", "请选择输出目录", self)
            return

        mode = "frame" if self._mode_group.checkedId() == 1 else "time"
        interval = float(self.interval_spin.value())
        fmt = self.fmt_combo.currentText()

        if mode == "time":
            set_float("extract_time_interval", interval)
        else:
            set_float("extract_frame_interval", interval)

        self.extract_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.status_label.setText("正在抽帧...")

        self._worker = ExtractWorker(video, out, mode, interval, fmt)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.progress.connect(self.progress.setValue)
        self._worker.start()

    def _on_finished(self, result: dict) -> None:
        self.extract_btn.setEnabled(True)
        self.progress.setVisible(False)
        self.status_label.setText(
            f"✅ 完成！共抽取 {result['extracted']} 帧 → {result['output_dir']}"
        )
        info(
            "抽帧完成",
            f"总帧数: {result['total_frames']}\n"
            f"FPS: {result['fps']:.1f}\n"
            f"时长: {result['duration']:.1f}s\n"
            f"抽取: {result['extracted']} 帧\n"
            f"输出: {result['output_dir']}",
            self,
        )

    def _on_error(self, err_msg: str) -> None:
        self.extract_btn.setEnabled(True)
        self.progress.setVisible(False)
        self.status_label.setText("❌ 抽帧失败")
        error("抽帧失败", err_msg, self)

    # ---- 持久化 ----

    def _load_settings(self) -> None:
        path = get_str("extract_video_path")
        self.video_edit.setText(path)
        self._load_video(path)

        self.out_edit.setText(get_str("extract_output_dir"))

        # 格式
        fmt = get_str("extract_format", "jpg")
        self.fmt_combo.setCurrentText(fmt)

        # 模式
        mode = get_str("extract_mode", "time")
        if mode == "frame":
            self.frame_radio.setChecked(True)
        else:
            self.time_radio.setChecked(True)
        self._on_settings_changed()  # 触发一次以设置正确的控件状态
