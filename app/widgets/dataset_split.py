from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    PushButton, PrimaryPushButton, LineEdit, ProgressBar,
    BodyLabel, StrongBodyLabel, SubtitleLabel,
    Slider, CheckBox, CardWidget,
)

from app.services.splitter import split_dataset, IMAGE_EXTS, find_pairs
from app.utils.config import get_str, set_str, set_bool, set_int, get_int, get_bool
from app.utils.message import warning, confirm, info, error
from app.utils.worker import Worker

import shutil

from app.utils.logger import get_logger

logger = get_logger(__name__)


class SplitWorker(Worker):
    """后台执行划分"""

    def __init__(self, image_dir: Path, label_dir: Path, output_dir: Path, train_r: float, val_r: float,
                 test_r: float, mode: str) -> None:
        super().__init__()
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.output_dir = output_dir
        self.train_r = train_r
        self.val_r = val_r
        self.test_r = test_r
        self.mode = mode

    def do_work(self) -> dict[str, int]:
        return split_dataset(
            self.image_dir,
            self.label_dir,
            self.output_dir,
            self.train_r,
            self.val_r,
            self.test_r,
            mode=self.mode,
            progress_callback=lambda p: self.progress.emit(p)
        )


class DatasetSplitPanel(QWidget):
    """数据集划分面板"""

    status_message = Signal(str)

    def __init__(self):
        super().__init__()
        self.setObjectName("dataset_split_panel")
        self._worker: SplitWorker | None = None

        self._setup_ui()
        self._load_settings()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)

        # ---- 标题 ----
        layout.addWidget(SubtitleLabel("数据集划分"))

        # ---- 路径设置 ----
        layout.addWidget(StrongBodyLabel("路径设置"))
        path_card = CardWidget()
        path_outer = QVBoxLayout(path_card)
        path_form = QFormLayout()
        path_outer.addLayout(path_form)

        self.img_edit = LineEdit()
        img_btn = PushButton("📁")
        img_btn.clicked.connect(lambda: self._browse_dir(self.img_edit, "img_dir"))
        img_row = QHBoxLayout()
        img_row.addWidget(self.img_edit, 1)
        img_row.addWidget(img_btn)
        path_form.addRow(BodyLabel("图片目录:"), img_row)

        self.lbl_edit = LineEdit()
        lbl_btn = PushButton("📁")
        lbl_btn.clicked.connect(lambda: self._browse_dir(self.lbl_edit, "lbl_dir"))
        lbl_row = QHBoxLayout()
        lbl_row.addWidget(self.lbl_edit, 1)
        lbl_row.addWidget(lbl_btn)
        path_form.addRow(BodyLabel("标签目录:"), lbl_row)

        self.out_edit = LineEdit()
        out_btn = PushButton("📁")
        out_btn.clicked.connect(lambda: self._browse_dir(self.out_edit, "out_dir"))
        out_row = QHBoxLayout()
        out_row.addWidget(self.out_edit, 1)
        out_row.addWidget(out_btn)
        path_form.addRow(BodyLabel("输出目录:"), out_row)

        self.pair_status = BodyLabel("")
        path_outer.addWidget(self.pair_status)

        layout.addWidget(path_card)

        # ---- 划分比例 ----
        layout.addWidget(StrongBodyLabel("划分比例"))
        ratio_card = CardWidget()
        ratio_layout = QVBoxLayout(ratio_card)

        self.train_slider = Slider(Qt.Orientation.Horizontal)
        self.train_slider.setRange(50, 90)
        self.train_slider.setValue(80)
        self.val_slider = Slider(Qt.Orientation.Horizontal)
        self.val_slider.setRange(1, 40)
        self.val_slider.setValue(10)
        self.test_slider = Slider(Qt.Orientation.Horizontal)
        self.test_slider.setRange(1, 40)
        self.test_slider.setValue(10)

        self.ratio_label = BodyLabel("训练集 80%  |  验证集 10%  |  测试集 10%")

        for sl in (self.train_slider, self.val_slider, self.test_slider):
            sl.valueChanged.connect(self._on_ratios_changed)

        ratio_layout.addWidget(BodyLabel("训练集比例:"))
        ratio_layout.addWidget(self.train_slider)
        ratio_layout.addWidget(BodyLabel("验证集比例:"))
        ratio_layout.addWidget(self.val_slider)
        ratio_layout.addWidget(BodyLabel("测试集比例:"))
        ratio_layout.addWidget(self.test_slider)
        ratio_layout.addWidget(self.ratio_label)

        layout.addWidget(ratio_card)

        # ---- 底部操作栏 ----
        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(12)

        bottom_row.addStretch()
        self.move_checkbox = CheckBox("剪切模式")
        self.move_checkbox.stateChanged.connect(
            lambda: set_bool("split_move_mode", self.move_checkbox.isChecked())
        )
        bottom_row.addWidget(self.move_checkbox)

        self.check_btn = PushButton("查看具体配对情况")
        self.check_btn.clicked.connect(self._on_check_detail)
        bottom_row.addWidget(self.check_btn)

        self.split_btn = PrimaryPushButton("开始划分")
        self.split_btn.clicked.connect(self._on_split)
        bottom_row.addWidget(self.split_btn)

        layout.addLayout(bottom_row)

        self.progress = ProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        layout.addStretch()

    # ---- 路径 ----
    def _browse_dir(self, edit: LineEdit, key: str) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择目录", edit.text())
        if path:
            edit.setText(path)
            set_str(key, path)
            self._on_ratios_changed()
            self._refresh_pair_status()

    # ---- 配对检查 ----
    def _current_dirs(self) -> tuple[Path | None, Path | None]:
        img = Path(self.img_edit.text()) if self.img_edit.text().strip() else None
        lbl = Path(self.lbl_edit.text()) if self.lbl_edit.text().strip() else None
        if img and img.is_dir() and lbl and lbl.is_dir():
            return img, lbl
        return None, None

    def _refresh_pair_status(self) -> None:
        """静默刷新配对状态标签"""
        img_dir, lbl_dir = self._current_dirs()
        if img_dir is None or lbl_dir is None:
            self.pair_status.setText("")
            return
        pairs, missing_label, missing_image = find_pairs(img_dir, lbl_dir)
        parts = [f"共 {len(pairs) + len(missing_label)} 张图片，{len(pairs)} 张已配对"]
        if missing_label:
            parts.append(f"⚠ {len(missing_label)} 张缺标签")
        if missing_image:
            parts.append(f"⚠ {len(missing_image)} 个标签缺图片")
        if not missing_label and not missing_image:
            parts.append("✅ 全部配对")
        self.pair_status.setText("  |  ".join(parts))

    def _on_check_detail(self) -> None:
        """点击查看配对详情"""
        img_dir, lbl_dir = self._current_dirs()
        if img_dir is None or lbl_dir is None:
            warning("提示", "请先选择有效的图片目录和标签目录", self)
            return
        pairs, missing_label, missing_image = find_pairs(img_dir, lbl_dir)

        lines = [
            f"图片目录: {img_dir}",
            f"标签目录: {lbl_dir}",
            "",
            f"━━━━━━━━━━━━━━━━━━━━",
            f"  ✅ 配对成功: {len(pairs)} 对",
            f"  ⚠ 缺少标签: {len(missing_label)} 张",
            f"  ⚠ 缺少图片: {len(missing_image)} 个",
            f"━━━━━━━━━━━━━━━━━━━━",
        ]

        if missing_label:
            lines.append(f"\n━━━ 缺少标签的图片 ({len(missing_label)} 张) ━━━")
            for f in missing_label:
                lines.append(f"  ✗ {f.name}")
            # if len(missing_label) > 20:
            #     lines.append(f"  ... 等共 {len(missing_label)} 张")

        if missing_image:
            lines.append(f"\n━━━ 缺少图片的标签 ({len(missing_image)} 个) ━━━")
            for f in missing_image:
                lines.append(f"  ✗ {f.name}")
            # if len(missing_image) > 20:
            #     lines.append(f"  ... 等共 {len(missing_image)} 个")

        if not missing_label and not missing_image:
            lines.append("\n🎉 所有图片和标签均配对")

        info("配对检查结果", "\n".join(lines), self)

    def _on_ratios_changed(self):
        t = self.train_slider.value()
        v = self.val_slider.value()
        s = self.test_slider.value()
        total = t + v + s

        img_dir, lbl_dir = self._current_dirs()
        pair_count = 0
        if img_dir and lbl_dir:
            pairs, _, _ = find_pairs(img_dir, lbl_dir)
            pair_count = len(pairs)
        train_n = round(pair_count * t / total)
        val_n = round(pair_count * v / total)
        test_n = pair_count - train_n - val_n

        if pair_count > 0:
            self.ratio_label.setText(
                f"训练集 {t * 100 / total:.0f}% ({train_n} 张)  |  "
                f"验证集 {v * 100 / total:.0f}% ({val_n} 张)  |  "
                f"测试集 {s * 100 / total:.0f}% ({test_n} 张)"
            )
        else:
            self.ratio_label.setText(
                f"训练集 {t * 100 / total:.0f}%  |  "
                f"验证集 {v * 100 / total:.0f}%  |  "
                f"测试集 {s * 100 / total:.0f}%"
            )
        set_int("split_train", t)
        set_int("split_val", v)
        set_int("split_test", s)

    def _load_settings(self):
        self.img_edit.setText(get_str("img_dir"))
        self.lbl_edit.setText(get_str("lbl_dir"))
        self.out_edit.setText(get_str("out_dir"))

        if get_int("split_train") > 0:
            self.train_slider.setValue(get_int("split_train"))
            self.val_slider.setValue(get_int("split_val"))
            self.test_slider.setValue(get_int("split_test"))

        self.move_checkbox.setChecked(get_bool("split_move_mode"))

        self._refresh_pair_status()
        self._on_ratios_changed()

    def _on_split(self) -> None:
        img = Path(self.img_edit.text())
        lbl = Path(self.lbl_edit.text())
        out = Path(self.out_edit.text())
        mode = "move" if self.move_checkbox.isChecked() else "copy"

        if not img.is_dir():
            warning("错误", "请选择有效的图片目录", self)
            return
        if not lbl.is_dir():
            warning("错误", "请选择有效的标签目录", self)
            return
        if not out.is_dir():
            warning("错误", "请选择有效的输出目录", self)
            return

        out.mkdir(parents=True, exist_ok=True)

        contents = list(out.iterdir())
        if contents and mode == "move":
            img_count = sum(
                1 for f in out.rglob("*") if f.suffix.lower() in IMAGE_EXTS
            )
            if not confirm(
                    "确认清空",
                    f"输出目录已有 {img_count} 张图片，\n"
                    "继续操作将清空该目录下所有内容。\n\n"
                    "是否继续？",
                    self,
            ):
                return

            for item in out.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()

        t = self.train_slider.value()
        v = self.val_slider.value()
        s = self.test_slider.value()
        total = t + v + s

        self.split_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)

        self._worker = SplitWorker(img, lbl, out, t / total, v / total, s / total, mode)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.progress.connect(self.progress.setValue)
        self._worker.start()

    def _on_finished(self, result: dict) -> None:
        self.progress.setValue(100)
        self.split_btn.setEnabled(True)

        if not result["ok"]:
            msg = "配对不完整，无法划分：\n\n"
            if result["missing_label"]:
                msg += f"⚠ 缺少标签的图片 ({len(result['missing_label'])} 张):\n"
                for f in result["missing_label"][:10]:
                    msg += f"  • {f.name}\n"
                if len(result["missing_label"]) > 10:
                    msg += f"  ... 等共 {len(result['missing_label'])} 个\n"
            if result["missing_image"]:
                msg += f"\n⚠ 缺少图片的标签 ({len(result['missing_image'])} 个):\n"
                for f in result["missing_image"][:10]:
                    msg += f"  • {f.name}\n"
                if len(result["missing_image"]) > 10:
                    msg += f"  ... 等共 {len(result['missing_image'])} 个\n"
            warning("配对不完整", msg, self)
            self.progress.setVisible(False)
            return

        msg = (
            f"划分完成！\n\n"
            f"训练集: {result['train']} 张\n"
            f"验证集: {result['val']} 张\n"
            f"测试集: {result['test']} 张"
        )
        info("完成", msg, self)
        self.progress.setVisible(False)

    def _on_error(self, err: str) -> None:
        self.split_btn.setEnabled(True)
        self.progress.setVisible(False)
        error("错误", f"划分失败:\n{err}", self)
