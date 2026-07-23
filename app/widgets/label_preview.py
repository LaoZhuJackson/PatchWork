"""F4: Label 标注预览面板"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QFormLayout,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    BodyLabel,
    StrongBodyLabel,
    SubtitleLabel,
    CardWidget,
)

from app.services.label_reader import IMAGE_EXTS, parse_yolo_label
from app.utils.config import get_str
from app.utils.message import info
from app.widgets.image_browser import ImageBrowser
from app.widgets.path_browser import PathBrowser


class LabelPreviewPanel(QWidget):
    """Label 标注预览面板"""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("label_preview_panel")

        self._label_map: dict[str, Path] = {}

        self._setup_ui()
        self._load_settings()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(24, 24, 24, 24)

        # ---- 标题 ----
        layout.addWidget(SubtitleLabel("Label 标注预览"))

        # ---- 路径设置 ----
        layout.addWidget(StrongBodyLabel("路径设置"))
        path_card = CardWidget()
        path_form = QFormLayout(path_card)

        self.img_browser = PathBrowser(
            label="", mode="dir",
            placeholder="选择图片所在的文件夹...",
            config_key="label_preview_img_dir",
        )
        self.img_browser.path_changed.connect(lambda _: self._reload())
        path_form.addRow(BodyLabel("图片目录:"), self.img_browser)

        self.lbl_browser = PathBrowser(
            label="", mode="dir",
            placeholder="选择标签所在的文件夹...",
            config_key="label_preview_lbl_dir",
        )
        self.lbl_browser.path_changed.connect(lambda _: self._reload())
        path_form.addRow(BodyLabel("标签目录:"), self.lbl_browser)

        self.info_label = BodyLabel("")
        path_form.addRow(BodyLabel(""), self.info_label)

        layout.addWidget(path_card)

        # ---- 图片浏览器 ----
        self.browser = ImageBrowser()
        self.browser.image_selected.connect(self._on_image_selected)
        layout.addWidget(self.browser, 1)

    # ---- 路径 ----

    def _reload(self) -> None:
        img_dir = self.img_browser.path
        lbl_dir = self.lbl_browser.path

        img_path = Path(img_dir) if img_dir else None
        lbl_path = Path(lbl_dir) if lbl_dir else None

        if img_path is None or not img_path.is_dir():
            return

        self._load_dataset(img_path, lbl_path if lbl_path and lbl_path.is_dir() else None)

    def _load_dataset(self, img_dir: Path, lbl_dir: Path | None) -> None:
        self._label_map.clear()

        images = sorted(
            f for f in img_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS
        )

        if not images:
            self.info_label.setText("❌ 图片目录下未找到图片")
            self.browser.clear()
            return

        missing_label: list[str] = []
        if lbl_dir:
            for img in images:
                label = lbl_dir / f"{img.stem}.txt"
                if label.exists():
                    self._label_map[img.stem] = label
                else:
                    missing_label.append(img.name)

        paired = len(self._label_map)
        msg = f"共 {len(images)} 张图片，{paired} 张已配对"
        if missing_label:
            msg += f"，{len(missing_label)} 张缺少标签"
            detail = (
                f"以下 {len(missing_label)} 张图片缺少对应标签，\n"
                f"将仅显示原图（无标注叠加）:\n\n"
            )
            for name in missing_label[:10]:
                detail += f"  • {name}\n"
            if len(missing_label) > 10:
                detail += f"  ... 等共 {len(missing_label)} 张\n"
            info("配对提示", detail, self)
        self.info_label.setText(msg)

        self.browser.set_images(images, select_index=0)

    # ---- 图片选中 ----

    def _on_image_selected(self, idx: int, path: Path) -> None:
        pixmap = self.browser.current_pixmap
        if pixmap is None or pixmap.isNull():
            return

        label_path = self._label_map.get(path.stem)
        if label_path:
            annotations = parse_yolo_label(label_path, pixmap.width(), pixmap.height())
        else:
            annotations = []

        self.browser.show_annotations(annotations)

    # ---- 持久化 ----

    def _load_settings(self) -> None:
        self.img_browser.path = get_str("label_preview_img_dir")
        self.lbl_browser.path = get_str("label_preview_lbl_dir")

        img_dir = get_str("label_preview_img_dir")
        lbl_dir = get_str("label_preview_lbl_dir")
        if img_dir:
            p_img = Path(img_dir)
            if p_img.is_dir():
                p_lbl = Path(lbl_dir) if lbl_dir else None
                self._load_dataset(p_img, p_lbl if p_lbl and p_lbl.is_dir() else None)
