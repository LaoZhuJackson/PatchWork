"""IR-VIS 控制点标注面板"""
from __future__ import annotations

from pathlib import Path

import cv2
from PySide6.QtCore import Qt, QPointF, Signal
from PySide6.QtGui import QColor, QPixmap, QKeyEvent
from PySide6.QtWidgets import (
    QFormLayout,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    BodyLabel, CardWidget, PrimaryPushButton, ProgressBar,
    PushButton, StrongBodyLabel, SubtitleLabel, InfoBar, InfoBarPosition,
)

from app.services.irvis_annotator import (
    IRVISState, IRVISPair,
    scan_pairs, load_annotations, save_annotations,
)
from app.utils.config import get_str, set_str
from app.utils.message import error, info, warning, confirm
from app.utils.worker import Worker
from app.widgets.image_viewer import ImageViewer
from app.widgets.path_browser import PathBrowser
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ── 常量 ──
GREEN = QColor(0, 200, 80)     # 十字标颜色
RED = QColor(255, 60, 60)      # 序号颜色
YELLOW = QColor(255, 200, 0)   # pending IR 点十字标
CROSS_SIZE = 20     # 十字线半长（像素）


# ═══════════════════════════════════════════════════════════════════════
# 控制点绘制工具
# ═══════════════════════════════════════════════════════════════════════

def _draw_cross(
    viewer: ImageViewer,
    x: float, y: float,
    cross_color: QColor,
    label: str = "",
    label_color: QColor = RED,
) -> None:
    """在 viewer 上绘制十字标记 + 可选序号

    Args:
        cross_color: 十字线颜色
        label_color: 序号文字颜色（默认红色）
    """
    points = [
        QPointF(x - CROSS_SIZE, y),
        QPointF(x + CROSS_SIZE, y),
    ]
    viewer.add_polygon(points, cross_color, line_width=2.0)
    points_v = [
        QPointF(x, y - CROSS_SIZE),
        QPointF(x, y + CROSS_SIZE),
    ]
    viewer.add_polygon(points_v, cross_color, line_width=2.0)

    if label:
        viewer.add_text(QPointF(x + 14, y - 10), label, label_color, size=14)


def _refresh_overlays(
    ir_viewer: ImageViewer,
    vis_viewer: ImageViewer,
    state: IRVISState,
    ir_scale: float = 1.0,
    vis_scale: float = 1.0,
) -> None:
    """清除并重绘两个 viewer 上的所有控制点标记

    state 中存储的是原始图像坐标，绘制时需要除以 scale 映射到显示坐标。
    """
    ir_viewer.clear_overlays()
    vis_viewer.clear_overlays()

    # 已完成的对 — 绿色十字 + 红色编号
    for i, (ir_pt, vis_pt) in enumerate(zip(state.ir_pts, state.vis_pts)):
        num = str(i + 1)
        _draw_cross(ir_viewer, ir_pt[0] / ir_scale, ir_pt[1] / ir_scale, GREEN, num)
        _draw_cross(vis_viewer, vis_pt[0] / vis_scale, vis_pt[1] / vis_scale, GREEN, num)

    # pending IR 点 — 金色 + 无编号
    if state.pending_ir is not None:
        x, y = state.pending_ir
        _draw_cross(ir_viewer, x / ir_scale, y / ir_scale, YELLOW)


# ═══════════════════════════════════════════════════════════════════════
# 后台 Worker：批量扫描
# ═══════════════════════════════════════════════════════════════════════

class ScanWorker(Worker):
    """后台扫描 IR/VIS 目录配对"""

    def __init__(self, ir_dir: str, vis_dir: str) -> None:
        super().__init__()
        self.ir_dir = ir_dir
        self.vis_dir = vis_dir

    def do_work(self) -> tuple[list[IRVISPair], dict]:
        return scan_pairs(self.ir_dir, self.vis_dir)


# ═══════════════════════════════════════════════════════════════════════
# 面板
# ═══════════════════════════════════════════════════════════════════════

class IRVISAnnotatorPanel(QWidget):
    """IR-VIS 控制点标注面板

    用法:
      1. 选择 IR 目录和 VIS 目录 → 自动扫描配对
      2. 左键在 IR 图点一个位置 → 左键在 VIS 图点对应位置 → 成对
      3. 右键撤销最后一对，中键拖拽平移，滚轮缩放
      4. 键盘 R/D=下一帧  Q/A=上一帧  S=保存  Ctrl+Z=撤销
    """

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("irvis_annotator_panel")

        self._state: IRVISState | None = None
        self._scan_worker: ScanWorker | None = None
        self._load_path: str = ""   # 已有的 .npz 文件路径
        self._ir_scale: float = 1.0
        self._vis_scale: float = 1.0

        self._setup_ui()
        self._load_settings()

    # ── UI 搭建 ──

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(24, 24, 24, 24)

        layout.addWidget(SubtitleLabel("IR-VIS 控制点标注"))

        # ---- 路径设置 ----
        layout.addWidget(StrongBodyLabel("数据集路径"))
        path_card = CardWidget()
        path_form = QFormLayout(path_card)

        # IR 目录
        self.ir_browser = PathBrowser(
            label="", mode="dir",
            placeholder="选择 IR（红外）图像目录...",
            config_key="irvis_ir_dir",
        )
        self.ir_browser.path_changed.connect(self._on_dir_changed)
        path_form.addRow(BodyLabel("IR 目录:"), self.ir_browser)

        # VIS 目录
        self.vis_browser = PathBrowser(
            label="", mode="dir",
            placeholder="选择 VIS（可见光）图像目录...",
            config_key="irvis_vis_dir",
        )
        self.vis_browser.path_changed.connect(self._on_dir_changed)
        path_form.addRow(BodyLabel("VIS 目录:"), self.vis_browser)

        # 标注输出
        self.npz_browser = PathBrowser(
            label="", mode="file",
            file_filter="NPZ Files (*.npz);;All Files (*)",
            placeholder="标注文件保存路径（.npz）...",
            config_key="irvis_npz_path",
        )
        path_form.addRow(BodyLabel("标注文件:"), self.npz_browser)

        layout.addWidget(path_card)

        # ---- 标注预览 ----
        layout.addWidget(StrongBodyLabel("标注视图 (左键点选, 中键拖拽, 滚轮缩放)"))
        preview_card = CardWidget()
        preview_layout = QHBoxLayout(preview_card)

        # IR 视图
        ir_col = QVBoxLayout()
        ir_label = BodyLabel("IR (红外)")
        ir_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ir_col.addWidget(ir_label)
        self.ir_viewer = ImageViewer()
        self.ir_viewer.setMinimumHeight(200)
        self.ir_viewer.set_interaction_mode("pick")
        self.ir_viewer.clicked.connect(self._on_ir_clicked)
        ir_col.addWidget(self.ir_viewer, 1)
        preview_layout.addLayout(ir_col, 1)

        # VIS 视图
        vis_col = QVBoxLayout()
        vis_label = BodyLabel("VIS (可见光)")
        vis_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        vis_col.addWidget(vis_label)
        self.vis_viewer = ImageViewer()
        self.vis_viewer.setMinimumHeight(200)
        self.vis_viewer.set_interaction_mode("pick")
        self.vis_viewer.clicked.connect(self._on_vis_clicked)
        vis_col.addWidget(self.vis_viewer, 1)
        preview_layout.addLayout(vis_col, 1)

        layout.addWidget(preview_card, 1)

        # ---- 状态栏 ----
        info_row = QHBoxLayout()
        self.frame_label = BodyLabel("未开始")
        info_row.addWidget(self.frame_label, 1)
        self.points_label = BodyLabel("")
        info_row.addWidget(self.points_label)
        layout.addLayout(info_row)

        # ---- 操作按钮 ----
        btn_row = QHBoxLayout()

        self.undo_btn = PushButton("撤销 (Ctrl+Z)")
        self.undo_btn.setToolTip("撤销最后一对控制点")
        self.undo_btn.clicked.connect(self._on_undo)
        self.undo_btn.setEnabled(False)
        btn_row.addWidget(self.undo_btn)

        btn_row.addStretch()

        self.prev_btn = PushButton("◀ 上一帧 (Q/A)")
        self.prev_btn.clicked.connect(self._on_prev)
        self.prev_btn.setEnabled(False)
        btn_row.addWidget(self.prev_btn)

        self.next_btn = PushButton("下一帧 (R/D) ▶")
        self.next_btn.clicked.connect(self._on_next)
        self.next_btn.setEnabled(False)
        btn_row.addWidget(self.next_btn)

        btn_row.addStretch()

        self.save_btn = PrimaryPushButton("保存标注 (S)")
        self.save_btn.clicked.connect(self._on_save)
        self.save_btn.setEnabled(False)
        btn_row.addWidget(self.save_btn)

        layout.addLayout(btn_row)

        self.progress = ProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

    # ── 目录切换 → 扫描配对 ──

    def _on_dir_changed(self, _: str) -> None:
        self._try_scan()

    def _try_scan(self) -> None:
        """两个目录都选好后自动扫描"""
        ir_dir = self.ir_browser.path.strip()
        vis_dir = self.vis_browser.path.strip()
        if not ir_dir or not vis_dir:
            return

        if self._scan_worker is not None and self._scan_worker.isRunning():
            return

        self._set_inputs_enabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.frame_label.setText("正在扫描配对...")

        self._scan_worker = ScanWorker(ir_dir, vis_dir)
        self._scan_worker.finished.connect(self._on_scan_done)
        self._scan_worker.error.connect(self._on_scan_error)
        self._scan_worker.start()

    def _on_scan_done(self, result: tuple[list[IRVISPair], dict]) -> None:
        pairs, stats = result
        self._set_inputs_enabled(True)
        self.progress.setVisible(False)

        if not pairs:
            error(
                "未找到配对",
                f"IR {stats['ir_count']} 张, VIS {stats['vis_count']} 张, "
                f"配对 0 对。\n\n检查文件名是否包含 _T_帧号 和 _V_帧号。",
                self,
            )
            self.frame_label.setText("无可配对图像")
            return

        # 尝试加载已有 .npz（优先使用用户设定的路径，否则默认 IR 目录下）
        npz_path = self.npz_browser.path.strip()
        if not npz_path:
            npz_path = str(Path(self.ir_browser.path) / "irvis_annotations.npz")
            self.npz_browser.path = npz_path
        self._load_path = npz_path
        existing = load_annotations(self._load_path)

        # 创建状态对象
        self._state = IRVISState(pairs)
        self._state.annotations = existing
        self._state._load_current()

        # 显示第一帧
        self._show_current_frame()

        unpaired_msg = ""
        if stats["unpaired_ir"]:
            unpaired_msg += f"\n未配对 IR: {len(stats['unpaired_ir'])} 张"
        if stats["unpaired_vis"]:
            unpaired_msg += f"\n未配对 VIS: {len(stats['unpaired_vis'])} 张"

        InfoBar.success(
            title="扫描完成",
            content=f"配对 {len(pairs)} 对, 已有标注 {len(existing)} 帧{unpaired_msg}",
            orient=Qt.Orientation.Horizontal,
            isClosable=True,
            position=InfoBarPosition.TOP,
            duration=3000,
            parent=self,
        )

    def _on_scan_error(self, err: str) -> None:
        self._set_inputs_enabled(True)
        self.progress.setVisible(False)
        self.frame_label.setText("扫描失败")
        error("扫描失败", err, self)

    # ── 帧显示 ──

    def _show_current_frame(self) -> None:
        """加载当前帧的 IR/VIS 图像到两个 viewer"""
        if self._state is None:
            return

        pair = self._state.current_pair
        ir_orig = cv2.imread(str(pair.ir_path))
        vis_orig = cv2.imread(str(pair.vis_path))

        if ir_orig is None:
            error("读取失败", f"无法读取 IR 图像:\n{pair.ir_path}", self)
            return
        if vis_orig is None:
            error("读取失败", f"无法读取 VIS 图像:\n{pair.vis_path}", self)
            return

        # 记录原始尺寸和缩放比（用于坐标映射）
        self._ir_orig_h, self._ir_orig_w = ir_orig.shape[:2]
        self._vis_orig_h, self._vis_orig_w = vis_orig.shape[:2]

        # 统一高度 (720px) 便于并排对比
        h_target = 720
        ir_img = _resize_to_height(ir_orig, h_target)
        vis_img = _resize_to_height(vis_orig, h_target)
        self._ir_scale = self._ir_orig_h / h_target
        self._vis_scale = self._vis_orig_h / h_target

        self.ir_viewer.set_image(_cv_to_qpixmap(ir_img))
        self.vis_viewer.set_image(_cv_to_qpixmap(vis_img))

        # 重绘控制点（坐标从原图空间映射到显示空间）
        _refresh_overlays(self.ir_viewer, self.vis_viewer, self._state,
                          self._ir_scale, self._vis_scale)

        # 更新状态栏
        self._update_status()

        # 按钮状态
        total = self._state.total_pairs
        annotated = self._state.count_annotated()
        self.prev_btn.setEnabled(self._state.idx > 0)
        self.next_btn.setEnabled(self._state.idx < total - 1)
        self.save_btn.setEnabled(annotated > 0 or self._state.current_point_count() > 0)
        self.undo_btn.setEnabled(
            self._state.current_point_count() > 0 or self._state.pending_ir is not None
        )

    def _update_status(self) -> None:
        if self._state is None:
            return
        n = self._state.current_point_count()
        total = self._state.total_pairs
        annotated = self._state.count_annotated()
        pending = " (+1 pending)" if self._state.pending_ir else ""

        self.frame_label.setText(
            f"帧: {self._state.frame}  ({self._state.idx + 1}/{total})  |  "
            f"已标注: {annotated} 帧"
        )
        self.points_label.setText(f"当前: {n} 对{pending}")

    # ── 点击处理 ──

    def _on_ir_clicked(self, scene_pos: QPointF) -> None:
        if self._state is None:
            return
        # 显示坐标 → 原始图像坐标
        orig_x = scene_pos.x() * self._ir_scale
        orig_y = scene_pos.y() * self._ir_scale
        added, _ = self._state.add_point(orig_x, orig_y, is_ir=True)
        if added:
            _refresh_overlays(self.ir_viewer, self.vis_viewer, self._state,
                              self._ir_scale, self._vis_scale)
            self._update_status()
            self.undo_btn.setEnabled(True)

    def _on_vis_clicked(self, scene_pos: QPointF) -> None:
        if self._state is None:
            return
        # 没有 pending IR 点时忽略 VIS 点击
        if self._state.pending_ir is None:
            return
        # 显示坐标 → 原始图像坐标
        orig_x = scene_pos.x() * self._vis_scale
        orig_y = scene_pos.y() * self._vis_scale
        added, ir_pt = self._state.add_point(orig_x, orig_y, is_ir=False)
        if added and ir_pt:
            n = len(self._state.ir_pts)
            logger.info(
                "[%d] IR (%.0f, %.0f) ↔ VIS (%.0f, %.0f)",
                n, ir_pt[0], ir_pt[1], orig_x, orig_y,
            )
            _refresh_overlays(self.ir_viewer, self.vis_viewer, self._state,
                              self._ir_scale, self._vis_scale)
            self._update_status()
            self.save_btn.setEnabled(True)

    # ── 按钮操作 ──

    def _on_undo(self) -> None:
        if self._state is None:
            return
        if self._state.undo():
            _refresh_overlays(self.ir_viewer, self.vis_viewer, self._state,
                              self._ir_scale, self._vis_scale)
            self._update_status()
            self.undo_btn.setEnabled(
                self._state.current_point_count() > 0 or self._state.pending_ir is not None
            )

    def _on_next(self) -> None:
        if self._state is None:
            return
        if self._state.go_next():
            self._show_current_frame()
        else:
            InfoBar.info(
                title="已是最后一帧",
                content="",
                orient=Qt.Orientation.Horizontal,
                isClosable=True,
                duration=1500,
                position=InfoBarPosition.TOP,
                parent=self,
            )

    def _on_prev(self) -> None:
        if self._state is None:
            return
        if self._state.go_prev():
            self._show_current_frame()
        else:
            InfoBar.info(
                title="已是第一帧",
                content="",
                orient=Qt.Orientation.Horizontal,
                isClosable=True,
                duration=1500,
                position=InfoBarPosition.TOP,
                parent=self,
            )

    def _on_save(self) -> None:
        if self._state is None:
            return
        # 先保存当前帧的标注
        self._state._save_current()

        save_path = self.npz_browser.path.strip() or self._load_path
        count = save_annotations(
            save_path,
            self._state.annotations,
            self._state.pairs,
        )
        if count > 0:
            self._load_path = save_path
            InfoBar.success(
                title="保存成功",
                content=f"已保存 {count} 帧标注 → {save_path}",
                orient=Qt.Orientation.Horizontal,
                isClosable=True,
                duration=3000,
                position=InfoBarPosition.TOP,
                parent=self,
            )
        else:
            warning("无标注数据", "请先添加控制点再保存。", self)

    # ── 键盘快捷键 ──

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if self._state is None:
            super().keyPressEvent(event)
            return

        key = event.key()
        ctrl = event.modifiers() & Qt.KeyboardModifier.ControlModifier

        if ctrl and key == Qt.Key.Key_Z:
            self._on_undo()
        elif key in (Qt.Key.Key_R, Qt.Key.Key_D):
            self._on_next()
        elif key in (Qt.Key.Key_Q, Qt.Key.Key_A):
            self._on_prev()
        elif key == Qt.Key.Key_S:
            self._on_save()
        elif key == Qt.Key.Key_U:
            self._on_undo()
        else:
            super().keyPressEvent(event)

    # ── 输入锁 ──

    def _set_inputs_enabled(self, enabled: bool) -> None:
        """扫描期间禁用输入"""
        self.ir_browser.setEnabled(enabled)
        self.vis_browser.setEnabled(enabled)
        self.npz_browser.setEnabled(enabled)
        self.undo_btn.setEnabled(enabled and self._state is not None)
        self.prev_btn.setEnabled(enabled and self._state is not None)
        self.next_btn.setEnabled(enabled and self._state is not None)
        self.save_btn.setEnabled(enabled and self._state is not None)

    # ── 持久化 ──

    def _load_settings(self) -> None:
        self.ir_browser.path = get_str("irvis_ir_dir")
        self.vis_browser.path = get_str("irvis_vis_dir")
        self.npz_browser.path = get_str("irvis_npz_path")
        # 如果两个路径都已持久化，启动时自动扫描
        if self.ir_browser.path.strip() and self.vis_browser.path.strip():
            self._try_scan()


# ═══════════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════════

def _resize_to_height(img, h: int):
    """保持宽高比缩放到目标高度"""
    if img.shape[0] == h:
        return img
    w = int(img.shape[1] * h / img.shape[0])
    return cv2.resize(img, (w, h))


def _cv_to_qpixmap(img) -> QPixmap:
    """OpenCV BGR → QPixmap (RGB)"""
    from PySide6.QtGui import QImage
    h, w, c = img.shape
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    qimg = QImage(rgb.data, w, h, c * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)
