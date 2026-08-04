"""IR-VIS 控制点标注：文件配对 + 标注状态管理 + npz 持久化"""
from __future__ import annotations

import io
import re
from pathlib import Path

import numpy as np

from app.utils.logger import get_logger

logger = get_logger(__name__)

# ── 文件名匹配 ──
# IR: xxx_xxx_T_000042.jpg      VIS: xxx_xxx_V_000042.jpg
#     前缀部分可不同              前缀需一致才能配对
IR_RE = re.compile(r"^(.+)_T_(\d{6})\.\w+$", re.IGNORECASE)
VIS_RE = re.compile(r"^(.+)_V_(\d{6})\.\w+$", re.IGNORECASE)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


# ── 配对结果 ──

class IRVISPair:
    """一对 IR/VIS 图像"""
    __slots__ = ("frame", "ir_path", "vis_path")

    def __init__(self, frame: str, ir_path: Path, vis_path: Path) -> None:
        self.frame = frame          # 帧号字符串, 如 "000042"
        self.ir_path = ir_path      # IR 图像路径
        self.vis_path = vis_path    # VIS 图像路径


# ── 扫描配对 ──

def scan_pairs(
    ir_dir: str | Path,
    vis_dir: str | Path,
) -> tuple[list[IRVISPair], dict]:
    """按帧号配对 IR/VIS 图像

    Returns:
        pairs: 按帧号排序的配对列表
        stats: {
            ir_count: int, vis_count: int, paired: int,
            unpaired_ir: list[str], unpaired_vis: list[str],
        }
    """
    ir_dir = Path(ir_dir)
    vis_dir = Path(vis_dir)

    # 扫描 IR
    ir_map: dict[str, Path] = {}
    for f in ir_dir.iterdir():
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
            if m := IR_RE.match(f.name):
                ir_map[m.group(2)] = f

    # 扫描 VIS
    vis_map: dict[str, Path] = {}
    for f in vis_dir.iterdir():
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
            if m := VIS_RE.match(f.name):
                vis_map[m.group(2)] = f

    # 配对 — 取帧号交集
    common = sorted(set(ir_map) & set(vis_map))
    pairs = [IRVISPair(f, ir_map[f], vis_map[f]) for f in common]

    unpaired_ir = sorted(set(ir_map) - set(vis_map))
    unpaired_vis = sorted(set(vis_map) - set(ir_map))

    stats = {
        "ir_count": len(ir_map),
        "vis_count": len(vis_map),
        "paired": len(pairs),
        "unpaired_ir": unpaired_ir,
        "unpaired_vis": unpaired_vis,
    }

    logger.info(
        "IR-VIS scan: IR=%d VIS=%d paired=%d",
        stats["ir_count"], stats["vis_count"], stats["paired"],
    )

    return pairs, stats


# ── 标注状态管理 ──

class IRVISState:
    """单次标注会话的状态机

    维护:
      - 当前帧索引
      - annotations dict: {frame: {"ir": [(x,y),...], "vis": [(x,y),...]}}
      - pending_ir: 等待 VIS 配对的临时 IR 点（同一时刻最多一个）

    帧切换时会自动 save_current → load_current。
    """

    def __init__(self, pairs: list[IRVISPair]) -> None:
        if not pairs:
            raise ValueError("配对列表为空，无法开始标注")

        self.pairs = pairs
        self.annotations: dict[str, dict] = {}
        self.idx = 0

        # 当前帧状态
        self.ir_pts: list[tuple[float, float]] = []
        self.vis_pts: list[tuple[float, float]] = []
        self.pending_ir: tuple[float, float] | None = None

    # ── 属性 ──

    @property
    def current_pair(self) -> IRVISPair:
        return self.pairs[self.idx]

    @property
    def frame(self) -> str:
        return self.current_pair.frame

    @property
    def total_pairs(self) -> int:
        return len(self.pairs)

    # ── 帧导航 ──

    def go_next(self) -> bool:
        """切换到下一帧。返回 False 表示已是最后一帧"""
        if self.idx < len(self.pairs) - 1:
            self._save_current()
            self.idx += 1
            self._load_current()
            return True
        return False

    def go_prev(self) -> bool:
        """切换到上一帧。返回 False 表示已是第一帧"""
        if self.idx > 0:
            self._save_current()
            self.idx -= 1
            self._load_current()
            return True
        return False

    # ── 控制点操作 ──

    def add_point(
        self, x: float, y: float, is_ir: bool,
    ) -> tuple[bool, tuple | None]:
        """添加一个控制点

        Args:
            x, y: 图像坐标（相对图像左上角）
            is_ir: True=点击的是 IR 图, False=VIS 图

        Returns:
            (added, pair)
              - added: 操作是否有效
              - pair: 如果是 VIS 点击且完成了配对, 返回 (ir_x, ir_y) 用于日志
        """
        if is_ir:
            # IR 图点击 → 存为 pending, 等待 VIS 配对
            self.pending_ir = (x, y)
            return True, None

        # VIS 图点击 → 需要已有 pending IR 点
        if self.pending_ir is not None:
            ir_pt = self.pending_ir
            self.ir_pts.append(ir_pt)
            self.vis_pts.append((x, y))
            self.pending_ir = None
            self._save_current()
            return True, ir_pt

        return False, None

    def undo(self) -> bool:
        """撤销最近的操作

        优先撤销 pending IR 点（还没配对）, 否则撤销最后一对已完成的点。
        """
        if self.pending_ir is not None:
            self.pending_ir = None
            return True
        if self.ir_pts:
            self.ir_pts.pop()
            self.vis_pts.pop()
            self._save_current()
            return True
        return False

    def clear_current(self) -> None:
        """清空当前帧所有控制点"""
        self.ir_pts.clear()
        self.vis_pts.clear()
        self.pending_ir = None
        self._save_current()

    # ── 统计 ──

    def count_annotated(self) -> int:
        """已标注帧数（至少有一对控制点的帧）"""
        return sum(
            1 for p in self.pairs
            if p.frame in self.annotations and self.annotations[p.frame]["ir"]
        )

    def current_point_count(self) -> int:
        """当前帧的控制点对数"""
        return len(self.ir_pts)

    # ── 内部 ──

    def _save_current(self) -> None:
        """把当前帧的点列表写入 annotations dict"""
        if self.ir_pts:
            self.annotations[self.frame] = {
                "ir": list(self.ir_pts),
                "vis": list(self.vis_pts),
            }
        else:
            self.annotations.pop(self.frame, None)

    def _load_current(self) -> None:
        """从 annotations dict 恢复当前帧的点列表"""
        anno = self.annotations.get(self.frame, {"ir": [], "vis": []})
        self.ir_pts = list(anno["ir"])
        self.vis_pts = list(anno["vis"])
        self.pending_ir = None


# ── .npz 持久化 ──

def load_annotations(path: str | Path) -> dict[str, dict]:
    """从 .npz 文件加载标注

    Returns:
        {frame: {"ir": [(x,y),...], "vis": [(x,y),...]}, ...}
        文件不存在或损坏时返回空 dict
    """
    path = Path(path)
    if not path.is_file():
        return {}

    try:
        # 用 BytesIO 绕过 np.load 在 Windows 上对中文路径的 C 层 fopen 问题
        data = np.load(io.BytesIO(path.read_bytes()), allow_pickle=True)
    except (OSError, ValueError) as exc:
        logger.warning("无法加载标注文件 %s: %s", path, exc)
        return {}

    annotations: dict[str, dict] = {}
    for frame in data.get("frames", []):
        frame = str(frame)
        ir_key, vis_key = f"ir_{frame}", f"vis_{frame}"
        if ir_key in data and vis_key in data:
            annotations[frame] = {
                "ir": [tuple(float(v) for v in p) for p in data[ir_key]],
                "vis": [tuple(float(v) for v in p) for p in data[vis_key]],
            }

    logger.info("加载 %d 帧已有标注 → %s", len(annotations), path)
    return annotations


def save_annotations(
    path: str | Path,
    annotations: dict[str, dict],
    pairs: list[IRVISPair],
) -> int:
    """保存标注到 .npz 文件

    Returns:
        实际保存的帧数（至少有一对控制点的帧）
    """
    path = Path(path)
    data: dict[str, np.ndarray] = {}
    saved_frames: list[str] = []

    for pair in pairs:
        frame = pair.frame
        anno = annotations.get(frame)
        if not anno or not anno.get("ir"):
            continue
        data[f"ir_{frame}"] = np.array(anno["ir"], dtype=np.float32)
        data[f"vis_{frame}"] = np.array(anno["vis"], dtype=np.float32)
        saved_frames.append(frame)

    if not saved_frames:
        logger.warning("无标注数据，跳过保存")
        return 0

    data["frames"] = np.array(saved_frames, dtype=str)

    path.parent.mkdir(parents=True, exist_ok=True)
    # BytesIO 绕开 np.savez_compressed 在 Windows 上中文路径的 C 层 fopen 问题
    buf = io.BytesIO()
    np.savez_compressed(buf, **data)
    path.write_bytes(buf.getvalue())
    logger.info("保存 %d 帧标注 → %s", len(saved_frames), path)
    return len(saved_frames)
