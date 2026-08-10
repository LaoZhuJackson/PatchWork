"""ICAFusion 双流推理引擎：YOLOv5 TransFusion 模型加载 + VIS/IR 双图推理"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PySide6.QtCore import QRectF

from app.services.label_reader import get_color

# ── 延迟导入：外部在构造 ICAFusionEngine 时设置 icafusion_root ──
_icafusion_imports: dict = {}
_icafusion_root_added: str = ""


def _ensure_imports(icafusion_root: str) -> None:
    """将 ICAFusion 根目录加入 sys.path，惰性导入所需的 YOLOv5 工具函数。"""
    global _icafusion_root_added

    root = str(Path(icafusion_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
        _icafusion_root_added = root

    if _icafusion_imports:
        return

    # 导入 ICAFusion / YOLOv5 工具（注释掉可能失败的非必要导入）
    from models.experimental import attempt_load                 # noqa: E402
    from utils.general import non_max_suppression, scale_coords  # noqa: E402
    from utils.datasets import letterbox                        # noqa: E402
    from utils.torch_utils import select_device                 # noqa: E402

    _icafusion_imports.update({
        "attempt_load": attempt_load,
        "non_max_suppression": non_max_suppression,
        "scale_coords": scale_coords,
        "letterbox": letterbox,
        "select_device": select_device,
    })


def _imread(path: str | Path) -> np.ndarray:
    """Windows 兼容的图片读取（绕过 cv2.imread 中文路径 bug）。"""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


# ──────────────────────────────────────────────
class ICAFusionEngine:
    """ICAFusion 双流目标检测推理引擎。

    与 ``InferenceEngine`` 保持相同的 annotation dict 输出格式，
    可以直接喂给 ``ImageBrowser.show_annotations()``。
    """

    def __init__(self, icafusion_root: str) -> None:
        self._icafusion_root = str(Path(icafusion_root).resolve())
        _ensure_imports(self._icafusion_root)

        self._model: torch.nn.Module | None = None
        self._model_path: str = ""
        self._class_names: dict[int, str] = {}
        self._device: torch.device | None = None
        self._stride: int = 32
        self._half: bool = False

    # ── 属性 ──────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def model_path(self) -> str:
        return self._model_path

    @property
    def class_names(self) -> dict[int, str]:
        return self._class_names

    @property
    def device_str(self) -> str:
        return str(self._device) if self._device else ""

    @property
    def root(self) -> str:
        return self._icafusion_root

    # ── 模型加载 ─────────────────────────

    def load_model(self, weights_path: str, device: str = "") -> None:
        """加载 ICAFusion 权重文件。

        Args:
            weights_path: .pt 权重路径
            device: 设备字符串，如 ``"0"``, ``"cpu"``
        """
        _ensure_imports(self._icafusion_root)

        select_device_fn = _icafusion_imports["select_device"]
        attempt_load_fn = _icafusion_imports["attempt_load"]

        dev = select_device_fn(device)
        self._half = dev.type != "cpu"

        model = attempt_load_fn(weights_path, map_location=dev)
        self._stride = int(model.stride.max())

        if self._half:
            model.half()

        model.eval()
        self._model = model
        self._model_path = str(weights_path)
        self._device = dev

        # YOLOv5 的 names 可能是 list 或 dict，统一规整为 {class_id: name}
        names = model.module.names if hasattr(model, "module") else model.names
        if isinstance(names, (list, tuple)):
            self._class_names = {i: name for i, name in enumerate(names)}
        else:
            self._class_names = names

    # ── 推理 ─────────────────────────────

    def infer(
        self,
        vis_path: str | Path,
        ir_path: str | Path,
        conf: float = 0.25,
        iou: float = 0.45,
        img_size: int = 1280,
    ) -> tuple[list[dict], list[dict]]:
        """对一对 VIS + IR 图像推理。

        Returns:
            ``(vis_annotations, ir_annotations)`` —— 两组 annotation dict，
            可直接用于 ``ImageBrowser.show_annotations()``。
        """
        if self._model is None:
            raise RuntimeError("模型未加载")

        _ensure_imports(self._icafusion_root)
        letterbox_fn = _icafusion_imports["letterbox"]
        nms_fn = _icafusion_imports["non_max_suppression"]
        scale_fn = _icafusion_imports["scale_coords"]

        # ── 读图 ──
        im0_vis = _imread(vis_path)
        im0_ir = _imread(ir_path)

        # ── 预处理：letterbox + BGR→RGB + HWC→CHW ──
        img_vis, ratio_vis, pad_vis = letterbox_fn(
            im0_vis, img_size, stride=self._stride
        )
        img_ir, _, _ = letterbox_fn(im0_ir, img_size, stride=self._stride)

        img_vis = img_vis[:, :, ::-1].transpose(2, 0, 1)         # BGR→RGB, HWC→CHW
        img_vis = np.ascontiguousarray(img_vis)
        img_ir = img_ir[:, :, ::-1].transpose(2, 0, 1)
        img_ir = np.ascontiguousarray(img_ir)

        # ── → Tensor ──
        vis_t = torch.from_numpy(img_vis).to(self._device)
        ir_t = torch.from_numpy(img_ir).to(self._device)
        vis_t = vis_t.half() if self._half else vis_t.float()
        ir_t = ir_t.half() if self._half else ir_t.float()
        vis_t = vis_t / 255.0
        ir_t = ir_t / 255.0
        if vis_t.ndimension() == 3:
            vis_t = vis_t.unsqueeze(0)
            ir_t = ir_t.unsqueeze(0)

        # ── Forward ──
        with torch.no_grad():
            pred = self._model(vis_t, ir_t)[0]

        # ── NMS ──
        det = nms_fn(pred, conf, iou)[0]

        # ── 构造 annotation dict ──
        vis_anns: list[dict] = []
        ir_anns: list[dict] = []

        if len(det):
            # 坐标还原：letterbox 空间 → 原图空间
            det[:, :4] = scale_fn(
                img_vis.shape[1:], det[:, :4], im0_vis.shape
            ).round()

            for *xyxy, conf_val, cls_id in det:
                x1, y1, x2, y2 = [float(v) for v in xyxy]
                cid = int(cls_id)
                score = float(conf_val)
                label = (
                    f"{self._class_names.get(cid, f'class_{cid}')} {score:.2f}"
                )

                ann = {
                    "type": "bbox",
                    "rect": QRectF(x1, y1, x2 - x1, y2 - y1),
                    "class_id": cid,
                    "color": get_color(cid),
                    "label": label,
                }
                vis_anns.append(ann)
                ir_anns.append(ann)

        return vis_anns, ir_anns
