"""SenseNova U1.5 云端图片编辑 API 客户端（红框控制目标合成）

对应 YOLO-Synth-U15 的第②步：把本地 U1.5 8B 推理替换为商汤云端 API。
用户画的红框坐标即标注（零成本），模型在框内"种"出目标并自动移除红框。

接口契约（已实测）:
    POST https://token.sensenova.cn/v1/images/edits
    model: sensenova-u1.5-lite
    images: [{"image_url": data:image/png;base64,...}]
    size: "auto" 或 "WxH"（宽高为 32 倍数、[512,4096]、宽高比≤3:1）
    返回: json["data"][0]["url"]（24h 临时链接，需立即下载）
"""
from __future__ import annotations

import base64
import io
import os
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

BASE_URL = "https://token.sensenova.cn/v1/images/edits"
DEFAULT_MODEL = "sensenova-u1.5-lite"

MIN_SIZE = 512      # API 允许的最小边长
MAX_SIZE = 4096     # API 允许的最大边长
STEP = 32           # 尺寸必须是 32 的倍数
MAX_ASPECT = 3.0    # 宽高比上限

RED = (255, 0, 0)

# 默认类别：光伏面板裂缝（提示词在最小测试中已验证）
DEFAULT_CLASSES = [
    {
        "id": 0,
        "name": "光伏面板裂缝",
        "prompt": "在光伏面板上生成细长的裂缝，符合自然光伏板会产生的裂缝",
        "detector_class": "crack",
    },
]


def resolve_key() -> str:
    """API Key 解析：环境变量 SENSENOVA_API_KEY -> 同目录 sensenova_key.txt"""
    key = os.environ.get("SENSENOVA_API_KEY", "").strip()
    if key:
        return key
    kf = Path(__file__).resolve().parent / "sensenova_key.txt"
    if kf.exists():
        key = kf.read_text(encoding="utf-8").strip()
        if key:
            return key
    return ""


def nearest_valid_size(w: int, h: int) -> tuple[int, int]:
    """最近的有效 API 尺寸：宽高为 32 倍数、[512,4096]、宽高比≤3:1"""
    w = min(max(int(w), MIN_SIZE), MAX_SIZE)
    h = min(max(int(h), MIN_SIZE), MAX_SIZE)

    def _round32(v: int) -> int:
        return max(MIN_SIZE, min(MAX_SIZE, int(round(v / STEP)) * STEP))

    w = _round32(w)
    h = _round32(h)

    if w > h * MAX_ASPECT:
        w = _round32(h * MAX_ASPECT)
    if h > w * MAX_ASPECT:
        h = _round32(w * MAX_ASPECT)
    return w, h


def draw_boxes(image: Image.Image, boxes, with_index: bool = False) -> Image.Image:
    """在图上画纯红矩形（线宽随图自适应），多框时左上角画红色编号。

    boxes: list of (x1, y1, x2, y2) 像素坐标
    移植自 YOLO-Synth 的 synth_layout.draw_boxes
    """
    d = ImageDraw.Draw(image)
    W, H = image.size
    lw = max(3, W // 400)
    fs = max(16, W // 40)
    try:
        font = ImageFont.truetype("arial.ttf", fs)
    except OSError:
        font = ImageFont.load_default()
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        d.rectangle([x1, y1, x2, y2], outline=RED, width=lw)
        if with_index:
            d.text((x1 + lw, max(0, y1 - fs - lw)), str(i + 1),
                   fill=RED, font=font)
    return image


def build_prompt(items: list[tuple[str, str]]) -> str:
    """四段式红框编辑提示词：定位 -> 内容 -> 框内约束 -> 移除红框+保持不变

    items: list of (类别名, 内容描述)
    移植自 YOLO-Synth 的 synth_layout.build_prompt
    """
    if len(items) == 1:
        _, desc = items[0]
        return (f"在红框内生成{desc}。"
                f"物体必须完全位于红框内部，不得超出红框边界。"
                f"生成完成后移除红框。"
                f"保持红框以外的背景、光照、色调与构图完全不变。")
    parts = "；".join(
        f"在编号{i + 1}的红框内生成{desc}" for i, (_, desc) in enumerate(items))
    return (f"图中共有{len(items)}个红框，每个红框左上角标有红色编号。{parts}。"
            f"每个物体都必须完全位于各自红框内部，不得超出边界，物体之间不得互相遮挡错位。"
            f"生成完成后移除所有红框和编号。"
            f"保持红框以外的背景、光照、色调与构图完全不变。")


def image_to_data_url(image: Image.Image) -> str:
    """PIL 图 -> base64 data-URL（RGB PNG）"""
    image = image.convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def edit_image(
    image,
    prompt: str,
    boxes=None,
    api_key: str = "",
    model: str = DEFAULT_MODEL,
    keep_size: bool = True,
    watermark: bool = False,
    prompt_extend: bool = False,
    timeout: int = 180,
) -> tuple[np.ndarray, dict]:
    """调云端图片编辑 API，返回 (编辑后 RGB ndarray, meta)。

    Args:
        image: 输入图（RGB ndarray / PIL Image / 文件路径）
        prompt: 编辑提示词
        boxes: list of (x1,y1,x2,y2) 像素坐标；多框时自动画红色编号
        keep_size: True=传最近 32 倍数尺寸并在返回后精确回原分辨率；
                   False=用 "auto"（模型会放大，不保证同分辨率）
        meta: {http_status, size_used, elapsed_s, response}
    """
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image).convert("RGB")
    elif not isinstance(image, Image.Image):
        raise TypeError("image 需为 ndarray / PIL Image / 路径")

    orig_w, orig_h = image.size

    # 1) 画红框（在副本上，不影响原图）
    if boxes:
        draw_boxes(image, boxes, with_index=len(boxes) > 1)

    # 2) 尺寸
    if keep_size:
        size_w, size_h = nearest_valid_size(orig_w, orig_h)
        size = f"{size_w}x{size_h}"
    else:
        size = "auto"

    # 3) 请求
    payload = {
        "model": model,
        "images": [{"image_url": image_to_data_url(image)}],
        "prompt": prompt,
        "n": 1,
        "size": size,
        "watermark": watermark,
        "prompt_extend": prompt_extend,
        "response_format": "url",
    }
    headers = {
        "Authorization": f"Bearer {api_key or resolve_key()}",
        "Content-Type": "application/json",
    }

    t0 = time.time()
    resp = requests.post(BASE_URL, json=payload, headers=headers, timeout=timeout)
    meta = {"http_status": resp.status_code, "size_used": size,
            "elapsed_s": round(time.time() - t0, 1)}

    if resp.status_code != 200:
        try:
            err = resp.json()
        except ValueError:
            err = {"raw": resp.text[:500]}
        raise RuntimeError(
            f"接口返回 {resp.status_code}: {err}\n"
            f"提示：检查 API Key 是否正确、model 名是否可用。")

    data = resp.json()
    item = (data.get("data") or [{}])[0]
    url = item.get("url")
    if url and url.startswith("data:"):
        out = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB")
    elif url:
        dl = requests.get(url, timeout=timeout)
        dl.raise_for_status()
        out = Image.open(io.BytesIO(dl.content)).convert("RGB")
    else:
        raise RuntimeError(f"响应中未找到图片 url: {data}")

    meta["response"] = data

    # 4) keep_size：精确回原分辨率
    if keep_size and (out.size != (orig_w, orig_h)):
        out = out.resize((orig_w, orig_h), Image.LANCZOS)

    return np.asarray(out), meta
