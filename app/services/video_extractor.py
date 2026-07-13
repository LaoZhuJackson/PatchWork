"""视频抽帧逻辑"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable

import cv2


def extract_frames(video_path: str | Path, output_dir: str | Path, mode: str = "time", interval: float = 1.0,
                   fmt: str = "jpg", progress_callback: Callable[[int], None] | None = None) -> dict:
    """从视频中抽帧。

    Args:
      video_path: 视频文件路径
      output_dir: 输出目录
      mode: "time" 按秒抽帧 / "frame" 按帧间隔抽帧
      interval: 间隔（秒或帧数）
      fmt: 输出格式 "jpg" 或 "png"
      progress_callback: 进度回调 0-100

    Returns:
      {"total_frames": int, "fps": float, "duration": float, "extracted": int, "output_dir": Path}
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频文件: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration = total_frames / fps if fps > 0 else 0

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_name = Path(video_path).stem
    ext = fmt.lower().lstrip(".")

    # 计算多少帧取一帧
    if mode == "time":
        step = max(0.01, int(interval * fps))
    else:
        step = max(1, int(interval))

    # 抽帧
    extracted = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            out_path = output_dir / f"{video_name}_{frame_idx:06d}.{ext}"
            cv2.imwrite(str(out_path), frame)
            extracted += 1
        frame_idx += 1
        if progress_callback and total_frames > 0:
            progress_callback(int(frame_idx / total_frames * 100))

    cap.release()

    return {
        "total_frames": total_frames,
        "fps": fps,
        "duration": duration,
        "extracted": extracted,
        "output_dir": output_dir,
    }
