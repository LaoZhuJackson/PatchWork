"""推理适配器抽象基类"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

class InferenceAdapter(ABC):
    """统一推理接口，所有推理方式实现此抽象类"""

    @property
    @abstractmethod
    def name(self) -> str:
        """显示名称，如 'YOLO 普通推理'"""
        ...

    @abstractmethod
    def load_model(self) -> None:
        """加载模型（后台线程调用）"""
        ...

    @abstractmethod
    def infer(self, image_path: Path) -> list[dict]:
        """对单张图片推理，返回统一标注列表"""
        ...

    @property
    def config_summary(self) -> str:
        """参数摘要，表格列头用"""
        return ""