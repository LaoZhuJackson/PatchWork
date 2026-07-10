# PatchWork

PV 缺陷检测工具集成桌面应用，基于 PySide6 + QFluentWidgets。

## 功能模块

| 模块 | 说明 | 状态 |
|------|------|------|
| 📦 数据集划分 | 图片/标签按比例划分 train/val/test，含配对检查 + 剪切/复制模式 | ✅ |
| 🔍 模型推理 | YOLO 模型加载 + 点击推理 + 检测框/多边形叠加预览 (det/seg) | ✅ |
| 🏷️ Label 预览 | YOLO 标签反归一化 + 在图片上叠加框/多边形 | ✅ |
| 📤 导出 ONNX | YOLO .pt → ONNX 格式导出，自动读取 imgsz | ✅ |
| 🎬 视频抽帧 | 按时长/帧间隔抽取关键帧 | ⏳ |
| 🎯 最优置信度 | 按类别寻找最优置信度阈值 | ⏳ |
| 🖥️ GPU 监控 | SSH 远程 nvidia-smi 监控 | ⏳ |
| 🔧 X-AnyLabeling | subprocess 唤醒外部标注工具 | ⏳ |

## 技术栈

- **Python 3.10+**
- **GUI**: PySide6 + QFluentWidgets (Fluent Design)
- **推理**: ultralytics + supervision
- **打包**: PyInstaller
- **SSH**: paramiko

## 安装

```bash
conda create -n patchwork python=3.10
conda activate patchwork
pip install -e .
```

## 运行

```bash
python main.py
```

## 目录结构

```
PatchWork/
├── main.py                  # 入口
├── pyproject.toml           # 依赖声明
├── app/
│   ├── main_window.py       # 主窗口 + 导航
│   ├── widgets/             # GUI 面板（每个功能一个文件）
│   ├── services/            # 业务逻辑（无 UI）
│   └── utils/               # 工具（日志/配置/Worker/消息框）
└── resources/               # 图标等资源
```

## License

GPL-3.0
