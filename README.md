# PatchWork

视觉检测工具集成桌面应用，基于 PySide6 + QFluentWidgets，支持浅色/深色主题切换。

## 功能模块

| 模块 | 说明 | 状态 |
|------|------|------|
| 🏠 首页 | 功能总览 + 模块卡片 | ✅ |
| 📦 数据集划分 | 图片/标签按比例 train/val/test，含配对检查 + 剪切/复制模式 | ✅ |
| 📄 NDJSON转换 | NDJSON 格式数据集 → YOLO 格式 | ✅ |
| 🔍 模型推理 | YOLO 模型加载 + 点击推理 + det/seg 叠加预览，支持 conf/iou 调节 | ✅ |
| 🏷️ 开放词汇检测 | YOLOE 文本提示零样本检测，支持任意类别名 | ✅ |
| 🔬 SAHI 切片推理 | 高分辨率图片切片检测，小目标增强 | ✅ |
| 📊 推理对比 | 多推理方式对比评测，输出每类 P/R/F1/AP50 表格 | ✅ |
| 🏷️ Label 预览 | YOLO 标注文件可视化预览 | ✅ |
| 📤 导出 ONNX | YOLO .pt → ONNX，支持自定义 imgsz/simplify/dynamic | ✅ |
| 🎬 视频抽帧 | 按秒/帧间隔抽帧，支持 jpg/png，实时预估产出 | ✅ |
| 🖥️ GPU 监控 | nvidia-smi / gpustat / HTTP 三种方式远程监控 GPU | ✅ |
| 🔧 X-AnyLabeling | subprocess 唤醒外部标注工具 | ✅ |
| 🎯 视频跟踪 | BoT-SORT / ByteTrack 视频目标跟踪 + 漏检补偿 | ⏳ |

## 技术栈

- **Python 3.10+**
- **GUI**: PySide6 + QFluentWidgets (Fluent Design, 浅/深色主题)
- **推理**: ultralytics (YOLO/YOLOE) + supervision，支持 detect/segment
- **开放词汇**: YOLOE + mobileclip 文本编码
- **SSH**: paramiko (GPU 监控)
- **打包**: PyInstaller
- **日志**: logging + 全局异常捕获 (faulthandler)
- **视频**: opencv-python
- **数据集**: ultralytics NDJSON converter

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
├── main.py                  # 入口 + 日志初始化 + 全局异常捕获
├── pyproject.toml           # 依赖声明
├── app/
│   ├── main_window.py       # FluentWindow 主窗口 + 导航 + 主题切换
│   ├── widgets/             # GUI 面板（每功能一个文件）
│   │   ├── home_panel.py    # 首页: Banner + 模块卡片
│   │   ├── image_browser.py # 通用: 缩略图列表 + 预览 + 导航按钮
│   │   ├── thumbnail_list.py# 通用: 懒加载缩略图列表
│   │   ├── image_viewer.py  # 通用: QGraphicsView 缩放/叠加
│   │   └── path_browser.py  # 通用: 文件/目录浏览控件
│   ├── services/            # 业务逻辑（无 UI）
│   └── utils/               # 工具（日志/配置/Worker/消息框/异常捕获）
└── resources/               # 图标等资源
```

## License

GPL-3.0
