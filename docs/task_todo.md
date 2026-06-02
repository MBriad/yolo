# YOLO 目标检测 — 实战学习项目

## 概述

通过 7 个阶段 22 个任务，从零到部署掌握 YOLO 目标检测。

**环境约束：**
- NVIDIA GPU 8-12GB 显存，本地训练主力
- Google Colab 仅作备选（需要更大显存时）
- `uv` 管理开发环境，最终导出 `requirements.txt` 用于 pip 部署

**CLAUDE.md 纪律（全程遵守）：**
- 每个任务 = 单一目标，不蔓延
- 每个任务有 checkbox 验收标准，通过再进入下一步
- 源文件保持 300 行以内，测试文件 500 行以内
- 简洁 > 聪明，能用 > 好看

---

## Phase 0: 环境基础

> 目标：一个可运行的 Python 环境，`uv run python -c "from ultralytics import YOLO"` 成功，且 `torch.cuda.is_available()` 返回 `True`。

---

### Task 0.1: 初始化 uv 项目并固定 Python 版本

**学习目标：** 理解 `uv` 项目初始化、Python 版本锁定。

**预计时间：** 15 分钟。

**设计说明：**
- 使用 Python 3.11（ultralytics 和 torch 测试最充分的版本区间 3.10-3.12）
- `uv init` 创建 `pyproject.toml` 和 `.python-version`
- 不要手动创建虚拟环境 — uv 自动管理

**步骤：**
1. 在项目根目录打开终端
2. 运行 `uv init --python 3.11 --name yolo-learning`
3. 查看 `pyproject.toml` — 确认有 `[project]` 表，含 name、version、requires-python
4. 运行 `uv python list` 确认 3.11 是解析到的版本
5. 运行 `uv sync` — 应输出 "Resolved 0 packages"（尚无依赖）

**成功标准（验证通过才进入下一步）：**
- [ ] `pyproject.toml` 存在，`name = "yolo-learning"`
- [ ] `.python-version` 内容为 `3.11`
- [ ] `uv sync` 无错误完成
- [ ] `uv run python --version` 输出 `Python 3.11.x`

**CLAUDE.md 纪律检查：**
- 精确：只初始化项目，不添加依赖，不创建源文件

---

### Task 0.2: 安装核心依赖（GPU 版本）

**学习目标：** 理解 `uv add` 和依赖解析。知道一个 YOLO 项目的核心包有哪些。确保 torch 是 CUDA 版本。

**预计时间：** 20 分钟。

**设计说明：**
- `ultralytics` 会拉取 `torch`（CPU 版）、`numpy`、`opencv-python`、`pillow`、`pyyaml`
- 需要**覆盖安装** torch 的 CUDA 版本，否则 GPU 不可用
- 推荐 CUDA 12.1 或 12.4 版本的 torch（兼容性最好）

**步骤：**
1. 先添加 ultralytics：`uv add "ultralytics>=8.0,<9.0"`
2. 添加 pillow（显式声明）：`uv add pillow`
3. 安装 CUDA 版 torch（覆盖 CPU 版）：
   ```bash
   uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```
   > 如果 CUDA 版本不同，去 https://pytorch.org/get-started/locally/ 查看对应命令
4. 验证 GPU 可用：
   ```bash
   uv run python -c "import torch; print(f'Torch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
   ```
   预期：`CUDA available: True`，并显示你的 GPU 名称
5. 验证 ultralytics：`uv run python -c "from ultralytics import YOLO; print('OK')"`

**成功标准（验证通过才进入下一步）：**
- [ ] `torch.cuda.is_available()` 返回 `True`
- [ ] 终端显示正确的 GPU 名称和 CUDA 版本
- [ ] `from ultralytics import YOLO` 无报错
- [ ] `pyproject.toml` 的 dependencies 中列出 `ultralytics` 和 `pillow`

**CLAUDE.md 纪律检查：**
- 简洁：一个 `uv add` 添加核心包。不要提前安装 onnx/fastapi 等后续阶段的依赖

---

### Task 0.3: 创建项目目录结构

**学习目标：** 建立清晰扁平的模块结构。每个文件一个职责。

**预计时间：** 15 分钟。

**设计说明：**
- `src/` 放所有应用代码，每个 .py 文件可直接 `uv run python src/<script>.py`
- `scripts/` 放一次性工具（下载器、验证器）
- `data/` 和 `models/` 用 .gitignore 忽略（大文件不入库）
- `notebooks/` 放 Colab 笔记本（备选方案）
- 不要创建 `utils/` 或 `common/` 模块 — 单次使用的代码内联即可

**步骤：**
1. 创建目录树：
   ```
   Yolo/
   ├── src/
   │   └── __init__.py
   ├── scripts/
   ├── notebooks/
   ├── tests/
   │   └── __init__.py
   ├── data/
   │   └── README.md          # "Datasets go here. See docs/task_todo.md."
   └── models/
       └── README.md          # "Trained models go here. Gitignored."
   ```
2. 创建 `.gitignore`：
   ```gitignore
   # Python
   __pycache__/
   *.pyc
   .venv/

   # Data and models (large binaries)
   data/*
   !data/README.md
   models/*
   !models/README.md

   # Ultralytics training output
   runs/

   # Environment
   .env
   *.egg-info/
   ```
3. 验证：`git status` 应显示新文件，但 `data/` 和 `models/` 下只有 README

**成功标准（验证通过才进入下一步）：**
- [ ] 目录树与上面一致
- [ ] `.gitignore` 存在且 `data/*` 和 `models/*` 被正确忽略
- [ ] `src/__init__.py` 和 `tests/__init__.py` 存在（空文件即可）

**CLAUDE.md 纪律检查：**
- 精确：只创建结构。不要写任何源代码

---

## Phase 1: 首次 YOLO 推理

> 目标：下载预训练模型，在一张图片上跑推理，看到检测框。这是证明整个管线能跑通的"Aha moment"。

---

### Task 1.1: 下载模型并在一张图片上推理

**学习目标：** 使用 `ultralytics.YOLO` API。理解 `.pt` 文件自动下载。看到原始预测输出。

**预计时间：** 20 分钟。

**设计说明：**
- `YOLO('yolov8n.pt')` 首次运行自动下载 nano 模型（~6 MB）
- 模型返回 `Results` 对象列表，每张输入图片一个
- 用你磁盘上任意一张 JPEG/PNG 图片测试。没有的话去 [Ultralytics Assets](https://ultralytics.com/images/bus.jpg) 下载
- 脚本控制在 50 行以内

**步骤：**
1. 创建 `src/infer_image.py`
2. 写脚本：
   ```python
   from ultralytics import YOLO

   model = YOLO('yolov8n.pt')  # 首次运行自动下载
   results = model('path/to/your/image.jpg')
   print(results[0])
   ```
3. 运行：`uv run python src/infer_image.py`
4. 观察输出：应打印检测到的类别、边界框坐标、置信度
5. 将下载的模型移到正确位置：
   - `mv yolov8n.pt models/yolov8n.pt`（Windows: `move yolov8n.pt models\yolov8n.pt`）
   - 更新脚本路径为 `models/yolov8n.pt`

**成功标准（验证通过才进入下一步）：**
- [ ] 脚本无错误运行
- [ ] 控制台输出至少一个检测结果（boxes、confidence）
- [ ] `models/yolov8n.pt` 存在（~6 MB）
- [ ] 脚本加载 `models/yolov8n.pt` 正常工作

**CLAUDE.md 纪律检查：**
- 目标驱动：成功 = 在控制台看到检测数据。不是漂亮的展示，下一步再做

---

### Task 1.2: 解析并结构化输出结果

**学习目标：** 理解 `Results` 对象 API。用代码提取 boxes、class names、confidence scores。

**预计时间：** 20 分钟。

**设计说明：**
- `results[0].boxes` 是 `Boxes` 对象，含 `.xyxy`（像素坐标）、`.cls`（类别索引）、`.conf`（置信度）
- `results[0].names` 是 `{class_id: class_name}` 字典
- 张量转 Python 列表用 `.tolist()` 或 `.cpu().numpy()`
- 不要加画图功能，只做结构化输出

**步骤：**
1. 扩展 `src/infer_image.py`，添加解析逻辑：
   ```python
   for box in results[0].boxes:
       class_id = int(box.cls.item())
       class_name = results[0].names[class_id]
       confidence = float(box.conf.item())
       x1, y1, x2, y2 = box.xyxy[0].tolist()
       print(f"{class_name}: {confidence:.2f} @ [{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")
   ```
2. 在同一张测试图上运行
3. 确认输出是人类可读的格式

**成功标准（验证通过才进入下一步）：**
- [ ] 每个检测结果为：`class_name: 0.XX @ [x1,y1,x2,y2]`
- [ ] 类别名可读（如 "person"、"car"），非数字 ID
- [ ] 置信度在 0.0 到 1.0 之间

**CLAUDE.md 纪律检查：**
- 精确：只加解析逻辑。不动模型加载和图片读取部分

---

### Task 1.3: 保存带标注框的结果图片

**学习目标：** 用 ultralytics 内置的 `plot()` 方法渲染检测框并保存。不需要手写 OpenCV 画图。

**预计时间：** 15 分钟。

**设计说明：**
- `results[0].plot()` 返回 numpy 数组（BGR 格式），已画好框
- 用 `cv2.imwrite()` 保存（ultralytics 依赖已包含 opencv）
- 也可以用 `model(image, save=True)` 自动保存到 `runs/detect/predict/`，但手动保存理解更深

**步骤：**
1. 在 `src/infer_image.py` 末尾加两行：
   ```python
   import cv2, os
   annotated = results[0].plot()  # numpy array, BGR
   os.makedirs("data/outputs", exist_ok=True)
   cv2.imwrite("data/outputs/annotated.jpg", annotated)
   print("Saved to data/outputs/annotated.jpg")
   ```
2. 运行脚本
3. 打开 `data/outputs/annotated.jpg` — 确认框和标签正确

**成功标准（验证通过才进入下一步）：**
- [ ] `data/outputs/annotated.jpg` 存在且能打开
- [ ] 检测框位置准确，标签含类别名和置信度

**CLAUDE.md 纪律检查：**
- 简洁：用 `results[0].plot()` 内置方法，不手写 OpenCV `rectangle` + `putText`

---

## Phase 2: 规模化推理

> 目标：从单张图片扩展到批量目录、视频文件、摄像头。视频/摄像头部分会用到 OpenCV。

---

### Task 2.1: 批量推理目录中的图片

**学习目标：** 一次处理多张图片。理解批处理和顺序处理的区别。汇总统计数据。

**预计时间：** 25 分钟。

**设计说明：**
- `model()` 接受路径列表，返回 `Results` 列表
- 创建一个小测试目录，放 3-5 张图片
- 打印每张图片的摘要和最终汇总

**步骤：**
1. 创建 `src/infer_batch.py`
2. 用 `glob.glob("data/test_images/*.jpg")` 收集图片路径
3. 建一个测试目录并放几张图（从网上下载或用自己的）
4. 运行 `results = model(image_paths)` 一次调用处理全部
5. 遍历结果，统计每张图的检测数
6. 打印："image1.jpg: 5 detections" ... "Total: 23 objects across 5 images"

**成功标准（验证通过才进入下一步）：**
- [ ] 脚本处理 3+ 张图片无报错
- [ ] 每张图有摘要，末尾有汇总统计
- [ ] 文件不超过 80 行

**CLAUDE.md 纪律检查：**
- 简洁：一个函数扫目录，一个函数打印汇总

---

### Task 2.2: 视频文件推理

**学习目标：** 用 OpenCV 逐帧读取视频，YOLO 推理每帧，写入带检测框的输出视频。这是第一次直接使用 OpenCV。

**预计时间：** 40 分钟。

**设计说明：**
- OpenCV `VideoCapture` 读帧，`VideoWriter` 写帧
- YOLO 每帧单独推理
- 编码：Windows 上 `.mp4` 用 `mp4v`
- FPS：从 `cap.get(cv2.CAP_PROP_FPS)` 读取，拿不到默认 30
- GPU 推理每帧 ~10-30ms，视频处理会很快
- 没有视频文件的话去 Pexels/Pixabay 下载一个短的

**步骤：**
1. 创建 `src/infer_video.py`
2. 打开输入视频：`cv2.VideoCapture(input_path)`
3. 读取 FPS、帧尺寸
4. 创建 `cv2.VideoWriter` 用于输出
5. 循环：`while True: ret, frame = cap.read(); if not ret: break`
6. 每帧推理：`results = model(frame, verbose=False)`，然后 `annotated = results[0].plot()`
7. 写入输出视频
8. 每 N 帧打印进度（如 "Frame 100/500"）
9. 释放资源，打印总耗时和平均 FPS

**成功标准（验证通过才进入下一步）：**
- [ ] 脚本在视频文件上运行无报错
- [ ] 输出视频可播放，检测框可见
- [ ] 控制台显示进度和最终 FPS
- [ ] 文件不超过 150 行

**CLAUDE.md 纪律检查：**
- 不过早优化：单帧推理即可，不要加跳帧、多线程、批处理

---

### Task 2.3: 摄像头实时推理

**学习目标：** YOLO 应用于实时摄像画面。体验 GPU 推理的实时性能。

**预计时间：** 15 分钟。

**设计说明：**
- 把 `cv2.VideoCapture(文件路径)` 换成 `cv2.VideoCapture(0)`（默认摄像头）
- 用 `cv2.imshow()` 显示带检测框的画面
- 按 'q' 退出
- GPU 下 YOLOv8n 可达 30+ FPS，画面流畅

**步骤：**
1. 在 `src/infer_video.py` 基础上创建 `src/infer_webcam.py`（~40 行）
2. `cap = cv2.VideoCapture(0)`
3. 循环：读帧 → 推理 → `cv2.imshow('YOLO', annotated)`
4. `if cv2.waitKey(1) & 0xFF == ord('q'): break`
5. 释放摄像头并关闭窗口

**成功标准（验证通过才进入下一步）：**
- [ ] 摄像头窗口打开，实时显示检测框
- [ ] FPS 流畅（15+ FPS 即可，30+ 更好）
- [ ] 按 'q' 干净退出

**CLAUDE.md 纪律检查：**
- 简洁：一个循环，一个 `imshow`。不画 FPS 计数器，不加录制功能

---

## Phase 3: 理解 YOLO 数据格式

> 目标：学会 YOLO 数据集在磁盘上的组织方式。这是"用预训练模型"到"训练自己模型"的桥梁。

---

### Task 3.1: 下载并探索 COCO128 数据集

**学习目标：** 理解 YOLO 目录约定：`images/` 和 `labels/` 目录并行，每个图片对应一个 .txt 标签文件。读懂 `dataset.yaml`。

**预计时间：** 25 分钟。

**设计说明：**
- COCO128 是 COCO 的 128 张图片子集，已是 YOLO 格式，是 ultralytics 训练的默认"hello world"数据集
- 下载地址：`https://ultralytics.com/assets/coco128.zip`（6.8 MB）
- 手动下载探索，而不用 ultralytics 内置下载器 — 这样才能理解目录布局
- 每个标签 .txt 文件每行一个目标：`class_id cx cy w h`（全部归一化到 [0,1]）

**步骤：**
1. 创建 `scripts/download_coco128.py`（~20 行）：
   ```python
   import urllib.request, zipfile, os
   url = "https://ultralytics.com/assets/coco128.zip"
   dest = "data/coco128.zip"
   urllib.request.urlretrieve(url, dest)
   with zipfile.ZipFile(dest, 'r') as z: z.extractall("data/")
   os.remove(dest)
   print("Extracted to data/coco128/")
   ```
2. 运行脚本
3. 手动探索：
   - 打开 `data/coco128/dataset.yaml` — 看懂 `path`、`train`、`val`、`names` 字段
   - 打开几个 `data/coco128/labels/train2017/*.txt` — 确认格式 `class_id cx cy w h`
   - 对比一个标签文件和它对应的图片
4. 思考：为什么坐标要归一化到 [0,1]？

**成功标准（验证通过才进入下一步）：**
- [ ] `data/coco128/` 存在，含 `images/` 和 `labels/` 子目录
- [ ] 能用自己话解释：标签文件里 5 个数字代表什么，为什么归一化
- [ ] 理解 `dataset.yaml` 的作用

**CLAUDE.md 纪律检查：**
- 目标驱动：目标是理解，不是写代码。下载脚本是临时的

---

### Task 3.2: 手写标注可视化器

**学习目标：** 读取 YOLO 标签文件，把归一化坐标转回像素坐标，画到对应图片上。这巩固你对数据格式的理解。这里需要少量 OpenCV。

**预计时间：** 40 分钟。

**设计说明：**
- 对 `data/coco128/images/train2017/` 中每张图片，找 `data/coco128/labels/train2017/` 中同名 .txt
- 解析每行：`class_id cx cy w h`
- 归一化转像素：`x_center = cx * img_width`，同理 y、w、h
- 中心+尺寸转对角：`x1 = x_center - w/2`，`y1 = y_center - h/2`
- 用 OpenCV 画框：`cv2.rectangle()` 和 `cv2.putText()`
- 用 `dataset.yaml` 映射 class_id → 类别名

**步骤：**
1. 创建 `src/visualize_data.py`
2. 用 `yaml.safe_load()` 读取 `dataset.yaml`
3. 对前 5 张训练图：
   a. 用 `cv2.imread()` 加载图片
   b. 读取对应标签文件
   c. 解析每行，转换坐标，画矩形
   d. 保存到 `data/outputs/coco128_viz/`
4. 打开保存的图片，和 Task 1.3 的推理结果对比

**成功标准（验证通过才进入下一步）：**
- [ ] 5 张标注图保存成功，标注框正确
- [ ] 框位置准确（和图片中物体对齐）
- [ ] 类别名（来自 dataset.yaml）显示在框上
- [ ] 文件不超过 120 行

**CLAUDE.md 纪律检查：**
- 简洁：一个函数读标签，一个函数画框，一个函数遍历图片

---

### Task 3.3: 写一个数据集验证器

**学习目标：** 程序化验证 YOLO 数据集是否合法。在训练前捕获常见错误。

**预计时间：** 25 分钟。

**设计说明：**
- 检查：孤儿标签（无对应图）、孤儿图片（无对应标签）、空标签文件、损坏图片、越界坐标
- 打印清晰汇总报告
- 这个脚本将在 Phase 4 自定义数据集上复用

**步骤：**
1. 创建 `scripts/validate_dataset.py`
2. 接受 `dataset_dir` 参数（默认 `data/coco128`）
3. 读取 `dataset.yaml` 找到 `train` 和 `val` 路径
4. 对每个 split：
   - 检查每张图有对应标签文件
   - 检查每个标签文件有对应图片
   - 验证每行标签恰好 5 个值
   - 验证所有值在合法范围（class_id >= 0，坐标在 [0,1]）
   - 用 PIL 尝试打开每张图检查损坏
5. 打印报告：
   ```
   === COCO128 Validation Report ===
   Images: 128 | Labels: 128 | Total boxes: 928
   Orphan images: 0 | Orphan labels: 0
   Empty labels: 0 | Invalid coords: 0 | Corrupted: 0
   === PASSED ===
   ```

**成功标准（验证通过才进入下一步）：**
- [ ] 验证器在 COCO128 上运行并报告 PASSED
- [ ] 理解每项检查防御什么问题
- [ ] 脚本可复用（参数化目录，非硬编码）

**CLAUDE.md 纪律检查：**
- 简洁：打印到控制台，不写日志文件，不需要配置文件

---

## Phase 4: 构建自己的数据集

> 目标：从零创建一个自定义数据集。这是最耗时的阶段（大量手工操作）。选择一个单类别以降低工作量。

---

### Task 4.1: 选定类别并收集图片

**学习目标：** 理解数据集大小、多样性、质量对模型性能的影响。设定合理期望。

**预计时间：** 60-90 分钟（主要是手工操作）。

**设计说明：**
- 选**一个类别**。例如："person"、"cat"、"car"、"bottle"、"chair"、"traffic light"
  挑一个你能轻松找到 50+ 张图片的类别
- 图片来源：自己拍的照片、Google Images、Pexels、Unsplash、COCO 子集提取
- 最低要求：50 张训练、10 张验证。推荐：100+ 训练、20+ 验证
- 图片用 JPEG、建议 640×640 左右（不强制，YOLO 训练时会自动 resize）
- 目录结构：
  ```
  data/custom/
  ├── dataset.yaml
  ├── images/
  │   ├── train/
  │   └── val/
  └── labels/
      ├── train/
      └── val/
  ```

**步骤：**
1. 决定你的单类别，写下来
2. 收集 50-100 张训练图、10-20 张验证图
3. 训练图放入 `data/custom/images/train/`
4. 验证图放入 `data/custom/images/val/`
5. 可选：统一命名为 `img_001.jpg`、`img_002.jpg` 等

**成功标准（验证通过才进入下一步）：**
- [ ] `data/custom/images/train/` 有 50+ 张 JPEG
- [ ] `data/custom/images/val/` 有 10+ 张 JPEG
- [ ] 所有图片能正常打开

**CLAUDE.md 纪律检查：**
- 这一步几乎不涉及代码。交付物是一个图片目录。不要过度自动化收集过程

---

### Task 4.2: 创建 YOLO 格式标注

**学习目标：** 使用标注工具标注目标。理解标注的实践挑战（遮挡、边界情况、一致性）。

**预计时间：** 60-120 分钟（手工标注）。

**设计说明：**
- 推荐免费工具（选一个）：
  - **[makesense.ai](https://makesense.ai)** — 浏览器操作，无需安装，直接导出 YOLO 格式（推荐）
  - **LabelImg** — 桌面应用，`pip install labelImg`
  - **Roboflow** — 网页平台，免费额度，可导出 YOLO 格式
- 标注要点：
  1. 对每个目标实例画紧贴的边界框
  2. 分配类别标签（本项目只有一个类别，所以 class_id 始终为 0）
  3. 不确定的目标（严重遮挡、极小）跳过
  4. 导出 YOLO 格式：每张图一个 .txt 文件
- 每行格式：`0 cx cy w h`

**步骤：**
1. 打开 makesense.ai
2. 导入训练图片
3. 创建一个标签（你的单类别）
4. 标注所有训练图片 — 耐心但不需要像素级完美
5. 导出 YOLO 格式
6. 将 .txt 文件放入 `data/custom/labels/train/`
7. 对验证图片重复上述流程，放入 `data/custom/labels/val/`
8. 运行 `uv run python scripts/validate_dataset.py data/custom/` 修复所有报错

**成功标准（验证通过才进入下一步）：**
- [ ] 每张训练图有对应 .txt 标签文件
- [ ] 每张验证图有对应 .txt 标签文件
- [ ] `scripts/validate_dataset.py` 在自定义数据集上报告 PASSED
- [ ] 手动抽查 5 张图：打开图片，确认标注框正确

**CLAUDE.md 纪律检查：**
- 目标驱动：成功 = 验证报告干净。不要反复修改"够好"的标注

---

### Task 4.3: 创建 dataset.yaml 并验证

**学习目标：** 写正确的 `dataset.yaml`。用 ultralytics 内置检查器确认训练就绪。

**预计时间：** 10 分钟。

**设计说明：**
- `dataset.yaml` 格式：
  ```yaml
  path: data/custom
  train: images/train
  val: images/val
  nc: 1
  names: ['your_class_name']
  ```
- 路径用正斜杠（Windows 也兼容）
- `nc` 必须等于 `len(names)`

**步骤：**
1. 创建 `data/custom/dataset.yaml`
2. 用 ultralytics 检查：
   ```bash
   uv run python -c "from ultralytics.data.utils import check_det_dataset; check_det_dataset('data/custom/dataset.yaml')"
   ```
3. 如果打印数据集统计无报错，数据集就绪

**成功标准（验证通过才进入下一步）：**
- [ ] `dataset.yaml` 存在，`nc` 和 `names` 正确
- [ ] `check_det_dataset()` 无报错
- [ ] 打印的统计显示你的图片数和类别数

**CLAUDE.md 纪律检查：**
- 精确：只创建一个 YAML 文件，不写配置生成器

---

## Phase 5: GPU 训练

> 目标：在自定义数据集上训练 YOLO。先从 nano 模型开始跑通流程，再尝试 small/medium 提升精度。

---

### Task 5.1: GPU 训练第一轮

**学习目标：** 跑通完整训练循环。理解 epochs、batch size、learning rate。看懂训练日志。得到一个真正工作的模型。

**预计时间：** 40 分钟（10-20 分钟实际训练）。

**设计说明：**
- 用 YOLOv8n 作为起始权重
- 训练参数：`epochs=50`、`imgsz=640`、`batch=16`、`device=0`（GPU）
- 8-12GB 显存足够 YOLOv8n + batch=16，训练很快
- 输出在 `runs/detect/train/`，每次运行新建编号目录
- 50-100 张图 + 50 epochs，预计 10-20 分钟

**步骤：**
1. 创建 `src/train.py`（~40 行）：
   ```python
   from ultralytics import YOLO

   model = YOLO('models/yolov8n.pt')
   results = model.train(
       data='data/custom/dataset.yaml',
       epochs=50,
       imgsz=640,
       batch=16,
       device=0,
       name='custom_train_v1',
       exist_ok=True,
   )
   ```
2. 运行：`uv run python src/train.py`
3. 观察每个 epoch 输出的指标：
   - `box_loss`（train + val）：边界框预测有多差？
   - `cls_loss`（train + val）：类别预测有多差？
   - `mAP50`、`mAP50-95`：核心评估指标（越高越好）
4. 训练完成后打开 `runs/detect/custom_train_v1/`：
   - `results.png`：loss 和 metric 曲线
   - `confusion_matrix.png`：预测 vs 真实
   - `val_batch0_pred.jpg`：验证集预测可视化
5. 用训练好的模型推理测试图：
   ```bash
   uv run python -c "from ultralytics import YOLO; m = YOLO('runs/detect/custom_train_v1/weights/best.pt'); print(m('path/to/test.jpg')[0].boxes)"
   ```

**成功标准（验证通过才进入下一步）：**
- [ ] 训练无错误完成
- [ ] `runs/detect/custom_train_v1/` 包含 `weights/best.pt` 和 `weights/last.pt`
- [ ] `results.png` 和验证图片已生成
- [ ] 训练好的模型在测试图上能检测到你标注的目标
- [ ] 能用一句话解释 `mAP50` 和 `box_loss` 代表什么

**CLAUDE.md 纪律检查：**
- 目标驱动：目标是看到完整训练循环。不要调超参数，不要试图提升精度。那是下一个任务

---

### Task 5.2: 超参数实验与模型对比

**学习目标：** 实验不同的 epochs 数量、模型大小。用 `model.val()` 系统化对比结果。做出信息充分的模型选择。

**预计时间：** 45 分钟。

**设计说明：**
- 尝试 YOLOv8s（small，~22 MB），比 nano（~6 MB）更大但更准
- 增加 epochs 到 100，观察 mAP 是否继续提升还是过拟合
- 调整 batch size 利用显存（8-12GB 跑 v8s 可用 batch=16 或 24）
- 对比表：模型大小、mAP50、mAP50-95、推理速度

**步骤：**
1. 扩展 `src/train.py` 支持参数配置，或写独立脚本
2. 至少跑 3 组实验，例如：

   | 实验 | 模型 | epochs | batch | 预期 |
   |------|------|--------|-------|------|
   | v1 | yolov8n | 50 | 16 | baseline |
   | v2 | yolov8n | 100 | 16 | 更高 mAP |
   | v3 | yolov8s | 50 | 16 | 更大模型对比 |

3. 每组训练后用 `model.val()` 获取数值指标
4. 在几张固定测试图上做可视化对比
5. 选一个最佳模型用于部署，写下理由（一句话）

**成功标准（验证通过才进入下一步）：**
- [ ] 至少完成 3 组训练实验
- [ ] 每组都有 mAP50、mAP50-95 数值
- [ ] 选定部署模型，有书面理由
- [ ] 最佳模型权重复制到 `models/custom_best.pt`

**CLAUDE.md 纪律检查：**
- 目标驱动：产出是一个决定（选哪个模型）。不要建复杂评估框架

---

### Task 5.3: （可选）Google Colab 训练

**学习目标：** 了解云端 GPU 训练流程。备用于需要更大显存或更长训练时间时。

**预计时间：** 45 分钟。

**设计说明：**
- Colab 免费 T4 GPU（~16 GB 显存），适合跑更大的 batch 或更大的模型
- 当你的数据集 >500 张或想尝试 YOLOv8m/l 时有用
- 这是可选项 — 如果本地 GPU 已满足需求可以跳过

**步骤：**
1. 创建 `notebooks/train_on_colab.ipynb`
2. Cell 1 安装：`!pip install ultralytics`
3. Cell 2 上传数据集：zip `data/custom/`，上传到 Colab，`!unzip custom_dataset.zip`
4. Cell 3 验证 GPU：`!nvidia-smi`
5. Cell 4 训练：`model.train(data='...', epochs=100, imgsz=640, batch=32, device=0)`
6. Cell 5 展示结果
7. Cell 6 下载模型：`files.download('runs/detect/train/weights/best.pt')`

**成功标准：**
- [ ] Colab 训练完成
- [ ] 模型下载到本地 `models/`
- [ ] 比较 Colab 模型和本地模型的 mAP

**CLAUDE.md 纪律检查：**
- 简洁：Colab notebook 按 cell 执行，不是生产脚本

---

## Phase 6: 模型导出与优化

> 目标：将训练好的 PyTorch 模型转换为 ONNX 格式，用 ONNX Runtime 推理。对比性能。

---

### Task 6.1: 导出 ONNX 模型

**学习目标：** 使用 ultralytics 内置的 export。理解 ONNX 是什么以及为什么部署需要它。

**预计时间：** 15 分钟。

**设计说明：**
- ultralytics 一行导出：`model.export(format='onnx')`
- ONNX 文件会保存在 .pt 同目录，同名 .onnx
- ONNX 好处：跨平台、不需要 Python 即可推理（但本学习项目保持 Python）

**步骤：**
1. `uv add onnx`
2. 创建 `src/export_onnx.py`（~15 行）：
   ```python
   from ultralytics import YOLO
   model = YOLO('models/custom_best.pt')
   model.export(format='onnx', imgsz=640, simplify=True, opset=12)
   print("Exported to models/custom_best.onnx")
   ```
3. 运行脚本
4. 对比文件大小：`ls -lh models/custom_best.pt models/custom_best.onnx`

**成功标准（验证通过才进入下一步）：**
- [ ] `models/custom_best.onnx` 存在
- [ ] 导出一行 `export()` 调用完成
- [ ] 能用一句话解释 ONNX 解决的问题

**CLAUDE.md 纪律检查：**
- 简洁：一个 `export()` 调用。不手动配置 opset 和输入输出名

---

### Task 6.2: 用 ONNX Runtime 推理

**学习目标：** 不依赖 PyTorch 做推理。对比 ONNX Runtime 和 PyTorch 的速度。

**预计时间：** 25 分钟。

**设计说明：**
- ultralytics 直接接受 .onnx 文件：`YOLO('model.onnx')` — 自动使用 ONNX Runtime
- 这是最简单的方式，一行代码改动
- 可选深入：直接用 `onnxruntime.InferenceSession` 做低层推理（需要手动预处理/后处理）

**步骤：**
1. `uv add onnxruntime`
2. 通过 ultralytics 用 ONNX 模型：
   ```python
   from ultralytics import YOLO
   model = YOLO('models/custom_best.onnx')  # 自动用 ONNX Runtime
   results = model('path/to/test.jpg')
   print(results[0].boxes)
   ```
3. 简单的速度对比：写一个循环计时，PyTorch vs ONNX Runtime
4. （可选）用 `onnxruntime.InferenceSession` 直接推理 — 学习低层 API

**成功标准（验证通过才进入下一步）：**
- [ ] ONNX 模型通过 ultralytics 推理成功
- [ ] 检测结果和 PyTorch 模型一致（或几乎一致）
- [ ] 记录了 ONNX Runtime vs PyTorch 的速度差异

**CLAUDE.md 纪律检查：**
- 简洁：先用 ultralytics 封装方式。只有想学习低层 API 时才写直接推理代码

---

### Task 6.3: （可选）尝试 OpenVINO 导出

**学习目标：** 了解 ONNX 之外的导出格式。OpenVINO 是 Intel 的推理引擎，可能在 Intel CPU 上更快。

**预计时间：** 15 分钟。

**设计说明：**
- OpenVINO 专为 Intel 硬件优化（你的 i7-13620H 也受益）
- `model.export(format='openvino')` 一行搞定
- 这是调研性质的任务 — 导出、记文件大小、继续前进

**步骤：**
1. `uv add openvino`
2. `model.export(format='openvino')`
3. 对比三种格式的文件大小和推理速度

**成功标准：**
- [ ] 至少一种额外格式导出成功
- [ ] 心中有数：ONNX = 跨平台，OpenVINO = Intel 优化

**CLAUDE.md 纪律检查：**
- 这是可选项。如果更想推进 Phase 7 部署，可以跳过

---

## Phase 7: 部署

> 目标：将模型封装为 FastAPI Web 服务，接收图片上传，返回 JSON 检测结果。这是"生产"交付物。

---

### Task 7.1: 构建 FastAPI 推理服务

**学习目标：** 创建目标检测的 REST API。理解 ML 服务的请求/响应模式。

**预计时间：** 60 分钟。

**设计说明：**
- FastAPI 而非 Flask：内置文件上传（UploadFile）、自动生成 OpenAPI 文档（/docs）、async 支持
- 模型**启动时加载一次**（全局变量），不是每次请求重新加载
- 两个端点：`POST /detect` 上传图片返回 JSON，`GET /health` 健康检查
- 模型可以是 .pt 或 .onnx（建议 ONNX 用于部署）
- 控制在 200 行以内

**步骤：**
1. `uv add fastapi uvicorn python-multipart`
2. 创建 `src/server.py`：
   ```python
   from fastapi import FastAPI, UploadFile, File
   from ultralytics import YOLO
   import cv2, numpy as np

   app = FastAPI(title="YOLO Detection API")
   model = None

   @app.on_event("startup")
   async def load_model():
       global model
       model = YOLO("models/custom_best.pt")  # 或 .onnx

   @app.get("/health")
   async def health():
       return {"status": "ok", "model_loaded": model is not None}

   @app.post("/detect")
   async def detect(file: UploadFile = File(...)):
       contents = await file.read()
       nparr = np.frombuffer(contents, np.uint8)
       img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
       results = model(img, verbose=False)[0]
       detections = []
       for box in results.boxes:
           detections.append({
               "class": results.names[int(box.cls.item())],
               "confidence": round(float(box.conf.item()), 4),
               "bbox": [round(x, 1) for x in box.xyxy[0].tolist()],
           })
       return {
           "filename": file.filename,
           "detections": detections,
           "count": len(detections),
       }
   ```
3. 启动：`uv run uvicorn src.server:app --reload --host 0.0.0.0 --port 8000`
4. 浏览器打开 `http://localhost:8000/docs` — 查看自动生成的 Swagger UI

**成功标准（验证通过才进入下一步）：**
- [ ] 服务无错误启动
- [ ] `GET /health` 返回 `{"status": "ok", "model_loaded": true}`
- [ ] `/docs` 可访问，显示两个端点
- [ ] 文件不超过 200 行

**CLAUDE.md 纪律检查：**
- 简洁：两个端点。不加认证、限流、数据库 — 这些是生产关注点，超出学习范围

---

### Task 7.2: 用 curl 和 Python 客户端测试服务

**学习目标：** 从客户端角度验证 API。理解 HTTP multipart form 上传。

**预计时间：** 25 分钟。

**步骤：**
1. curl 测试：
   ```bash
   curl -X POST -F "file=@data/coco128/images/train2017/000000000009.jpg" http://localhost:8000/detect
   ```
   验证返回 JSON 含 class、confidence、bbox
2. 创建 `tests/test_server.py`：
   ```python
   import requests

   BASE = "http://localhost:8000"

   def test_health():
       r = requests.get(f"{BASE}/health")
       assert r.status_code == 200
       assert r.json()["status"] == "ok"

   def test_detect():
       with open("data/coco128/images/train2017/000000000009.jpg", "rb") as f:
           r = requests.post(f"{BASE}/detect", files={"file": f})
       assert r.status_code == 200
       data = r.json()
       assert "detections" in data
       assert "count" in data
       for d in data["detections"]:
           assert all(k in d for k in ["class", "confidence", "bbox"])
           assert 0 <= d["confidence"] <= 1
       print(f"Found {data['count']} detections")

   if __name__ == "__main__":
       test_health()
       test_detect()
       print("All tests passed!")
   ```
3. 运行：`uv run python tests/test_server.py`

**成功标准（验证通过才进入下一步）：**
- [ ] curl 返回有效 JSON
- [ ] Python 测试脚本全部通过
- [ ] 响应含 `filename`、`detections`（list）、`count`

**CLAUDE.md 纪律检查：**
- 目标驱动：测试验证的是契约（响应结构、类型），不是模型精度

---

### Task 7.3: 导出 requirements.txt 并写部署文档

**学习目标：** 从 `uv`（开发）过渡到 `pip`（部署）。理解 lock 文件和 requirements 文件的区别。

**预计时间：** 15 分钟。

**设计说明：**
- `uv export --format requirements-txt --output-file requirements.txt` 生成 pip 兼容文件
- 这个文件锁定精确版本（类似 `uv.lock`），但可用于任何 pip 环境
- 部署命令：`pip install -r requirements.txt && uvicorn src.server:app --host 0.0.0.0 --port 8000`
- 注意：部署时需确保 torch 是 CUDA 版本（`requirements.txt` 中 CUDA 索引需手动处理）

**步骤：**
1. 运行：`uv export --format requirements-txt --output-file requirements.txt`
2. 检查输出：前 20 行应显示精确版本（不是范围）
3. 如果 `requirements.txt` 中的 torch 是 CPU 版，手动替换为 CUDA 版链接或加注释说明
4. （可选）在干净环境测试：
   ```bash
   python -m venv /tmp/test_deploy
   # 激活后 pip install -r requirements.txt
   # python -c "from ultralytics import YOLO; print('OK')"
   ```
5. 在 `docs/` 中创建 `deploy.md`（控制在 30 行以内）：
   ```markdown
   # Deployment

   1. `pip install -r requirements.txt`
   2. 确保 torch 是 CUDA 版本（需 GPU 时）
   3. 放置模型到 `models/custom_best.pt`
   4. `uvicorn src.server:app --host 0.0.0.0 --port 8000`
   5. API 文档: http://localhost:8000/docs
   ```

**成功标准（验证通过才进入下一步）：**
- [ ] `requirements.txt` 存在，含精确版本号
- [ ] `docs/deploy.md` 存在，部署指令清晰
- [ ] 可以用 `requirements.txt` + pip 启动服务（不需要 uv）

**CLAUDE.md 纪律检查：**
- 精确：只加一个 30 行的部署文档。不创建详尽的运维手册

---

## 附录 A: 项目文件清单

```
Yolo/
├── CLAUDE.md                    # 项目纪律 + 部署说明
├── .gitignore                   # 忽略 data/*, models/*, runs/, .venv/
├── .python-version              # 固定 Python 3.11
├── pyproject.toml               # uv 项目配置 + 依赖
├── requirements.txt             # pip 部署依赖（Task 7.3）
├── uv.lock                      # uv 锁定文件（自动生成）
│
├── docs/
│   ├── task_todo.md             # 本文件 — 任务指南
│   └── deploy.md                # 部署文档（Task 7.3）
│
├── notebooks/
│   └── train_on_colab.ipynb     # Colab 训练笔记本（Task 5.3 可选）
│
├── src/
│   ├── __init__.py
│   ├── infer_image.py           # Task 1.1-1.3: 单图推理
│   ├── infer_batch.py           # Task 2.1: 批量推理
│   ├── infer_video.py           # Task 2.2: 视频推理
│   ├── infer_webcam.py          # Task 2.3: 摄像头推理
│   ├── visualize_data.py        # Task 3.2: 标注可视化
│   ├── train.py                 # Task 5.1-5.2: GPU 训练
│   ├── export_onnx.py           # Task 6.1: ONNX 导出
│   └── server.py                # Task 7.1: FastAPI 推理服务
│
├── scripts/
│   ├── download_coco128.py      # Task 3.1: 下载 COCO128
│   └── validate_dataset.py      # Task 3.3: 数据集验证器
│
├── tests/
│   ├── __init__.py
│   └── test_server.py           # Task 7.2: API 集成测试
│
├── data/
│   ├── README.md
│   ├── coco128/                 # Task 3.1: 下载的数据集
│   ├── custom/                  # Task 4.1-4.3: 你的数据集
│   │   ├── dataset.yaml
│   │   ├── images/
│   │   │   ├── train/
│   │   │   └── val/
│   │   └── labels/
│   │       ├── train/
│   │       └── val/
│   ├── test_images/             # Task 2.1: 测试图片
│   └── outputs/                 # 所有输出图片
│
└── models/
    ├── README.md
    ├── yolov8n.pt               # Task 1.1: COCO 预训练 nano 模型
    └── custom_best.pt           # Task 5.2: 你训练的最佳模型
```

## 附录 B: 标注工具速查

| 工具 | 类型 | 费用 | 说明 |
|------|------|------|------|
| [makesense.ai](https://makesense.ai) | Web | 免费 | 最简单，直接导出 YOLO 格式 |
| LabelImg | 桌面 | 免费 | 经典工具，`pip install labelImg` |
| Roboflow | Web | 免费额度 | 数据集管理 + 标注一站式 |
| CVAT | Web/桌面 | 免费（自部署）| 专业级，更复杂 |

## 附录 C: Google Colab 速查

1. 访问 https://colab.research.google.com/
2. File → New notebook
3. Runtime → Change runtime type → T4 GPU
4. 上传 `custom_dataset.zip` 到文件浏览器（左侧栏）
5. 解压：`!unzip custom_dataset.zip -d /content/data/`
6. 安装：`!pip install ultralytics`
7. 训练：
   ```python
   from ultralytics import YOLO
   model = YOLO('yolov8n.pt')
   model.train(data='/content/data/dataset.yaml', epochs=100, imgsz=640, batch=32)
   ```
8. 下载模型：
   ```python
   from google.colab import files
   files.download('runs/detect/train/weights/best.pt')
   ```

## 附录 D: 排错指南

| 问题 | 可能原因 | 解决 |
|------|----------|------|
| `ImportError: cannot import 'YOLO'` | ultralytics 未安装 | `uv add ultralytics` |
| `CUDA out of memory` | batch 太大 | 减 `batch` 到 8 或 4 |
| `CUDA not available` | torch 是 CPU 版本 | 重新安装 CUDA 版 torch |
| Training loss is NaN | 学习率太高 | 默认 lr 即可，先检查数据集 |
| `cv2.imshow()` 卡住 | GUI 后端问题 | 改用 `cv2.imwrite()` |
| ONNX export 失败 | opset 不匹配 | 尝试 `opset=11` |
| Server 返回 500 | 模型路径错误 | 检查 `models/custom_best.pt` 是否存在 |
| `check_det_dataset()` 报路径错 | dataset.yaml 路径格式 | Windows 也用正斜杠 `/` |

## 附录 E: 术语表

| 术语 | 定义 |
|------|------|
| **mAP** | mean Average Precision — YOLO 标准评估指标（越高越好）|
| **mAP50** | IoU 阈值 0.5 的 mAP（宽松）|
| **mAP50-95** | IoU 0.5 到 0.95 步长平均的 mAP（严格）|
| **IoU** | Intersection over Union — 两个框的重叠程度 |
| **NMS** | Non-Maximum Suppression — 去重检测框 |
| **YOLO 格式** | 归一化标注：`class_id cx cy w h`（均在 [0,1]）|
| **ONNX** | Open Neural Network Exchange — 跨平台模型格式 |
| **预训练** | 在大数据集（COCO）上训练好的模型，作为起点 |
| **微调** | 在预训练模型上用新数据继续训练 |
| **Epoch** | 完整遍历一次训练数据集 |
| **Batch size** | 每次更新权重前处理的图片数 |

---

## 学习路径总结

```
Phase 0: 环境    (3 tasks, ~1h)    → uv + GPU torch 就绪
Phase 1: 首次推理 (3 tasks, ~1h)    → 看到检测框
Phase 2: 规模化   (3 tasks, ~1.5h)  → 批量 + 视频 + 摄像头
Phase 3: 数据格式 (3 tasks, ~2h)    → 理解 YOLO 标注
Phase 4: 自定义数据 (3 tasks, ~4h)  → 构建自己的数据集
Phase 5: GPU 训练  (3 tasks, ~2h)   → 本地 GPU 训练出可用模型
Phase 6: 导出     (3 tasks, ~1h)    → ONNX + 格式探索
Phase 7: 部署     (3 tasks, ~2h)    → FastAPI 服务上线
                         ----------
                         ~14.5 小时活跃工作
```

开始吧。从 Phase 0 Task 0.1 做起，一个任务一个勾，不跳步。
