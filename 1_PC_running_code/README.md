# PC OpenCV Experiment — PnP 位姿解算 & 透视校正图形测量

本目录是论文《基于透视校正的平面目标图形识别与尺寸测量方法研究》的 PC 端实验代码。

与 Jetson/MaixCam 竞赛代码保持独立。本 PC 版支持：PnP 位姿解算、距离修正、
透视校正、内部图形（三角形/正方形/圆形）识别与测量，以及 iPhone 等相机标定接入。

## 目录结构

```
1_PC_running_code/
├── data/
│   ├── input/          ← 放置待处理的图片
│   ├── output/         ← 结果输出目录（自动创建）
│   └── calib/          ← 相机标定文件（.npz / .json）
├── src/
│   └── pc_opencv_experiment.py   ← 主程序
├── requirements.txt
└── README.md
```

## 环境配置

```powershell
cd C:\Git_Program\25-NUEDC-C-lite
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r 1_PC_running_code\requirements.txt
```

如果需要 resize iPhone 高分辨率照片，还需安装 Pillow：

```powershell
pip install Pillow
```

## 使用示例

### 1️⃣ 处理单张图片

```powershell
python 1_PC_running_code\src\pc_opencv_experiment.py --input image.jpg --output 1_PC_running_code\data\output
```

### 2️⃣ 批量处理文件夹

```powershell
python 1_PC_running_code\src\pc_opencv_experiment.py --input 1_PC_running_code\data\input --output 1_PC_running_code\data\output
```

### 3️⃣ 使用相机标定文件（iPhone 等）

```powershell
python 1_PC_running_code\src\pc_opencv_experiment.py --input data\input --output data\output --calib-file data\calib\iphone_calib.npz
```

### 4️⃣ 手动指定相机内参

```powershell
python 1_PC_running_code\src\pc_opencv_experiment.py --input data\input --output data\output --camera-matrix 1158.86 1183.95 357.72 200.27
```

### 5️⃣ 处理 iPhone 高分辨率照片（自动缩放）

```powershell
python 1_PC_running_code\src\pc_opencv_experiment.py --input data\input --output data\output --resize 1920 1440 --camera-matrix 1500 1500 960 720
```

### 6️⃣ 从 USB 相机采集

```powershell
python 1_PC_running_code\src\pc_opencv_experiment.py --camera 0 --capture data\input\usb_capture.jpg
```

### 7️⃣ 带距离修正标定

```powershell
python 1_PC_running_code\src\pc_opencv_experiment.py --input data\input --output data\output --calib-csv data\calib\distance_calib.csv
```

距离标定 CSV 格式（`distance_calib.csv`）：

```csv
real_mm,raw_z_mm
1000,1023
1500,1518
2000,2035
```

## 输出文件

| 文件 | 说明 |
|---|---|
| `pose_results.csv` | 每帧的 PnP 位姿结果（X/Y/Z坐标、偏航/俯仰/滚转角、修正距离） |
| `measurements.csv` | 内部图形检测结果（形状、像素尺寸、实际尺寸 mm²） |
| `*_raw.png` | 原始图像 + 外框角点标注 + 3D坐标轴 + 位姿文本 |
| `*_binary.png` | 二值化图像（用于外框检测） |
| `*_warped.png` | 透视校正后的目标板 + 内部图形标注 |

## 关键技术参数

| 参数 | 值 |
|---|---|
| A4 外框尺寸 (PC) | 168.1 × 255.1 mm |
| A4 外框尺寸 (Jetson) | 170.5 × 260.5 mm |
| 透视校正像素宽度 | 800 px |
| PnP 方法 | SOLVEPNP_IPPE |
| 距离测量范围 | 100–200 cm |

### 已知相机内参

**Jetson 相机 (640×480):**
- fx=585.57, fy=598.96, cx=326.52, cy=226.96

**MaixCam 相机 (640×480):**
- fx=1158.86, fy=1183.95, cx=357.72, cy=200.27

**iPhone 相机:** 需要标定，请使用 `--calib-file` 或 `--camera-matrix` 指定
