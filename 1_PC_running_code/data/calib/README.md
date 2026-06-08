# iPhone 相机标定工作流程

## 你需要的材料
- [x] **棋盘格标定板** — 已生成: `chessboard.png`（A4 可打印，8×6 格子，25mm/格，200×150mm）
- [x] **硬纸板** — 贴棋盘格用，保持平坦
- [x] **iPhone** — 拍照用
- [ ] **卷尺** — 测量距离用（后面做距离修正时需要）

---

## Step 1: 打印并制作标定板

1. 用 A4 纸打印 `chessboard.png`（300 DPI，实际尺寸 200×150mm）
2. 贴在硬纸板上，确保表面平整无褶皱

---

## Step 2: 拍摄标定照片

用 iPhone 拍摄 15-25 张棋盘格照片，存到 `iphone_images/` 目录。

拍摄要点：

| 要求 | 说明 |
|---|---|
| **数量** | 15-25 张 |
| **棋盘格位置** | 覆盖画面左上/右上/中心/左下/右下 |
| **距离** | 30cm ~ 200cm |
| **倾斜角度** | 0°（正面）~ 45° |
| **光照** | 均匀，避免反光 |
| **清晰度** | 确保棋盘格清晰可见 |
| **格式** | 保持 iPhone 原始分辨率（如 4032×3024） |

建议拍摄方案（20 张）：

```
正面 (0°):  30cm, 50cm, 80cm, 100cm, 150cm, 200cm  → 6 张
左倾 30°:   50cm, 100cm, 150cm                       → 3 张
右倾 30°:   50cm, 100cm, 150cm                       → 3 张
上倾 30°:   100cm                                     → 1 张
下倾 30°:   100cm                                     → 1 张
四角倾斜:   各方向 100cm 各 1-2 张                    → 6 张
```

---

## Step 3: 运行标定

```powershell
Set-Location "C:\Git_Program\25-NUEDC-C-lite\1_PC_running_code"

# 标准标定（棋盘格 9×7 内角点，25mm）
& "C:\Users\29787\AppData\Local\Programs\Python\Python312\python.exe" tools\calibrate_camera.py `
    --input data/calib/iphone_images/ `
    --output data/calib/iphone_calib.npz `
    --show
```

加入 `--show` 参数可以逐张查看角点检测结果。

正常输出示例：
```
Calibration result:
  RMS re-projection error: 0.3542 pixels
  Camera matrix:
    [[2925.34, 0, 2013.56],
     [0, 2928.12, 1510.78],
     [0, 0, 1]]
```

**RMS 重投影误差 < 0.5 像素** 说明标定质量良好。

---

## Step 4: 用标定结果做实验

```powershell
# 拍摄 A4 目标板照片，存到 data/input/
# 然后：
& "C:\Users\29787\AppData\Local\Programs\Python\Python312\python.exe" src\pc_opencv_experiment.py `
    --input data/input/ `
    --output data/output/ `
    --calib-file data/calib/iphone_calib.npz `
    --resize 1920 1440
```

> **为什么 --resize?**
> iPhone 原始照片 4032×3024 太大，处理慢。
> 脚本会自动缩放到 1920×1440。
> 注意：`--camera-matrix` 中的内参也要对应缩放后的分辨率。
> 如果使用 `--calib-file`，脚本会自动匹配，无需手动换算。

---

## Step 5: 距离修正标定（可选，但推荐）

在多个已知距离拍摄 A4 目标板，记录 PnP 输出的 `D_raw` 和真实距离：

```csv
real_mm,raw_z_mm
1000,1023
1250,1280
1500,1535
1750,1790
2000,2045
```

保存到 `data/calib/distance_calib.csv`，然后：

```powershell
& "C:\Users\29787\AppData\Local\Programs\Python\Python312\python.exe" src\pc_opencv_experiment.py `
    --input data/input/ `
    --output data/output/ `
    --calib-file data/calib/iphone_calib.npz `
    --resize 1920 1440 `
    --calib-csv data/calib/distance_calib.csv
```

---

## 常见问题

**Q: 角点检测失败率很高？**
- 检查棋盘格是否清晰对焦
- 检查是否过度倾斜（>60°）
- 检查光照是否太暗/太亮

**Q: RMS 误差 > 1.0？**
- 可能某些图片质量差，去掉最差的几张
- 检查棋盘格是否变形（没贴平）

**Q: iPhone 照片和 resize 后的内参关系？**
- 如果原始照片 4032×3024，缩放至 1920×1440：
  - fx' = fx × 1920 / 4032
  - fy' = fy × 1920 / 4032
  - cx' = cx × 1920 / 4032
  - cy' = cy × 1440 / 3024
- 使用 `--calib-file` 时脚本自动处理
