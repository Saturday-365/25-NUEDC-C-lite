# 25-NUEDC-C-lite

2025 年全国大学生电子设计竞赛 **C 题**——基于单目视觉的目标物测量装置。

本仓库包含完整竞赛代码（Jetson / MaixCam）以及论文《基于透视校正的平面目标图形识别与尺寸测量方法研究》的 PC 端实验代码和排版工具。

---

## 项目结构

```
├── 0_SoftWare/
│   ├── jetson/          ← Jetson Orin NX (C++/OpenCV/OnnxRuntime)
│   └── maixcam/         ← MaixCam K210 (Python/maix)
├── 1_PC_running_code/   ← PC 端实验代码 + 标定工具 ★ (当前工作目录)
├── 2_HardWare/          ← 硬件资料
├── 3_Document/          ← 论文文档
└── ignore/              ← 资源图片与排版脚本
```

---

## 快速开始（PC 实验）

### 1. 环境配置

```powershell
cd C:\Git_Program\25-NUEDC-C-lite\1_PC_running_code
# Python 3.12 安装路径
& "C:\Users\29787\AppData\Local\Programs\Python\Python312\python.exe" -m pip install opencv-python numpy Pillow
```

### 2. 生成测试图片

```powershell
& "C:\Users\29787\AppData\Local\Programs\Python\Python312\python.exe" tools\generate_test_images.py --output data/input
```

### 3. 运行实验

```powershell
& "C:\Users\29787\AppData\Local\Programs\Python\Python312\python.exe" src\pc_opencv_experiment.py --input data/input --output data/output --camera-matrix 1500 1500 960 720
```

### 4. iPhone 相机标定（首次使用需做）

详见 `1_PC_running_code/data/calib/README.md`

---

## 关键技术参数

| 参数 | 值 |
|---|---|
| A4 外框尺寸 (PC) | 168.1 × 255.1 mm |
| A4 外框尺寸 (Jetson) | 170.5 × 260.5 mm |
| 透视校正像素宽度 | 800 px |
| PnP 方法 | `SOLVEPNP_IPPE` |
| 距离测量范围 | 100–200 cm |

## 核心算法流程

```
摄像头采集 → 外框检测(OTSU二值化→找最大四边形→角点排序)
          → PnP位姿解算(solvePnP → X/Y/Z坐标 + 欧拉角)
          → 距离修正(分段线性插值)
          → 透视校正(getPerspectiveTransform)
          → 内部图形检测(三角形/正方形/圆形分类)
          → 输出D(距离)和x(图形尺寸)
```

| 情况 | 图形形状 | 重叠情况 |
| :-: | :-: | :-: |
| 1.多边形角点数=3<br>2.轮廓面积/最小外接圆面积<阈值 | 三角形 | 不重叠 |
| 1.多边形角点数=3<br>2.轮廓面积/最小外接圆面积>阈值 | 圆形 | 不重叠 |
| 1.多边形角点数=4<br>2.轮廓面积/最小外接矩形面积<阈值1 <br>3.轮廓面积/最小外接圆形面积<阈值2 | 无 | 重叠 |
| 1.多边形角点数=4<br>2.轮廓面积/最小外接矩形面积>阈值1<br>3.轮廓面积/最小外接圆形面积<阈值2 | 正方形 | 不重叠 |
| 1.多边形角点数=4<br>2.轮廓面积/最小外接圆形面积>阈值2 | 圆形 | 不重叠 |
| 1.多边形角点数>4<br>2.轮廓面积/最小外接圆形面积<阈值 | 无 | 重叠 |
| 1.多边形角点数>4<br>2.轮廓面积/最小外接圆形面积>阈值 | 圆形 | 不重叠 |

<b>2.</b>若为基础图形：<b>真实边长(mm) = 边框真实周长(mm)*(内部图形轮廓周长(像素) 或 直径(像素))/边框轮廓边长(像素)</b>

### 重叠正方形分割
<b>1.</b> 对判定为重叠的轮廓进行精度更高的多边形近似，对多边形的角点进行分类，以角点为圆心，做圆形掩膜与上内部图形图像，计算黑色像素数量和总像素数量，计算黑色像素数量/总像素数量比值，若比值小于阈值POINT_SORT_LIMIT，则判定为<b>正方形边框角点</b>，反之则为<b>正方形相交的交点</b>

<b>2.</b> 连接两个矩形相交处的两个交点，使重叠N个正方形轮廓可以分解为N个离散轮廓,如下图：
![边框](ignore/res/manyRectSplit.png)

<b>3.</b> 遍历所有离散轮廓，以单个离散轮廓构造掩膜，对各个轮廓做<b>精细的多边形近似</b>，<b>按1中方法对角点进行分类</b>

<b>4.</b> 对于离散轮廓，有以下几种情况：

| 情况 | 边长计算方法 |
| :-: | :-: |
| 边框角点数=3 | 计算边框角点间的距离，最长的为对角线，边长=非对角线的两点距离 |
| 边框角点数=2<br>（这两个角点为中间无交点） | 这两个边框角点为非对角线点，边长=这两点距离 |
| 边框角点数=2<br>（这两个角点为中间有交点） | 这两个边框角点为对角线点，边长=这两点距离/sqrt(2) |

## 目前需改进方向
<b>1.</b> 需要重构

<b>2.</b> 优化完善重叠正方形分割2的实现，理论没有问题，只是代码实现只实现了一半，导致泛用性不佳

<b>3.</b> 将图像二值化修改为HSV颜色空间二值化，降低光线影响

# [<font color=#0b88bb>🐧要做一辈子嵌入式开发!!!!!🐧</font>](https://github.com/Geek-Egret)

