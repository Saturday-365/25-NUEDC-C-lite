# 25-NUEDC-C-lite 项目指南

## 项目概要
2025年全国大学生电子设计竞赛C题——"基于单目视觉的目标物测量装置"。
单摄像头测量竖立A4纸（2cm黑框）的距离D（100-200cm）和内部图形尺寸x。

## 项目结构
- `0_SoftWare/jetson/` — Jetson Orin NX (C++/OpenCV/OnnxRuntime) 嵌入式端代码
- `0_SoftWare/maixcam/` — MaixCam K210 (Python/maix) 嵌入式端代码
- `1_PC_running_code/` — PC端实验代码 (Python/OpenCV)
- `3_Document/` — 论文文档和排版输出
- `ignore/` — 资源图片和排版脚本

## 关键技术参数
- A4板尺寸: 168.1 × 255.1 mm（PC代码）/ 170.5 × 260.5 mm（Jetson代码）
- Jetson 相机内参: fx=585.57, fy=598.96, cx=326.52, cy=226.96（640×480分辨率）
- MaixCam 相机内参: fx=1158.86, fy=1183.95, cx=357.72, cy=200.27（640×480分辨率）
- iPhone相机需要重新标定
- PnP方法: SOLVEPNP_IPPE（平面目标最优）
- 透视校正像素: warp_width=800px

## 主要技术路线
1. 外框检测 → 二值化(OTSU) → 找最大四边形 → 角点排序
2. PnP位姿解算 → solvePnP → tvec(X/Y/Z坐标) + rvec→欧拉角
3. 距离修正 → 分段线性插值（多校准点）
4. 透视校正 → getPerspectiveTransform → 像素/mm比例 → 图形尺寸x

## PC代码关键函数
- `preprocess_for_black_contours()` — 二值化预处理
- `find_board_quad()` — A4外框四边形检测
- `order_quad_points()` — 角点排序（和/差法）
- `warp_board()` — 透视校正
- `classify_inner_shape()` — 内部分类（三角形/正方形/圆形）
- `detect_inner_shapes()` — 内部图形检测与测量

## 论文信息
- 标题: 基于透视校正的平面目标图形识别与尺寸测量方法研究
- 实际内容: PnP位姿解算（已撰写）+ 透视校正图形测量（需补充）
- 作者: 宋嘉诚 学号2315102026 通信工程
- 排版脚本: ignore/tools/build_paper_docx.py
- 输出格式: docx（python-docx）

## 构建和运行
### PC端实验
```bash
cd 1_PC_running_code
pip install opencv-python numpy
python src/pc_opencv_experiment.py --input data/input/ --output data/output/
```

### Jetson端
```bash
cd 0_SoftWare/jetson/GE
bash compile.sh
```

## 编码约定
- Python: 类型注解, dataclass, snake_case函数名
- C++: OpenCV命名空间, PascalCase函数名, ge命名空间
- 论文Markdown: 标准引用式, $$数学公式, 代码块用```text
