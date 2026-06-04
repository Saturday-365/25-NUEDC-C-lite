# PC OpenCV experiment

This folder contains the PC-side experiment code for the course paper:

基于透视校正的平面目标图形识别与尺寸测量方法研究

The original Jetson/MaixCam competition code is kept unchanged. This PC version
uses a normal USB camera or saved images and keeps only the image-processing
pipeline needed by the paper.

## Setup

```powershell
cd C:\Git_Program\25-NUEDC-C-lite
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r pc_opencv\requirements.txt
```

## Prepare target images

Print or draw an A4 target board with:

- a black rectangular outer border;
- inner triangle, square, and circle with known dimensions;
- a clean white background when possible.

Save test images under:

```text
pc_opencv\data\input
```

Suggested names:

```text
angle_00.jpg
angle_15.jpg
angle_30.jpg
distance_100cm.jpg
distance_150cm.jpg
distance_200cm.jpg
```

## Run on saved images

```powershell
python pc_opencv\src\pc_opencv_experiment.py --input pc_opencv\data\input --output pc_opencv\data\output
```

## Capture from USB camera

```powershell
python pc_opencv\src\pc_opencv_experiment.py --camera 0 --capture pc_opencv\data\input\usb_capture.jpg
```

Then process the captured image:

```powershell
python pc_opencv\src\pc_opencv_experiment.py --input pc_opencv\data\input\usb_capture.jpg --output pc_opencv\data\output
```

## Outputs

The script writes:

- `measurements.csv`: detected shape type, pixel size, measured size, and area;
- `*_raw.png`: source image with detected target board corners;
- `*_binary.png`: binary image used for outer contour detection;
- `*_warped.png`: perspective-corrected target board with detected inner shapes.

Use these outputs as figures and data tables in the paper.
