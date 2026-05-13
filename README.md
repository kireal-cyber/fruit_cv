# Fruit Ripeness Classification

Computer Vision course project. Classifies banana ripeness into 3 classes: Unripe / Ripe / Overripe using a 5-stage OpenCV pipeline.

## Pipeline
Image → Enhance → Segment → Clean → Detect → Decide

## Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install opencv-python numpy matplotlib
```

## Run
Place images in `images/` folder, then:
```bash
python3 main.py
python3 visualize.py
```

## Results
- Test dataset: 15 images (5 per class)
- Accuracy: 15/15 (100%)

## Methods
| Stage | Method |
|---|---|
| Enhance | CLAHE + GaussianBlur |
| Segment | HSV color thresholding |
| Clean | Morphological Opening + Closing |
| Detect | findContours + boundingRect |
| Decide | Mean Hue (Unripe >32, Ripe 20-32, Overripe <20) |
