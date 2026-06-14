# Fruit Ripeness Classification

Computer Vision course project. Automatically classifies fruit ripeness (Unripe / Ripe / Overripe) for bananas, apples, and mangoes using a 6-stage pipeline combining OpenCV and Deep Learning.

## Pipeline

Image -> Enhance -> Segment -> Clean -> Detect -> Identify -> Decide -> Result

| Stage | Method | Library |
|---|---|---|
| Enhance | CLAHE + GaussianBlur | OpenCV |
| Segment | HSV color thresholding | OpenCV |
| Clean | Morphological Opening + Closing | OpenCV |
| Detect | findContours + boundingRect | OpenCV |
| Identify | Random Forest (79% accuracy) | scikit-learn |
| Decide | ResNet-50 Transfer Learning (84.8% accuracy) | PyTorch |

## Setup

    git clone https://github.com/kireal-cyber/fruit_cv.git
    cd fruit_cv
    python3 -m venv venv
    source venv/bin/activate
    pip install opencv-python numpy matplotlib scikit-learn torch torchvision

## Run

Place fruit images in images/ folder, then:

    python3 main.py
    python3 visualize.py
    open results/

## Results

- Test dataset: 33 images (banana / apple / mango, 3 classes each)
- Fruit identification accuracy: 51.5% (Random Forest)
- Ripeness classification accuracy: 84.8% (ResNet-50)

| Fruit | Unripe | Ripe | Overripe | Total |
|---|---|---|---|---|
| Banana | 5/5 | 5/5 | 5/5 | 100% |
| Apple | 3/3 | 3/3 | 0/3 | 67% |
| Mango | 3/3 | 3/3 | 1/3 | 78% |
| Total | | | | 84.8% |

## Training (optional)

To retrain the models:

    python3 train_classifier.py
    python3 train_resnet.py

Requires dataset in dataset/archive-3/ folder.

## Project Structure

    fruit_cv/
    |-- images/              <- input images
    |-- results/             <- output (6 stages per image)
    |-- pipeline/
    |   |-- enhance.py       <- Stage 1: CLAHE + blur
    |   |-- segment.py       <- Stage 2: HSV masking
    |   |-- clean.py         <- Stage 3: morphology
    |   |-- detect.py        <- Stage 4: contours
    |   |-- identify.py      <- Stage 5: Random Forest
    |   |-- decide.py        <- Stage 6: ResNet-50
    |-- main.py              <- runs full pipeline
    |-- visualize.py         <- creates summary images
    |-- train_classifier.py  <- trains Random Forest
    |-- train_resnet.py      <- trains ResNet-50
    |-- fruit_classifier.pkl <- trained RF model
    |-- resnet_ripeness.pth  <- trained ResNet model
