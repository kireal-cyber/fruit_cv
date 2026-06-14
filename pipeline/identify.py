import cv2
import numpy as np
import pickle
import os

# Загружаем модель один раз при импорте
_model_path = os.path.join(os.path.dirname(__file__), "../fruit_classifier.pkl")
with open(_model_path, "rb") as f:
    _data = pickle.load(f)
_model  = _data["model"]
_scaler = _data["scaler"]

def extract_features(img, mask):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    total = cv2.countNonZero(mask)
    if total < 100:
        total = img.shape[0] * img.shape[1]

    features = []

    ranges = [
        ([0,  60, 60], [10, 255,255]),
        ([160,60, 60], [180,255,255]),
        ([8,  60, 60], [18, 255,255]),
        ([15, 40, 40], [35, 255,255]),
        ([36, 40, 40], [85, 255,255]),
        ([85, 40, 40], [130,255,255]),
        ([0,  0,  0],  [180,255, 60]),
        ([0,  0,  0],  [180, 50,255]),
    ]
    for lower, upper in ranges:
        m = cv2.inRange(hsv, np.array(lower), np.array(upper))
        count = cv2.countNonZero(cv2.bitwise_and(m, m, mask=mask))
        features.append(count / total * 100)

    for ch in range(3):
        channel = hsv[:,:,ch]
        vals = channel[mask > 0]
        if len(vals) > 0:
            features.extend([float(vals.mean()), float(vals.std())])
        else:
            features.extend([0.0, 0.0])

    for ch in range(3):
        channel = lab[:,:,ch]
        vals = channel[mask > 0]
        if len(vals) > 0:
            features.append(float(vals.mean()))
        else:
            features.append(0.0)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        aspect = max(w,h) / max(min(w,h), 1)
        area = cv2.contourArea(largest)
        hull_area = cv2.contourArea(cv2.convexHull(largest))
        solidity = area / max(hull_area, 1)
        features.extend([aspect, solidity])
    else:
        features.extend([1.0, 1.0])

    return features

def identify_fruit(original_img, cleaned_mask, contour):
    img_resized = cv2.resize(original_img, (300, 300))
    mask_resized = cv2.resize(cleaned_mask, (300, 300))

    features = extract_features(img_resized, mask_resized)
    features_scaled = _scaler.transform([features])
    fruit_type = _model.predict(features_scaled)[0]

    # Вероятность предсказания
    proba = _model.predict_proba(features_scaled)[0]
    confidence = max(proba) * 100
    print(f"   ML: {fruit_type} ({confidence:.1f}% уверенность)")

    return fruit_type
