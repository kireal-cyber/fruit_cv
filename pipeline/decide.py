import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
import os

# Загружаем модель один раз при импорте
_model_path = os.path.join(os.path.dirname(__file__), "../resnet_ripeness.pth")
_checkpoint = torch.load(_model_path, map_location="cpu")
_classes = _checkpoint["classes"]  # ['Overipe', 'Ripe', 'Unripe']
_num_classes = _checkpoint["num_classes"]

_device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

_resnet = models.resnet50(weights=None)
_resnet.fc = nn.Linear(_resnet.fc.in_features, _num_classes)
_resnet.load_state_dict(_checkpoint["model_state"])
_resnet = _resnet.to(_device)
_resnet.eval()

_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Цвета для каждого класса
_colors = {
    "Unripe":  (0, 180, 0),    # зелёный
    "Ripe":    (0, 180, 180),  # жёлтый
    "Overripe":(0, 0, 200),    # красный
}

def decide(original_img, cleaned_mask, fruit_type="unknown"):
    """
    Классификация зрелости через ResNet-50.
    Вырезаем фрукт по маске и подаём в нейросеть.
    """
    # Вырезаем фрукт по bounding box маски
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        # Добавляем отступ 10px
        pad = 10
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(original_img.shape[1], x + w + pad)
        y2 = min(original_img.shape[0], y + h + pad)
        crop = original_img[y1:y2, x1:x2]
    else:
        crop = original_img

    if crop.size == 0:
        crop = original_img

    # Конвертируем BGR → RGB для torchvision
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    # Прогоняем через ResNet
    tensor = _transform(crop_rgb).unsqueeze(0).to(_device)
    with torch.no_grad():
        output = _resnet(tensor)
        proba  = torch.softmax(output, dim=1)[0]
        pred_idx = torch.argmax(proba).item()

    raw_label = _classes[pred_idx]
    confidence = proba[pred_idx].item() * 100

    # Нормализуем метку (в датасете 'Overipe' вместо 'Overripe')
    if raw_label == "Overipe":
        label = "Overripe"
    else:
        label = raw_label

    color = _colors.get(label, (128, 128, 128))
    print(f"   ResNet: {label} ({confidence:.1f}% уверенность)")

    return label, color, confidence
