import cv2
import numpy as np
import os
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

def extract_features(img, mask):
    """
    Извлекаем признаки фрукта для классификатора.
    Используем цветовой профиль + форму контура.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    total = cv2.countNonZero(mask)
    if total == 0:
        return None

    # Цветовые признаки
    ranges = [
        ([0,  60, 60], [10, 255, 255]),   # красный 1
        ([160,60, 60], [180,255,255]),     # красный 2
        ([8,  60, 60], [18, 255,255]),     # оранжевый
        ([15, 40, 40], [35, 255,255]),     # жёлтый
        ([36, 40, 40], [85, 255,255]),     # зелёный
        ([85, 40, 40], [130,255,255]),     # голубой/синий
        ([0,  0,  0],  [180,50, 255]),     # ненасыщенный (серый/белый)
        ([0,  0,  0],  [180,255,60]),      # тёмный
    ]

    features = []
    for lower, upper in ranges:
        m = cv2.inRange(hsv, np.array(lower), np.array(upper))
        count = cv2.countNonZero(cv2.bitwise_and(m, m, mask=mask))
        features.append(count / total * 100)

    # Средние значения HSV внутри маски
    mean_hsv = cv2.mean(hsv, mask=mask)
    features.extend([mean_hsv[0], mean_hsv[1], mean_hsv[2]])

    return features

# Загружаем все фото и извлекаем признаки
X = []  # признаки
y = []  # метки (banana/apple/mango)

label_map = {}
for fname in os.listdir("images"):
    if not fname.lower().endswith((".jpg",".jpeg",".png")):
        continue

    # Определяем метку по имени файла
    fname_lower = fname.lower()
    if "apple" in fname_lower:
        label = "apple"
    elif "mango" in fname_lower:
        label = "mango"
    elif any(x in fname_lower for x in ["unripe","ripe","overripe"]):
        label = "banana"
    else:
        continue

    img = cv2.imread(f"images/{fname}")
    if img is None:
        continue

    # Простая маска — берём всё кроме белого фона
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0,30,30]), np.array([180,255,255]))
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    features = extract_features(img, mask)
    if features:
        X.append(features)
        y.append(label)
        print(f"  {fname}: {label}")

X = np.array(X)
y = np.array(y)

print(f"\nВсего образцов: {len(X)}")
print(f"Классы: {np.unique(y, return_counts=True)}")

# Обучаем KNN классификатор
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_scaled, y)

# Проверяем точность через cross-validation
scores = cross_val_score(knn, X_scaled, y, cv=3)
print(f"\nТочность классификатора: {scores.mean()*100:.1f}% (+/- {scores.std()*100:.1f}%)")

# Сохраняем модель
with open("fruit_classifier.pkl", "wb") as f:
    pickle.dump({"model": knn, "scaler": scaler}, f)
print("Модель сохранена: fruit_classifier.pkl")
