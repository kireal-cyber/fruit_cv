import cv2
import numpy as np
import os

def make_summary(name):
    """
    Собирает 6 картинок одного изображения в одну сводную.
    Это именно то что нужно показать в отчёте и на защите.
    """
    steps = [
        (f"results/{name}_0_original.jpg",  "1. Original"),
        (f"results/{name}_1_enhanced.jpg",  "2. Enhanced"),
        (f"results/{name}_2_mask.jpg",      "3. Mask"),
        (f"results/{name}_3_cleaned.jpg",   "4. Cleaned"),
        (f"results/{name}_4_detected.jpg",  "5. Detected"),
        (f"results/{name}_5_final.jpg",     "6. Decision"),
    ]

    SIZE = (300, 300)  # размер каждой картинки в сетке
    images = []

    for path, title in steps:
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((SIZE[1], SIZE[0], 3), np.uint8)
        else:
            img = cv2.resize(img, SIZE)

        # Если картинка чёрно-белая (маска) — переводим в BGR
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Добавляем заголовок сверху
        header = np.zeros((40, SIZE[0], 3), np.uint8)
        header[:] = (40, 40, 40)  # тёмно-серый фон
        cv2.putText(header, title, (8, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cell = np.vstack([header, img])
        images.append(cell)

    # Собираем 2 ряда по 3 картинки
    row1 = np.hstack(images[0:3])
    row2 = np.hstack(images[3:6])
    summary = np.vstack([row1, row2])

    out_path = f"results/{name}_SUMMARY.jpg"
    cv2.imwrite(out_path, summary)
    print(f"Сохранено: {out_path}")

# Запускаем для всех обработанных изображений
for f in os.listdir("results"):
    if f.endswith("_0_original.jpg"):
        name = f.replace("_0_original.jpg", "")
        make_summary(name)
