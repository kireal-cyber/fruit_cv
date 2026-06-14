import cv2
import numpy as np

def segment(enhanced_img):
    """
    Выделяем фрукт из фона.
    Захватываем широкий диапазон цветов: зелёный, жёлтый, 
    оранжевый, красный, коричневый.
    """
    hsv = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2HSV)

    # Зелёный, жёлтый, оранжевый (Hue 10-95)
    mask1 = cv2.inRange(hsv, np.array([10, 40, 40]), np.array([95, 255, 255]))

    # Красный (Hue 0-10 и 160-180) — два диапазона потому что
    # красный в HSV находится на обоих концах шкалы
    mask2 = cv2.inRange(hsv, np.array([0, 40, 40]), np.array([10, 255, 255]))
    mask3 = cv2.inRange(hsv, np.array([160, 40, 40]), np.array([180, 255, 255]))

    # Объединяем все маски
    mask = cv2.bitwise_or(mask1, mask2)
    mask = cv2.bitwise_or(mask, mask3)

    return mask
