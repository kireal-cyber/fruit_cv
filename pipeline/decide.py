import cv2

def decide(original_img, cleaned_mask):
    hsv = cv2.cvtColor(original_img, cv2.COLOR_BGR2HSV)
    mean_val = cv2.mean(hsv, mask=cleaned_mask)
    hue = mean_val[0]

    # Границы подобраны по реальным данным:
    # Overripe: Hue < 20  (коричневый/тёмный)
    # Ripe:     Hue 20-32 (жёлтый)
    # Unripe:   Hue > 32  (зелёный)
    if hue > 32:
        label = "Unripe"
        color = (0, 180, 0)
    elif hue >= 20:
        label = "Ripe"
        color = (0, 180, 180)
    else:
        label = "Overripe"
        color = (0, 0, 200)

    return label, color, hue
