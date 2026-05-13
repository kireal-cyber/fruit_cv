import cv2

def detect(original_img, cleaned_mask):
    contours, _ = cv2.findContours(
        cleaned_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    result = original_img.copy()
    if not contours:
        print("ВНИМАНИЕ: контуры не найдены!")
        return result, None
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return result, largest
