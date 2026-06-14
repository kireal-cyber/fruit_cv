import cv2
import os
from pipeline.enhance   import enhance
from pipeline.segment   import segment
from pipeline.clean     import clean
from pipeline.detect    import detect
from pipeline.identify  import identify_fruit
from pipeline.decide    import decide

def run_pipeline(image_path):
    name = os.path.splitext(os.path.basename(image_path))[0]
    print(f"\n--- Обрабатываем: {name} ---")

    img = cv2.imread(image_path)
    if img is None:
        print(f"ОШИБКА: не удалось загрузить {image_path}")
        return None
    cv2.imwrite(f"results/{name}_0_original.jpg", img)

    enhanced = enhance(img)
    cv2.imwrite(f"results/{name}_1_enhanced.jpg", enhanced)
    print("✓ Enhance")

    mask = segment(enhanced)
    cv2.imwrite(f"results/{name}_2_mask.jpg", mask)
    print("✓ Segment")

    cleaned = clean(mask)
    cv2.imwrite(f"results/{name}_3_cleaned.jpg", cleaned)
    print("✓ Clean")

    detected, contour = detect(img, cleaned)
    cv2.imwrite(f"results/{name}_4_detected.jpg", detected)
    print("✓ Detect")

    fruit_type = identify_fruit(img, cleaned, contour)
    print(f"✓ Identify → {fruit_type}")

    label, color, pct_green = decide(img, cleaned, fruit_type)
    print(f"✓ Decide → {label}")

    final = detected.copy()
    cv2.putText(final, f"{fruit_type}: {label}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.imwrite(f"results/{name}_5_final.jpg", final)

    return fruit_type, label

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    results = {}
    for filename in os.listdir("images"):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join("images", filename)
            result = run_pipeline(path)
            if result:
                results[filename] = result

    print("\n=== ИТОГО ===")
    for fname, (ftype, label) in results.items():
        print(f"  {fname}: {ftype} → {label}")
