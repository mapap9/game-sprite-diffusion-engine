import os
import cv2
import numpy as np

def audit_processed_data(folder_path):
    file_list = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    print(f"📊 Auditing {len(file_list)} processed assets...")

    boxy_count = 0
    irregular_count = 0

    for filename in file_list:
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Threshold to count active object pixels vs black background
        _, thresh = cv2.threshold(img, 15, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            x, y, w, h = cv2.boundingRect(contours[0])
            aspect_ratio = float(w) / h
            extent = float(cv2.contourArea(contours[0])) / (w * h) if (w * h) > 0 else 0

            # If the item fills out its bounding box like a solid square/rectangle
            if 0.8 <= aspect_ratio <= 1.2 and extent > 0.75:
                boxy_count += 1
            else:
                irregular_count += 1

    print(f"\n📈 Results:")
    print(f"   └── 📦 Square/Boxy Assets (Books, Chests, Scrolls): {boxy_count} ({boxy_count/len(file_list)*100:.1f}%)")
    print(f"   └── ⚔️ Irregular Assets (Weapons, Characters, Animals): {irregular_count} ({irregular_count/len(file_list)*100:.1f}%)")

if __name__ == "__main__":
    audit_processed_data("data/processed")
