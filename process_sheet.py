import os
import cv2
import numpy as np
from glob import glob

def batch_extraction_engine(input_dir, output_dir, target_size=32):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Find all image assets in the raw directory
    valid_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
    image_paths = []
    for ext in valid_extensions:
        image_paths.extend(glob(os.path.join(input_dir, ext)))

    print(f"📂 Found {len(image_paths)} raw sprite sheets to process.")

    global_count = 0

    for sheet_idx, image_path in enumerate(image_paths):
        sheet_name = os.path.basename(image_path)
        print(f"🎬 Processing Sheet [{sheet_idx + 1}/{len(image_paths)}]: {sheet_name}")

        # Read the raw image
        img = cv2.imread(image_path)
        if img is None:
            print(f"⚠️ Warning: Could not read {sheet_name}. Skipping.")
            continue

        h_orig, w_orig, _ = img.shape

        # ⛔ WATERMARK SANITIZATION:
        # Clone the sheet matrix and black out the bottom-right watermark sector
        # so the contour engine never sees the Gemini logo artifacts.
        masked_img = img.copy()
        watermark_y = int(h_orig * 0.86)
        watermark_x = int(w_orig * 0.84)
        masked_img[watermark_y:, watermark_x:] = 0

        # Convert to grayscale and threshold to binary
        gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)

        # MORPHOLOGICAL CLEANUP:
        # Erode edges with a 2x2 structural element to break accidental diagonal contacts
        kernel = np.ones((2, 2), np.uint8)
        eroded = cv2.erode(thresh, kernel, iterations=1)

        # Find contours based on the isolated, eroded structural paths
        contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        sheet_extracted_count = 0

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # FILTER NOISE FRAGMENTS & ACCIDENTS
            # Enforce strict pixel bounding boundaries
            if w < 10 or h < 10 or w > 55 or h > 55:
                continue

            # Crop the target asset out of the ORIGINAL image to preserve raw un-eroded data
            sprite_crop = img[y:y+h, x:x+w]

            # Initialize a clean, standardized black template canvas
            canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)

            # Proportionally downscale the asset if it exceeds the 32x32 bounding frame
            if w > target_size or h > target_size:
                scale = min(target_size / w, target_size / h)
                sprite_crop = cv2.resize(sprite_crop, (int(w * scale), int(h * scale)))
                h, w, _ = sprite_crop.shape

            # Compute exact coordinate offsets to drop the item dead-center on the black background
            pad_x = (target_size - w) // 2
            pad_y = (target_size - h) // 2
            canvas[pad_y:pad_y+h, pad_x:pad_x+w] = sprite_crop

            # Save out with a global continuous file index across all sheets
            output_filename = f"sprite_{global_count:05d}.png"
            cv2.imwrite(os.path.join(output_dir, output_filename), canvas)

            global_count += 1
            sheet_extracted_count += 1

        print(f"   └── ✅ Extracted {sheet_extracted_count} clean sprites from {sheet_name}")

    print(f"\n🏁 System Complete. Generated a unified dataset of {global_count} training assets.")

if __name__ == "__main__":
    batch_extraction_engine("data/raw", "data/processed")
