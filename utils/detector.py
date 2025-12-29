import cv2
import os
import numpy as np
from ultralytics import YOLO

# Load YOLO model once
model = YOLO(r"E:\Yolo train\yolov8s\runs\detect\train4\weights\best.pt")

# Class names (match your training)
CLASS_NAMES = {
    0: "Bladder",
    1: "Kidney",
    2: "Ureter"
}


def upscale_back_to_original(orig_img, crop_img, bbox):
    """Upscale crop back to original image size with black background."""
    x1, y1, x2, y2 = bbox
    H, W, _ = orig_img.shape

    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    # Resize crop into bbox area
    resized = cv2.resize(crop_img, (x2 - x1, y2 - y1), interpolation=cv2.INTER_CUBIC)

    # Place crop
    canvas[y1:y2, x1:x2] = resized

    return canvas


def detect_and_crop(img_path, save_dir):
    """
    Runs YOLO detection, saves:
        - annotated image
        - cropped organs
        - upscaled crops (placed back in full image)
    Returns dictionary for Flask.
    """

    # Output folders
    crop_dir = os.path.join(save_dir, "crops")
    upscale_dir = os.path.join(save_dir, "upscaled")
    os.makedirs(crop_dir, exist_ok=True)
    os.makedirs(upscale_dir, exist_ok=True)

    # Run model
    results = model(img_path)
    res = results[0]

    # Save annotated detection result
    annotated_img = res.plot()
    annotated_path = os.path.join(save_dir, "annotated.jpg")
    cv2.imwrite(annotated_path, annotated_img)

    # Load original image
    orig = cv2.imread(img_path)

    # Crop counter
    counter = {"Bladder": 0, "Kidney": 0, "Ureter": 0}

    # Store crop paths for classifier
    all_crops = {
        "Bladder": [],
        "Kidney": [],
        "Ureter": []
    }

    # Loop through detected boxes
    for box in res.boxes.data:
        x1, y1, x2, y2, conf, cls = box.tolist()
        cls = int(cls)
        organ = CLASS_NAMES[cls]

        counter[organ] += 1

        # Normal crop (small crop)
        crop = orig[int(y1):int(y2), int(x1):int(x2)]
        crop_path = os.path.join(crop_dir, f"{organ}_{counter[organ]}.png")
        cv2.imwrite(crop_path, crop)

        # Upscaled (full size with black bg)
        upscaled = upscale_back_to_original(
            orig, crop, (int(x1), int(y1), int(x2), int(y2))
        )
        upscale_path = os.path.join(upscale_dir, f"{organ}_{counter[organ]}_upscaled.png")
        cv2.imwrite(upscale_path, upscaled)

        # Save both paths in dictionary
        all_crops[organ].append({
            "crop": crop_path,
            "upscaled": upscale_path,
            "bbox": (int(x1), int(y1), int(x2), int(y2))
        })

    return annotated_path, all_crops
