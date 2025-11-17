# img_preprocess.py
import os
from facenet_pytorch import MTCNN
import cv2
from PIL import Image
import numpy as np

# global detector instance
mtcnn = MTCNN(keep_all=False, device='cpu')

def load_image(image_path):
    """Load image as RGB uint8."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def detect_and_crop_face(image):
    """Detect the largest face and return the cropped region as RGB uint8."""
    if isinstance(image, str):
        image = load_image(image)
        if image is None:
            return None

    try:
        pil_img = Image.fromarray(image)
        boxes, probs = mtcnn.detect(pil_img)
    except Exception as e:
        print("Detection error:", e)
        return None

    if boxes is None or len(boxes) == 0 or probs[0] < 0.9:
        return None

    x1, y1, x2, y2 = boxes[0].astype(int)

    # bounds check
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    face = image[y1:y2, x1:x2]
    return face  # RGB uint8

def process_dataset(input_dir, output_dir, dim=(224, 224)):
    """Process a flat directory of images and write outputs into output_dir.
    Creates output_dir if it doesn't exist.
    """
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(input_dir):
        if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        path = os.path.join(input_dir, fname)
        face = detect_and_crop_face(path)
        if face is None:
            print(f"❌ No face detected in {fname}")
            continue

        # Resize to specified dimensions
        resized = cv2.resize(face, dim)

        save_path = os.path.join(output_dir, fname)
        cv2.imwrite(save_path, cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))

        print(f"✔ Processed: {fname}")


def process_dataset_tree(input_root, output_root, dim=(224, 224)):
    """Walk input_root recursively and process images while preserving
    the relative subdirectory structure under output_root.

    - input_root: directory containing class/category subfolders with images
    - output_root: destination root; subfolders will be created to mirror input
    - dim: output image size (w, h)
    """
    for dirpath, dirnames, filenames in os.walk(input_root):
        # Compute relative path from the input root to current folder
        rel = os.path.relpath(dirpath, input_root)
        # Determine corresponding output directory
        out_dir = os.path.join(output_root, rel) if rel != "." else output_root
        os.makedirs(out_dir, exist_ok=True)

        # Process supported image files in this folder
        for fname in filenames:
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            in_path = os.path.join(dirpath, fname)
            face = detect_and_crop_face(in_path)
            if face is None:
                print(f"❌ No face detected in {os.path.join(rel, fname)}")
                continue

            resized = cv2.resize(face, dim)
            out_path = os.path.join(out_dir, fname)
            cv2.imwrite(out_path, cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))
            print(f"✔ Processed: {os.path.join(rel, fname)} -> {os.path.relpath(out_path, output_root)}")

