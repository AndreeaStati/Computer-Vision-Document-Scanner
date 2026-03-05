import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

# base functions and utils 

# ────────────────────────────────────────────── 1. LOAD
def load_image(path):
    if not os.path.exists(path):
        print(f"Error: File '{path}' not found.")
        return None
    img = cv2.imread(path)
    if img is None:
        print(f"Error: Could not decode image.")
        return None
    print(f"Successfully loaded: {path}  |  shape: {img.shape}")
    return img


# ────────────────────────────────────────────── 2. DISPLAY
def display_step(image, title="Image Step", is_gray=False):
    if image is None:
        return
    plt.figure(figsize=(12, 8))
    if is_gray:
        plt.imshow(image, cmap='gray')
    else:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# ────────────────────────────────────────────── 3. GRAYSCALE
def convert_to_grayscale(image):
    if image is None:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print("Grayscale conversion complete.")
    return gray

# ────────────────────────────────────────────── 4. THRESHOLDING 
def apply_thresholding(gray_image):
    if gray_image is None:
        return None
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print("Otsu thresholding complete.")
    return binary


if __name__ == "__main__":

    path = 'dataset/1.jpg'

    # 1. Load
    image = load_image(path)
    if image is None:
        exit(1)
    display_step(image, "1. Original Image")

    orig = image.copy()

    # Resize la 800px inaltime pentru procesare
    HEIGHT_TARGET = 800.0
    ratio = image.shape[0] / HEIGHT_TARGET
    image_resized = cv2.resize(image, (int(image.shape[1] / ratio), int(HEIGHT_TARGET)))

    # 2. Grayscale
    gray = convert_to_grayscale(image_resized)
    display_step(gray, "2. Grayscale", is_gray=True)

    # 3. Thresholding --> img binara
    binary_mask = apply_thresholding(gray)
    display_step(binary_mask, "4. Binary Image (Otsu)", is_gray=True)
