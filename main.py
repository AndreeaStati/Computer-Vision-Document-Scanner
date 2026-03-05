import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

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

if __name__ == "__main__":

    path = 'dataset/1.jpg'

    image = load_image(path)
    if image is None:
        exit(1)
    display_step(image, "1. Original Image")

    orig = image.copy()
