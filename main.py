import cv2
import matplotlib.pyplot as plt
import os

def display_image(image_path):

    if not os.path.exists(image_path):
        print(f"Error: The file '{image_path}' was not found.")
        return

    img = cv2.imread(image_path)

    if img is None:
        print("Error: Could not decode the image. It might be corrupted or an unsupported format.")
        return

    # Convert BGR to RGB
    # OpenCV uses BGR by default, but Matplotlib expects RGB.
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.imshow(img_rgb)
    plt.title("Image Loaded Successfully")
    #plt.axis('off')  # Hide pixel coordinates
    plt.show()

display_image("dataset/0144.jpg")