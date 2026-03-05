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

# pre-procesarea Imaginii
# ────────────────────────────────────────────── 3. GRAYSCALE
def convert_to_grayscale(image):
    if image is None:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print("Grayscale conversion complete.")
    return gray


# ────────────────────────────────────────────── 4. CLAHE
def apply_clahe(gray_image, clip_limit=2.0, tile_grid=(8, 8)):
    if gray_image is None:
        return None
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    enhanced = clahe.apply(gray_image)
    print(f"CLAHE applied.")
    return enhanced


# ────────────────────────────────────────────── 5. THRESHOLDING 
def apply_thresholding(gray_image):
    if gray_image is None:
        return None
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print("Otsu thresholding complete.")
    return binary


# ────────────────────────────────────────────── 6. EDGE DETECTION
def enhance_for_contours(gray_image):
    if gray_image is None:
        return None
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edged, kernel, iterations=2)
    print("Edge detection complete.")
    return dilated

# detectarea si extragerea Documentului
# ────────────────────────────────────────────── 7. FIND DOCUMENT CONTOUR
def is_rectangular_enough(approx, angle_tolerance=40):
    pts = approx.reshape(4, 2).astype("float32")
    for i in range(4):
        p1 = pts[(i - 1) % 4]
        p2 = pts[i]
        p3 = pts[(i + 1) % 4]
        v1 = p1 - p2
        v2 = p3 - p2
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
        if abs(angle - 90) > angle_tolerance:
            return False
    return True

def find_document_contour(edged_image, image_shape):
    MAX_CANDIDATES = 10
    EPSILONS = [0.01, 0.02, 0.03, 0.04, 0.05]   # de la fin la grosier
    MAX_AREA_RATIO = 0.97   # singura restricție: să nu fie chenarul imaginii
    MIN_AREA_RATIO = 0.10   # cel puțin 10% din imagine

    img_area = image_shape[0] * image_shape[1]

    contours, _ = cv2.findContours(
        edged_image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None, []

    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for c in sorted_contours[:MAX_CANDIDATES]:
        area = cv2.contourArea(c)
        ratio = area / img_area
        if ratio < MIN_AREA_RATIO or ratio > MAX_AREA_RATIO:
            continue

        peri = cv2.arcLength(c, True)
        for eps in EPSILONS:
            approx = cv2.approxPolyDP(c, eps * peri, True)
            if len(approx) == 4:
                print(f"4-corner contour found (epsilon={eps}, area_ratio={ratio:.2f}).")
                return approx, sorted_contours

    print("No 4-corner contour found. Fallback: minAreaRect.")
    for c in sorted_contours[:MAX_CANDIDATES]:
        area = cv2.contourArea(c)
        ratio = area / img_area
        if ratio < MIN_AREA_RATIO or ratio > MAX_AREA_RATIO:
            continue
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        contour_box = box.astype("int").reshape(4, 1, 2)
        print(f"minAreaRect fallback used (area_ratio={ratio:.2f}).")
        return contour_box, sorted_contours

    print("Error: no valid contour found.")
    return None, sorted_contours


# ────────────────────────────────────────────── 8. ORDER POINTS
def order_points(pts):
    pts = pts.reshape(4, 2).astype("float32")
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]    # top-left
    rect[2] = pts[np.argmax(s)]    # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # top-right
    rect[3] = pts[np.argmax(diff)] # bottom-left
    return rect


# ────────────────────────────────────────────── 9. PERSPECTIVE TRANSFORM
def perspective_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA  = np.linalg.norm(br - bl)
    widthB  = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    print(f"Perspective transform complete → {warped.shape[1]}x{warped.shape[0]}")
    return warped


# ────────────────────────────────────────────── 10. QUALITY SCORE
def calculate_quality_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Sharpness
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Skew via Hough Lines
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
    skew_deg = 0.0
    if lines is not None:
        angles = []
        for rho, theta in lines[:, 0]:
            angle = np.degrees(theta) - 90
            if -45 < angle < 45:
                angles.append(angle)
        if angles:
            skew_deg = float(np.median(angles))

    status = "Clear" if sharpness > 100 else "Blurry/Motion"
    print(f"Quality → sharpness: {sharpness:.2f} ({status}) | skew: {skew_deg:.2f}°")
    return {"sharpness": sharpness, "skew_deg": skew_deg, "status": status}


# ────────────────────────────────────────────── PIPELINE

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

    # 3. CLAHE
    gray_clahe = apply_clahe(gray)
    display_step(gray_clahe, "3. CLAHE Enhanced", is_gray=True)

    # 4. Thresholding --> img binara
    binary_mask = apply_thresholding(gray_clahe)
    display_step(binary_mask, "4. Binary Image (Otsu)", is_gray=True)

    # 5. Edge detection
    edged = enhance_for_contours(gray_clahe)
    display_step(edged, "5. Edges (Canny + Dilation)", is_gray=True)

    # 6. Find contour
    doc_contour, all_contours = find_document_contour(edged, image_resized.shape)

    # top 10 contururi
    img_debug = image_resized.copy()
    cv2.drawContours(img_debug, all_contours[:10], -1, (0, 255, 255), 2)
    display_step(img_debug, "6a. Debug – Top 10 Contours")

    if doc_contour is not None:
        img_selection = image_resized.copy()
        cv2.drawContours(img_selection, [doc_contour], -1, (0, 255, 0), 3)
        display_step(img_selection, "6b. Final Selected Contour")

        # 7. Perspective transform pe originalul la rezolutie completa
        pts_orig = doc_contour.reshape(4, 2).astype("float32") * ratio
        warped = perspective_transform(orig, pts_orig)
        display_step(warped, "7. Warped (Perspective Corrected)")

        # 8. Quality score
        scores = calculate_quality_score(warped)
        label = (f"8. Final Result  |  "
                 f"Sharpness: {scores['sharpness']:.1f} ({scores['status']})  |  "
                 f"Skew: {scores['skew_deg']:.1f}°")
        display_step(warped, label)

    else:
        print("Pipeline stopped: no valid document contour found.")
