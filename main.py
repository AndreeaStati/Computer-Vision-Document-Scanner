import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import csv


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


# ────────────────────────────────────────────── W5 BINARIZATION AFTER RECTIFICATION
def apply_adaptive_thresholding(gray_image):
    if gray_image is None:
        return None
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 21, 10
    )
    print("Adaptive thresholding complete.")
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


# ────────────────────────────────────────────── W4 FALLBACK: HOUGH + CORNERS
def hough_corner_fallback(edged_image, image_shape):
    lines = cv2.HoughLinesP(
        edged_image,
        1,
        np.pi / 180,
        threshold=80,
        minLineLength=int(min(image_shape[:2]) * 0.25),
        maxLineGap=20
    )

    if lines is None:
        print("Hough fallback failed: no lines detected.")
        return None

    line_mask = np.zeros_like(edged_image)
    for line in lines[:, 0]:
        x1, y1, x2, y2 = line
        cv2.line(line_mask, (x1, y1), (x2, y2), 255, 3)

    line_mask = cv2.dilate(line_mask, np.ones((5, 5), np.uint8), iterations=2)

    corners = cv2.goodFeaturesToTrack(
        line_mask,
        maxCorners=40,
        qualityLevel=0.01,
        minDistance=20
    )

    if corners is None or len(corners) < 4:
        print("Hough fallback failed: not enough corners detected.")
        return None

    points = corners.reshape(-1, 2).astype(np.float32)
    rect = cv2.minAreaRect(points)
    box = cv2.boxPoints(rect).astype("int").reshape(4, 1, 2)

    print(f"Hough + corner fallback used ({len(lines)} lines, {len(points)} corners).")
    return box


# ────────────────────────────────────────────── 7. FIND DOCUMENT CONTOUR
def find_document_contour(edged_image, image_shape):
    MAX_CANDIDATES = 10
    EPSILONS = [0.01, 0.02, 0.03, 0.04, 0.05]
    MAX_AREA_RATIO = 0.97
    MIN_AREA_RATIO = 0.10

    img_area = image_shape[0] * image_shape[1]

    contours, _ = cv2.findContours(
        edged_image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        fallback = hough_corner_fallback(edged_image, image_shape)
        return fallback, [], "hough_corners" if fallback is not None else "none"

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
                return approx, sorted_contours, "contour"

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
        return contour_box, sorted_contours, "minAreaRect"

    fallback = hough_corner_fallback(edged_image, image_shape)
    if fallback is not None:
        return fallback, sorted_contours, "hough_corners"

    print("Error: no valid contour found.")
    return None, sorted_contours, "none"


# ────────────────────────────────────────────── 8. ORDER POINTS
def order_points(pts):
    pts = pts.reshape(4, 2).astype("float32")
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


# ────────────────────────────────────────────── 9. PERSPECTIVE TRANSFORM
def perspective_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
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

    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

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


# ────────────────────────────────────────────── HELPER: to RGB for display
def to_rgb(img, is_gray=False):
    """Converts image to RGB numpy array for matplotlib display."""
    if img is None:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    if is_gray or len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ────────────────────────────────────────────── PROCESS ONE IMAGE
def process_image(path, save_outputs=False, output_dir='outputs_week6'):
    """
    Runs the full pipeline on one image.
    Returns:
        steps  -> list of (title, rgb_image) tuples for display
        report -> dict with basic results for batch mode
    """
    steps = []
    report = {
        "image_name": os.path.basename(path),
        "detected": False,
        "method": "",
        "sharpness": "",
        "skew_deg": "",
        "status": ""
    }

    image = load_image(path)
    if image is None:
        return steps, report
    steps.append(("1. Original", to_rgb(image)))

    orig = image.copy()

    HEIGHT_TARGET = 800.0
    ratio = image.shape[0] / HEIGHT_TARGET
    image_resized = cv2.resize(image, (int(image.shape[1] / ratio), int(HEIGHT_TARGET)))

    gray = convert_to_grayscale(image_resized)
    steps.append(("2. Grayscale", to_rgb(gray, is_gray=True)))

    gray_clahe = apply_clahe(gray)
    steps.append(("3. CLAHE", to_rgb(gray_clahe, is_gray=True)))

    binary_mask = apply_thresholding(gray_clahe)
    steps.append(("4. Binary (Otsu)", to_rgb(binary_mask, is_gray=True)))

    edged = enhance_for_contours(gray_clahe)
    steps.append(("5. Edges (Canny+Dilation)", to_rgb(edged, is_gray=True)))

    doc_contour, all_contours, method = find_document_contour(edged, image_resized.shape)
    img_debug = image_resized.copy()
    cv2.drawContours(img_debug, all_contours[:10], -1, (0, 255, 255), 2)
    steps.append(("6a. Top 10 Contours", to_rgb(img_debug)))

    if doc_contour is not None:
        report["detected"] = True
        report["method"] = method

        img_selection = image_resized.copy()
        cv2.drawContours(img_selection, [doc_contour], -1, (0, 255, 0), 3)
        steps.append((f"6b. Selected Boundary ({method})", to_rgb(img_selection)))

        pts_orig = doc_contour.reshape(4, 2).astype("float32") * ratio
        warped = perspective_transform(orig, pts_orig)
        steps.append(("7. Perspective Corrected", to_rgb(warped)))

        warped_gray = convert_to_grayscale(warped)
        otsu_scan = apply_thresholding(warped_gray)
        adaptive_scan = apply_adaptive_thresholding(warped_gray)

        steps.append(("8. Scan (Otsu)", to_rgb(otsu_scan, is_gray=True)))
        steps.append(("9. Scan (Adaptive)", to_rgb(adaptive_scan, is_gray=True)))

        quality = calculate_quality_score(warped)
        report["sharpness"] = f"{quality['sharpness']:.2f}"
        report["skew_deg"] = f"{quality['skew_deg']:.2f}"
        report["status"] = quality["status"]

        if save_outputs:
            os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(path))[0]
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_warped.jpg"), warped)
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_otsu.jpg"), otsu_scan)
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_adaptive.jpg"), adaptive_scan)
    else:
        steps.append(("6b. No contour found", to_rgb(image_resized)))
        print("Pipeline stopped: no valid document contour found.")

    return steps, report


# ────────────────────────────────────────────── DISPLAY ALL STEPS IN GRID
def show_steps_grid(steps, image_name, image_index, total_images):
    n = len(steps)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 3.5))
    fig.suptitle(
        f"[{image_index}/{total_images}]  {image_name}   —   Enter = următoarea imagine  |  Q / Esc = ieșire",
        fontsize=12, fontweight='bold'
    )
    plt.subplots_adjust(top=0.93)

    axes_flat = axes.flatten() if rows > 1 else (axes if cols > 1 else [axes])

    for i, (title, img) in enumerate(steps):
        axes_flat[i].imshow(img)
        axes_flat[i].set_title(title, fontsize=9)
        axes_flat[i].axis('off')

    for j in range(n, len(axes_flat)):
        axes_flat[j].axis('off')

    plt.tight_layout()

    quit_flag = [False]

    def on_key(event):
        if event.key == 'enter':
            plt.close(fig)
        elif event.key in ('q', 'escape'):
            quit_flag[0] = True
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

    return quit_flag[0]


# ────────────────────────────────────────────── SAVE BATCH REPORT
def save_summary_csv(reports, output_dir='outputs_week6'):
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'summary_week6.csv')

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image_name", "detected", "method", "sharpness", "skew_deg", "status"]
        )
        writer.writeheader()
        writer.writerows(reports)

    print(f"Summary saved to: {csv_path}")


# ────────────────────────────────────────────── MAIN
if __name__ == "__main__":
    dataset_dir = 'dataset'
    output_dir = 'outputs_week6'

    supported_ext = ('.jpg', '.jpeg', '.png')
    image_files = sorted([
        f for f in os.listdir(dataset_dir)
        if f.lower().endswith(supported_ext)
    ])

    if not image_files:
        print(f"No images found in '{dataset_dir}/'.")
        exit(1)

    total = len(image_files)
    print(f"Found {total} image(s) in '{dataset_dir}/'.\n")

    reports = []

    for idx, filename in enumerate(image_files, start=1):
        path = os.path.join(dataset_dir, filename)
        print(f"\n{'='*60}")
        print(f"Processing image {idx}/{total}: {filename}")
        print(f"{'='*60}")

        steps, report = process_image(path, save_outputs=True, output_dir=output_dir)
        reports.append(report)

        if steps:
            should_quit = show_steps_grid(steps, filename, idx, total)
            if should_quit:
                print("\nExecution stopped by user (Q / Escape).")
                break
        else:
            print(f"Skipping display for {filename} (no steps generated).")

    save_summary_csv(reports, output_dir=output_dir)
    print("\nAll images processed.")
