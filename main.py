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

# ────────────────────────────────────────────── 2. GRAYSCALE
def convert_to_grayscale(image):
    if image is None:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print("Grayscale conversion complete.")
    return gray

# ────────────────────────────────────────────── 3. CLAHE
def apply_clahe(gray_image, clip_limit=2.0, tile_grid=(8, 8)):
    if gray_image is None:
        return None
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    enhanced = clahe.apply(gray_image)
    print(f"CLAHE applied.")
    return enhanced

# ────────────────────────────────────────────── 4. THRESHOLDING
def apply_thresholding(gray_image):
    if gray_image is None:
        return None
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print("Otsu thresholding complete.")
    return binary

# ────────────────────────────────────────────── 5. ADAPTIVE THRESHOLDING
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

# ────────────────────────────────────────────── 6a. EDGE DETECTION ORIGINAL
def enhance_for_contours(gray_image):
    if gray_image is None:
        return None
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edged, kernel, iterations=2)
    return dilated

# ────────────────────────────────────────────── 6b. EDGE DETECTION ROBUST (fara zgomot)
def enhance_for_contours_robust(gray_image):
    if gray_image is None:
        return None
    blurred = cv2.bilateralFilter(gray_image, 9, 75, 75)
    edged = cv2.Canny(blurred, 30, 100)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    
    dilated = cv2.dilate(closed, np.ones((3, 3), np.uint8), iterations=1)
    return dilated

# ────────────────────────────────────────────── 6c. EDGE DETECTION ADAPTIVE
def enhance_for_contours_adaptive(gray_image):
    if gray_image is None:
        return None
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    thresh = cv2.bitwise_not(thresh)
    
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    return closed

# ────────────────────────────────────────────── HELPER: VALIDARE CONTUR
def is_good_contour(contour, image_shape):
    if contour is None or len(contour) != 4:
        return False
    
    area = cv2.contourArea(contour)
    img_area = image_shape[0] * image_shape[1]
    
    if not (0.15 < (area / img_area) < 0.95):
        return False
        
    if not cv2.isContourConvex(contour):
        return False
        
    rect = cv2.minAreaRect(contour)
    (w, h) = rect[1]
    if min(w, h) == 0: 
        return False
    aspect_ratio = max(w, h) / min(w, h)
    
    if aspect_ratio > 2.5: 
        return False
        
    return True


# ────────────────────────────────────────────── 7. HOUGH CORNER FALLBACK 
def hough_corner_fallback(edged_image, image_shape):
    lines = cv2.HoughLinesP(
        edged_image, 1, np.pi / 180, threshold=80,
        minLineLength=int(min(image_shape[:2]) * 0.25), maxLineGap=20
    )

    if lines is None:
        return None

    line_mask = np.zeros_like(edged_image)
    for line in lines[:, 0]:
        x1, y1, x2, y2 = line
        cv2.line(line_mask, (x1, y1), (x2, y2), 255, 3)

    line_mask = cv2.dilate(line_mask, np.ones((5, 5), np.uint8), iterations=2)

    corners = cv2.goodFeaturesToTrack(
        line_mask, maxCorners=40, qualityLevel=0.01, minDistance=20
    )

    if corners is None or len(corners) < 4:
        return None

    points = corners.reshape(-1, 2).astype(np.float32)
    rect = cv2.minAreaRect(points)
    box = cv2.boxPoints(rect).astype("int").reshape(4, 1, 2)
    return box

# ────────────────────────────────────────────── 8. FIND DOCUMENT (MULTI-PASS)
def find_document_contour(gray_image, image_shape):
    MAX_CANDIDATES = 10
    EPSILONS = [0.02, 0.04, 0.06, 0.08, 0.10] 
    img_area = image_shape[0] * image_shape[1]

    def extract_contour_from_edges(edges_img):
        contours, _ = cv2.findContours(edges_img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        for c in sorted_contours[:MAX_CANDIDATES]:
            peri = cv2.arcLength(c, True)
            for eps in EPSILONS:
                approx = cv2.approxPolyDP(c, eps * peri, True)
                if is_good_contour(approx, image_shape):
                    return approx, sorted_contours
        return None, sorted_contours

    edges_1 = enhance_for_contours_robust(gray_image)
    contour_1, all_1 = extract_contour_from_edges(edges_1)
    if contour_1 is not None:
        print("Gasit prin: Strategia 1 (Bilateral+Close)")
        return contour_1, all_1, "Bilateral+Close", edges_1

    edges_adaptive = enhance_for_contours_adaptive(gray_image)
    contour_adap, all_adap = extract_contour_from_edges(edges_adaptive)
    if contour_adap is not None:
        print("Gasit prin: Strategia 2 (Adaptive Edge)")
        return contour_adap, all_adap, "Adaptive Edge", edges_adaptive

    blurred = cv2.GaussianBlur(gray_image, (11, 11), 0)
    _, thresh_img = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contour_3, all_3 = extract_contour_from_edges(thresh_img)
    if contour_3 is not None:
        print("Gasit prin: Strategia 3 (Otsu Blob)")
        return contour_3, all_3, "Otsu Blob", thresh_img

    print("Încerc fallback cu MinAreaRect pe pata Otsu.")
    contours_otsu, _ = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours_otsu:
        sorted_otsu = sorted(contours_otsu, key=cv2.contourArea, reverse=True)
        for c in sorted_otsu[:5]:
            area = cv2.contourArea(c)
            if 0.15 < (area / img_area) < 0.95:
                rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rect).astype("int").reshape(4, 1, 2)
                return box, sorted_otsu, "MinAreaRect (Otsu)", thresh_img
                
    print("Incerc fallback cu MinAreaRect pe limitele exterioare (Edges).")
    for c in all_1[:10]:
        rect = cv2.minAreaRect(c)
        w, h = rect[1]
        rect_area = w * h  
        
        # Verificăm dacă cutia care încadrează aceste margini e destul de mare
        if 0.15 < (rect_area / img_area) < 0.95:
            aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 100
            if aspect_ratio < 3.0:
                box = cv2.boxPoints(rect).astype("int").reshape(4, 1, 2)
                return box, all_1, "MinAreaRect (Edges)", edges_1

    print("Eroare: Niciun contur valid nu a fost gasit in nicio strategie.")
    return None, all_1, "none", edges_1

# ────────────────────────────────────────────── 9. ORDER POINTS
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

# ────────────────────────────────────────────── 10. PERSPECTIVE TRANSFORM
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

# ────────────────────────────────────────────── 11. QUALITY SCORE
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
    print(f"Quality --> sharpness: {sharpness:.2f} ({status}) | skew: {skew_deg:.2f}°")
    return {"sharpness": sharpness, "skew_deg": skew_deg, "status": status}

# ────────────────────────────────────────────── HELPER: to RGB
def to_rgb(img, is_gray=False):
    if img is None:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    if is_gray or len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ────────────────────────────────────────────── PROCESS ONE IMAGE
def process_image(path, save_outputs=False, output_dir='outputs_week6'):
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
    
    orig = image.copy()
    HEIGHT_TARGET = 800.0
    ratio = image.shape[0] / HEIGHT_TARGET
    image_resized = cv2.resize(image, (int(image.shape[1] / ratio), int(HEIGHT_TARGET)))

    steps.append(("1. Original", to_rgb(image_resized)))

    gray = convert_to_grayscale(image_resized)
    steps.append(("2. Grayscale", to_rgb(gray, is_gray=True)))

    doc_contour, all_contours, method, best_edges = find_document_contour(gray, image_resized.shape)
    
    steps.append((f"3. Edges/Mask ({method})", to_rgb(best_edges, is_gray=True)))

    img_debug = image_resized.copy()
    cv2.drawContours(img_debug, all_contours[:10], -1, (0, 255, 255), 2)
    steps.append(("4. Top 10 Contours", to_rgb(img_debug)))

    if doc_contour is not None:
        report["detected"] = True
        report["method"] = method

        img_selection = image_resized.copy()
        cv2.drawContours(img_selection, [doc_contour], -1, (0, 255, 0), 3)
        steps.append((f"5. Boundary ({method})", to_rgb(img_selection)))

        pts_orig = doc_contour.reshape(4, 2).astype("float32") * ratio
        warped = perspective_transform(orig, pts_orig)
        steps.append(("6. Perspective Corrected", to_rgb(warped)))

        warped_gray = convert_to_grayscale(warped)
        warped_clahe = apply_clahe(warped_gray)
        steps.append(("7. Crop + CLAHE (Text Enhance)", to_rgb(warped_clahe, is_gray=True)))

        otsu_scan = apply_thresholding(warped_clahe)
        adaptive_scan = apply_adaptive_thresholding(warped_clahe)

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
        steps.append(("5. No contour found", to_rgb(image_resized)))
        print("Pipeline stopped: no valid document contour found.")

    return steps, report

# ────────────────────────────────────────────── DISPLAY 
def show_steps_grid(steps, image_name, image_index, total_images):
    n = len(steps)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 3.5))
    fig.suptitle(
        f"[{image_index}/{total_images}]  {image_name}   —   Enter = urmatoarea imagine  |  Q / Esc = iesire",
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

# ────────────────────────────────────────────── SAVE REPORT 
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
    
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
        
    image_files = sorted([
        f for f in os.listdir(dataset_dir)
        if f.lower().endswith(supported_ext)
    ])

    if not image_files:
        print(f"No images found in '{dataset_dir}/'. Please add images to test.")
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
