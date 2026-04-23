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
        print("Error: Could not decode image.")
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
    print("CLAHE applied.")
    return enhanced


# ────────────────────────────────────────────── 4. OTSU THRESHOLDING
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


# ────────────────────────────────────────────── 6b. EDGE DETECTION ROBUST
# Metoda noua pentru detectarea marginilor documentelor 
# Pentru detectia documentelor mai dificile (margini incomoplete/fundaluri mai zgomotoase)
def enhance_for_contours_robust(gray_image):
    if gray_image is None:
        return None
    #bilateralFilter ca sa reducem zgomotul fara sa pierdem muchiile
    blurred = cv2.bilateralFilter(gray_image, 9, 75, 75) 
    edged = cv2.Canny(blurred, 30, 100) #Canny cu alte praguri
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel) # inchidem golurile din contur
    dilated = cv2.dilate(closed, np.ones((3, 3), np.uint8), iterations=1)
    return dilated


# ────────────────────────────────────────────── 6c. EDGE DETECTION ADAPTIVE
# Aplica binarizare adaptiva si operatii morfologice pentru a evidentia mai bine documentul
# in cond de iluminare neuniforma / fundal neuninform
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
# [MODIFICARE] prag mai mic de arie pentru documente mai departate
# [MODIFICARE] fill_ratio simplu ca sa respinga mai bine bucati de text
def is_good_contour(contour, image_shape, min_area_ratio=0.08):
    if contour is None or len(contour) != 4:
        return False

    area = cv2.contourArea(contour)
    img_area = image_shape[0] * image_shape[1]
    if not (min_area_ratio < (area / img_area) < 0.95):
        return False

    if not cv2.isContourConvex(contour):
        return False

    rect = cv2.minAreaRect(contour)
    w, h = rect[1]
    if min(w, h) == 0:
        return False

    aspect_ratio = max(w, h) / min(w, h)
    if aspect_ratio > 3.0:
        return False

    rect_area = w * h
    if rect_area <= 0:
        return False

    fill_ratio = area / rect_area
    if fill_ratio < 0.45:
        return False

    return True


# ────────────────────────────────────────────── 7. HOUGH CORNER FALLBACK
def hough_corner_fallback(edged_image, image_shape):
    lines = cv2.HoughLinesP(
        edged_image, 1, np.pi / 180, threshold=80,
        minLineLength=int(min(image_shape[:2]) * 0.20), maxLineGap=20
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
    contour_2, all_2 = extract_contour_from_edges(edges_adaptive)
    if contour_2 is not None:
        print("Gasit prin: Strategia 2 (Adaptive Edge)")
        return contour_2, all_2, "Adaptive Edge", edges_adaptive

    blurred = cv2.GaussianBlur(gray_image, (11, 11), 0)
    _, thresh_img = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contour_3, all_3 = extract_contour_from_edges(thresh_img)
    if contour_3 is not None:
        print("Gasit prin: Strategia 3 (Otsu Blob)")
        return contour_3, all_3, "Otsu Blob", thresh_img

    # [MODIFICARE W4] fallback-ul Hough + corners este folosit efectiv
    edges_hough = enhance_for_contours(gray_image)
    hough_box = hough_corner_fallback(edges_hough, image_shape)
    if hough_box is not None and is_good_contour(hough_box, image_shape, min_area_ratio=0.05):
        print("Gasit prin: Strategia 4 (Hough+Corners)")
        contours_h, _ = cv2.findContours(edges_hough.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours_h = sorted(contours_h, key=cv2.contourArea, reverse=True)
        return hough_box, contours_h, "Hough+Corners", edges_hough

    print("Incerc fallback cu MinAreaRect pe pata Otsu.")
    contours_otsu, _ = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours_otsu:
        sorted_otsu = sorted(contours_otsu, key=cv2.contourArea, reverse=True)
        for c in sorted_otsu[:5]:
            area = cv2.contourArea(c)
            if 0.08 < (area / img_area) < 0.95:
                rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rect).astype("int").reshape(4, 1, 2)
                if is_good_contour(box, image_shape):
                    return box, sorted_otsu, "MinAreaRect (Otsu)", thresh_img

    print("Incerc fallback cu MinAreaRect pe limitele exterioare (Edges).")
    for c in all_1[:10]:
        rect = cv2.minAreaRect(c)
        w, h = rect[1]
        rect_area = w * h

        if 0.08 < (rect_area / img_area) < 0.95:
            aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 100
            if aspect_ratio < 3.0:
                box = cv2.boxPoints(rect).astype("int").reshape(4, 1, 2)
                if is_good_contour(box, image_shape):
                    return box, all_1, "MinAreaRect (Edges)", edges_1

    print("Eroare: Niciun contur valid nu a fost găsit.")
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
    print(f"Quality → sharpness: {sharpness:.2f} ({status}) | skew: {skew_deg:.2f}°")
    return {"sharpness": sharpness, "skew_deg": skew_deg, "status": status}


# ────────────────────────────────────────────── 12. [W7] DE-SHADOWING 
# reducerea umbrelor din document
def apply_deshadow(gray_image):
    if gray_image is None:
        return None

    h, w = gray_image.shape[:2]
    k = max(21, (min(h, w) // 12) | 1)
    background = cv2.GaussianBlur(gray_image, (k, k), 0)
    result = cv2.divide(gray_image, background, scale=255)

    print("De-shadowing applied.")
    return result


# ────────────────────────────────────────────── 13. [W9] COMPARATIE SIMPLA OTSU vs ADAPTIVE
def black_ratio(binary_image):
    total = binary_image.shape[0] * binary_image.shape[1]
    return 1.0 - (cv2.countNonZero(binary_image) / total)

# Alegem intre Otsu si Adaptive in functie de care e mai apropiata ca proportie de negru de 0.15
def choose_preferred_scan(otsu_scan, adaptive_scan):
    otsu_ratio = black_ratio(otsu_scan)
    adaptive_ratio = black_ratio(adaptive_scan)

    # alegere simplă: vrem o imagine nici prea albă, nici prea neagră
    if abs(adaptive_ratio - 0.15) < abs(otsu_ratio - 0.15):
        print("Preferred scan: Adaptive")
        return "Adaptive", adaptive_scan, otsu_ratio, adaptive_ratio

    print("Preferred scan: Otsu")
    return "Otsu", otsu_scan, otsu_ratio, adaptive_ratio


# ────────────────────────────────────────────── 14. [W8/W9] ETICHETE SIMPLE PENTRU EXPERIMENTE
def estimate_scene_conditions(gray_image):
    mean_brightness = float(np.mean(gray_image))
    edges = cv2.Canny(gray_image, 50, 150)
    edge_density = float(np.count_nonzero(edges) / edges.size)

    if mean_brightness < 90:
        lighting = "dark"
    elif mean_brightness > 180:
        lighting = "bright"
    else:
        lighting = "normal"

    background = "cluttered" if edge_density > 0.08 else "simple"

    return lighting, background, mean_brightness, edge_density


# ────────────────────────────────────────────── 15. [W8/W9] AUTO-REJECT
# Respingerea imaginii daca e prea neclara/inclinata sau are binarizare slaba
def decide_rejection(quality, preferred_ratio):
    reasons = []

    if quality["sharpness"] < 60:
        reasons.append("low_sharpness")
    if abs(quality["skew_deg"]) > 20:
        reasons.append("high_skew")
    if not (0.03 < preferred_ratio < 0.35):
        reasons.append("poor_binarization")

    if reasons:
        return "Rejected", ",".join(reasons)
    return "Accepted", "-"


# ────────────────────────────────────────────── HELPER: to RGB
def to_rgb(img, is_gray=False):
    if img is None:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    if is_gray or len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ────────────────────────────────────────────── 16. PROCESS ONE IMAGE
def process_image(path, save_outputs=False, output_dir='outputs_week9'):
    steps = []
    report = {
        "image_name": os.path.basename(path),
        "detected": False,
        "method": "",
        "preferred_scan": "",
        "sharpness": "",
        "skew_deg": "",
        "status": "",
        "decision": "",
        "reject_reason": "",
        "lighting": "",
        "background": "",
        "otsu_black_ratio": "",
        "adaptive_black_ratio": ""
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

    lighting, background, mean_brightness, edge_density = estimate_scene_conditions(gray)
    report["lighting"] = lighting
    report["background"] = background

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

        # [MODIFICARE W7] de-shadowing simplu înainte de CLAHE
        warped_deshadow = apply_deshadow(warped_gray)
        steps.append(("7. De-shadowed", to_rgb(warped_deshadow, is_gray=True)))

        warped_clahe = apply_clahe(warped_deshadow)
        steps.append(("8. CLAHE", to_rgb(warped_clahe, is_gray=True)))

        otsu_scan = apply_thresholding(warped_clahe)
        adaptive_scan = apply_adaptive_thresholding(warped_clahe)
        steps.append(("9. Scan (Otsu)", to_rgb(otsu_scan, is_gray=True)))
        steps.append(("10. Scan (Adaptive)", to_rgb(adaptive_scan, is_gray=True)))

        preferred_name, preferred_scan, otsu_ratio, adaptive_ratio = choose_preferred_scan(otsu_scan, adaptive_scan)
        report["preferred_scan"] = preferred_name
        report["otsu_black_ratio"] = f"{otsu_ratio:.4f}"
        report["adaptive_black_ratio"] = f"{adaptive_ratio:.4f}"

        steps.append((f"11. Preferred Scan ({preferred_name})", to_rgb(preferred_scan, is_gray=True)))

        quality = calculate_quality_score(warped)
        report["sharpness"] = f"{quality['sharpness']:.2f}"
        report["skew_deg"] = f"{quality['skew_deg']:.2f}"
        report["status"] = quality["status"]

        preferred_ratio = adaptive_ratio if preferred_name == "Adaptive" else otsu_ratio
        decision, reject_reason = decide_rejection(quality, preferred_ratio)
        report["decision"] = decision
        report["reject_reason"] = reject_reason

        if save_outputs:
            os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(path))[0]
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_warped.jpg"), warped)
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_deshadow.jpg"), warped_deshadow)
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_otsu.jpg"), otsu_scan)
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_adaptive.jpg"), adaptive_scan)
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_preferred.jpg"), preferred_scan)
    else:
        steps.append(("5. No contour found", to_rgb(image_resized)))
        report["decision"] = "Rejected"
        report["reject_reason"] = "no_contour"
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


# ────────────────────────────────────────────── 17. SAVE SUMMARY CSV
def save_summary_csv(reports, output_dir='outputs_week9'):
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'summary_week9.csv')

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_name", "detected", "method", "preferred_scan",
                "sharpness", "skew_deg", "status",
                "decision", "reject_reason",
                "lighting", "background",
                "otsu_black_ratio", "adaptive_black_ratio"
            ]
        )
        writer.writeheader()
        writer.writerows(reports)

    print(f"Summary saved to: {csv_path}")


# ────────────────────────────────────────────── 18. [W9] EXPERIMENTS CSV 
def save_experiments_csv(reports, output_dir='outputs_week9'):
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'experiments_week9.csv')

    grouped = {}
    for r in reports:
        key = (r["lighting"], r["background"])
        if key not in grouped:
            grouped[key] = {"total": 0, "accepted": 0}

        grouped[key]["total"] += 1
        if r["decision"] == "Accepted":
            grouped[key]["accepted"] += 1

    rows = []
    for (lighting, background), info in grouped.items():
        total = info["total"]
        accepted = info["accepted"]
        rate = (accepted / total) * 100 if total else 0.0

        rows.append({
            "lighting": lighting,
            "background": background,
            "total_images": total,
            "accepted_images": accepted,
            "accepted_rate_%": f"{rate:.2f}"
        })

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["lighting", "background", "total_images", "accepted_images", "accepted_rate_%"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Experiments saved to: {csv_path}")


# ────────────────────────────────────────────── MAIN
if __name__ == "__main__":
    dataset_dir = 'dataset'
    output_dir = 'outputs_week9'
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
    save_experiments_csv(reports, output_dir=output_dir)
    print("\nAll images processed.")
