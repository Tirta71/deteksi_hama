import cv2
import numpy as np


def resize_to_640(img_bgr, imgsz):
    if img_bgr is None:
        return None

    return cv2.resize(
        img_bgr,
        (imgsz, imgsz),
        interpolation=cv2.INTER_LANCZOS4,
    )


def preprocessing_contour_masking(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    lower_green = np.array([30, 50, 50], dtype=np.uint8)
    upper_green = np.array([85, 255, 255], dtype=np.uint8)

    hsv_mask = cv2.inRange(
        hsv,
        lower_green,
        upper_green,
    )

    kernel = np.ones((7, 7), np.uint8)

    morphology = cv2.morphologyEx(
        hsv_mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=2,
    )

    morphology = cv2.morphologyEx(
        morphology,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=3,
    )

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        morphology,
        connectivity=8,
    )

    if num_labels <= 1:
        black = np.zeros_like(img_bgr)
        return img_bgr, hsv_mask, morphology, black, img_bgr

    largest_label = 1
    largest_area = stats[1, cv2.CC_STAT_AREA]

    for i in range(2, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        if area > largest_area:
            largest_area = area
            largest_label = i

    largest_component_mask = np.zeros_like(morphology)
    largest_component_mask[labels == largest_label] = 255

    largest_component_mask = cv2.morphologyEx(
        largest_component_mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2,
    )

    largest_component_mask = cv2.GaussianBlur(
        largest_component_mask,
        (5, 5),
        0,
    )

    _, largest_component_mask = cv2.threshold(
        largest_component_mask,
        127,
        255,
        cv2.THRESH_BINARY,
    )

    contours, _ = cv2.findContours(
        largest_component_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        black = np.zeros_like(img_bgr)
        return img_bgr, hsv_mask, morphology, black, img_bgr

    largest = max(contours, key=cv2.contourArea)

    if cv2.contourArea(largest) < 100:
        black = np.zeros_like(img_bgr)
        return img_bgr, hsv_mask, morphology, black, img_bgr

    contour_mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)

    cv2.drawContours(
        contour_mask,
        [largest],
        -1,
        255,
        thickness=cv2.FILLED,
    )

    contour_masking = cv2.bitwise_and(
        img_bgr,
        img_bgr,
        mask=contour_mask,
    )

    x, y, w, h = cv2.boundingRect(largest)

    leaf_crop = contour_masking[y:y + h, x:x + w]

    if leaf_crop.size == 0:
        leaf_crop = contour_masking

    return img_bgr, hsv_mask, morphology, contour_masking, leaf_crop


def split_grid_with_offsets(img_bgr, rows=4, cols=4):
    h, w = img_bgr.shape[:2]

    cell_h = h // rows
    cell_w = w // cols

    grids = []

    for r in range(rows):
        for c in range(cols):
            x1 = c * cell_w
            y1 = r * cell_h

            x2 = w if c == cols - 1 else (c + 1) * cell_w
            y2 = h if r == rows - 1 else (r + 1) * cell_h

            grid_img = img_bgr[y1:y2, x1:x2]

            grids.append({
                "grid": (r * cols) + c + 1,
                "img": grid_img,
                "x_offset": x1,
                "y_offset": y1,
                "w": x2 - x1,
                "h": y2 - y1,
            })

    return grids


def make_grid_preview(img_bgr, rows=4, cols=4):
    preview = img_bgr.copy()

    h, w = preview.shape[:2]

    cell_h = h // rows
    cell_w = w // cols

    for r in range(1, rows):
        y = r * cell_h

        cv2.line(
            preview,
            (0, y),
            (w, y),
            (0, 0, 255),
            2,
        )

    for c in range(1, cols):
        x = c * cell_w

        cv2.line(
            preview,
            (x, 0),
            (x, h),
            (0, 0, 255),
            2,
        )

    return preview


def make_zoom_crop_preview(grids, grid_rows, grid_cols, imgsz):
    previews = []

    for item in grids:
        grid_img = item["img"]

        if grid_img is None or grid_img.size == 0:
            continue

        zoomed = resize_to_640(grid_img, imgsz)

        zoomed_small = cv2.resize(
            zoomed,
            (220, 220),
            interpolation=cv2.INTER_AREA,
        )

        zoomed_small = cv2.copyMakeBorder(
            zoomed_small,
            3,
            3,
            3,
            3,
            cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        )

        previews.append(zoomed_small)

    if not previews:
        return None

    total = grid_rows * grid_cols

    while len(previews) < total:
        previews.append(np.zeros_like(previews[0]))

    rows = []
    idx = 0

    for _ in range(grid_rows):
        row_imgs = []

        for _ in range(grid_cols):
            row_imgs.append(previews[idx])
            idx += 1

        rows.append(np.hstack(row_imgs))

    return np.vstack(rows)


def make_yolo_grid_preview(grid_result_images, grid_rows, grid_cols):
    previews = []

    for img in grid_result_images:
        if img is None or img.size == 0:
            continue

        small = cv2.resize(
            img,
            (220, 220),
            interpolation=cv2.INTER_AREA,
        )

        small = cv2.copyMakeBorder(
            small,
            3,
            3,
            3,
            3,
            cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        )

        previews.append(small)

    if not previews:
        return None

    total = grid_rows * grid_cols

    while len(previews) < total:
        previews.append(np.zeros_like(previews[0]))

    rows = []
    idx = 0

    for _ in range(grid_rows):
        row_imgs = []

        for _ in range(grid_cols):
            row_imgs.append(previews[idx])
            idx += 1

        rows.append(np.hstack(row_imgs))

    return np.vstack(rows)


def nms_merge_boxes(boxes, scores, score_thr=0.0, nms_thr=0.5):
    if len(boxes) == 0:
        return []

    xywh = []

    for box in boxes:
        x1, y1, x2, y2 = box

        xywh.append([
            int(x1),
            int(y1),
            int(x2 - x1),
            int(y2 - y1),
        ])

    indices = cv2.dnn.NMSBoxes(
        bboxes=xywh,
        scores=scores,
        score_threshold=score_thr,
        nms_threshold=nms_thr,
    )

    if len(indices) == 0:
        return []

    return np.array(indices).flatten().tolist()
