import os
import time

import cv2

from simasha.services.image_processing_service import (
    make_grid_preview,
    make_yolo_grid_preview,
    make_zoom_crop_preview,
    nms_merge_boxes,
    preprocessing_contour_masking,
    resize_to_640,
    split_grid_with_offsets,
)
from simasha.utils.file_utils import save_process_image


def get_class_name(names, cls_id):
    if isinstance(names, dict):
        return names.get(cls_id, str(cls_id))

    if isinstance(names, list) and 0 <= cls_id < len(names):
        return names[cls_id]

    return str(cls_id)


def summarize_result(ultra_res, duration_ms, model):
    boxes = ultra_res.boxes
    names = ultra_res.names or model.names

    n = int(boxes.shape[0]) if boxes is not None else 0

    classes = []
    confs = []

    if n > 0:
        for i in range(n):
            cls_id = int(boxes.cls[i].item())
            conf = float(boxes.conf[i].item())

            classes.append(get_class_name(names, cls_id))
            confs.append(conf)

    avg_conf = round(sum(confs) / len(confs), 4) if confs else 0.0

    return {
        "num_boxes": n,
        "classes": classes,
        "avg_conf": avg_conf,
        "duration_ms": int(round(duration_ms)),
    }


def run_yolo_and_save(img_bgr, out_path, model, config):
    t0 = time.perf_counter()

    res = model.predict(
        source=img_bgr,
        conf=config["CONF_THR"],
        iou=config["IOU_THR"],
        imgsz=config["IMGSZ"],
        verbose=False,
    )

    duration_ms = (time.perf_counter() - t0) * 1000.0

    rendered = res[0].plot()
    cv2.imwrite(out_path, rendered)

    return summarize_result(res[0], duration_ms, model)


def detect_method_and_save(img_bgr, out_path, model, config):
    t0 = time.perf_counter()

    original_img, hsv_mask, morphology_img, contour_masking, leaf_img = (
        preprocessing_contour_masking(img_bgr)
    )

    process_images = {}

    process_images["original"] = save_process_image(
        original_img,
        config["RESULTS_DIR"],
        "process_01_original",
    )

    process_images["hsv_mask"] = save_process_image(
        hsv_mask,
        config["RESULTS_DIR"],
        "process_02_hsv",
    )

    process_images["morphology"] = save_process_image(
        morphology_img,
        config["RESULTS_DIR"],
        "process_03_morphology",
    )

    process_images["contour_masking"] = save_process_image(
        contour_masking,
        config["RESULTS_DIR"],
        "process_04_contour_masking",
    )

    grids = split_grid_with_offsets(
        leaf_img,
        rows=config["GRID_ROWS"],
        cols=config["GRID_COLS"],
    )

    grid_preview = make_grid_preview(
        leaf_img,
        rows=config["GRID_ROWS"],
        cols=config["GRID_COLS"],
    )

    process_images["grid"] = save_process_image(
        grid_preview,
        config["RESULTS_DIR"],
        "process_05_grid",
    )

    zoom_crop_preview = make_zoom_crop_preview(
        grids,
        config["GRID_ROWS"],
        config["GRID_COLS"],
        config["IMGSZ"],
    )

    if zoom_crop_preview is not None:
        process_images["zoom_crop"] = save_process_image(
            zoom_crop_preview,
            config["RESULTS_DIR"],
            "process_06_zoom_crop",
        )
    else:
        process_images["zoom_crop"] = process_images["grid"]

    all_boxes = []
    all_scores = []
    all_cls_ids = []

    grid_yolo_images = []
    per_grid_results = []

    for item in grids:
        grid_index = item["grid"]
        grid_img = item["img"]

        if grid_img is None or grid_img.size == 0:
            continue

        gh, gw = grid_img.shape[:2]

        zoomed_grid = resize_to_640(grid_img, config["IMGSZ"])

        result = model.predict(
            source=zoomed_grid,
            conf=config["CONF_THR"],
            iou=config["IOU_THR"],
            imgsz=config["IMGSZ"],
            verbose=False,
        )

        boxes = result[0].boxes

        grid_rendered = result[0].plot()
        grid_yolo_images.append(grid_rendered)

        grid_result_path = save_process_image(
            grid_rendered,
            config["RESULTS_DIR"],
            f"grid_{grid_index:02d}_yolo",
        )

        grid_classes = []
        grid_scores = []

        if boxes is not None and len(boxes) > 0:
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()

                x1, y1, x2, y2 = xyxy

                scale_x = gw / config["IMGSZ"]
                scale_y = gh / config["IMGSZ"]

                x1 = x1 * scale_x + item["x_offset"]
                x2 = x2 * scale_x + item["x_offset"]
                y1 = y1 * scale_y + item["y_offset"]
                y2 = y2 * scale_y + item["y_offset"]

                conf = float(boxes.conf[i].item())
                cls_id = int(boxes.cls[i].item())
                class_name = get_class_name(model.names, cls_id)

                grid_classes.append(class_name)
                grid_scores.append(conf)

                all_boxes.append([x1, y1, x2, y2])
                all_scores.append(conf)
                all_cls_ids.append(cls_id)

        per_grid_results.append({
            "grid": grid_index,
            "image": grid_result_path,
            "num_boxes": len(grid_scores),
            "classes": grid_classes,
            "avg_conf": (
                round(sum(grid_scores) / len(grid_scores), 4)
                if grid_scores
                else 0.0
            ),
        })

    yolo_grid_preview = make_yolo_grid_preview(
        grid_yolo_images,
        config["GRID_ROWS"],
        config["GRID_COLS"],
    )

    if yolo_grid_preview is not None:
        process_images["yolo_detection"] = save_process_image(
            yolo_grid_preview,
            config["RESULTS_DIR"],
            "process_07_yolo_detection",
        )
    else:
        process_images["yolo_detection"] = process_images["zoom_crop"]

    keep_indices = nms_merge_boxes(
        boxes=all_boxes,
        scores=all_scores,
        score_thr=0.0,
        nms_thr=config["IOU_THR"],
    )

    final_result = leaf_img.copy()

    final_classes = []
    final_scores = []

    names = model.names

    for idx in keep_indices:
        x1, y1, x2, y2 = all_boxes[idx]
        score = all_scores[idx]
        cls_id = all_cls_ids[idx]

        class_name = get_class_name(names, cls_id)

        final_classes.append(class_name)
        final_scores.append(score)

        x1 = int(max(0, x1))
        y1 = int(max(0, y1))
        x2 = int(min(final_result.shape[1] - 1, x2))
        y2 = int(min(final_result.shape[0] - 1, y2))

        cv2.rectangle(
            final_result,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        label = f"{class_name} {score:.2f}"

        cv2.putText(
            final_result,
            label,
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    cv2.imwrite(out_path, final_result)

    process_images["final_result"] = out_path.replace(os.sep, "/")

    duration_ms = (time.perf_counter() - t0) * 1000.0

    avg_conf = (
        round(sum(final_scores) / len(final_scores), 4)
        if final_scores
        else 0.0
    )

    return {
        "num_boxes": len(keep_indices),
        "classes": final_classes,
        "avg_conf": avg_conf,
        "duration_ms": int(round(duration_ms)),
        "grid": f"{config['GRID_ROWS']}x{config['GRID_COLS']}",
        "method": (
            "HSV Color Masking + Morphological Operation + "
            "Contour Masking + Griding + Zoom-In Crop + "
            "YOLOv8 Detection Per Grid + Hasil Akhir"
        ),
        "process_images": process_images,
        "per_grid_results": per_grid_results,
    }
