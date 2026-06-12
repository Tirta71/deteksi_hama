import os

import cv2
from flask import current_app, jsonify, request, session
from werkzeug.utils import secure_filename

from simasha.services.history_service import create_detection_result
from simasha.services.image_processing_service import resize_to_640
from simasha.services.yolo_service import detect_method_and_save, run_yolo_and_save
from simasha.utils.file_utils import cleanup_dir, unique_name
from simasha.validators.upload_validator import validate_upload


def handle_detect_both():
    config = current_app.config
    model = current_app.extensions["yolo_model"]

    f = request.files.get("file")
    upload_error = validate_upload(f)

    if upload_error:
        return jsonify({
            "status": "error",
            "message": upload_error,
        }), 400

    fname = secure_filename(f.filename)
    save_path = os.path.join(config["UPLOAD_FOLDER"], fname)
    f.save(save_path)

    img = cv2.imread(save_path)

    if img is None:
        return jsonify({
            "status": "error",
            "message": "Gagal membaca gambar",
        }), 400

    img640 = resize_to_640(img, config["IMGSZ"])

    out_full_name = unique_name("full")
    out_full_path = os.path.join(
        config["RESULTS_DIR"],
        out_full_name,
    )

    full_summary = run_yolo_and_save(
        img640,
        out_full_path,
        model,
        config,
    )

    yolo_full_url = f"/{out_full_path.replace(os.sep, '/')}"

    final_name = unique_name("final_method")
    final_path = os.path.join(
        config["RESULTS_DIR"],
        final_name,
    )

    crop_summary = detect_method_and_save(
        img,
        final_path,
        model,
        config,
    )

    yolo_crop_url = f"/{final_path.replace(os.sep, '/')}"

    create_detection_result(
        user_id=session.get("user_id"),
        filename=fname,
        full_result_image=out_full_path.replace(os.sep, "/"),
        full_summary=full_summary,
        crop_result_image=final_path.replace(os.sep, "/"),
        crop_summary=crop_summary,
    )

    cleanup_dir(config["CROPS_DIR"])
    cleanup_dir(config["RESULTS_DIR"])

    return jsonify({
        "status": "success",
        "filename": fname,
        "yolo_full": yolo_full_url,
        "yolo_crop": yolo_crop_url,
        "full_summary": full_summary,
        "crop_summary": crop_summary,
        "process_images": crop_summary.get("process_images"),
        "per_grid_results": crop_summary.get("per_grid_results", []),
    })
