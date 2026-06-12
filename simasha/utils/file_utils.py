import os
import time
import uuid

import cv2


def ensure_runtime_dirs(config):
    os.makedirs(config["STATIC_DIR"], exist_ok=True)
    os.makedirs(config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(config["CROPS_DIR"], exist_ok=True)
    os.makedirs(config["RESULTS_DIR"], exist_ok=True)


def unique_name(prefix, ext=".jpg"):
    return f"{prefix}_{int(time.time())}_{uuid.uuid4().hex[:8]}{ext}"


def save_process_image(img, results_dir, prefix):
    name = unique_name(prefix)
    path = os.path.join(results_dir, name)

    cv2.imwrite(path, img)

    return path.replace(os.sep, "/")


def cleanup_dir(dir_path, max_age_hours=24, keep_min=100):
    try:
        files = [
            os.path.join(dir_path, f)
            for f in os.listdir(dir_path)
        ]

        files = [
            f for f in files
            if os.path.isfile(f)
        ]

        if len(files) <= keep_min:
            return

        now = time.time()

        for f in files:
            try:
                age_h = (now - os.path.getmtime(f)) / 3600.0

                if age_h > max_age_hours:
                    os.remove(f)

            except Exception:
                pass

    except Exception:
        pass
