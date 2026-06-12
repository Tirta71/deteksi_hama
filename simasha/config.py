import os

from dotenv import load_dotenv


load_dotenv()


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-change-me")

    YOLO_WEIGHTS = os.environ.get(
        "YOLO_WEIGHTS",
        "runs/detect/train/weights/best2.pt",
    )

    CONF_THR = float(os.environ.get("CONF_THR", 0.7))
    IOU_THR = float(os.environ.get("IOU_THR", 0.5))
    IMGSZ = int(os.environ.get("IMGSZ", 640))

    GRID_ROWS = int(os.environ.get("GRID_ROWS", 4))
    GRID_COLS = int(os.environ.get("GRID_COLS", 4))

    STATIC_DIR = "static"
    UPLOAD_FOLDER = os.path.join(STATIC_DIR, "uploads")
    CROPS_DIR = os.path.join(STATIC_DIR, "crops")
    RESULTS_DIR = os.path.join(STATIC_DIR, "results")
    LOG_PATH = os.path.join(RESULTS_DIR, "detect_log.jsonl")

    DB_HOST = os.environ.get("DB_HOST", "localhost")
    DB_PORT = int(os.environ.get("DB_PORT", 3306))
    DB_NAME = os.environ.get("DB_NAME", "deteksi_hama")
    DB_USER = os.environ.get("DB_USER", "root")
    DB_PASSWORD = os.environ.get("DB_PASSWORD", "")
    DB_CHARSET = os.environ.get("DB_CHARSET", "utf8mb4")
    DB_POOL_NAME = os.environ.get("DB_POOL_NAME", "simasha_pool")
    DB_POOL_SIZE = int(os.environ.get("DB_POOL_SIZE", 5))
