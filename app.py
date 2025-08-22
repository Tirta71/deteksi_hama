import os, time, uuid, json
from datetime import datetime, timedelta
from flask import Flask, render_template, request, send_from_directory, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
from werkzeug.utils import secure_filename

# ================== KONFIG ==================
YOLO_WEIGHTS = os.environ.get("YOLO_WEIGHTS", "runs/detect/train/weights/best.pt")
CONF_THR = float(os.environ.get("CONF_THR", 0.7))
IOU_THR  = float(os.environ.get("IOU_THR", 0.5))
IMGSZ    = int(os.environ.get("IMGSZ", 640))

app = Flask(__name__)

# Struktur folder
app.config['STATIC_DIR']     = 'static'
app.config['UPLOAD_FOLDER']  = os.path.join(app.config['STATIC_DIR'], 'uploads')
app.config['CROPS_DIR']      = os.path.join(app.config['STATIC_DIR'], 'crops')
app.config['RESULTS_DIR']    = os.path.join(app.config['STATIC_DIR'], 'results')

# (Tidak membatasi ukuran upload di Flask; kalau di depan ada Nginx/IIS, limiti di sana)
os.makedirs(app.config['STATIC_DIR'], exist_ok=True)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['CROPS_DIR'], exist_ok=True)
os.makedirs(app.config['RESULTS_DIR'], exist_ok=True)

LOG_PATH = os.path.join(app.config['RESULTS_DIR'], "detect_log.jsonl")

# Muat YOLO
model = YOLO(YOLO_WEIGHTS)

# ============== UTIL ==============
def unique_name(prefix, ext=".jpg"):
    return f"{prefix}_{int(time.time())}_{uuid.uuid4().hex[:8]}{ext}"

def resize_to_640(img_bgr):
    if img_bgr is None:
        return None
    return cv2.resize(img_bgr, (IMGSZ, IMGSZ), interpolation=cv2.INTER_LINEAR)

def summarize_result(ultra_res, duration_ms: float):
    b = ultra_res.boxes
    n = int(b.shape[0]) if b is not None else 0
    classes, confs = [], []
    if n > 0:
        names = ultra_res.names or {}
        for i in range(n):
            cls_id = int(b.cls[i].item())
            classes.append(names.get(cls_id, str(cls_id)))
            confs.append(float(b.conf[i].item()))
    avg_conf = round(sum(confs)/len(confs), 4) if confs else 0.0
    return {
        "num_boxes": n,
        "classes": classes,
        "avg_conf": avg_conf,
        "duration_ms": int(round(duration_ms))
    }

def run_yolo_and_save(img_bgr, out_path):
    t0 = time.perf_counter()
    res = model.predict(source=img_bgr, conf=CONF_THR, iou=IOU_THR, imgsz=IMGSZ, verbose=False)
    duration_ms = (time.perf_counter() - t0) * 1000.0
    rendered = res[0].plot()  # ndarray BGR
    cv2.imwrite(out_path, rendered)
    return summarize_result(res[0], duration_ms)

def append_log(entry: dict):
    entry["ts_iso"] = datetime.utcnow().isoformat() + "Z"
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def cleanup_dir(dir_path, max_age_hours=24, keep_min=100):
    try:
        files = [os.path.join(dir_path, f) for f in os.listdir(dir_path)]
        files = [f for f in files if os.path.isfile(f)]
        if len(files) <= keep_min: return
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

def parse_iso(s: str):
    if not s:
        return None
    try:
        if s.endswith("Z"):
            s = s[:-1]
        return datetime.fromisoformat(s)
    except Exception:
        return None

def fmt_local(iso_str: str) -> str:
    dt = parse_iso(iso_str)
    if not dt:
        return "-"
    # Tambah 7 jam biar jadi WIB
    dt = dt + timedelta(hours=7)
    return dt.strftime("%Y-%m-%d %H:%M:%S")

# ============== ROUTES ==============
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/detect_both", methods=["POST"])
def detect_both():
    """
    multipart/form-data:
      - file : gambar asli (wajib)
      - crop : file gambar hasil crop (opsional, image/jpeg/png)
    """
    f = request.files.get("file")
    if not f or f.filename == "":
        return jsonify({"status": "error", "message": "No file"}), 400

    fname = secure_filename(f.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
    f.save(save_path)

    img = cv2.imread(save_path)
    if img is None:
        return jsonify({"status": "error", "message": "Gagal membaca gambar"}), 400

    # ===== FULL DETECTION =====
    img640 = resize_to_640(img)
    out_full_name = unique_name("full")
    out_full_path = os.path.join(app.config['RESULTS_DIR'], out_full_name)
    full_summary = run_yolo_and_save(img640, out_full_path)
    yolo_full_url = f"/{out_full_path.replace(os.sep, '/')}"

    append_log({
        "mode": "full",
        "filename": fname,
        "result_image": out_full_path.replace(os.sep, "/"),
        "summary": full_summary
    })

    # ===== CROP DETECTION (opsional) =====
    yolo_crop_url = None
    crop_summary = None
    crop_file = request.files.get("crop")
    if crop_file and crop_file.filename:
        crop_bytes = crop_file.read()
        arr = np.frombuffer(crop_bytes, dtype=np.uint8)
        crop_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if crop_img is None:
            return jsonify({"status": "error", "message": "Gagal decode crop"}), 400

        if crop_img.shape[:2] != (IMGSZ, IMGSZ):
            crop_img = resize_to_640(crop_img)

        crop_name = unique_name("crop_det")
        out_crop_path = os.path.join(app.config['RESULTS_DIR'], crop_name)
        crop_summary = run_yolo_and_save(crop_img, out_crop_path)
        yolo_crop_url = f"/{out_crop_path.replace(os.sep, '/')}"

        append_log({
            "mode": "crop",
            "filename": fname,
            "result_image": out_crop_path.replace(os.sep, "/"),
            "summary": crop_summary
        })

    cleanup_dir(app.config['CROPS_DIR'])
    cleanup_dir(app.config['RESULTS_DIR'])

    return jsonify({
        "status": "success",
        "filename": fname,
        "yolo_full": yolo_full_url,
        "yolo_crop": yolo_crop_url,
        "full_summary": full_summary,
        "crop_summary": crop_summary
    })

@app.route("/hasil_deteksi", methods=["GET"])
def hasil_deteksi():
    """
    Baca detect_log.jsonl -> gabungkan entri per filename (full & crop),
    lalu render ke tabel (modal 'Lihat detail' menampilkan gambar).
    """
    pairs = {}
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                fname = row.get("filename") or "unknown"
                mode  = row.get("mode")
                if not mode:
                    continue
                if fname not in pairs:
                    pairs[fname] = {}
                pairs[fname][mode] = row

    rows = []
    for fname, modes in pairs.items():
        full = modes.get("full")
        crop = modes.get("crop")
        full_sum = (full or {}).get("summary") or {}
        crop_sum = (crop or {}).get("summary") or {}

        rows.append({
            "filename": fname,

            "full_time": fmt_local((full or {}).get("ts_iso")),
            "full_img": (full or {}).get("result_image"),
            "full_classes": ", ".join(full_sum.get("classes", [])) or "-",
            "full_avgconf": full_sum.get("avg_conf", 0.0),
            "full_speed": full_sum.get("duration_ms", 0),

            "crop_time": fmt_local((crop or {}).get("ts_iso")),
            "crop_img": (crop or {}).get("result_image"),
            "crop_classes": ", ".join(crop_sum.get("classes", [])) or "-",
            "crop_avgconf": crop_sum.get("avg_conf", 0.0),
            "crop_speed": crop_sum.get("duration_ms", 0),
        })

    # urutkan terbaru (pakai ts_iso yang ada; fallback ke very old)
    def latest_dt(item):
        f = parse_iso((pairs[item["filename"]].get("full") or {}).get("ts_iso", ""))
        c = parse_iso((pairs[item["filename"]].get("crop") or {}).get("ts_iso", ""))
        return max([d for d in [f, c] if d is not None] or [datetime.min])
    rows.sort(key=latest_dt, reverse=True)

    return render_template("hasil_deteksi.html", rows=rows)

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)
