import os, base64, time, uuid
from flask import Flask, render_template, request, send_from_directory, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
from werkzeug.utils import secure_filename

# ================== KONFIG ==================
YOLO_WEIGHTS = os.environ.get("YOLO_WEIGHTS", "runs/detect/train/weights/best.pt")
CONF_THR = 0.7
IOU_THR  = 0.5
IMGSZ    = 640

app = Flask(__name__)

# Struktur folder
app.config['STATIC_DIR']     = 'static'
app.config['UPLOAD_FOLDER']  = os.path.join(app.config['STATIC_DIR'], 'uploads')
app.config['CROPS_DIR']      = os.path.join(app.config['STATIC_DIR'], 'crops')
app.config['RESULTS_DIR']    = os.path.join(app.config['STATIC_DIR'], 'results')

# Pastikan semua folder ada
for d in [app.config['STATIC_DIR'], app.config['UPLOAD_FOLDER'], app.config['CROPS_DIR'], app.config['RESULTS_DIR']]:
    os.makedirs(d, exist_ok=True)

model = YOLO(YOLO_WEIGHTS)

# ============== UTIL ==============
def resize_to_640(img_bgr):
    """Resize langsung ke 640x640 (tanpa letterbox/padding)."""
    if img_bgr is None:
        return None
    return cv2.resize(img_bgr, (IMGSZ, IMGSZ), interpolation=cv2.INTER_LINEAR)

def run_yolo_and_save(img_bgr, out_path):
    """Jalankan YOLO dan simpan hasil render bounding box ke out_path."""
    res = model.predict(source=img_bgr, conf=CONF_THR, iou=IOU_THR, imgsz=IMGSZ, verbose=False)
    rendered = res[0].plot()  # BGR ndarray
    cv2.imwrite(out_path, rendered)

def unique_name(prefix, ext=".jpg"):
    """Bikin nama file unik biar nggak ketimpa."""
    return f"{prefix}_{int(time.time())}_{uuid.uuid4().hex[:8]}{ext}"

# Menghapus File Lama
def cleanup_dir(dir_path, max_age_hours=24, keep_min=100):
    try:
        files = [os.path.join(dir_path, f) for f in os.listdir(dir_path)]
        files = [f for f in files if os.path.isfile(f)]
        if len(files) <= keep_min:
            return
        now = time.time()
        for f in files:
            age_h = (now - os.path.getmtime(f)) / 3600.0
            if age_h > max_age_hours:
                try:
                    os.remove(f)
                except Exception:
                    pass
    except Exception:
        pass

# ============== ROUTES ==============
@app.route("/", methods=["GET","POST"])
def index():
    if request.method == "POST":
        f = request.files.get("file")
        if f and f.filename:
            fname = secure_filename(f.filename)
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
            f.save(save_path)

            img = cv2.imread(save_path)
            if img is None:
                return render_template("index.html",
                                       filename=None,
                                       has_result=False,
                                       error="Gagal membaca gambar.")

            # Resize fill 640x640 dan deteksi
            img640 = resize_to_640(img)

            # Simpan hasil prediksi FULL ke folder results/ dengan nama unik
            full_name = unique_name("full")
            out_full_path = os.path.join(app.config['RESULTS_DIR'], full_name)
            run_yolo_and_save(img640, out_full_path)

            # URL untuk <img> (Flask secara default melayani /static/*)
            yolo_full_url = f"/{out_full_path.replace(os.sep, '/')}"

            # Bersihin file lama (opsional)
            cleanup_dir(app.config['RESULTS_DIR'])

            return render_template("index.html",
                                   filename=fname,
                                   has_result=True,
                                   yolo_full_url=yolo_full_url,
                                   yolo_crop_url=None)
        # tidak ada file
        return render_template("index.html", filename=None, has_result=False)

    # GET: tidak perlu hapus hasil—cukup tidak kirim URL apa pun
    return render_template("index.html", filename=None, has_result=False)

@app.route("/detect_crop", methods=["POST"])
def detect_crop():
    """Terima crop base64 (seharusnya 640x640), deteksi & kembalikan URL hasil."""
    data = request.json or {}
    img_b64 = data.get("image", "")
    if not img_b64:
        return jsonify({"status":"error","message":"No image"}), 400

    try:
        # ambil setelah "data:image/..;base64,"
        if "," in img_b64:
            img_b64 = img_b64.split(",")[1]
        img_bytes = base64.b64decode(img_b64)

        # Simpan juga versi crop mentah (opsional)
        raw_name = unique_name("crop_raw", ext=".png")
        raw_path = os.path.join(app.config['CROPS_DIR'], raw_name)
        with open(raw_path, "wb") as f:
            f.write(img_bytes)

        # decode langsung ke numpy -> BGR
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        crop = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({"status":"error","message":f"Decode error: {e}"}), 400

    if crop is None:
        return jsonify({"status":"error","message":"Gagal decode crop"}), 400

    # pastikan 640x640 (fill)
    if crop.shape[0] != IMGSZ or crop.shape[1] != IMGSZ:
        crop = resize_to_640(crop)

    # Simpan hasil deteksi CROP ke results/ dengan nama unik
    crop_name = unique_name("crop_det")
    out_crop_path = os.path.join(app.config['RESULTS_DIR'], crop_name)
    run_yolo_and_save(crop, out_crop_path)

    # URL untuk <img>
    yolo_crop_url = f"/{out_crop_path.replace(os.sep, '/')}"

    # Bersihin file lama (opsional)
    cleanup_dir(app.config['CROPS_DIR'])
    cleanup_dir(app.config['RESULTS_DIR'])

    return jsonify({"status":"success", "yolo_crop": yolo_crop_url})

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)

