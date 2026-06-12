# Deteksi Hama SIMASHA

Aplikasi Flask untuk deteksi hama berbasis YOLOv8, login/register user, dan penyimpanan riwayat hasil deteksi ke MySQL.

## Setup

1. Install dependency:

   ```powershell
   pip install -r requirements.txt
   ```

2. Buat file `.env` dari contoh:

   ```powershell
   copy .env.example .env
   ```

3. Sesuaikan koneksi MySQL di `.env`.

4. Import schema database di MySQL/phpMyAdmin:

   - `database/users_schema.sql`
   - `database/detection_results_schema.sql`

5. Jalankan aplikasi:

   ```powershell
   python app.py
   ```

6. Buka browser:

   ```text
   http://127.0.0.1:8080
   ```

## Catatan

- File `.env` tidak ikut dipush karena berisi secret dan konfigurasi lokal.
- Model YOLO default dibaca dari `YOLO_WEIGHTS` di `.env`.
