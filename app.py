import os
from flask import Flask, render_template, request, jsonify, send_from_directory
from ultralytics import YOLO
from PIL import Image
import io
import uuid # Import modul uuid untuk membuat nama file unik

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Direktori untuk menyimpan file yang diunggah dan hasil deteksi
# Di Render, ini akan berada di sistem file ephemeral (sementara)
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'

# Pastikan folder-folder ini ada
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Memuat model YOLOv11
# Pastikan file 'best.pt' berada di direktori yang sama dengan 'app.py'
model = None
class_names = [] # List untuk menyimpan nama-nama kelas
try:
    model = YOLO("best.pt")
    print("Model YOLOv11 berhasil dimuat!")
    if model.names: # Cek apakah model memiliki nama kelas
        class_names = list(model.names.values())
        print(f"Nama-nama kelas yang dimuat: {class_names}")
    else:
        print("Model tidak memiliki nama kelas yang terdefinisi.")
except Exception as e:
    print(f"Error memuat model best.pt: {e}")
    model = None # Set model ke None jika gagal dimuat, untuk penanganan error

@app.route('/')
def index():
    """
    Route utama untuk menampilkan halaman web.
    Merender file index.html dari folder templates.
    Mengirimkan daftar nama kelas ke template.
    """
    return render_template('index.html', class_names=class_names)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Route untuk melakukan prediksi deteksi objek pada gambar yang diunggah.
    Menerima permintaan POST dengan file gambar.
    """
    # Periksa apakah model berhasil dimuat
    if model is None:
        return jsonify({'error': 'Model AI tidak dapat dimuat. Silakan periksa file best.pt Anda.'}), 500

    # Periksa apakah ada file dalam request
    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada bagian file dalam request'}), 400
    
    file = request.files['file']
    
    # Periksa apakah nama file kosong
    if file.filename == '':
        return jsonify({'error': 'Tidak ada file yang dipilih'}), 400
    
    if file:
        try:
            # Baca gambar dari stream biner
            img_bytes = file.read()
            # Buka gambar menggunakan Pillow dan konversi ke RGB
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

            # Lakukan inferensi menggunakan model YOLOv11
            # results akan berisi objek Results dari Ultralytics
            results = model.predict(img)

            detected_character = "Tidak Dikenal"
            confidence = 0.0
            processed_image_url = ""

            # Proses hasil deteksi
            if results:
                # Ambil objek Results pertama (karena kita memprediksi satu gambar)
                res = results[0] 
                
                # Dapatkan bounding boxes, class IDs, dan confidences
                boxes = res.boxes 
                
                if len(boxes) > 0:
                    # Ambil deteksi dengan confidence tertinggi
                    best_detection = boxes[boxes.conf.argmax()]
                    
                    # Dapatkan ID kelas dan confidence
                    class_id = int(best_detection.cls)
                    confidence = float(best_detection.conf)

                    # Dapatkan nama kelas dari model.names (yang dimuat dari data.yaml saat pelatihan)
                    if class_id < len(class_names): # Pastikan class_id valid
                        detected_character = class_names[class_id]
                    else:
                        detected_character = "Karakter Tidak Dikenal (ID Kelas Tidak Dikenal)"
                
                # Gambar bounding box dan label pada gambar asli
                # res.plot() mengembalikan gambar dalam format NumPy array (BGR)
                im_bgr = res.plot()  
                    # Konversi dari BGR (OpenCV/Ultralytics default) ke RGB untuk Pillow
                im_rgb = Image.fromarray(im_bgr[..., ::-1]) 

                # Buat nama file unik untuk gambar hasil
                unique_filename = f"detected_image_{uuid.uuid4().hex}.jpg"
                processed_image_path = os.path.join(RESULTS_FOLDER, unique_filename)
                im_rgb.save(processed_image_path)
                
                # Buat URL untuk gambar yang diproses agar bisa diakses oleh frontend
                processed_image_url = f'/results/{unique_filename}'

                return jsonify({
                    'character_name': detected_character,
                    'confidence': confidence,
                    'image_url': processed_image_url
                })
            else:
                # Jika tidak ada deteksi yang ditemukan
                return jsonify({'character_name': 'Tidak Terdeteksi', 'confidence': 0.0, 'image_url': ''})

        except Exception as e:
            # Tangani error selama proses prediksi
            return jsonify({'error': str(e)}), 500

@app.route('/results/<filename>')
def serve_results_image(filename):
    """
    Route untuk melayani file gambar dari folder results.
    Digunakan oleh frontend untuk menampilkan gambar hasil deteksi.
    """
    return send_from_directory(RESULTS_FOLDER, filename)

# Dapatkan port dari environment variable, default ke 5000 jika tidak ada
# Ini penting untuk deployment di Render
port = int(os.environ.get("PORT", 5000))

# Jalankan aplikasi Flask
if __name__ == '__main__':
    # debug=True akan otomatis me-reload server saat ada perubahan kode
    # dan memberikan pesan error yang lebih detail di konsol
    # host='0.0.0.0' agar bisa diakses dari luar container/server Render
    app.run(debug=True, host='0.0.0.0', port=port)
