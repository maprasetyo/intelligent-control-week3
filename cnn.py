import cv2
import numpy as np
from keras.models import load_model
import os
import json

# Pastikan model tersedia
model_path = 'Intelligent-Control-Week3/cnn_model.h5'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file '{model_path}' tidak ditemukan!")

# Load model yang telah dilatih
model = load_model(model_path)

# Coba muat label kelas dari file JSON jika tersedia
class_labels_path = "Intelligent-Control-Week3/class_labels.json"
if os.path.exists(class_labels_path):
    with open(class_labels_path, "r") as f:
        class_dict = json.load(f)
        class_labels = [label for label, index in sorted(class_dict.items(), key=lambda item: item[1])]  # Urutkan berdasarkan indeks
else:
    print("File class_labels.json tidak ditemukan! Membuat file baru dengan label default.")
    class_labels = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']  # Sesuaikan dengan jumlah kelas model Anda
    class_dict = {label: i for i, label in enumerate(class_labels)}
    with open(class_labels_path, "w") as f:
        json.dump(class_dict, f)
    print(f"Class Labels: {class_labels}")  # Debugging untuk melihat isi class_labels


# Periksa kesesuaian jumlah kelas dengan output model
output_shape = model.output_shape[1]  # Ambil jumlah output neurons
if len(class_labels) != output_shape:
    raise ValueError(f"Jumlah class_labels ({len(class_labels)}) tidak sesuai dengan output model ({output_shape})! Periksa kembali jumlah kelas.")

# Inisialisasi kamera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mode Night Vision dengan konversi ke skala abu-abu
    night_vision = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    night_vision = cv2.applyColorMap(night_vision, cv2.COLORMAP_JET)

    # Preprocessing gambar untuk model
    img = cv2.resize(frame, (150, 150))  # Sesuaikan ukuran input model
    img = img.astype("float32") / 255.0  # Normalisasi
    img = np.expand_dims(img, axis=0)  # Tambahkan dimensi batch

    # Prediksi kelas
    pred = model.predict(img)
    label_index = np.argmax(pred)
    label = class_labels[label_index] if label_index < len(class_labels) else "Unknown"

    print(f"Prediksi mentah: {pred}, Kelas: {label}")  # Debugging output model

    # Tampilkan hasil pada frame
    cv2.putText(frame, f'Class: {label}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Frame', frame)
    cv2.imshow('Night Vision', night_vision)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Tutup kamera dan jendela OpenCV
cap.release()
cv2.destroyAllWindows()
