import os
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, Label, Button
from PIL import Image, ImageTk
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# === Label dan RGB dasar ===
class_labels = ["Merah", "Hijau", "Biru", "Kuning", "Hitam"]
warna_rgb = {
    "Merah": [255, 0, 0],
    "Hijau": [0, 255, 0],
    "Biru": [0, 0, 255],
    "Kuning": [255, 255, 0],
    "Hitam": [0, 0, 0],
}
# === Fungsi membuat dataset warna ===
def generate_data(jumlah_per_warna=200):
    X, y = [], []
    for i, warna in enumerate(class_labels):
        rgb = warna_rgb[warna]
        for _ in range(jumlah_per_warna):
            img = np.ones((64, 64, 3), dtype=np.uint8) * np.array(rgb, dtype=np.uint8)
            noise = np.random.randint(0, 30, (64, 64, 3), dtype=np.uint8)
            img = np.clip(img + noise, 0, 255)
            X.append(img)
            y.append(i)
    return np.array(X), to_categorical(np.array(y), num_classes=len(class_labels))

# === Load atau latih model ===
model_path = "model_warna_cnn.h5"
if os.path.exists(model_path):
    model = load_model(model_path)
else:
    X, y = generate_data()
    X = X.astype("float32") / 255.0

    model = Sequential([
        Conv2D(16, (3,3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D(),
        Conv2D(32, (3,3), activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(len(class_labels), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)
    model.save(model_path)

# === Prediksi warna dari gambar ===
def kenali_warna_dengan_cnn(image_path):
    img = Image.open(image_path).resize((64, 64)).convert("RGB")
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    pred = model.predict(img_array)
    index = np.argmax(pred)
    warna = class_labels[index]
    mean_rgb = np.array(img).mean(axis=(0, 1)).astype(int)
    return warna, mean_rgb

# === GUI ===
def buka_gambar():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
    if not file_path:
        return

    try:
        img = Image.open(file_path).convert("RGB")
        img_resized = img.resize((300, 300))
        tk_img = ImageTk.PhotoImage(img_resized)

        label_gambar.config(image=tk_img)
        label_gambar.image = tk_img

        warna, rgb = kenali_warna_dengan_cnn(file_path)
        hasil_rgb.config(text=f"RGB: {rgb.tolist()}")
        hasil_warna.config(text=f"Warna: {warna}")
        kotak_preview.config(bg=rgb_to_hex(rgb))

    except Exception as e:
        messagebox.showerror("Error", f"Gagal membuka gambar: {str(e)}")

def rgb_to_hex(rgb):
    return "#%02x%02x%02x" % tuple(rgb)

root = tk.Tk()
root.title("Deteksi Warna Otomatis")
root.geometry("700x700")
root.configure(bg="#ffc0cb")  # Pink muda

FONT_UTAMA = ("Comic Sans MS", 13, "bold")

judul = Label(root, text="Pengenalan Warna dari Gambar", font=("Comic Sans MS", 25, "bold"),  fg="#4B0082", bg="#ffc0cb")
judul.pack(pady=30)

btn_buka = Button(root, text="Pilih Gambar", command=buka_gambar, font=("Times New Roman", 12, "bold"), bg="#3498db", fg="white",
                  activebackground="#2980b9", relief="raised", padx=20, pady=10)
btn_buka.pack(pady=15)

label_gambar = Label(root, bg="#e6f2ff")
label_gambar.pack()

hasil_rgb = Label(root, text="Nilai RGB: -", font=("Times New Roman", 14, "bold"), fg="#34495e", bg="#e6f2ff")
hasil_rgb.pack(pady=5)

hasil_warna = Label(root, text="Warna: -", font=("Arial", 12, "bold"))
hasil_warna.pack(pady=(5,15))

kotak_preview = Label(root, text="", width=20, height=2,  relief="solid", bd=2, bg="white")
kotak_preview.pack(pady=(20))

root.mainloop()