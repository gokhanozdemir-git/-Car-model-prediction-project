"""
============================================================
 FLASK WEB ARAYÜZÜ - CANLI ARABA GÖVDE TİPİ TAHMİNİ
============================================================
 - Görüntü yükleme (file upload + drag&drop)
 - Tahmin sonucu (sınıf adı + güven skoru)
 - 8 sınıf için olasılık dağılımı (bar chart - JS tarafında render)
 - Yüklenen görsel + sonuç yan yana

 Çalıştırma:
     python app.py
     -> http://127.0.0.1:5000
============================================================
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import io
import time
import base64
from pathlib import Path

import torch
from flask import Flask, render_template, request, jsonify
from PIL import Image
from torchvision import transforms

from train import build_model, IMAGENET_MEAN, IMAGENET_STD


# ============================================================
# Sabitler
# ============================================================
CKPT_PATH = Path(__file__).resolve().parent / "saved_model" / "car_body_classifier.pt"

# Klasör adından kullanıcıya gösterilecek Türkçe isim
PRETTY_NAME = {
    "SUV":           "SUV",
    "VAN":           "VAN",
    "STATION_WAGON": "Station Wagon",
    "MICRO":         "Micro",
    "F1":            "Açık Tekerlekli (F1)",
    "SEDAN":         "Sedan",
    "HATCHBACK":     "Hatchback",
    "PICKUP":        "Pick-Up",
}


# ============================================================
# Modeli bir kez yükle (uygulama başlangıcında)
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Cihaz: {device}")

if not Path(CKPT_PATH).exists():
    print(f"[UYARI] Checkpoint bulunamadı: {CKPT_PATH}")
    print( "        Önce 'python train.py' ile modeli eğitin.")
    model = None
    ckpt  = None
else:
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model = build_model(num_classes=len(ckpt["folder_classes"]),
                        dropout=0.3).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"[OK] Model yüklendi: {CKPT_PATH}")

# Transform (eğitimle bire bir aynı eval transform)
IMG_SIZE = (ckpt["img_size"] if ckpt else 224)
TF = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.14)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

FOLDER_CLASSES = (ckpt["folder_classes"] if ckpt else
                  ["SUV", "VAN", "STATION_WAGON", "MICRO",
                   "F1", "SEDAN", "HATCHBACK", "PICKUP"])


# ============================================================
# Flask app
# ============================================================
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024   # 16 MB üst sınır


@app.route("/")
def index():
    # JS tarafına sınıf isimlerini geç (bar chart için)
    pretty_list = [PRETTY_NAME.get(c, c) for c in FOLDER_CLASSES]
    return render_template("index.html",
                           class_names=pretty_list,
                           raw_classes=FOLDER_CLASSES)


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model henüz eğitilmemiş. "
                                 "Önce train.py'yi çalıştırın."}), 500

    if "image" not in request.files:
        return jsonify({"error": "Görüntü dosyası bulunamadı."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Dosya seçilmedi."}), 400

    try:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Görüntü açılamadı: {e}"}), 400

    # ---- Tahmin (zamanlama dahil) ----
    t0 = time.time()
    with torch.no_grad():
        x = TF(img).unsqueeze(0).to(device)
        logits = model(x)
        probs  = torch.softmax(logits, dim=1)[0].cpu().tolist()
    elapsed_ms = (time.time() - t0) * 1000

    # En yüksek olasılıklı sınıf
    top_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
    folder_name = FOLDER_CLASSES[top_idx]
    pretty      = PRETTY_NAME.get(folder_name, folder_name)
    proj_label  = ckpt["idx_to_label"][top_idx]   # 1..8 proje numarası

    # JSON dön
    return jsonify({
        "predicted_class":      pretty,
        "predicted_class_raw":  folder_name,
        "project_label":        int(proj_label),
        "confidence":           float(probs[top_idx]),
        "probabilities":        [float(p) for p in probs],
        "class_names":          [PRETTY_NAME.get(c, c) for c in FOLDER_CLASSES],
        "inference_time_ms":    round(elapsed_ms, 2),
    })


@app.route("/health")
def health():
    return jsonify({
        "status":       "ok",
        "model_loaded": model is not None,
        "device":       str(device),
        "num_classes":  len(FOLDER_CLASSES),
    })


if __name__ == "__main__":
    # debug=False -> demo sırasında auto-reload kapalı (daha hızlı)
    app.run(host="0.0.0.0", port=5000, debug=False)
