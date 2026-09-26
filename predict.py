"""
============================================================
 TAHMİN SCRIPTI - HOCALARIN TEST SCRIPTI İLE UYUMLU
============================================================
 Hocaların Test.txt scripti şu formatta dosya bekliyor:

     image1.jpg | Pred: 6
     image2.jpg | Pred: 1

 Test klasörü yapısı:
     testdata/1/...  testdata/2/...  ...  testdata/8/...

 Kullanım:
     python predict.py --test_dir <test_klasoru>  --out preds.txt
============================================================
"""

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from train import build_model, IMAGENET_MEAN, IMAGENET_STD


# Geçerli görüntü uzantıları
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


# ============================================================
# Yardımcı: modeli yükle
# ============================================================
def load_model(ckpt_path: str, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_model(num_classes=len(ckpt["folder_classes"]),
                        dropout=0.3).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def build_tf(img_size: int):
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ============================================================
# Tek dosya tahmini -> proje numaralandırması (1..8)
# ============================================================
@torch.no_grad()
def predict_single(image_path: Path, model, tf, idx_to_label, device):
    img = Image.open(image_path).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)
    logits = model(x)
    probs  = torch.softmax(logits, dim=1)[0].cpu()
    pred_idx = int(probs.argmax().item())
    proj_label = idx_to_label[pred_idx]   # 1..8 (test script uyumlu)
    confidence = float(probs[pred_idx].item())
    return proj_label, confidence, probs.tolist()


# ============================================================
# Klasör -> preds.txt
# ============================================================
def main(test_dir: str, ckpt_path: str, out_file: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_model(ckpt_path, device)
    tf = build_tf(ckpt["img_size"])
    idx_to_label = ckpt["idx_to_label"]
    # JSON'dan dönerken key'ler str olabilir -> int'e zorla
    idx_to_label = {int(k): int(v) for k, v in idx_to_label.items()}

    test_dir = Path(test_dir)

    # Test verisi hem düz klasör hem de sayısal alt klasörler (1-8) desteklenir
    img_paths = []
    for entry in sorted(test_dir.rglob("*")):
        if entry.is_file() and entry.suffix.lower() in IMG_EXTS:
            img_paths.append(entry)

    if not img_paths:
        print(f"[UYARI] {test_dir} içinde görüntü bulunamadı.")
        return

    lines = []
    for p in img_paths:
        try:
            label, conf, _ = predict_single(p, model, tf, idx_to_label, device)
            line = f"{p.name} | Pred: {label} "
            lines.append(line)
            print(f"  {p.name:40s} -> Pred: {label}  (conf={conf:.3f})")
        except Exception as e:
            print(f"[HATA] {p.name}: {e}")

    Path(out_file).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[OK] {len(lines)} tahmin '{out_file}' dosyasına yazıldı.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_dir", required=True,
                    help="Test görüntülerinin bulunduğu klasör")
    ap.add_argument("--ckpt", default="saved_model/car_body_classifier.pt")
    ap.add_argument("--out",  default="preds.txt")
    args = ap.parse_args()
    main(args.test_dir, args.ckpt, args.out)
