"""
============================================================
 MODEL DEĞERLENDİRME SCRIPTI
============================================================
 Bir 'eval_set/' klasörü (her sınıf için alt-klasör) üzerinde
 - Accuracy
 - Precision (per-class, macro, weighted)
 - Recall    (per-class, macro, weighted)
 - F1-Score  (per-class, macro, weighted)   <-- 1. öncelikli metrik
 - Normalize edilmiş 8x8 Confusion Matrix
 hesaplar ve raporlanması için PNG/CSV çıktıları üretir.

 Bu script, kendi tuttuğunuz HOLD-OUT setinde
 çalıştırılmak içindir (sunum öncesi kendi içsel testiniz).
 Hocaların paylaştığı resmi test_script.py için ayrıca
 predict.py kullanılır (preds.txt üretir).
============================================================
"""

import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from train import build_model, IMAGENET_MEAN, IMAGENET_STD, CLASS_TO_LABEL


# ============================================================
# 1) Test transform - augmentation YOK, sadece resize+normalize
# ============================================================
def build_eval_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ============================================================
# 2) Modeli checkpoint'ten yükle
# ============================================================
def load_checkpoint(ckpt_path: str, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_model(num_classes=len(ckpt["folder_classes"]),
                        dropout=0.3).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


# ============================================================
# 3) Veri kümesi üzerinden tahmin et
# ============================================================
@torch.no_grad()
def collect_predictions(model, loader, device):
    y_true, y_pred = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        out  = model(imgs).argmax(1).cpu().numpy()
        y_pred.extend(out.tolist())
        y_true.extend(labels.numpy().tolist())
    return np.array(y_true), np.array(y_pred)


# ============================================================
# 4) Normalized Confusion Matrix grafiği (8x8 heatmap)
# ============================================================
def plot_confusion_matrix(y_true, y_pred, class_names, out_path: Path):
    cm = confusion_matrix(y_true, y_pred,
                          labels=list(range(len(class_names))))
    # Satır toplamına böl -> her gerçek sınıf için yüzde dağılımı
    cm_norm = cm.astype(np.float32) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    plt.figure(figsize=(9, 7))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={"label": "Oran"}, vmin=0, vmax=1)
    plt.xlabel("Tahmin Edilen Sınıf")
    plt.ylabel("Gerçek Sınıf")
    plt.title("Normalized Confusion Matrix")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"[OK] Normalized confusion matrix -> {out_path}")
    return cm_norm


# ============================================================
# 5) Tüm metrikleri raporla
# ============================================================
def report_metrics(y_true, y_pred, class_names, out_dir: Path):
    acc = accuracy_score(y_true, y_pred)

    p_per, r_per, f_per, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names))),
        zero_division=0
    )
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro",    zero_division=0
    )
    p_w, r_w, f_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    print("\n=========== METRİK ÖZETİ ===========")
    print(f"Accuracy              : {acc:.4f}")
    print(f"Macro    Precision/Recall/F1 : "
          f"{p_macro:.4f} / {r_macro:.4f} / {f_macro:.4f}")
    print(f"Weighted Precision/Recall/F1 : "
          f"{p_w:.4f} / {r_w:.4f} / {f_w:.4f}")
    print("\n----- Sınıf bazlı -----")
    print(f"{'Sınıf':<16} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    for i, cname in enumerate(class_names):
        print(f"{cname:<16} {p_per[i]:>10.4f} {r_per[i]:>10.4f} "
              f"{f_per[i]:>10.4f} {sup[i]:>10d}")

    # Detaylı raporu da JSON'a yaz - LaTeX raporunda kullanmak için
    report = classification_report(
        y_true, y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names, zero_division=0, output_dict=True,
    )
    summary = {
        "accuracy": acc,
        "macro":    {"precision": p_macro, "recall": r_macro, "f1": f_macro},
        "weighted": {"precision": p_w,     "recall": r_w,     "f1": f_w},
        "per_class": {
            cname: {"precision": p_per[i], "recall": r_per[i],
                    "f1": f_per[i], "support": int(sup[i])}
            for i, cname in enumerate(class_names)
        },
        "sklearn_classification_report": report,
    }
    with open(out_dir / "metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[OK] Metrik özeti JSON'a yazıldı -> {out_dir/'metrics_summary.json'}")
    return summary


# ============================================================
# 6) MAIN
# ============================================================
def main(eval_dir: str = "eval_set",
         ckpt_path: str = "saved_model/car_body_classifier.pt",
         out_dir:   str = "plots"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir); out_dir.mkdir(exist_ok=True, parents=True)

    model, ckpt = load_checkpoint(ckpt_path, device)
    img_size = ckpt["img_size"]
    tf = build_eval_transform(img_size)

    ds = datasets.ImageFolder(eval_dir, transform=tf)
    # ÖNEMLİ: eval_set klasörlerinin isimleri TRAIN ile aynı olmalı,
    # aksi halde indeksler uyuşmaz. Aşağıda doğrula:
    if ds.classes != ckpt["folder_classes"]:
        print(f"[UYARI] Eval klasör sınıfları    : {ds.classes}")
        print(f"        Eğitim klasör sınıfları  : {ckpt['folder_classes']}")
        print( "        İsimler birebir aynı olmalı!")

    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)
    y_true, y_pred = collect_predictions(model, loader, device)

    plot_confusion_matrix(y_true, y_pred, ds.classes,
                          out_dir / "confusion_matrix.png")
    report_metrics(y_true, y_pred, ds.classes, out_dir)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--eval_dir", default="eval_set",
                   help="Her sınıf için alt-klasör içeren değerlendirme seti")
    p.add_argument("--ckpt", default="saved_model/car_body_classifier.pt")
    p.add_argument("--out",  default="plots")
    args = p.parse_args()
    main(args.eval_dir, args.ckpt, args.out)
