"""
============================================================
 ARABA GÖVDE TİPİ SINIFLANDIRMA - EĞİTİM SCRIPTI
============================================================
 Kocaeli Üniversitesi - Yazılım Laboratuvarı II - Proje III

 Model      : MobileNetV3-Large (transfer learning, ImageNet weights)
 Sınıflar   : 8  (SUV, VAN, STATION_WAGON, MICRO, F1,
                  SEDAN, HATCHBACK, PICKUP)
 Input      : 224x224 RGB
 Framework  : PyTorch
============================================================
"""

import os
import json
import time
import copy
import random
import winsound
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt


# ============================================================
# 1) KONFİGÜRASYON
# ============================================================
# Tüm hiperparametreleri tek noktada toplamak,
# tekrarlanabilirlik ve raporda açıklayabilmek için önemlidir.
CONFIG = {
    "data_dir":      "dataset",         # Klasör başına bir sınıf
    "save_dir":      "saved_model",
    "plot_dir":      "plots",
    "img_size":      224,               # MobileNetV3'ün doğal girişi
    "batch_size":    32,
    "epochs":        20,
    "lr_head":       1e-3,              # Sınıflandırıcı başı LR'si
    "lr_backbone":   1e-4,              # Önceden eğitilmiş gövde için daha düşük
    "weight_decay":  4e-4,
    "dropout":       0.5,
    "val_split":     0.15,              # %15 validation, %85 train
    "patience":      10,                # EarlyStopping sabır değeri
    "label_smooth":  0.1,               # Aşırı güveni hafifletir
    "num_workers":   4,
    "seed":          42,
    "use_amp":       True,              # Mixed precision -> daha hızlı + az VRAM
}

# Sınıf indeksleri TEST SCRIPTI ile birebir uyumlu olmalı:
# 1=SUV, 2=VAN, 3=STATION_WAGON, 4=MICRO, 5=F1,
# 6=SEDAN, 7=HATCHBACK, 8=PICKUP
CLASS_TO_LABEL = {
    "SUV": 1, "VAN": 2, "STATION_WAGON": 3, "MICRO": 4,
    "F1":  5, "SEDAN": 6, "HATCHBACK": 7, "PICKUP": 8,
}

# CrossEntropyLoss class weights — düşük F1'li sınıflara daha yüksek ceza
LABEL_WEIGHTS = {
    1: 2.0,   # SUV           F1=0.77
    2: 1.0,   # VAN           F1=0.92
    3: 2.0,   # STATION_WAGON F1=0.59 (Sedan ile karışıyor)
    4: 1.5,   # MICRO         F1=0.82
    5: 1.0,   # F1            F1=0.99
    6: 1.5,   # SEDAN         F1=0.72 (Hatchback ile karışıyor)
    7: 2.0,   # HATCHBACK     F1=0.61
    8: 2.0,   # PICKUP        F1=0.59
}


# ============================================================
# 2) TEKRARLANABİLİRLİK (REPRODUCIBILITY)
# ============================================================
def set_seed(seed: int) -> None:
    """Aynı çalıştırmaların aynı sonucu vermesi için tüm RNG'leri sabitler."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# 3) VERİ DÖNÜŞÜMLERİ (AUGMENTATION + NORMALIZATION)
# ============================================================
# ImageNet istatistikleri - önceden eğitilmiş model bunlara göre normalize bekler
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def build_transforms(img_size: int):
    """Eğitim ve değerlendirme için transform pipeline'larını döner.

    Eğitim için augmentation: model overfit etmesin diye veriyi her epoch'ta
    biraz farklı görür.  Validation/Test için yalnızca deterministik
    resize+normalize yapılır.
    """
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.65, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3,
                               saturation=0.3, hue=0.08),
        transforms.RandomRotation(degrees=15),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.4),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.4, scale=(0.02, 0.2)),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),   # 256 -> 224 klasik şema
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    return train_tf, eval_tf


# ============================================================
# 4) VERİ KÜMESİNİ HAZIRLAMA
# ============================================================
def prepare_dataloaders(cfg):
    """ImageFolder ile veriyi yükler, train/val olarak böler.

    NOT: 'test' verisi proje şartnamesi gereği eğitim sürecinde KESİNLİKLE
    kullanılmaz - sadece sunum sırasında verilen örnekler test setini oluşturur.
    """
    train_tf, eval_tf = build_transforms(cfg["img_size"])

    # Tüm veri seti, sınıf indekslerini ImageFolder belirler.
    # CLASS_TO_LABEL ile uyum sağlamak için sonradan eşleme yapacağız.
    full_set = datasets.ImageFolder(cfg["data_dir"], transform=train_tf)

    # ImageFolder'ın bulduğu sınıf adlarını yazdır - sanity check
    print(f"[INFO] Tespit edilen sınıflar (ImageFolder sırası): "
          f"{full_set.classes}")
    print(f"[INFO] Toplam görüntü sayısı: {len(full_set)}")

    # Sınıf başına örnek sayısını yazdır - dengesizliği erken yakalamak için
    class_counts = {c: 0 for c in full_set.classes}
    for _, lbl in full_set.samples:
        class_counts[full_set.classes[lbl]] += 1
    print("[INFO] Sınıf başına görüntü sayısı:")
    for c, n in class_counts.items():
        print(f"        {c:18s}: {n}")

    # Train/Validation ayır
    val_n   = int(len(full_set) * cfg["val_split"])
    train_n = len(full_set) - val_n
    train_ds, val_ds = random_split(
        full_set, [train_n, val_n],
        generator=torch.Generator().manual_seed(cfg["seed"])
    )

    # Validation için augmentation OLMAMALI -> kopyada transform'u değiştir
    # random_split alt-datasetin .dataset alanı orijinaldir, paylaşılır.
    # Bu yüzden val için eval_tf'li ayrı bir referans oluşturuyoruz:
    val_ds_eval = copy.copy(val_ds)
    val_ds_eval.dataset = copy.copy(full_set)
    val_ds_eval.dataset.transform = eval_tf

    # WeightedRandomSampler ile sınıf dengesizliğini azaltıyoruz
    train_targets = [full_set.targets[i] for i in train_ds.indices]
    class_sample_count = np.bincount(train_targets, minlength=len(full_set.classes))
    # Az örnekli sınıflara daha yüksek ağırlık
    weights_per_class = 1.0 / np.maximum(class_sample_count, 1)
    sample_weights = [weights_per_class[t] for t in train_targets]
    sampler = torch.utils.data.WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        sampler=sampler,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds_eval,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )
    return train_loader, val_loader, full_set.classes


# ============================================================
# 5) MODEL: MobileNetV3-Large (Transfer Learning)
# ============================================================
def build_model(num_classes: int, dropout: float) -> nn.Module:
    """ImageNet'te önceden eğitilmiş MobileNetV3-Large yükler ve
    classifier başını 8-sınıf için yeniden tanımlar.

    MobileNetV3 tercih sebebi:
    - <20 MB boyut (proje 95 MB sınırının çok altında)
    - SE bloklarıyla yüksek doğruluk
    - Hızlı çıkarım -> arayüzde gerçek-zamanlı tahmin
    """
    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)

    # Orijinal classifier:  Linear(960,1280) -> Hardswish -> Dropout -> Linear(1280,1000)
    in_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 1280),
        nn.Hardswish(inplace=True),
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(1280, num_classes),    # 8 araba gövde tipi
    )
    return model


# ============================================================
# 6) EĞİTİM DÖNGÜSÜ
# ============================================================
class EarlyStopping:
    """Validation loss belirli epoch boyunca iyileşmezse eğitimi durdurur."""
    def __init__(self, patience: int = 7, min_delta: float = 1e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter   = 0
        self.stop      = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
            return True              # iyileşme var -> modeli kaydet
        self.counter += 1
        if self.counter >= self.patience:
            self.stop = True
        return False


def run_epoch(model, loader, criterion, optimizer, device, scaler, train: bool):
    """Tek bir epoch yürütür. train=True ise grad+backward yapar."""
    model.train(train)
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for imgs, labels in loader:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        # Mixed precision: float16'da forward, kayıp hesabı float32'de
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            logits = model(imgs)
            loss   = criterion(logits, labels)

        if train:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        total_loss    += loss.item() * imgs.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total_samples += imgs.size(0)

    return total_loss / total_samples, total_correct / total_samples


def train_model(cfg):
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Kullanılan cihaz: {device}")

    # --- Veri ---
    train_loader, val_loader, folder_classes = prepare_dataloaders(cfg)

    # --- Model ---
    model = build_model(num_classes=len(folder_classes),
                        dropout=cfg["dropout"]).to(device)

    # --- Optimizer: backbone'a daha düşük, classifier başına daha yüksek LR ---
    head_params, backbone_params = [], []
    for name, p in model.named_parameters():
        (head_params if name.startswith("classifier") else backbone_params).append(p)
    optimizer = optim.AdamW(
        [
            {"params": backbone_params, "lr": cfg["lr_backbone"]},
            {"params": head_params,     "lr": cfg["lr_head"]},
        ],
        weight_decay=cfg["weight_decay"]
    )

    # --- Scheduler: cosine annealing -> sonlara doğru daha küçük adım ---
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    # --- Loss: class weights ile az örnekli/zor sınıflar ağırlıklandırılır ---
    class_weights = torch.tensor(
        [LABEL_WEIGHTS[CLASS_TO_LABEL[fc]] for fc in folder_classes],
        dtype=torch.float32, device=device
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=cfg["label_smooth"])

    # --- AMP scaler ---
    scaler = torch.cuda.amp.GradScaler() if (cfg["use_amp"] and device.type == "cuda") else None

    # --- EarlyStopping & history ---
    stopper = EarlyStopping(patience=cfg["patience"])
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    best_state = None
    Path(cfg["save_dir"]).mkdir(exist_ok=True, parents=True)
    Path(cfg["plot_dir"]).mkdir(exist_ok=True, parents=True)

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer,
                                    device, scaler, train=True)
        with torch.no_grad():
            vl_loss, vl_acc = run_epoch(model, val_loader, criterion, optimizer,
                                        device, scaler=None, train=False)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        improved = stopper.step(vl_loss)
        if improved:
            best_state = copy.deepcopy(model.state_dict())

        print(f"[{epoch:02d}/{cfg['epochs']}] "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
              f"val_loss={vl_loss:.4f} val_acc={vl_acc:.4f} | "
              f"lr={optimizer.param_groups[0]['lr']:.2e} "
              f"({time.time()-t0:.1f}s)"
              + ("  *best*" if improved else ""))
        winsound.Beep(1000, 400)

        if stopper.stop:
            print(f"[INFO] EarlyStopping: {cfg['patience']} epoch boyunca "
                  f"iyileşme yok. Eğitim sonlandırıldı.")
            break

    # --- En iyi modeli yükle & kaydet ---
    if best_state is not None:
        model.load_state_dict(best_state)

    # ImageFolder sıralaması (alfabetik) ile yarışmadaki etiketleri eşle
    idx_to_label = {}
    for idx, folder_name in enumerate(folder_classes):
        idx_to_label[idx] = CLASS_TO_LABEL[folder_name]

    save_path = Path(cfg["save_dir"]) / "car_body_classifier.pt"
    torch.save({
        "state_dict":   model.state_dict(),
        "folder_classes": folder_classes,    # ImageFolder sırası
        "idx_to_label": idx_to_label,        # 0..7 -> 1..8 (proje numaralandırması)
        "img_size":     cfg["img_size"],
        "mean":         IMAGENET_MEAN,
        "std":          IMAGENET_STD,
        "architecture": "mobilenet_v3_large",
    }, save_path)
    print(f"[OK] Model kaydedildi: {save_path}")
    print(f"     Dosya boyutu: {save_path.stat().st_size / 1e6:.2f} MB")

    # --- History'i de dosyaya yaz (evaluate.py kullanacak) ---
    with open(Path(cfg["save_dir"]) / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # --- Grafikleri çiz ---
    plot_curves(history, cfg["plot_dir"])
    return model, history


# ============================================================
# 7) GRAFİKLER (LOSS & ACCURACY)
# ============================================================
def plot_curves(history, out_dir):
    epochs = range(1, len(history["train_loss"]) + 1)

    # Training & Validation Loss
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss",      linewidth=2)
    plt.plot(epochs, history["val_loss"],   label="Validation Loss", linewidth=2)
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(Path(out_dir) / "loss_curve.png", dpi=150)
    plt.close()

    # Training & Validation Accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [a*100 for a in history["train_acc"]], label="Train Acc",      linewidth=2)
    plt.plot(epochs, [a*100 for a in history["val_acc"]],   label="Validation Acc", linewidth=2)
    plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)")
    plt.title("Training & Validation Accuracy")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(Path(out_dir) / "accuracy_curve.png", dpi=150)
    plt.close()
    print(f"[OK] Loss & accuracy grafikleri kaydedildi -> {out_dir}/")


# ============================================================
# 8) ENTRY POINT
# ============================================================
if __name__ == "__main__":
    train_model(CONFIG)
