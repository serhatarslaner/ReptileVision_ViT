import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageFile
from datasets import load_dataset
from transformers import ViTImageProcessor, ViTForImageClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score

# --------- BOZUK GÖRSEL DAYANIKLILIĞI ----------
# Bazı yarım dosyalar için tolerans (istersen kapatabilirsin)
ImageFile.LOAD_TRUNCATED_IMAGES = True
# ------------------------------------------------

# --- 1. AYARLAR ---
DATASET_ROOT = r"C:\Users\serha\OneDrive\Desktop\Reptiles\archive"
MODEL_NAME = "google/vit-base-patch16-224"
OUTPUT_DIR = "./reptile_vit_out"
EPOCHS = 20
BATCH_SIZE = 16

# --- 2. GPU KONTROLÜ ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n📢 Kullanılan Cihaz: {device.upper()}")
if device == "cuda":
    print(f"✅ Ekran Kartı: {torch.cuda.get_device_name(0)}")

# --- 3. DATASET: PATH ÜZERİNDEN OKU (BOZUKLARI SKIP EDEBİLELİM) ---
train_dir = os.path.join(DATASET_ROOT, "train")
test_dir  = os.path.join(DATASET_ROOT, "test")
valid_dir = os.path.join(DATASET_ROOT, "valid")

data_files = {
    "train": os.path.join(train_dir, "**"),
    "test": os.path.join(test_dir, "**"),
    "validation": os.path.join(valid_dir, "**"),
}

print(f"\n📂 Veri seti okunuyor: {DATASET_ROOT}")
# keep_in_memory=False default, path bilgisi kalsın
ds = load_dataset("imagefolder", data_files=data_files)

labels = ds["train"].features["label"].names
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for i, label in enumerate(labels)}
print(f"📊 Toplam Sınıf Sayısı: {len(labels)}")
print("Sınıflar:", labels)

processor = ViTImageProcessor.from_pretrained(MODEL_NAME)

# --- 4) BOZUK GÖRSELİ YAKALAYAN "SAFE" DATASET WRAPPER ---
class SafeDataset(torch.utils.data.Dataset):
    """
    HF dataset'ten 'image' + 'label' alır.
    image decode/pil load hatası olursa None döndürür (collate_fn filtreler).
    """
    def __init__(self, hf_ds, processor, max_retry=3):
        self.ds = hf_ds
        self.processor = processor
        self.max_retry = max_retry

    def __len__(self):
        return len(self.ds)

    def _encode(self, img, label):
        img = img.convert("RGB")
        enc = self.processor(img, return_tensors="pt")
        return {
            "pixel_values": enc["pixel_values"].squeeze(0),
            "labels": int(label),
        }

    def __getitem__(self, idx):
        # Bozuk dosyaya denk gelirse birkaç kere başka index dene
        tries = 0
        cur = idx
        while tries < self.max_retry:
            try:
                ex = self.ds[cur]
                img = ex["image"]   # burada decode olabilir
                lab = ex["label"]
                return self._encode(img, lab)
            except Exception as e:
                # bozuk -> retry: random başka bir örnek
                tries += 1
                cur = random.randint(0, len(self.ds) - 1)

        # hala olmadıysa: None -> collate_fn atacak
        return None

train_torch = SafeDataset(ds["train"], processor)
valid_torch = SafeDataset(ds["validation"], processor)

# --- 5. MODEL ---
model = ViTForImageClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(labels),
    id2label={str(k): v for k, v in id2label.items()},
    label2id={v: str(k) for k, v in id2label.items()},
    ignore_mismatched_sizes=True
).to(device)

# --- 6. METRİK ---
def compute_metrics(eval_pred):
    logits, y_true = eval_pred
    y_pred = np.argmax(logits, axis=1)
    return {"accuracy": accuracy_score(y_true, y_pred)}

# --- 7. COLLATE: None OLANLARI FİLTRELE (ÇÖKMEYİ ENGELLER) ---
def collate_fn(batch):
    # bozuklardan gelen None'ları at
    batch = [x for x in batch if x is not None]
    # çok kötü durumda batch tamamen boşsa -> en az 1 örnek uydur (trainer çökmemesi için)
    if len(batch) == 0:
        # rastgele sıfır tensörü ile "dummy" batch
        dummy = torch.zeros((3, 224, 224), dtype=torch.float32)
        return {"pixel_values": dummy.unsqueeze(0), "labels": torch.tensor([0])}

    pixel_values = torch.stack([x["pixel_values"] for x in batch])
    labels_t = torch.tensor([x["labels"] for x in batch], dtype=torch.long)
    return {"pixel_values": pixel_values, "labels": labels_t}

# --- 8. TRAINING ARGS ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    learning_rate=2e-5,
    weight_decay=0.01,

    remove_unused_columns=False,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=(device == "cuda"),

    dataloader_num_workers=0,
    logging_steps=50,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_torch,
    eval_dataset=valid_torch,
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
)

print("\n🏁 Eğitim Başlıyor...")
trainer.train()

print(f"\n💾 Model kaydediliyor: {OUTPUT_DIR}")
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

# --- 9. GRAFİK ---
history = trainer.state.log_history
train_loss = [x["loss"] for x in history if "loss" in x]
eval_loss  = [x["eval_loss"] for x in history if "eval_loss" in x]
eval_acc   = [x["eval_accuracy"] for x in history if "eval_accuracy" in x]

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_loss, label="Training Loss")
if eval_loss:
    plt.plot(np.linspace(0, len(train_loss), len(eval_loss)), eval_loss, label="Validation Loss")
plt.title("Kayıp (Loss)")
plt.legend()

plt.subplot(1, 2, 2)
if eval_acc:
    plt.plot(eval_acc, label="Validation Accuracy")
plt.title("Doğruluk (Accuracy)")
plt.legend()

plt.savefig("egitim_sonuclari.png")
print("📈 Grafik kaydedildi: egitim_sonuclari.png")
plt.show()
