- Serhat Arslaner   220502043 -


Reptile Image Classification (ViT) + Gradio Web UI

Bu proje, sürüngen görsellerini sınıflandırmak için eğitilmiş bir Vision Transformer (ViT) modelini kullanır ve sonucu kullanıcı dostu bir Gradio web arayüzü üzerinden gösterir.

Özellikler
- Bilgisayardan görsel yükleme
- Otomatik ön işleme (normalize / resize / crop) (modelin processor’ı ile)
- Tek tıkla tahmin alma
- Top-K olasılık tablosu (en olası sınıflar ve olasılıkları)
- Basit ve erişilebilir web arayüzü

Proje Yapısı
Reptiles/
├─ app_gradio.py            # Gradio arayüzü
├─ reptiles.py              # eğitim / veri hazırlama scripti
├─ requirements.txt
├─ reptile_vit_out/         # Eğitilmiş model + processor (from_pretrained ile yüklenir)
│  ├─ config.json
│  ├─ model.safetensors / pytorch_model.bin
│  ├─ preprocessor_config.json
│  └─ ...
└─ archive/
   ├─ train/
   ├─ valid/
   └─ test/

Kurulum

1) Ortam (önerilen)
Python 3.10+ önerilir.

Windows (PowerShell)
python -m venv .venv
.venv\Scripts\activate

macOS / Linux
python -m venv .venv
source .venv/bin/activate

2) Bağımlılıklar
pip install -r requirements.txt

Eğer requirements.txt yoksa:
pip install torch transformers pillow gradio pandas

Çalıştırma (Gradio)
Model klasörü varsayılan olarak reptile_vit_out beklenir.

python app_gradio.py

Çalışınca terminalde bir URL çıkar (genelde http://127.0.0.1:7860). Tarayıcıdan açıp görsel yükleyerek test edebilirsin.

Model Klasörü Seçmek (opsiyonel)
Model klasör adın farklıysa ortam değişkeni ile verebilirsin:

Windows (PowerShell)
$env:MODEL_DIR="reptile_vit_out"
python app_gradio.py

macOS / Linux
MODEL_DIR="reptile_vit_out" python app_gradio.py

Kullanım
1. “Görsel Yükle” alanından bir görsel seç
2. (İstersen) Top-K değerini ayarla
3. “Tahmin Et” butonuna bas
4. Sonuç: Tahmin sınıfı + olasılık ve Top-K tablo

Notlar / Sınırlamalar
- Model, sadece eğitildiği sınıfları tahmin eder. “Bilinmeyen sınıf” (unknown) eğitilmediyse, alakasız bir görselde de en yakın sınıfa zorlayabilir.
- En iyi sonuç için: net, tek nesne/sürüngen içeren, iyi ışıklı görseller kullan.

Sorun Giderme

1) reptile_vit_out bulunamadı
- reptile_vit_out/ klasörünün proje kökünde olduğundan emin ol.
- Alternatif olarak MODEL_DIR ile doğru yolu ver.

2) CUDA / GPU kullanımı
- Otomatik olarak GPU varsa kullanır. Yoksa CPU’da çalışır.
- Torch CUDA kurulumun yoksa CPU sürümüyle de çalışır (daha yavaş olabilir).

3) ModuleNotFoundError
- Sanal ortamın aktif olduğundan emin ol.
- pip install -r requirements.txt tekrar çalıştır.