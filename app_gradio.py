import os
import torch
import gradio as gr
import pandas as pd
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Model klasörü (reptiles.py OUTPUT_DIR ile aynı) :contentReference[oaicite:2]{index=2}
MODEL_DIR = os.environ.get("MODEL_DIR", "reptile_vit_out")

device = "cuda" if torch.cuda.is_available() else "cpu"

@torch.inference_mode()
def predict(img: Image.Image, top_k: int = 5):
    if img is None:
        return "Görsel yükleyin.", pd.DataFrame(columns=["class", "prob"])

    img = img.convert("RGB")

    # Model + processor yükle (local klasörden)
    processor = AutoImageProcessor.from_pretrained(MODEL_DIR)
    model = AutoModelForImageClassification.from_pretrained(MODEL_DIR).to(device)
    model.eval()

    inputs = processor(images=img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits[0], dim=-1)

    k = min(int(top_k), probs.numel())
    top_probs, top_ids = torch.topk(probs, k=k)

    id2label = model.config.id2label
    rows = []
    for p, idx in zip(top_probs.tolist(), top_ids.tolist()):
        label = id2label.get(str(idx), id2label.get(idx, str(idx)))
        rows.append({"class": label, "prob": float(p)})

    df = pd.DataFrame(rows)
    best = df.iloc[0]
    result_text = f"Tahmin: **{best['class']}** (olasılık: {best['prob']:.3f})"
    return result_text, df

with gr.Blocks(title="Reptile ViT Classifier") as demo:
    gr.Markdown("# Reptile Sınıflandırma (ViT)\nGörsel yükle → Tahmin al")

    with gr.Row():
        img_in = gr.Image(type="pil", label="Görsel Yükle")
        with gr.Column():
            topk = gr.Slider(1, 10, value=5, step=1, label="Top-K")
            btn = gr.Button("Tahmin Et")

    out_text = gr.Markdown()
    out_table = gr.Dataframe(label="Top-K Olasılıklar", headers=["class", "prob"], interactive=False)

    btn.click(fn=predict, inputs=[img_in, topk], outputs=[out_text, out_table])

demo.launch()
