#!/usr/bin/env python3
# app_or_demo/gradio_app.py
# FedTB-Nigeria Gradio Inference Demo
# DISCLAIMER: Research prototype. NOT for clinical use.
# Usage: pip install gradio && python app_or_demo/gradio_app.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import gradio as gr

from src.config import load_config
from src.paths import get_paths
from src.model import build_model
from src.data_utils import build_transforms

cfg    = load_config()
paths  = get_paths()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

val_transform = build_transforms(
    image_size=cfg["data"]["image_size"], split="val",
    normalize_mean=cfg["augmentation"]["normalize_mean"],
    normalize_std=cfg["augmentation"]["normalize_std"],
)

MODEL_LOADED = False
model = build_model(pretrained=False, num_classes=2, dropout=0.5)
model_path = paths["centralised_model_dir"] / "best_model.pth"
if model_path.exists():
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    MODEL_LOADED = True
    print(f"Model loaded from {model_path}")
else:
    print(f"WARNING: model not found at {model_path}. Run Notebook 07 first.")

DISCLAIMER = (
    "RESEARCH PROTOTYPE - NOT FOR CLINICAL USE.\n"
    "Do not use this output for diagnostic or treatment decisions.\n"
    "Always consult a qualified medical professional."
)


def predict_tb(image_pil):
    if not MODEL_LOADED:
        return {"Error": 1.0}, "Model not loaded. Run Notebook 07 first."
    if image_pil is None:
        return {"TB Negative": 0.5, "TB Positive": 0.5}, DISCLAIMER

    img_rgb = image_pil.convert("RGB")
    t = val_transform(img_rgb).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = torch.softmax(model(t), dim=1)[0]

    p_neg, p_pos = float(probs[0]), float(probs[1])
    conf = "High" if max(p_neg, p_pos) > 0.80 else "Low"
    interp = (
        f"TB Probability: {p_pos:.1%}\n"
        f"Confidence: {conf}\n\n"
        + DISCLAIMER
    )
    return {"TB Negative": p_neg, "TB Positive": p_pos}, interp


# Parameter placement for Gradio 6.x:
#   flagging_mode  → gr.Interface()   (renamed from allow_flagging in Gradio 5)
#   theme          → demo.launch()    (moved out of constructor in Gradio 6)
demo = gr.Interface(
    fn=predict_tb,
    inputs=gr.Image(type="pil", label="Upload Chest X-Ray Image"),
    outputs=[
        gr.Label(num_top_classes=2, label="TB Classification Probability"),
        gr.Textbox(label="Interpretation and Disclaimer", lines=8),
    ],
    title="FedTB-Nigeria: TB Detection Research Demo",
    description="RESEARCH PROTOTYPE ONLY - NOT FOR CLINICAL USE.",
    flagging_mode="never",            # Gradio 5+: replaces allow_flagging
)

if __name__ == "__main__":
    demo.launch(
        share=False,
        server_port=7860,
        theme=gr.themes.Soft(),       # Gradio 6+: moved here from Interface()
    )