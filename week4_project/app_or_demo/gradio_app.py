#!/usr/bin/env python3
# app_or_demo/gradio_app.py
# FedTB-Nigeria Gradio Inference Demo
# DISCLAIMER: Research prototype. NOT for clinical use.
# Usage: pip install gradio && python app_or_demo/gradio_app.py

import sys
import os
import socket
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

INTRO_TEXT = (
    "Upload a chest X-ray to see the model's research-only estimate of TB risk. "
    "If you do not have an image handy, try one of the example scans below."
)


def find_free_port(start_port: int = 7860, max_tries: int = 50) -> int:
    """Return the first available TCP port at or above start_port."""
    for port in range(start_port, start_port + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError(f"No free port found in range {start_port}-{start_port + max_tries - 1}")

def predict_tb(image_pil):
    if not MODEL_LOADED:
        return (
            {"TB Negative": 0.5, "TB Positive": 0.5},
            "Model not loaded. Run Notebook 07 first.",
            "Unknown",
            "Model unavailable",
        )
    if image_pil is None:
        return (
            {"TB Negative": 0.5, "TB Positive": 0.5},
            DISCLAIMER,
            "Ready",
            "Upload an image to begin",
        )
    img_rgb = image_pil.convert("RGB")
    t = val_transform(img_rgb).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(t), dim=1)[0]
    p_neg, p_pos = float(probs[0]), float(probs[1])
    conf = "High" if max(p_neg, p_pos) > 0.80 else "Low"
    tb_percent = p_pos * 100.0
    risk_label = "Higher" if p_pos >= 0.5 else "Lower"
    status = "TB-positive leaning" if p_pos >= 0.5 else "TB-negative leaning"
    interp = (
        f"Estimated TB risk: {tb_percent:.1f}%\n"
        f"Risk level: {risk_label}\n"
        f"Confidence: {conf}\n\n"
        + DISCLAIMER
    )
    return {"TB Negative": p_neg, "TB Positive": p_pos}, interp, status, f"Risk signal looks {status.lower()}."

examples = [
    str(paths["mock"] / "mock_0000_label0.png"),
    str(paths["mock"] / "mock_0001_label1.png"),
    str(paths["mock"] / "mock_0002_label0.png"),
]

with gr.Blocks(theme=gr.themes.Soft(), title="FedTB-Nigeria: TB Detection Research Demo") as demo:
    gr.Markdown(
        """
        # FedTB-Nigeria: TB Detection Research Demo
        Upload a chest X-ray to see the model's research-only estimate of TB risk.

        If you do not have an image handy, try one of the example scans below.
        """
    )

    with gr.Row():
        with gr.Column(scale=5):
            image_in = gr.Image(type="pil", label="Upload Chest X-Ray Image")
            gr.Examples(
                examples=examples,
                inputs=image_in,
                label="Try an example image",
            )
            submit_btn = gr.Button("Analyze Image", variant="primary")

        with gr.Column(scale=5):
            summary = gr.Markdown("### Result\nUpload an image to begin.")
            verdict = gr.Markdown("**Status:** Ready")
            output_label = gr.Label(num_top_classes=2, label="TB Classification Probability")
            output_text = gr.Textbox(label="Interpretation and Disclaimer", lines=9)

    with gr.Accordion("Important note", open=False):
        gr.Markdown(
            """
            This demo is intended for research and presentation only.
            It does not diagnose TB and should not be used for clinical decisions.
            Always consult a qualified medical professional.
            """
        )

    def run_prediction(image_pil):
        probs, text, status, summary_text = predict_tb(image_pil)
        return probs, text, f"**Status:** {status}", f"### Result\n{summary_text}"

    submit_btn.click(
        fn=run_prediction,
        inputs=image_in,
        outputs=[output_label, output_text, verdict, summary],
    )

if __name__ == "__main__":
    port = find_free_port(int(os.environ.get("PORT", 7860)))
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=port,
        theme=gr.themes.Soft(),
    )
