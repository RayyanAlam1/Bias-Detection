"""
Hugging Face Spaces entry point — Gradio UI
============================================
Model weights are loaded from HF Hub automatically.
To host your own weights: upload them with `huggingface-cli upload`
or set HF_MODEL_REPO in the Space secrets.
"""

import os
import torch
import numpy as np
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Config ────────────────────────────────────────────────────────────────────
HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "RayyanAlam123/roberta-large-bias-detector")
MAX_LENGTH    = 512
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

LABEL_INFO = {
    0: {"name": "🔵 Left",   "color": "#3b82f6"},
    1: {"name": "⚖️ Center", "color": "#8b5cf6"},
    2: {"name": "🔴 Right",  "color": "#ef4444"},
}

# ── Load model ────────────────────────────────────────────────────────────────
print(f"Loading model from HF Hub: {HF_MODEL_REPO}  (device={DEVICE})")
tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_REPO)
model = AutoModelForSequenceClassification.from_pretrained(
    HF_MODEL_REPO,
    low_cpu_mem_usage=True,
    device_map=DEVICE,
    dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
)
model.eval()
print("✅ Model ready.")


# ── Prediction function ───────────────────────────────────────────────────────
def predict(text: str):
    if not text or not text.strip():
        return "Please enter some text.", {}, ""

    inputs = tokenizer(
        text.strip(),
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    model_device = next(model.parameters()).device
    inputs = {k: v.to(model_device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=-1).squeeze().cpu().float().numpy()
    pred_label = int(np.argmax(probs))
    confidence = float(probs[pred_label]) * 100

    label_str  = LABEL_INFO[pred_label]["name"]
    probs_dict = {LABEL_INFO[i]["name"]: float(p) for i, p in enumerate(probs)}
    summary    = f"**{label_str}** bias detected with **{confidence:.1f}%** confidence."

    return summary, probs_dict, label_str


# ── Gradio UI ─────────────────────────────────────────────────────────────────
examples = [
    ["The Republican party proposed sweeping tax cuts that will boost economic growth and freedom."],
    ["Scientists released a new study on climate change impacts across multiple regions."],
    ["The socialist agenda being pushed by Democrats threatens the foundations of American liberty."],
    ["Both parties met yesterday to discuss the new infrastructure spending bill."],
]

with gr.Blocks(
    title="News Bias Classifier",
    theme=gr.themes.Soft(primary_hue="purple"),
    css="""
        .result-box { font-size: 1.1em; padding: 10px; border-radius: 8px; }
        footer { display: none !important; }
    """,
) as demo:
    gr.Markdown("""
    # 📰 News Bias Classifier
    Detect **political bias** in news articles using `roberta-large` fine-tuned on 10,020 balanced news articles.
    **Accuracy: 85.43%** | Left 🔵 · Center ⚖️ · Right 🔴
    """)

    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="News Article / Headline",
                placeholder="Paste a news article, headline, or paragraph here…",
                lines=8,
            )
            submit_btn = gr.Button("🔍 Analyse Bias", variant="primary")

        with gr.Column(scale=1):
            summary_out = gr.Markdown(label="Result", elem_classes=["result-box"])
            probs_out   = gr.Label(label="Confidence per class", num_top_classes=3)

    gr.Examples(examples=examples, inputs=text_input, label="Try an example")

    submit_btn.click(
        fn=predict,
        inputs=text_input,
        outputs=[summary_out, probs_out, gr.Textbox(visible=False)],
    )
    text_input.submit(
        fn=predict,
        inputs=text_input,
        outputs=[summary_out, probs_out, gr.Textbox(visible=False)],
    )

    gr.Markdown("""
    ---
    **Model**: [`RayyanAlam123/roberta-large-bias-detector`](https://huggingface.co/RayyanAlam123/roberta-large-bias-detector) |
    **Code**: [`GitHub`](https://github.com/RayyanAlam1/Bias-Detection)
    """)

if __name__ == "__main__":
    demo.launch()
