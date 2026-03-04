"""
News Bias Classifier - Flask Web App
=====================================
Best model: roberta-large  |  Accuracy: 85.43%  |  F1: 85.39%
Classes: 0=Left, 1=Center, 2=Right
"""

import os
import torch
import numpy as np
from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ─── Config ───────────────────────────────────────────────────────────────────
MODEL_DIR  = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "roberta-large-finetuned-v3", "checkpoint-502"
)
MAX_LENGTH = 512
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

LABEL_INFO = {
    0: {
        "name":        "Left",
        "emoji":       "🔵",
        "description": "Left-leaning political bias detected.",
        "color":       "#3b82f6",   # blue
    },
    1: {
        "name":        "Center",
        "emoji":       "⚖️",
        "description": "Centrist / balanced reporting detected.",
        "color":       "#8b5cf6",   # purple
    },
    2: {
        "name":        "Right",
        "emoji":       "🔴",
        "description": "Right-leaning political bias detected.",
        "color":       "#ef4444",   # red
    },
}

# ─── Load model once at startup ───────────────────────────────────────────────
print(f"Loading model from {MODEL_DIR} ...")
print(f"Device: {DEVICE}  |  CUDA: {torch.cuda.is_available()}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model     = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR,
    low_cpu_mem_usage=True,
    device_map=DEVICE,                                        # load straight to GPU — skips CPU RAM
    dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
)
model.eval()

print("✅ Model loaded and ready.")

# ─── Flask app ────────────────────────────────────────────────────────────────
app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"error": "No text provided."}), 400

    # Tokenise
    inputs = tokenizer(
        text,
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    # Move inputs to the same device as the model
    model_device = next(model.parameters()).device
    inputs = {k: v.to(model_device) for k, v in inputs.items()}

    # Inference
    with torch.no_grad():
        logits = model(**inputs).logits

    probs      = torch.softmax(logits, dim=-1).squeeze().cpu().float().numpy()
    pred_label = int(np.argmax(probs))
    confidence = float(probs[pred_label])

    result = {
        "label":       pred_label,
        "name":        LABEL_INFO[pred_label]["name"],
        "emoji":       LABEL_INFO[pred_label]["emoji"],
        "description": LABEL_INFO[pred_label]["description"],
        "color":       LABEL_INFO[pred_label]["color"],
        "confidence":  round(confidence * 100, 2),
        "probabilities": {
            LABEL_INFO[i]["name"]: round(float(p) * 100, 2)
            for i, p in enumerate(probs)
        },
    }
    return jsonify(result)


if __name__ == "__main__":
    print("\n" + "="*55)
    print("  News Bias Classifier Web App")
    print("  Model : roberta-large (85.43% accuracy)")
    print("  URL   : http://127.0.0.1:5000")
    print("="*55 + "\n")
    app.run(debug=False, host="0.0.0.0", port=5000)
