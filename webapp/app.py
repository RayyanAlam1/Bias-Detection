"""
News Bias Classifier - Flask Web App
=====================================
Best model: roberta-large  |  Accuracy: 85.43%  |  F1: 85.39%
Classes: 0=Left, 1=Center, 2=Right
"""

import os
import re
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import requests
import torch
import numpy as np
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification

# ─── Config ───────────────────────────────────────────────────────────────────
MODEL_DIR  = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "roberta-large-finetuned-v3", "checkpoint-502"
)
MODEL_REPO = os.getenv("HF_MODEL_REPO", "").strip()
MAX_LENGTH = 512
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
ROOT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEEDBACK_LOG_PATH = os.path.join(ROOT_DIR, "feedback_logs.jsonl")
USE_LEGACY_UI = os.getenv("USE_LEGACY_UI", "0") == "1"
MAX_SENTENCES = 12

LABEL_INFO = {
    0: {
        "name":        "Left",
        "description": "Left-leaning political bias detected.",
        "color":       "#3b82f6",
    },
    1: {
        "name":        "Center",
        "description": "Centrist / balanced reporting detected.",
        "color":       "#8b5cf6",
    },
    2: {
        "name":        "Right",
        "description": "Right-leaning political bias detected.",
        "color":       "#ef4444",
    },
}

# ─── Load model ───────────────────────────────────────────────────────────────
print(f"Device: {DEVICE}  |  CUDA: {torch.cuda.is_available()}")

if MODEL_REPO:
    print(f"Loading model from HF Hub: {MODEL_REPO} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_REPO,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    )
    model = model.to(DEVICE)
    if DEVICE == "cuda":
        model = model.half()
else:
    print(f"Loading model from {MODEL_DIR} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    # Build empty model from config (no weights yet — uses almost no RAM)
    config    = AutoConfig.from_pretrained(MODEL_DIR)
    model     = AutoModelForSequenceClassification.from_config(config)

    # Load weights directly into GPU tensors using safetensors (avoids pagefile)
    weights_path = os.path.join(MODEL_DIR, "model.safetensors")
    from safetensors.torch import load_file as st_load
    state_dict = st_load(weights_path, device=DEVICE)   # loads straight into GPU RAM

    # Older HuggingFace checkpoints store LayerNorm params as .gamma / .beta
    # but current transformers expects .weight / .bias — rename them so every
    # key maps correctly and strict=True can be used.
    renamed = {}
    for k, v in state_dict.items():
        if k.endswith(".gamma"):
            k = k[:-6] + ".weight"
        elif k.endswith(".beta"):
            k = k[:-5] + ".bias"
        renamed[k] = v
    del state_dict

    model.load_state_dict(renamed, strict=True)
    del renamed  # free the dict immediately

    model = model.to(DEVICE)
    if DEVICE == "cuda":
        model = model.half()  # fp16 saves ~700MB VRAM

model.eval()

print("✅ Model loaded and ready.")


# ─── Flask app ────────────────────────────────────────────────────────────────
app = Flask(__name__)


def _split_sentences(text: str):
    # Simple regex splitter keeps dependencies light and works well for prose.
    raw = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in raw if s and len(s.strip()) > 24]
    return sentences[:MAX_SENTENCES]


def _predict_text(text: str):
    if not text:
        return {"error": "No text provided."}, 400

    inputs = tokenizer(
        text,
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
    confidence = float(probs[pred_label])

    sentence_results = []
    for sentence in _split_sentences(text):
        s_inputs = tokenizer(
            sentence,
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        s_inputs = {k: v.to(model_device) for k, v in s_inputs.items()}
        with torch.no_grad():
            s_logits = model(**s_inputs).logits

        s_probs = torch.softmax(s_logits, dim=-1).squeeze().cpu().float().numpy()
        s_label = int(np.argmax(s_probs))
        sentence_results.append(
            {
                "text": sentence,
                "label": s_label,
                "name": LABEL_INFO[s_label]["name"],
                description": LABEL_INFO[s_label]["description"],
                "confidence": round(float(s_probs[s_label]) * 100, 2),
                "probabilities": {
                    LABEL_INFO[i]["name"]: round(float(p) * 100, 2)
                    for i, p in enumerate(s_probs)
                },
            }
        )

    sorted_idx = np.argsort(probs)[::-1]
    top1_idx = int(sorted_idx[0])
    top2_idx = int(sorted_idx[1])
    top1_conf_pct = round(float(probs[top1_idx]) * 100, 2)
    top2_conf_pct = round(float(probs[top2_idx]) * 100, 2)
    margin_pct = round(top1_conf_pct - top2_conf_pct, 2)

    # Use high-confidence sentences that align with the final label as evidence.
    evidence_sentences = sorted(
        [s for s in sentence_results if s["label"] == pred_label],
        key=lambda s: s["confidence"],
        reverse=True,
    )[:3]

    if evidence_sentences:
        reasoning_summary = (
            f"Predicted {LABEL_INFO[pred_label]['name']} because it has the highest "
            f"probability ({top1_conf_pct}%) with a {margin_pct}% margin over "
            f"{LABEL_INFO[top2_idx]['name']} ({top2_conf_pct}%). "
            f"Top supporting sentence confidence: {evidence_sentences[0]['confidence']}%."
        )
    else:
        reasoning_summary = (
            f"Predicted {LABEL_INFO[pred_label]['name']} because it has the highest "
            f"probability ({top1_conf_pct}%) with a {margin_pct}% margin over "
            f"{LABEL_INFO[top2_idx]['name']} ({top2_conf_pct}%)."
        )

    result = {
        "label": pred_label,
        "name": LABEL_INFO[pred_label]["name"],
        "": LABEL_INFO[pred_label][""],
        "description": LABEL_INFO[pred_label]["description"],
        "color": LABEL_INFO[pred_label]["color"],
        "confidence": round(confidence * 100, 2),
        "probabilities": {
            LABEL_INFO[i]["name"]: round(float(p) * 100, 2)
            for i, p in enumerate(probs)
        },
        "sentence_analysis": sentence_results,
        "sentence_count": len(sentence_results),
        "reasoning": {
            "summary": reasoning_summary,
            "top_prediction": {
                "label": top1_idx,
                "name": LABEL_INFO[top1_idx]["name"],
                "confidence": top1_conf_pct,
            },
            "runner_up": {
                "label": top2_idx,
                "name": LABEL_INFO[top2_idx]["name"],
                "confidence": top2_conf_pct,
            },
            "margin": margin_pct,
            "top_evidence_sentences": [
                {
                    "text": s["text"],
                    "name": s["name"],
                    "confidence": s["confidence"],
                }
                for s in evidence_sentences
            ],
        },
    }
    return result, 200


def _extract_article_text(url: str):
    parsed = requests.utils.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("URL must start with http:// or https://")

    response = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    title = (soup.title.string or "").strip() if soup.title else ""
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        tag.extract()

    body_container = soup.find("article") or soup.find("main") or soup.body
    if body_container is None:
        raise ValueError("Unable to parse content from URL")

    paragraphs = [p.get_text(" ", strip=True) for p in body_container.find_all("p")]
    text = "\n".join([p for p in paragraphs if len(p) > 40])

    if len(text) < 120:
        raise ValueError("Could not extract enough article text from URL")

    return title, text


def _append_feedback(entry):
    Path(os.path.dirname(FEEDBACK_LOG_PATH)).mkdir(parents=True, exist_ok=True)
    with open(FEEDBACK_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


@app.route("/")
def index():
    if USE_LEGACY_UI:
        return render_template("index_legacy.html")
    return render_template("index.html")


@app.route("/legacy")
def legacy():
    return render_template("index_legacy.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()
    result, status = _predict_text(text)
    return jsonify(result), status


@app.route("/predict_text", methods=["POST"])
def predict_text():
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()
    result, status = _predict_text(text)
    return jsonify(result), status


@app.route("/predict_url", methods=["POST"])
def predict_url():
    data = request.get_json(force=True)
    url = (data.get("url") or "").strip()
    if not url:
        return jsonify({"error": "No URL provided."}), 400

    try:
        title, article_text = _extract_article_text(url)
    except Exception as exc:
        return jsonify({"error": f"Could not process URL: {exc}"}), 400

    result, status = _predict_text(article_text)
    if status != 200:
        return jsonify(result), status

    result["source_url"] = url
    result["article_title"] = title
    result["extracted_chars"] = len(article_text)
    return jsonify(result), 200


@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.get_json(force=True)
    value = (data.get("feedback") or "").strip().lower()
    source_type = (data.get("source_type") or "text").strip().lower()
    source_value = (data.get("source_value") or "").strip()
    predicted_label = (data.get("predicted_label") or "").strip()

    if value not in ("agree", "disagree"):
        return jsonify({"error": "feedback must be 'agree' or 'disagree'"}), 400

    if source_type not in ("text", "url"):
        return jsonify({"error": "source_type must be 'text' or 'url'"}), 400

    payload_hash = hashlib.sha256(source_value.encode("utf-8")).hexdigest()[:16]
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "feedback": value,
        "source_type": source_type,
        "predicted_label": predicted_label,
        "payload_hash": payload_hash,
    }
    _append_feedback(entry)
    return jsonify({"ok": True})


if __name__ == "__main__":
    print("\n" + "="*55)
    print("  KhabarCheck Web App")
    print("  Model : roberta-large (85.43% accuracy)")
    print("  URL   : http://127.0.0.1:5000")
    print("="*55 + "\n")
    app.run(debug=False, host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
