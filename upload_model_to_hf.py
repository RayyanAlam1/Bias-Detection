"""
Upload the fine-tuned roberta-large model to Hugging Face Hub.
Run: python upload_model_to_hf.py

You will be prompted for your HF token (get it from https://huggingface.co/settings/tokens).
Make sure the token has WRITE access.
"""

from huggingface_hub import HfApi, login
import os

# ── Config ────────────────────────────────────────────────────────────────────
HF_USERNAME  = "RayyanAlam123"
MODEL_REPO   = f"{HF_USERNAME}/roberta-large-bias-detector"
CKPT_DIR     = r"C:\Users\22K-2127\Desktop\FYP\roberta-large-finetuned-v3\checkpoint-502"

# ── Model card ────────────────────────────────────────────────────────────────
MODEL_CARD = """---
language: en
license: mit
tags:
  - text-classification
  - political-bias
  - roberta
  - news
datasets:
  - custom
metrics:
  - accuracy
  - f1
model-index:
  - name: roberta-large-bias-detector
    results:
      - task:
          type: text-classification
        metrics:
          - type: accuracy
            value: 0.8543
          - type: f1
            value: 0.8539
---

# KhabarCheck - RoBERTa-Large Bias Model

Fine-tuned `roberta-large` for political bias classification in news articles for KhabarCheck.

## Model Details

| | |
|---|---|
| **Base model** | `roberta-large` |
| **Task** | 3-class text classification |
| **Classes** | `0` = Left 🔵, `1` = Center ⚖️, `2` = Right 🔴 |
| **Accuracy** | **85.43%** |
| **F1 (weighted)** | **85.39%** |
| **Max tokens** | 512 |

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_id = "RayyanAlam123/roberta-large-bias-detector"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)
model.eval()

text = "The president signed a new executive order on climate policy."
inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
with torch.no_grad():
    logits = model(**inputs).logits
probs = torch.softmax(logits, dim=-1).squeeze()
labels = ["Left", "Center", "Right"]
print(f"Prediction: {labels[probs.argmax()]}  ({probs.max()*100:.1f}% confidence)")
```

## Training Details

| Hyperparameter | Value |
|---|---|
| Learning rate | 1e-5 |
| LR scheduler | cosine |
| Effective batch size | 32 (4 × 8 grad accum) |
| Warmup | 8% |
| Mixed precision | bf16 |
| Early stopping patience | 3 |

## Dataset

10,020 balanced news articles — 3,340 each for Left, Center, Right.

## Live Demo

[Hugging Face Space](https://huggingface.co/spaces/RayyanAlam123/KhabarCheck)
"""

# ── Upload ────────────────────────────────────────────────────────────────────
print("=" * 55)
print("  Uploading model to Hugging Face Hub")
print(f"  Repo : {MODEL_REPO}")
print(f"  From : {CKPT_DIR}")
print("=" * 55)

# Login — use HF_TOKEN env var if set, otherwise prompt interactively
token = os.environ.get("HF_TOKEN")
if token:
    login(token=token, add_to_git_credential=False)
    print("✅ Logged in via HF_TOKEN env var.")
else:
    print("\nNo HF_TOKEN env var found.")
    print("Please enter your HF token (from https://huggingface.co/settings/tokens):")
    login()

api = HfApi()

# Create model repo (ok if already exists)
print(f"\n📁 Creating repo: {MODEL_REPO} ...")
api.create_repo(repo_id=MODEL_REPO, repo_type="model", exist_ok=True, private=False)

# Write model card
card_path = os.path.join(CKPT_DIR, "README.md")
with open(card_path, "w", encoding="utf-8") as f:
    f.write(MODEL_CARD)
print("📝 Model card written.")

# Upload all checkpoint files
print(f"\n⬆️  Uploading files from {CKPT_DIR} ...")
print("   (This will take a few minutes — model is ~1.3 GB)\n")

api.upload_folder(
    folder_path=CKPT_DIR,
    repo_id=MODEL_REPO,
    repo_type="model",
    commit_message="Upload fine-tuned roberta-large (85.43% accuracy, F1=85.39%)",
)

print("\n" + "=" * 55)
print(f"✅ Upload complete!")
print(f"   Model URL : https://huggingface.co/{MODEL_REPO}")
print("=" * 55)
