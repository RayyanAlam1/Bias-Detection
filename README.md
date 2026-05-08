---
title: KhabarCheck
: 📰
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
license: mit
short_description: Detect political bias in news with RoBERTa-Large
---

# KhabarCheck 📰

**Detect political bias in news articles** using a fine-tuned `roberta-large` model.

| | |
|---|---|
| **Model** | RoBERTa-Large (355M parameters) |
| **Accuracy** | 85.43% |
| **F1 Score** | 85.39% |
| **Classes** | Left  · Center  · Right  |
| **Dataset** | 10,020 balanced news articles |
| **Live Demo** | [HF Spaces](https://huggingface.co/spaces/RayyanAlam123/KhabarCheck) |

---

## 🚀 Local Setup

```powershell
git clone https://github.com/RayyanAlam1/Bias-Detection
cd Bias-Detection
pip install -r requirements.txt
python run_app.py        # Flask app → http://127.0.0.1:5000
```

---

## 📊 Performance

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Left 🔵 | 85.1% | 90.7% | 87.8% |
| Center ⚖️ | 85.9% | 82.6% | 84.2% |
| Right 🔴 | 85.4% | 82.9% | 84.1% |
| **Overall** | **85.4%** | **85.4%** | **85.4%** |

---

## 🗂️ Project Structure

```
Bias-Detection/
├── webapp/               # Flask web app
│   ├── app.py            # Backend (model loading + /predict endpoint)
│   ├── templates/
│   │   └── index.html    # Frontend UI
│   └── static/
│       └── style.css
├── Dockerfile            # HF Spaces deployment entrypoint
├── hf_app.py             # Legacy Gradio app for earlier deployment attempts
├── fine_tuning.ipynb     # Training notebook
├── train_classifier.py   # CLI training script
├── predict.py            # CLI prediction script
├── requirements.txt
├── run_app.py            # Local launcher
└── tests/
    └── test_app.py       # Unit tests
```

---

## ⚙️ CI/CD Pipeline

Every push to `main` automatically:
1. **Lints** the code with flake8
2. **Runs unit tests** with pytest (mocked model — no GPU needed)
3. **Deploys** to [Hugging Face Spaces](https://huggingface.co/spaces/RayyanAlam123/KhabarCheck)

---

## ✨ New App Features (Web UI)

- **Dual input mode**: analyze pasted text or a live article URL
- **Sentence-level breakdown**: class/confidence shown for up to 12 sentences
- **User feedback capture**: Agree/Disagree signals are logged to `feedback_logs.jsonl`

New API routes:
- `POST /predict_text`
- `POST /predict_url`
- `POST /feedback`

Backward compatibility:
- Existing `POST /predict` is still available.

---

## 🔁 Safe Revert / Legacy Mode

If anything goes wrong, you can immediately revert UI behavior:

1. Open legacy page directly: `http://127.0.0.1:5000/legacy`
2. Or start app in legacy mode:

```powershell
$env:USE_LEGACY_UI="1"
python run_app.py
```

To return to enhanced UI:

```powershell
$env:USE_LEGACY_UI="0"
python run_app.py
```

Legacy snapshots are saved in:
- `webapp/legacy_backup/index_2026-04-13.html`
- `webapp/legacy_backup/style_2026-04-13.css`
- `webapp/legacy_backup/app_2026-04-13.py`

Full backend rollback (PowerShell):

```powershell
Copy-Item .\webapp\legacy_backup\app_2026-04-13.py .\webapp\app.py -Force
Copy-Item .\webapp\templates\index_legacy.html .\webapp\templates\index.html -Force
Copy-Item .\webapp\static\style_legacy.css .\webapp\static\style.css -Force
python run_app.py
```

---

## 🏋️ Training Details

```
Model:        roberta-large
MAX_LENGTH:   512
LR:           1e-5  (cosine scheduler)
Eff. batch:   32 (batch=4 × grad_accum=8)
Warmup:       8%
Epochs:       10 with early stopping (patience=3)
Mixed prec.:  bf16
```

---

## 3) Fine-tune (legacy section)

```powershell
python train_classifier.py `
  --data_path "c:\Users\22K-2127\Desktop\Test\TEST\balanced_bias_news_dataset.csv" `
  --text_column "content_clean" `
  --label_column "label" `
  --model_name "bert-base-multilingual-cased" `
  --output_dir "outputs\bert_bias_classifier"
```

## 4) Predict (inference)

```powershell
python predict.py --model_dir "outputs\bert_bias_classifier" --text "Some example text here"
```

## Notes

- Your existing `config.json` suggests you’re doing **BERT sequence classification**, not a chat-style “LLM generation” model.
- If you want *true LLM* fine-tuning (e.g., Llama/Mistral) tell me your target model + GPU VRAM, and I’ll generate the LoRA/QLoRA setup for that.
...
