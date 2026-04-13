"""
test_center.py — Test 50 Center samples and show misclassifications clearly.
Run:  python test_center.py
"""
import torch, warnings, pandas as pd
warnings.filterwarnings("ignore")
from transformers import AutoTokenizer, AutoModelForSequenceClassification

CKPT = r"C:\Users\22K-2127\Desktop\FYP\roberta-large-finetuned-v3\checkpoint-502"
LABEL = {0: "Left  ", 1: "Center", 2: "Right "}

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(CKPT)
model = AutoModelForSequenceClassification.from_pretrained(CKPT)
model.eval()
print("Ready.\n")

df = pd.read_csv(r"C:\Users\22K-2127\Desktop\FYP\balanced_bias_news_dataset.csv")

N = 50
print(f"=== {N} CENTER samples (label=1) ===")
correct = 0
wrong_as_right = 0
wrong_as_left  = 0

for _, row in df[df["label"] == 1].head(N).iterrows():
    text = str(row["text_field"])
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).squeeze().tolist()
    pred  = logits.argmax().item()
    ok    = (pred == 1)
    if ok:
        correct += 1
    elif pred == 2:
        wrong_as_right += 1
    else:
        wrong_as_left  += 1

    status = "OK   " if ok else "WRONG"
    print(
        f"[{status}] Pred={LABEL[pred]}  "
        f"L={probs[0]*100:4.1f}%  C={probs[1]*100:4.1f}%  R={probs[2]*100:4.1f}%  |  "
        f"{text[:75]}"
    )

print()
print(f"Score           : {correct}/{N}  ({correct/N*100:.0f}%)")
print(f"Wrong as Right  : {wrong_as_right}")
print(f"Wrong as Left   : {wrong_as_left}")
