import argparse

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run inference with a fine-tuned classifier.")
    p.add_argument("--model_dir", required=True, help="Directory containing saved model + tokenizer.")
    p.add_argument("--text", required=True, help="Text to classify.")
    p.add_argument("--max_length", type=int, default=256)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device)
    model.eval()

    inputs = tokenizer(args.text, return_tensors="pt", truncation=True, max_length=args.max_length).to(device)
    with torch.no_grad():
        out = model(**inputs)
        probs = torch.softmax(out.logits, dim=-1).squeeze(0).cpu().tolist()
        pred = int(torch.argmax(out.logits, dim=-1).item())

    print({"predicted_label": pred, "probabilities": probs})


if __name__ == "__main__":
    main()

