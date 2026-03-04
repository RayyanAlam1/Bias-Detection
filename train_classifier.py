import argparse
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class Config:
    data_path: str
    text_column: str
    label_column: str
    model_name: str
    output_dir: str
    max_length: int
    batch_size: int
    learning_rate: float
    num_train_epochs: float
    weight_decay: float
    seed: int
    eval_split_ratio: float


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Fine-tune a Transformer for CSV text classification.")
    p.add_argument("--data_path", required=True, help="Path to CSV file.")
    p.add_argument("--text_column", default="content_clean", help="CSV column containing text.")
    p.add_argument("--label_column", default="label", help="CSV column containing numeric labels.")
    p.add_argument(
        "--model_name",
        default="bert-base-multilingual-cased",
        help="HF model name (or local directory) to fine-tune.",
    )
    p.add_argument("--output_dir", default=os.path.join("outputs", "model"), help="Where to save the model.")
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--num_train_epochs", type=float, default=3.0)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval_split_ratio", type=float, default=0.1)

    a = p.parse_args()
    return Config(
        data_path=a.data_path,
        text_column=a.text_column,
        label_column=a.label_column,
        model_name=a.model_name,
        output_dir=a.output_dir,
        max_length=a.max_length,
        batch_size=a.batch_size,
        learning_rate=a.learning_rate,
        num_train_epochs=a.num_train_epochs,
        weight_decay=a.weight_decay,
        seed=a.seed,
        eval_split_ratio=a.eval_split_ratio,
    )


def compute_metrics(eval_pred: Any) -> Dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro")),
    }


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)

    df = pd.read_csv(cfg.data_path)
    if cfg.text_column not in df.columns:
        raise ValueError(f"Missing text_column={cfg.text_column}. Found columns: {list(df.columns)}")
    if cfg.label_column not in df.columns:
        raise ValueError(f"Missing label_column={cfg.label_column}. Found columns: {list(df.columns)}")

    df = df[[cfg.text_column, cfg.label_column]].dropna()
    df[cfg.label_column] = df[cfg.label_column].astype(int)

    # Ensure labels are 0..N-1
    unique_labels = sorted(df[cfg.label_column].unique().tolist())
    label_map = {lab: i for i, lab in enumerate(unique_labels)}
    df[cfg.label_column] = df[cfg.label_column].map(label_map).astype(int)
    num_labels = len(unique_labels)

    dataset = Dataset.from_pandas(df, preserve_index=False)
    split = dataset.train_test_split(test_size=cfg.eval_split_ratio, seed=cfg.seed)
    train_ds = split["train"]
    eval_ds = split["test"]

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)

    def tokenize(batch: Dict[str, Any]) -> Dict[str, Any]:
        return tokenizer(
            batch[cfg.text_column],
            truncation=True,
            max_length=cfg.max_length,
        )

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=[cfg.text_column])
    eval_ds = eval_ds.map(tokenize, batched=True, remove_columns=[cfg.text_column])

    train_ds = train_ds.rename_column(cfg.label_column, "labels")
    eval_ds = eval_ds.rename_column(cfg.label_column, "labels")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name, num_labels=num_labels)

    args = TrainingArguments(
        output_dir=cfg.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        num_train_epochs=cfg.num_train_epochs,
        weight_decay=cfg.weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=cfg.seed,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    print(f"Saved model to: {cfg.output_dir}")
    print(f"Label mapping used (original->new): {label_map}")


if __name__ == "__main__":
    main()

