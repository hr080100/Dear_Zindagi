import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
import torch
from sklearn.metrics import classification_report

def train_emotion_model(model_name: str, train_path: str, val_path: str, output_dir: str):
    # Load and encode labels
    print("✅ Using device:", torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU only")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    le = LabelEncoder()
    train_df["labels"] = le.fit_transform(train_df["label"])
    val_df["labels"] = le.transform(val_df["label"])

    # Save label map
    os.makedirs(output_dir, exist_ok=True)
    pd.Series(le.classes_).to_csv(os.path.join(output_dir, "label_map.csv"), index_label="index")

    # Convert to HF datasets
    train_ds = Dataset.from_pandas(train_df[["text", "labels"]])
    val_ds = Dataset.from_pandas(val_df[["text", "labels"]])

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(example):
        return tokenizer(example["text"], truncation=True)

    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(le.classes_)
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return {"accuracy": (preds == labels).mean()}

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, "logs"),
        save_total_limit=1
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()

    # Save model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ Saved model: {model_name} to {output_dir}")

if __name__ == "__main__":
    # Example usage for RoBERTa
    train_emotion_model(
        model_name="roberta-base",
        train_path=".\\cleaned_data\\empathetic_train_cleaned.csv",
        val_path=".\\cleaned_data\\empathetic_valid_cleaned.csv",
        output_dir=".\\models\\empathetic\\roberta\\emotion_roberta_finetuned"
    )
