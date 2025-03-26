import os
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from transformers import (
    DistilBertTokenizerFast, 
    DistilBertForSequenceClassification
)
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_and_prepare_data(test_path, tokenizer, label_encoder):
    df = pd.read_csv(test_path)
    df["labels"] = label_encoder.fit_transform(df["label"])

    # Tokenize
    tokens = tokenizer(list(df["text"]), padding=True, truncation=True, return_tensors="pt")
    labels = torch.tensor(df["labels"].values)
    dataset = TensorDataset(tokens["input_ids"], tokens["attention_mask"], labels)
    return dataset, label_encoder


def evaluate_model(model, dataset, label_encoder, model_name, output_dir):
    model.to(device)
    model.eval()
    dataloader = DataLoader(dataset, batch_size=8)
    preds, truths = [], []

    for input_ids, attention_mask, labels in dataloader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            pred = torch.argmax(logits, dim=1)

        preds.extend(pred.cpu().numpy())
        truths.extend(labels.cpu().numpy())


    y_pred = label_encoder.inverse_transform(preds)
    y_true = label_encoder.inverse_transform(truths)

    # Save report
    os.makedirs(output_dir, exist_ok=True)
    report = classification_report(y_true, y_pred, digits=4)
    with open(os.path.join(output_dir, f"{model_name}_report.txt"), "w") as f:
        f.write(report)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=label_encoder.classes_)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.savefig(os.path.join(output_dir, f"{model_name}_confusion_matrix.png"))
    plt.close()

    return report


def compare_base_vs_finetuned_distilbert():
    test_path = ".\\cleaned_data\\empathetic_test_cleaned.csv"
    fine_tuned_dir = ".\\models\\empathetic\\emotion_distilbert_finetuned"
    base_model_name = "distilbert-base-uncased"
    output_dir_base = ".\\evaluation\\distilbert_base_vs_finetuned\\base_distilbert"
    output_dir_finetuned = ".\\evaluation\\distilbert_base_vs_finetuned\\finetuned_distilbert"

    tokenizer = DistilBertTokenizerFast.from_pretrained(base_model_name)
    label_encoder = LabelEncoder()

    dataset, le = load_and_prepare_data(test_path, tokenizer, label_encoder)

    print("\n📉 Base DistilBERT Performance:")
    base_model = DistilBertForSequenceClassification.from_pretrained(base_model_name, num_labels=len(le.classes_))
    print(evaluate_model(base_model, dataset, le, "base_distilbert", output_dir_base))

    print("\n📈 Fine-tuned DistilBERT Performance:")
    fine_model = DistilBertForSequenceClassification.from_pretrained(fine_tuned_dir)
    print(evaluate_model(fine_model, dataset, le, "finetuned_distilbert", output_dir_finetuned))


if __name__ == "__main__":
    compare_base_vs_finetuned_distilbert()
