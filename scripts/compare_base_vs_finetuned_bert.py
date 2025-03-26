import os
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification
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


def tokenize_data(df, tokenizer):
    dataset = Dataset.from_pandas(df[["text", "labels"]])
    dataset = dataset.map(lambda x: tokenizer(x["text"], truncation=True), batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return dataset


def evaluate_model(model, dataset, label_encoder):
    model.eval()
    predictions, true_labels = [], []

    for batch in dataset:
        input_ids = batch["input_ids"].unsqueeze(0)
        attention_mask = batch["attention_mask"].unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            pred = torch.argmax(outputs.logits, dim=1).item()

        predictions.append(pred)
        true_labels.append(batch["labels"])

    y_pred = label_encoder.inverse_transform(predictions)
    y_true = label_encoder.inverse_transform(true_labels)
    return classification_report(y_true, y_pred, digits=4)


if __name__ == "__main__":
    # File paths
    test_path = ".\\cleaned_data\\empathetic_test_cleaned.csv"
    fine_tuned_dir = ".\\models\\empathetic\\emotion_bert_finetuned"
    base_model_name = "bert-base-uncased"

    # Load data and tokenizer
    df, le = load_data(test_path)
    tokenizer = BertTokenizerFast.from_pretrained(base_model_name)
    dataset = tokenize_data(df, tokenizer)

    # Load base BERT (random classifier head)
    base_model = BertForSequenceClassification.from_pretrained(base_model_name, num_labels=len(le.classes_))
    print("\n📉 Base BERT Performance:")
    print(evaluate_model(base_model, dataset, le))

    # Load fine-tuned BERT
    fine_model = BertForSequenceClassification.from_pretrained(fine_tuned_dir)
    print("\n📈 Fine-tuned BERT Performance:")
    print(evaluate_model(fine_model, dataset, le))
