import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

def test_emotion_model(test_path, model_dir, report_dir):
    # Select GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # Load test data
    df = pd.read_csv(test_path)

    # Load label encoder mapping
    label_map_path = os.path.join(model_dir, "label_map.csv")
    label_names = pd.read_csv(label_map_path)["0"].tolist()
    le = LabelEncoder()
    le.fit(label_names)

    # Encode labels
    df["labels"] = le.transform(df["label"])

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    # Tokenize inputs
    tokens = tokenizer(list(df["text"]), padding=True, truncation=True, return_tensors="pt")
    labels = torch.tensor(df["labels"].values)
    dataset = TensorDataset(tokens["input_ids"], tokens["attention_mask"], labels)
    dataloader = DataLoader(dataset, batch_size=32)

    predictions, true_labels = [], []

    for input_ids, attention_mask, batch_labels in dataloader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        batch_labels = batch_labels.to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            pred = torch.argmax(outputs.logits, dim=1)

        predictions.extend(pred.cpu().numpy())
        true_labels.extend(batch_labels.cpu().numpy())

    # Decode labels
    y_pred = le.inverse_transform(predictions)
    y_true = le.inverse_transform(true_labels)

    # Create report directory
    os.makedirs(report_dir, exist_ok=True)

    # Classification report
    report_path = os.path.join(report_dir, "classification_report.txt")
    report_text = classification_report(y_true, y_pred, digits=4)
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"Saved classification report to: {report_path}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=le.classes_)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, "confusion_matrix.png"))
    plt.close()

    # Per-class bar chart
    cr_dict = classification_report(y_true, y_pred, output_dict=True)
    df_cr = pd.DataFrame(cr_dict).transpose().drop(["accuracy", "macro avg", "weighted avg"])
    df_cr[["precision", "recall", "f1-score"]].plot(kind="bar", figsize=(12, 6))
    plt.title("Per-class Metrics")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, "metrics_bar_chart.png"))
    plt.close()

    print(f"Visualizations saved to: {report_dir}")

if __name__ == "__main__":
    test_emotion_model(
        test_path=".\\cleaned_data\\empathetic\\empathetic_test_cleaned.csv",
        model_dir=".\\models\\empathetic\\bert\\emotion_bert_base_uncased_finetuned",
        report_dir=".\\evaluation\\bert\\bert_finetuned"
    )
