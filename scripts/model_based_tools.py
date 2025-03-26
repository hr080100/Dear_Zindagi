# Tools that use only user-trained models (transformers and traditional classifier)

from langchain.tools import Tool
import torch
import joblib
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datetime import datetime
import pandas as pd

# --- Tool 1: Emotion Classification (Transformer) ---
def classify_emotion(text: str) -> str:
    model_path = "models/empathetic/emotion_roberta_finetuned"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    predicted_class = torch.argmax(probs, dim=1).item()

    label_map_path = os.path.join(model_path, "label_map.csv")
    label_map = pd.read_csv(label_map_path).set_index("id")["label"].to_dict()
    return label_map.get(predicted_class, "Unknown")

emotion_classifier_tool = Tool(
    name="emotion_classifier",
    func=classify_emotion,
    description="Uses the user's fine-tuned RoBERTa model to classify emotion in a sentence."
)

# --- Tool 2: Reddit Mental Health Label Classifier ---
def classify_reddit_label(text: str) -> str:
    model_path = "models/reddit/reddit_model.pkl"
    vectorizer_path = "models/reddit/reddit_vectorizer.pkl"

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]

    label_map = {
        0: "Stress",
        1: "Depression",
        2: "Bipolar disorder",
        3: "Personality disorder",
        4: "Anxiety"
    }

    return label_map.get(prediction, "Unknown")

reddit_sentiment_tool = Tool(
    name="reddit_sentiment_classifier",
    func=classify_reddit_label,
    description="Uses the user's trained Reddit model to classify mental health condition labels."
)

# --- Tool 3: Log Emotion Result ---
def log_emotion_to_file(text: str, filename: str = "logs/emotion_log.txt") -> str:
    result = classify_emotion(text)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "a", encoding="utf-8") as f:
        f.write(f"{timestamp} - {result} - {text}\n")
    return f"Logged emotion: {result}"

log_emotion_tool = Tool(
    name="log_emotion",
    func=log_emotion_to_file,
    description="Logs the emotion classification result to a local file with timestamp."
)

# --- Tool 4: Log Reddit Sentiment Result ---
def log_reddit_label_to_file(text: str, filename: str = "logs/reddit_log.txt") -> str:
    result = classify_reddit_label(text)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "a", encoding="utf-8") as f:
        f.write(f"{timestamp} - {result} - {text}\n")
    return f"Logged Reddit sentiment: {result}"

log_reddit_tool = Tool(
    name="log_reddit_label",
    func=log_reddit_label_to_file,
    description="Logs the Reddit mental health label classification result to a local file."
)

# --- Tool 5: Compare All Transformer Models (Optional) ---
def compare_transformer_models(text: str) -> str:
    base_path = "models/empathetic"
    model_names = [
        ("emotion_bert_finetuned", "BERT"),
        ("emotion_distilbert_finetuned", "DistilBERT"),
        ("emotion_roberta_finetuned", "RoBERTa")
    ]

    results = []
    for folder, label in model_names:
        model_path = os.path.join(base_path, folder)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()

        label_map_path = os.path.join(model_path, "label_map.csv")
        label_map = pd.read_csv(label_map_path).set_index("id")["label"].to_dict()
        emotion = label_map.get(pred, "Unknown")
        results.append(f"{label}: {emotion}")

    return " | ".join(results)

compare_models_tool = Tool(
    name="compare_transformer_models",
    func=compare_transformer_models,
    description="Returns emotion predictions from all three transformer models for comparison."
)
