from langchain.tools import Tool
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from datetime import datetime
import joblib
import os
import pandas as pd

# You must ensure this is imported correctly
from chatbot_main import llm  # adjust path if needed


# --- Emotion Classification ---
def classify_emotion(text: str) -> str:
    model_path = ".\\models\\empathetic\\roberta\\emotion_roberta_finetuned"
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
    description="Uses the fine-tuned RoBERTa model to classify emotion in a sentence."
)


# --- Reddit Classifier ---
def classify_reddit_label(text: str) -> str:
    model_path = ".\\models\\reddit\\reddit_model.pkl"
    vectorizer_path = ".\\models\\reddit\\reddit_vectorizer.pkl"

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
    description="Classifies mental health condition labels using the trained Reddit model."
)


# --- Logging Tools ---
def log_emotion_to_file(text: str, filename: str = "logs\\emotion_log.txt") -> str:
    result = classify_emotion(text)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "a", encoding="utf-8") as f:
        f.write(f"{timestamp} - {result} - {text}\n")
    return f"Logged emotion: {result}"

log_emotion_tool = Tool(
    name="log_emotion",
    func=log_emotion_to_file,
    description="Logs emotion classification result to file."
)


def log_reddit_label_to_file(text: str, filename: str = "logs\\reddit_log.txt") -> str:
    result = classify_reddit_label(text)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "a", encoding="utf-8") as f:
        f.write(f"{timestamp} - {result} - {text}\n")
    return f"Logged Reddit sentiment: {result}"

log_reddit_tool = Tool(
    name="log_reddit_label",
    func=log_reddit_label_to_file,
    description="Logs Reddit label result to file."
)


# --- Model Comparison Tool ---
def compare_transformer_models(text: str) -> str:
    base_path = ".\\models\\empathetic\\"
    model_names = [
        ("bert\\emotion_bert_base_uncased_finetuned", "BERT"),
        ("distilbert\\emotion_distilbert_base_uncased_finetuned", "DistilBERT"),
        ("roberta\\emotion_roberta_base_uncased_finetuned", "RoBERTa")
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
    description="Returns predictions from BERT, DistilBERT, and RoBERTa."
)


# --- Gemini-Powered Dynamic Tools ---
def generate_gratitude_prompt(text):
    return llm.invoke(f"Generate a gratitude journaling prompt for: {text}")

generate_gratitude_prompt_tool = Tool(
    name="generate_gratitude_prompt",
    func=generate_gratitude_prompt,
    description="Gratitude prompt")

def generate_affirmation(text):
    return llm.invoke(f"Provide a supportive affirmation for someone who says: {text}")

generate_affirmation_tool = Tool(
    name="generate_affirmation",
    func=generate_affirmation,
    description="Dynamic affirmation")

def generate_checkin_prompt(text):
    return llm.invoke(f"Ask a check-in reflection question based on: {text}")

generate_checkin_prompt_tool = Tool(
    name="generate_checkin_prompt",
    func=generate_checkin_prompt,
    description="Check-in prompt")

def generate_cbt_prompt(text):
    return llm.invoke(f"Give a CBT-style question to help challenge negative thoughts from: {text}")

generate_cbt_prompt_tool = Tool(
    name="generate_cbt_prompt",
    func=generate_cbt_prompt,
    description="CBT-style question")

def suggest_dynamic_activities(text):
    return llm.invoke(f"Suggest a coping activity based on: {text}")

suggest_dynamic_activities_tool = Tool(
    name="suggest_coping_activity",
    func=suggest_dynamic_activities,
    description="Coping activity")

def generate_breathing_guide(text):
    return llm.invoke(f"Create a calming breathing guide for: {text}")

generate_breathing_guide_tool = Tool(
    name="generate_breathing_guide",
    func=generate_breathing_guide,
    description="Breathing guide")

def generate_inspirational_quote(text):
    return llm.invoke(f"Give an inspirational quote for someone feeling: {text}")

generate_inspirational_quote_tool = Tool(
    name="generate_inspirational_quote",
    func=generate_inspirational_quote,
    description="Inspirational quote")