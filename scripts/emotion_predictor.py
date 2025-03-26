import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import os

# Load model directory
MODEL_DIR = ".\\models\\empathetic\\emotion_distilbert"

# Load model + tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

# Load label mapping
label_map_path = os.path.join(MODEL_DIR, "label_map.csv")
label_names = pd.read_csv(label_map_path)["0"].tolist()
le = LabelEncoder()
le.fit(label_names)

def predict_emotion(text: str) -> str:
    """Predict the emotional label for a single input string"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction_id = torch.argmax(logits, dim=1).item()
    return le.inverse_transform([prediction_id])[0]

# ✅ Optional: Run directly
if __name__ == "__main__":
    user_input = input("Enter a sentence: ")
    emotion = predict_emotion(user_input)
    print(f"🧠 Predicted Emotion: {emotion}")
