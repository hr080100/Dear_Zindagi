from langchain.tools import Tool
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from datetime import datetime
import random
import os

# --- Emotion Prediction Tool ---
def predict_emotion(text: str) -> str:
    model_path = ".\\models\\empathetic\\emotion_roberta_finetuned"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    label_id = torch.argmax(probs, dim=1).item()

    label_map_path = os.path.join(model_path, "label_map.csv")
    label_map = {}
    if os.path.exists(label_map_path):
        with open(label_map_path, "r", encoding="utf-8") as f:
            for line in f:
                id_, label = line.strip().split(",")
                label_map[int(id_)] = label
    else:
        label_map = {
            0:"afraid",
            1:"angry",
            2:"annoyed",
            3:"anticipating",
            4:"anxious",
            5:"apprehensive",
            6:"ashamed",
            7:"caring",
            8:"confident",
            9:"content",
            10:"devastated",
            11:"disappointed",
            12:"disgusted",
            13:"embarrassed",
            14:"excited",
            15:"faithful",
            16:"furious",
            17:"grateful",
            18:"guilty",
            19:"hopeful",
            20:"impressed",
            21:"jealous",
            22:"joyful",
            23:"lonely",
            24:"nostalgic",
            25:"prepared",
            26:"proud",
            27:"sad",
            28:"sentimental",
            29:"surprised",
            30:"terrified",
            31:"trusting"
        }

    return label_map.get(label_id, "Unknown")

emotion_predictor_tool = Tool(
    name="emotion_classifier",
    func=predict_emotion,
    description="Analyzes user input and returns the predicted emotion."
)

# --- Gratitude Prompt Tool ---
def gratitude_prompt(_) -> str:
    prompts = [
        "What's something you're grateful for today?",
        "Think of a moment this week that brought you peace — what was it?",
        "Who made you smile recently?",
        "Name one thing you love about yourself."
    ]
    return random.choice(prompts)

gratitude_tool = Tool(
    name="gratitude_prompt",
    func=gratitude_prompt,
    description="Returns a journaling prompt to reflect on gratitude."
)

# --- Affirmation Tool ---
def daily_affirmation(_) -> str:
    affirmations = [
        "You are enough just as you are.",
        "Every emotion you feel is valid.",
        "This moment is tough, but you are tougher.",
        "Healing is not linear — and that's okay.",
        "You are not alone. You are deeply cared for."
    ]
    return random.choice(affirmations)

affirmation_tool = Tool(
    name="daily_affirmation",
    func=daily_affirmation,
    description="Returns a gentle, supportive affirmation."
)

# --- Save Journal Entry Tool ---
def save_to_txt(data: str, filename: str = "journal_entries.txt"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"--- Journal Entry ---\nTimestamp: {timestamp}\n\n{data}\n\n"
    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_text)
    return f"Journal entry successfully saved to {filename}"

save_journal_tool = Tool(
    name="save_journal_entry",
    func=save_to_txt,
    description="Saves a user's reflection or journal entry to a local text file."
)
