from langchain.tools import Tool
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from datetime import datetime
import random
import os

# --- Emotion Prediction Tool ---
def predict_emotion(text: str) -> str:
    model_path = "models/empathetic/emotion_roberta_finetuned"
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

# --- Mental Health Search Tool ---
from langchain_community.tools import DuckDuckGoSearchRun
search = DuckDuckGoSearchRun()
mental_health_search_tool = Tool(
    name="mental_health_search",
    func=search.run,
    description="Search the web for mental health support articles and techniques."
)

# --- Wikipedia Info Tool ---
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=10000)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)
wikipedia_info_tool = Tool(
    name="wikipedia_info",
    func=wiki_tool.run,
    description="Get general information from Wikipedia about mental health topics."
)

# --- Daily Check-In Tool ---
def daily_checkin(_) -> str:
    prompts = [
        "What emotions have you experienced most today?",
        "What's one thing that made you smile today?",
        "What is something you wish you had more of right now?",
        "Did anything feel overwhelming today?"
    ]
    return random.choice(prompts)

daily_checkin_tool = Tool(
    name="daily_checkin",
    func=daily_checkin,
    description="Returns a reflective question to help users check in with their emotional state."
)

# --- Suggest Activities Tool ---
def suggest_activities(_) -> str:
    suggestions = [
        "Try a 5-minute deep breathing exercise.",
        "Take a short walk, even if it's just around the room.",
        "Listen to your favorite song with full attention.",
        "Write down one thing you're proud of today.",
        "Call or message a friend to say hello."
    ]
    return random.choice(suggestions)

suggest_activities_tool = Tool(
    name="suggest_activities",
    func=suggest_activities,
    description="Recommends healthy activities to boost emotional well-being."
)

# --- Mood Tracker Tool ---
def mood_tracker(text: str, filename: str = "mood_log.txt") -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"{timestamp} - {text}\n"
    with open(filename, "a", encoding="utf-8") as f:
        f.write(entry)
    return f"Mood tracked: '{text}' at {timestamp}"

mood_tracker_tool = Tool(
    name="mood_tracker",
    func=mood_tracker,
    description="Tracks and logs the user's mood over time."
)

# --- Breathing Exercise Tool ---
def breathing_exercise(_) -> str:
    return (
        "Let's try the 4-7-8 breathing technique:\n"
        "1. Inhale for 4 seconds\n"
        "2. Hold your breath for 7 seconds\n"
        "3. Exhale slowly for 8 seconds\n"
        "Repeat this for 4 cycles to help calm your nervous system."
    )

breathing_exercise_tool = Tool(
    name="breathing_exercise",
    func=breathing_exercise,
    description="Guides the user through a calming breathing exercise."
)

# --- CBT Prompt Tool ---
def cbt_prompt(_) -> str:
    prompts = [
        "What evidence do you have that supports or contradicts this thought?",
        "If your friend had this thought, what would you say to them?",
        "Can you think of a more balanced or realistic perspective?",
        "What would happen if this thought were not true?"
    ]
    return random.choice(prompts)

cbt_prompt_tool = Tool(
    name="cbt_thought_challenger",
    func=cbt_prompt,
    description="Provides a CBT-style reflection question to challenge negative thoughts."
)

# --- Inspirational Quote Tool ---
def quote_tool(_) -> str:
    quotes = [
        "Keep taking time for yourself until you’re you again. – Lalah Delia",
        "You don’t have to control your thoughts. You just have to stop letting them control you. – Dan Millman",
        "You are not your illness. You have a name, a history, a personality. Staying yourself is part of the battle. – Julian Seifter",
        "Sometimes the people around you won’t understand your journey. They don’t need to, it’s not for them. – Joubert Botha",
        "Self-care is how you take your power back. – Lalah Delia"
    ]
    return random.choice(quotes)

inspirational_quote_tool = Tool(
    name="inspirational_quote",
    func=quote_tool,
    description="Returns an inspirational mental health quote."
)

# --- Emotion Definition Tool ---
def emotion_definition(emotion: str) -> str:
    definitions = {
        "Joy": "A feeling of great pleasure and happiness.",
        "Sadness": "Emotional pain associated with loss, disappointment, or helplessness.",
        "Anger": "A strong feeling of displeasure or hostility.",
        "Fear": "An unpleasant emotion caused by the belief that something is dangerous.",
        "Surprise": "A sudden feeling of wonder or astonishment.",
        "Love": "An intense feeling of deep affection.",
        "Guilt": "A feeling of responsibility for wrongdoing.",
        "Shame": "A painful feeling due to the consciousness of wrong or foolish behavior."
    }
    return definitions.get(emotion.capitalize(), "Definition not found.")

emotion_definition_tool = Tool(
    name="emotion_definition",
    func=emotion_definition,
    description="Returns a brief definition of a given emotion."
)
