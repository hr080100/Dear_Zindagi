# Dear Zindagi 🌟

**An Empathetic AI Chatbot for Mental Health Support**

---

## ✨ Overview

*Dear Zindagi* is an AI-powered mental health companion designed to offer empathetic, emotion-aware support to users navigating difficult thoughts, feelings, or everyday stress. Inspired by the spirit of the Bollywood movie of the same name, this chatbot combines a fine-tuned transformer model with Google’s Gemini Pro LLM to actively listen, respond compassionately, and guide users with journaling prompts, coping tools, and reflective questions.

---

## 🪤 Key Features

- **Empathetic LLM Conversations:** Powered by Gemini Pro with a warm, compassionate system prompt.
- **Emotion Detection:** Fine-tuned RoBERTa classifier detects user emotions using the Empathetic Dialogues dataset.
- **Mental Health Classification:** Logistic regression model classifies posts into mental health categories based on Reddit data.
- **Integrated Support Tools:** Affirmations, breathing guides, journaling prompts, gratitude check-ins, inspirational quotes, and CBT-based reflections.
- **Gradio UI:** A simple chat-based interface for users to interact in real-time.
- **Session Logging:** Logs user chats, emotional trends, and journaling entries for self-reflection.

---

## 📊 Models Used

1. **Emotion Classification Model**
   - Dataset: Empathetic Dialogues (Hugging Face)
   - Architecture: RoBERTa-base fine-tuned for emotion classification
   - Output: 28 emotion categories (e.g., proud, guilty, anxious, hopeful)

2. **Mental Health Label Classifier**
   - Dataset: Reddit mental health subreddit posts
   - Model: Logistic Regression + TF-IDF
   - Labels: Depression, Anxiety, Stress, Bipolar, Personality Disorder

3. **LLM**
   - Provider: Google Gemini Pro (via LangChain)
   - Used for dynamic tool invocation, personalized response generation, summarization

---

## 🎨 Tools & Technologies

- `Gradio`: for front-end chat interface
- `LangChain`: to wrap tools and create agents
- `Google Generative AI (Gemini Pro)`: for compassionate language generation
- `Transformers + HuggingFace`: for tokenizer/model loading
- `Pandas, sklearn, matplotlib`: for model evaluation and data handling
- `Python 3.11`

---

## 🎓 How It Works

1. User enters their name and is greeted via a Gemini-generated message.
2. The chat interface opens; user messages are emotion-classified in real-time.
3. Gemini uses this emotional context to generate a tailored response.
4. Tools like journaling, affirmations, or breathing exercises are invoked as needed.
5. Each session is logged for review; emotions are tracked turn-by-turn.

---

## 📦 Project Structure

```
Dear_Zindagi/
├── scripts/
│   ├── app_modified.py
│   ├── chatbot_engine.py
│   ├── tools.py
│   ├── clean_*.py
│   ├── compare_base_vs_finetuned_*.py
│   └── train_*.py
├── models/
│   └── empathetic/roberta/emotion_roberta_finetuned/
├── cleaned_data/
├── chat_logs/
├── README.md
└── requirements.txt
```

---

## 📖 How to Run

### Install Requirements
```bash
pip install -r requirements.txt
```

### Set Your API Key (Gemini)
One of the following methods can be utilised.

Method 1: 

Create a .env file and store all the API Keys there.

Method 2:
```bash
export GEMINI_API_KEY="your-key-here"  # Linux/macOS
set GEMINI_API_KEY="your-key-here"     # Windows
```

### Launch Chatbot
```bash
python scripts/app_modified.py
```

The chatbot will run locally at: [http://localhost:7860](http://localhost:7860)

---

## 🚀 Future Improvements
- Deploying publicly via HuggingFace or Streamlit Cloud
- Adding multilingual support
- Advanced memory and long-context awareness
- Scheduled daily check-ins or reminders

---

## 🎉 Credits
- **University of Michigan - Master of Applied Data Science (MADS)** Capstone Project
- Built by: Haider Rizvi
- Dataset sources: Hugging Face, Kaggle, r/MentalHealth
- AI model: Google Gemini via LangChain

---

## 📕 License
This project is for academic and educational purposes only. Not intended for clinical or diagnostic use.

---

*Dear Zindagi reminds us that it's okay to not be okay. And sometimes, all we need is someone to listen.* ❤️
