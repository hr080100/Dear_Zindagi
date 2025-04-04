from datetime import datetime
import os
import time
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import (
    emotion_classifier_tool,
    reddit_sentiment_tool,
    log_emotion_tool,
    log_reddit_tool,
    compare_models_tool,
    generate_affirmation_tool,
    generate_gratitude_prompt_tool,
    generate_cbt_prompt_tool,
    suggest_dynamic_activities_tool,
    generate_checkin_prompt_tool,
    generate_breathing_guide_tool,
    generate_inspirational_quote_tool,
    classify_emotion,
)
from tools import llm

# Define available tools
tools = [
    emotion_classifier_tool,
    reddit_sentiment_tool,
    log_emotion_tool,
    log_reddit_tool,
    compare_models_tool,
    generate_affirmation_tool,
    generate_gratitude_prompt_tool,
    generate_cbt_prompt_tool,
    suggest_dynamic_activities_tool,
    generate_checkin_prompt_tool,
    generate_breathing_guide_tool,
    generate_inspirational_quote_tool
]

base_prompt = f"""
                    You are a compassionate and supportive mental health chatbot named *Dear Zindagi*. (taking the name from the Bollywood movie of the same name that stars Shahrukh Khan). The user you are speaking with is named {{user_name}}.

                    Your role is to actively listen, provide emotional support, and offer thoughtful, evidence-based reflections. 
                    You are not a licensed therapist, and should never give medical or diagnostic advice.

                    Instead:
                    - Respond with empathy, warmth, and non-judgmental language.
                    - Ask thoughtful follow-up questions.
                    - Suggest healthy coping strategies (e.g., breathing exercises, journaling, grounding).
                    - Encourage the user to talk to a professional when needed.

                    Use a friendly, conversational tone — like a trusted friend who listens deeply. 
                    Help the user feel heard, supported, and understood. Make sure to respect user privacy and confidentiality and avoid sharing personal information. Give as much time and attention as needed to each user. If you need to take a break, let the user know you'll be back soon.
                    Keep your responses concise and focused on the user's needs. 
                    Avoid rushing or being generic.
                """

def get_agent(emotion: str):
    prompt = ChatPromptTemplate.from_messages([
        ("system", base_prompt + f"\nCurrent detected emotion: {emotion}"),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    agent = create_tool_calling_agent(llm=llm, prompt=prompt, tools=tools)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

# Global state
chat_history = []
session_log = []
emotion_log = []
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
os.makedirs("chat_logs/chat", exist_ok=True)
os.makedirs("chat_logs/emotion", exist_ok=True)
os.makedirs("chat_logs/summary", exist_ok=True)
os.makedirs("chat_logs/journal", exist_ok=True)

log_path = f"chat_logs/chat/{timestamp}.txt"
emotion_path = f"chat_logs/emotion/{timestamp}.csv"
summary_path = f"chat_logs/summary/{timestamp}.txt"
journal_path = f"chat_logs/journal/{timestamp}.txt"

# Main interaction logic
def chat_with_dear_zindagi(user_message: str, user_name: str):
    global chat_history, session_log, emotion_log

    if not user_message.strip():
        return "Please enter a message."

    # Emotion detection
    detected_emotion = classify_emotion(user_message)
    emotion_log.append(detected_emotion)

    # Update chat history
    chat_history.append(("human", user_message))
    session_log.append(f"{user_name}: {user_message}")

    # Create and run agent
    agent_executor = get_agent(detected_emotion)
    response = agent_executor.invoke({
        "query": user_message,
        "chat_history": chat_history,
        "user_name": user_name
    })

    # Extract bot reply
    bot_reply = response.get("output", str(response))
    chat_history.append(("ai", bot_reply))
    session_log.append(f"Dear Zindagi: {bot_reply}\n")

    # Save chat log
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(session_log))

    # Save emotion log
    with open(emotion_path, "w", encoding="utf-8") as ef:
        ef.write("Turn,Emotion\n")
        for i, emo in enumerate(emotion_log):
            ef.write(f"{i+1},{emo}\n")

    return bot_reply

# Optional journaling logic
def log_journal_entry(entry: str):
    with open(journal_path, "w", encoding="utf-8") as jf:
        jf.write(f"Journal Entry ({timestamp})\n\n{entry}\n")
    return "Your journal entry has been saved."

# Optional summary (called on session end)
def generate_summary():
    chat_text = "\n".join(session_log)
    summary_prompt = f"Please summarize the following mental health support chat:\n\n{chat_text}"
    summary = llm.invoke(summary_prompt)
    with open(summary_path, "w", encoding="utf-8") as sf:
        sf.write(f"Conversation Summary ({timestamp})\n\n{summary}\n")
    return summary
