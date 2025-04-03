# chatbot_main.py

from dotenv import load_dotenv
import os
import time
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
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
    classify_emotion,  # used directly
)

# Load API key and environment variables
load_dotenv()

# Initialize Gemini model
llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-pro-latest",
    google_api_key=os.getenv("GEMINI_API_KEY"),
)

# Ask for user's name
user_name = input("Hi there! I'm Dear Zindagi. What's your name?")
print(f"\nIt's lovely to meet you, {user_name}! Let's talk. Type 'exit' anytime to leave.\n")

# System prompt for chatbot
base_prompt = f"""
                    You are a compassionate and supportive mental health chatbot named *Dear Zindagi*. (taking the name from the Bollywood movie of the same name that stars Shahrukh Khan). The user you are speaking with is named {user_name}.

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

# Register tools
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

# Create agent executor dynamically
def get_executor(emotion: str):
    prompt = ChatPromptTemplate.from_messages([
        ("system", base_prompt + f"\nThe model has predicted that the user's current detected emotion: {emotion}\n"),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    agent = create_tool_calling_agent(llm=llm, prompt=prompt, tools=tools)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

# Initialize chat log and state
chat_history = []
session_log = []
emotion_log = []
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_path = f".\\chatbot_logs\\chat\\{timestamp}.txt"
os.makedirs(".\\chatbot_logs\\chat\\", exist_ok=True)

# Chat loop
while True:
    user_input = input(f"{user_name}: ")
    if user_input.lower() in ["exit", "quit"]:
        print("\nTake care. I'm always here when you need me.\n")
        break

    chat_history.append(("human", user_input))
    session_log.append(f"{user_name}: {user_input}")

    # Emotion detection
    detected_emotion = classify_emotion(user_input)
    emotion_log.append(detected_emotion)

    print("\nDear Zindagi is thinking...\n")
    time.sleep(1.5)

    # Generate response
    agent_executor = get_executor(detected_emotion)
    response = agent_executor.invoke({
        "query": user_input,
        "chat_history": chat_history
    })

    bot_reply = response.get("output", str(response))
    print("Dear Zindagi:\n")
    print(bot_reply)

    chat_history.append(("ai", bot_reply))
    session_log.append(f"Dear Zindagi: {bot_reply}\n")

# Save chat log
with open(log_path, "w", encoding="utf-8") as f:
    f.write("\n".join(session_log))
print(f"\nChat saved to {log_path}")

# Save emotion log
emotion_path = f".\\chatbot_logs\\emotion\\{timestamp}.csv"
os.makedirs(".\\chatbot_logs\\emotion\\", exist_ok=True)
with open(emotion_path, "w", encoding="utf-8") as ef:
    ef.write("Turn,Emotion\n")
    for i, emo in enumerate(emotion_log):
        ef.write(f"{i+1},{emo}\n")
print(f"Emotion trend saved to {emotion_path}")

# Gemini-generated summary
summary_prompt = f"Please summarize the following mental health chat between a user named {user_name} and the chatbot Dear Zindagi:\n\n" + "\n".join(session_log)
print("\nGenerating conversation summary...\n")
summary = llm.invoke(summary_prompt)
summary_path = f".\\chatbot_logs\\summary\\{timestamp}.txt"
os.makedirs(".\\chatbot_logs\\summary", exist_ok=True)
with open(summary_path, "w", encoding="utf-8") as sf:
    sf.write(f"Conversation Summary ({timestamp})\n\n{summary}\n")
print(f"Summary saved to {summary_path}")

# Journaling opportunity
journal = input("\nWould you like to write a short journal entry to reflect on today's chat? (yes/no): ")
if journal.lower() in ["yes", "y"]:
    entry = input("Great! What's on your mind?\n")
    journal_path = f".\\chatbot_logs\\journal\\{timestamp}.txt"
    os.makedirs("chatbot_logs/journal", exist_ok=True)
    with open(journal_path, "w", encoding="utf-8") as jf:
        jf.write(f"Journal Entry ({timestamp})\n\n{entry}\n")
    print(f"Your journal entry has been saved to {journal_path}")
