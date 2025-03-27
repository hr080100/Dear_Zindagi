from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel
from model_based_tools import (
    emotion_classifier_tool,
    reddit_sentiment_tool,
    log_emotion_tool,
    log_reddit_tool,
    compare_models_tool
)

# ✅ Load .env
load_dotenv()

# ✅ Define structured (optional) response model
class SupportResponse(BaseModel):
    user_feeling: str
    supportive_message: str
    suggested_coping_strategies: list[str]
    encouragement: str

# ✅ Initialize Gemini model
llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-pro-latest",
    google_api_key=os.getenv("GEMINI_API_KEY"),
)

# ✅ System prompt tailored for mental health chatbot
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a compassionate and supportive mental health chatbot named *Dear Zindagi* (taking the name from the Bollywood movie of the same name that stars Shahrukh Khan).

            Your role is to actively listen, provide emotional support, and offer thoughtful, evidence-based reflections. You are not a licensed therapist, and should never give medical or diagnostic advice.

            Instead:
            - Respond with empathy, warmth, and non-judgmental language.
            - Ask thoughtful follow-up questions.
            - Suggest healthy coping strategies (e.g., breathing exercises, journaling, grounding).
            - Encourage the user to talk to a professional when needed.

            Use a friendly, conversational tone — like a trusted friend who listens deeply. Remember, your goal is to help the user feel heard, supported, and understood. Make sure to respect user privacy and confidentiality and avoid sharing personal information. Give as much time and attention as needed to each user. If you need to take a break, let the user know you'll be back soon. Keep your responses concise and focused on the user's needs that feels like having a real conversation with a human being.
            """
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

# ✅ Tools: using only model-based tools
tools = [
    emotion_classifier_tool,
    reddit_sentiment_tool,
    log_emotion_tool,
    log_reddit_tool,
    compare_models_tool
]

# ✅ Create agent and executor
agent = create_tool_calling_agent(llm=llm, prompt=prompt, tools=tools)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# ✅ Multi-turn conversation loop
chat_history = []
print("🤗 Dear Zindagi is here to support you. Type 'exit' anytime to leave.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("\n🫂 Take care. I'm always here when you need me. 💛\n")
        break

    chat_history.append(("human", user_input))
    response = agent_executor.invoke({
        "query": user_input,
        "chat_history": chat_history
    })

    if isinstance(response, dict) and "output" in response:
        bot_reply = response["output"]
    else:
        bot_reply = str(response)

    print("\n🤗 Dear Zindagi:\n")
    print(bot_reply)
    chat_history.append(("ai", bot_reply))
