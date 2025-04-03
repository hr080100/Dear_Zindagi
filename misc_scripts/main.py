from dotenv import load_dotenv
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel

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

# ✅ Optional structured parser
# parser = PydanticOutputParser(pydantic_object=SupportResponse)

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

                Use a friendly, conversational tone — like a trusted friend who listens deeply. Remember, your goal is to help the user feel heard, supported, and understood. Make sure to respect user privacy and confidentiality and avoid sharing personal information. Give as much time and attention as needed to each user. If you need to take a break, let the user know you'll be back soon. Let your result be longer no problem with that but do not make is shorter than 200 words.

                If structured formatting is explicitly requested, respond using this layout:
            """
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)  # .partial(format_instructions=parser.get_format_instructions())
# {format_instructions} is a placeholder for structured response instructions

# ✅ (Optional) Tools – you can leave empty or add journaling, quote, etc.
tools = []

# ✅ Create agent
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

# ✅ Create executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# ✅ Prompt user
query = input("🧠 How are you feeling today? ")

# ✅ Run agent
response = agent_executor.invoke({"query": query})

# ✅ Display response (no need to parse if conversational)
if isinstance(response, dict) and "output" in response:
    print("\n🤗 Dear Zindagi:\n")
    print(response["output"])
else:
    print("\n🤗 Dear Zindagi:\n")
    print(response)
