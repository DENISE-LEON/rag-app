from langchain_anthropic import ChatAnthropic
import os
#for reading .env file
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")

llm = ChatAnthropic(
    model_name="claude-haiku-4-5",
    temperature=0,
    api_key=api_key,
)