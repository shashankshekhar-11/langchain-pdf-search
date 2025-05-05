from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
print("API Key:", openai_api_key)  # Debug: Verify the key is loaded

llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=openai_api_key)
response = llm.invoke("Hello, world!")
print(response.content)