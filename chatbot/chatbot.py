import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Load API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("Error: GOOGLE_API_KEY not found in .env file. Get it from https://makersuite.google.com/app/apikey")
    exit(1)

#verify model availability
try:
    genai.configure(api_key=GOOGLE_API_KEY)
    available_models = [m.name for m in genai.list_models()]
    model_name = "gemini-1.5-pro" if "models/gemini-1.5-pro" in available_models else "gemini-1.5-flash"
except Exception as e:
    print(f"Error checking models: {str(e)}")
    print("Check your API key and network connection.")
    exit(1)

# Initialize Google LLM
try:
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.7,
        google_api_key=GOOGLE_API_KEY
    )
except Exception as e:
    print(f"Error initializing Google LLM: {str(e)}")
    print("Try checking API key permissions or model availability.")
    exit(1)

# Save chat history to file
def save_chat_history(messages, filename="chat_history.txt"):
    try:
        with open(filename, "w", encoding="utf-8") as f:
            for msg in messages:
                sender = "You" if isinstance(msg, HumanMessage) else "Bot"
                f.write(f"{sender}: {msg.content}\n")
        print(f"Chat history saved to {filename}")
    except Exception as e:
        print(f"Error saving chat history: {str(e)}")

# Define the LangGraph workflow
workflow = StateGraph(state_schema=MessagesState)

# Define the function that calls the model
def call_model(state: MessagesState):
    # Create prompt with conversation history
    history = ""
    user_name = None
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            history += f"User: {msg.content}\n"
            # Check for name declaration
            if "my name is" in msg.content.lower():
                name_parts = msg.content.lower().split("my name is")
                if len(name_parts) > 1:
                    user_name = name_parts[1].strip().capitalize()
        else:
            history += f"Assistant: {msg.content}\n"
    
    # Handle name query
    if state["messages"][-1].content.lower().startswith("what is my name") and user_name:
        response = f"Your name is {user_name}!"
        return {"messages": AIMessage(content=response)}
    
    # Handle name declaration
    if "my name is" in state["messages"][-1].content.lower() and user_name:
        response = f"Hi {user_name}! It's nice to meet you. How can I help you today?"
        return {"messages": AIMessage(content=response)}
    
    # General query
    prompt_template = """
    You are a friendly, knowledgeable chatbot. Provide clear, concise, and engaging answers to the user's questions.
    Use the conversation history to maintain context and personalize responses if possible.
    If you don't know the answer, say so politely and offer to help with something else.
    
    Conversation history:
    {history}
    
    Current question: {question}
    Answer:
    """
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["history", "question"]
    )
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({
        "history": history,
        "question": state["messages"][-1].content
    })
    return {"messages": AIMessage(content=response)}

# Define the graph
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# Add memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# Main chatbot function
def main():
    # Configuration with thread_id for persistence
    config = {"configurable": {"thread_id": "abc123"}}
    
    # Welcome message
    print("\n=== General Chatbot ===")
    print("I'm here to chat about anything! Ask me questions, tell me jokes, or type 'quit' to exit.")
    print("Type 'save' to save our conversation to chat_history.txt.")
    print("======================")
    
    # Chat loop
    while True:
        query = input("\nYou: ").strip()
        
        if query.lower() == "quit":
            print("Goodbye! Thanks for chatting.")
            break
        elif query.lower() == "save":
            # Get current state
            state = app.get_state(config)
            save_chat_history(state.values["messages"])
            continue
        elif not query:
            print("Please enter a question or message.")
            continue
        
        try:
            # Invoke the graph with the query
            input_messages = [HumanMessage(content=query)]
            output = app.invoke({"messages": input_messages}, config)
            response = output["messages"][-1].content.strip()
            print(f"Bot: {response}")
        except Exception as e:
            error_msg = f"Error: {str(e)}. Please try again."
            print(error_msg)

if __name__ == "__main__":
    main()