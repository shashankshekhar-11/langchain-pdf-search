import os
import pdfplumber
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Load API key and project
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
if not GOOGLE_API_KEY:
    print("Error: GOOGLE_API_KEY not found in .env file. Get it from https://makersuite.google.com/app/apikey")
    exit(1)
if not GOOGLE_CLOUD_PROJECT:
    print("Error: GOOGLE_CLOUD_PROJECT not found in .env file. Set it in Google Cloud Console: https://console.cloud.google.com/projectselector2/home/dashboard")
    exit(1)

# Debug: Verify API key and project
print("Loaded API Key:", GOOGLE_API_KEY[:10] + "..." + GOOGLE_API_KEY[-4:])
print("Google Cloud Project:", GOOGLE_CLOUD_PROJECT)

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
        temperature=0.3,
        google_api_key=GOOGLE_API_KEY
    )
except Exception as e:
    print(f"Error initializing Google LLM: {str(e)}")
    print("Try checking API key permissions or model availability.")
    exit(1)

# PDF processing function
def process_pdf(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = " ".join([page.extract_text() or "" for page in pdf.pages])
        if not text.strip():
            print("Error: No text extracted from PDF. Please provide a valid PDF.")
            return None
        return text
    except Exception as e:
        print(f"Error reading PDF: {str(e)}")
        return None

# Main function
def main():
    # Specify the PDF file path
    pdf_path = "policy.pdf"  # Update with your PDF file name/path
    
    # Process the PDF
    print(f"\nProcessing PDF: {pdf_path}")
    text = process_pdf(pdf_path)
    if not text:
        exit(1)
    
    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = splitter.split_text(text)
    print(f"Split PDF into {len(chunks)} chunks")
    
    # Create vector store
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY,
            project=GOOGLE_CLOUD_PROJECT
        )
        # Convert chunks to Document objects for FAISS
        documents = [Document(page_content=chunk) for chunk in chunks]
        vector_store = FAISS.from_documents(documents, embeddings)
        print("Vector store created successfully")
    except Exception as e:
        print(f"Error creating vector store: {str(e)}")
        print("Ensure Vertex AI API is enabled and gcloud auth is set up (run: gcloud auth application-default login).")
        exit(1)
    
    # Define prompt template
    prompt_template = """
    You are an insurance policy assistant. Provide a concise answer (1-2 sentences) based on the policy context.
    
    If the answer is found in the context, provide the answer.
    If the answer is not found, respond: "I couldn't find this information in the policy."
    
    Context: {context}
    Question: {question}
    Answer:
    """
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # Create RAG chain
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {
            "context": vector_store.as_retriever(search_kwargs={"k": 2}) | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # Query loop
    while True:
        query = input("\nEnter your question (or 'quit' to exit): ")
        if query.lower() == "quit":
            print("Exiting...")
            break
        
        try:
            # Get answer
            answer = rag_chain.invoke(query)
            print("Answer:", answer.strip())
        
        except Exception as e:
            print(f"Error processing query: {str(e)}")
            print("Check model availability and API quotas.")

if __name__ == "__main__":
    main()