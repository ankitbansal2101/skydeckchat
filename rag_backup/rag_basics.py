import os
import glob
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# Load environment variables from .env file
load_dotenv()

# Set OpenAI API key from environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables.")
os.environ["OPENAI_API_KEY"] = openai_api_key

# Disable Chroma telemetry
os.environ["ALLOW_CHROMA_TELEMETRY"] = "FALSE"

# Get the current directory
current_dir = os.getcwd()

# Define directories
documents_dir = os.path.join(current_dir, "documents")

# Get all PDFs from documents
pdf_files = glob.glob(os.path.join(documents_dir, "*.pdf"))

if not pdf_files:
    raise FileNotFoundError("No PDF files found in the 'documents' directory.")

print("--- Found PDF files ---")
for pdf in pdf_files:
    print(pdf)

# Load all PDF documents
all_docs = []
for pdf_path in pdf_files:
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    all_docs.extend(docs)

# Split the documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
split_docs = text_splitter.split_documents(all_docs)

# Directory to persist embeddings
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

# Create embeddings and vector store if not exists
if not os.path.exists(persistent_directory):
    print("\n--- Creating embeddings and vector store ---")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma.from_documents(split_docs, embeddings, persist_directory=persistent_directory)
    print("--- Vector store created and persisted ---")
else:
    print("\n--- Loading existing vector store ---")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# Setup retriever and QA chain
retriever = db.as_retriever()
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
    retriever=retriever
)

# Ask questions
questions = [
    "How token calculation works"
]

print("\n--- Asking Questions ---")
for q in questions:
    try:
        answer = qa_chain.run(q)
        print(f"\nQ: {q}\nA: {answer}")
    except Exception as e:
        print(f"Error while processing question '{q}': {e}")
