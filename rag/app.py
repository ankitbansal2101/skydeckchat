from flask import Flask, render_template, request, session, redirect, url_for
import os
import glob
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for session management

# Load environment variables
load_dotenv()

# Set OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables.")
os.environ["OPENAI_API_KEY"] = openai_api_key

# Disable Chroma telemetry
os.environ["ALLOW_CHROMA_TELEMETRY"] = "FALSE"

# Setup LangChain components
current_dir = os.getcwd()
documents_dir = os.path.join(current_dir, "documents")
pdf_files = glob.glob(os.path.join(documents_dir, "*.pdf"))

if not pdf_files:
    raise FileNotFoundError("No PDF files found in the 'documents' directory.")

all_docs = []
for pdf_path in pdf_files:
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    all_docs.extend(docs)

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
split_docs = text_splitter.split_documents(all_docs)

persistent_directory = os.path.join(current_dir, "db", "chroma_db")

# If Chroma DB already exists, don't re-process documents.
if not os.path.exists(persistent_directory):
    # Rebuild Chroma DB from scratch with chunking and embedding.
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma.from_documents(split_docs, embeddings, persist_directory=persistent_directory)
else:
    # Load the existing Chroma DB (no chunking/embedding).
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

retriever = db.as_retriever()

# Initialize memory for chat history
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
    retriever=retriever,
    memory=memory
)

# Define routes
@app.route('/')
def home():
    # Initialize chat history in session if not already present
    if 'chat_history' not in session:
        session['chat_history'] = []
    return render_template('index.html', chat_history=session['chat_history'])

@app.route('/ask', methods=['POST'])
def ask_question():
    question = request.form.get('question')
    if not question.strip():  # Ensure the question is not empty or just whitespace
        return render_template('index.html', error="Please enter a valid question.", chat_history=session.get('chat_history', []))
    try:
        # Sync session chat history with ConversationBufferMemory
        memory.chat_memory.messages = session.get('chat_history', [])

        # Get the raw answer using the QA chain
        raw_answer = qa_chain.run(question)

        # Dynamically prettify the answer into bullet points
        prettified_answer = raw_answer.split(". ")  # Split the answer into sentences
        prettified_answer = [f"🔹 {sentence.strip()}" for sentence in prettified_answer if sentence.strip()]  # Add emojis and clean up

        # Join the prettified answer into an HTML list
        answer = "<ul>" + "".join([f"<li>{point}</li>" for point in prettified_answer]) + "</ul>"

        # Update chat history in session
        chat_entry = {"question": question, "answer": answer}
        session['chat_history'].append(chat_entry)

        # Sync updated session chat history back to ConversationBufferMemory
        memory.chat_memory.messages = session['chat_history']

        return render_template('index.html', question=question, answer=answer, chat_history=session['chat_history'])
    except Exception as e:
        return render_template('index.html', error=f"Error: {str(e)}", chat_history=session.get('chat_history', []))

@app.route('/exit', methods=['POST'])
def exit_conversation():
    # Clear the session and reset the memory
    session.clear()
    memory.chat_memory.clear()
    return redirect(url_for('home'))

# Run the app
if __name__ == '__main__':
    app.run(debug=True)