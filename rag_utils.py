import os
import json
import warnings
from datetime import datetime
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# Suppress PyPDF2 warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="PyPDF2")

# Blocked terms file
BLOCKED_FILE = "blocked_terms.json"

# Chat logs file
LOGS_FILE = "chat_logs.json"

# Admin logs file
ADMIN_LOGS_FILE = "admin_logs.json"

# Config file for models
CONFIG_FILE = "config.json"

def load_config():
    """Load model config from JSON file."""
    if not os.path.exists(CONFIG_FILE):
        # Default config if file doesn't exist
        default_config = {
            "generation_model": "gemma2:9b",
            "embedding_model": "nomic-embed-text"
        }
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=4)
        return default_config
    
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_blocked_terms():
    """Load blocked terms from JSON file."""
    if not os.path.exists(BLOCKED_FILE):
        # Default empty list if file doesn't exist
        default_terms = []
        with open(BLOCKED_FILE, 'w', encoding='utf-8') as f:
            json.dump(default_terms, f)
        return default_terms
    
    with open(BLOCKED_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_blocked_terms(terms):
    """Save blocked terms to JSON file."""
    with open(BLOCKED_FILE, 'w', encoding='utf-8') as f:
        json.dump(terms, f, indent=4)

def check_blocked(query):
    """Check if query contains any blocked term (case-insensitive substring)."""
    blocked_terms = load_blocked_terms()
    query_lower = query.lower()
    for term in blocked_terms:
        if term.lower() in query_lower:
            return True
    return False

def log_query(query, response, session_id):
    """Log query and response to JSON file."""
    timestamp = datetime.now().isoformat()
    log_entry = {
        "timestamp": timestamp,
        "query": query,
        "response": response,
        "session_id": session_id
    }
    
    logs = []
    if os.path.exists(LOGS_FILE):
        try:
            with open(LOGS_FILE, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:  # Only parse if file is not empty
                    logs = json.loads(content)
        except (json.JSONDecodeError, IOError):
            # If file is empty or invalid, start with empty list
            logs = []
    
    logs.append(log_entry)
    
    with open(LOGS_FILE, 'w', encoding='utf-8') as f:
        json.dump(logs, f, indent=4)

def log_admin_activity(action, username, details=None):
    """Log admin activity to JSON file."""
    timestamp = datetime.now().isoformat()
    log_entry = {
        "timestamp": timestamp,
        "action": action,
        "username": username,
        "details": details or ""
    }
    
    logs = []
    if os.path.exists(ADMIN_LOGS_FILE):
        try:
            with open(ADMIN_LOGS_FILE, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:  # Only parse if file is not empty
                    logs = json.loads(content)
        except (json.JSONDecodeError, IOError):
            # If file is empty or invalid, start with empty list
            logs = []
    
    logs.append(log_entry)
    
    with open(ADMIN_LOGS_FILE, 'w', encoding='utf-8') as f:
        json.dump(logs, f, indent=4)

def load_rag():
    config = load_config()
    embeddings = OllamaEmbeddings(model=config["embedding_model"])
    db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    llm = OllamaLLM(model=config["generation_model"])
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever())
    return qa

def query_rag(query, session_id):
    """Query RAG with blocked check and logging."""
    if check_blocked(query):
        blocked_msg = "Sorry, that query is not allowed. Please ask something else."
        log_query(query, blocked_msg, session_id)
        return blocked_msg
    
    qa = load_rag()
    try:
        response = qa.invoke({"query": query})["result"]  # Use invoke instead of run
        log_query(query, response, session_id)
        return response
    except Exception as e:
        error_msg = f"Error querying RAG: {str(e)}"
        log_query(query, error_msg, session_id)
        return error_msg

def ingest_documents(folder_path="./documents"):
    config = load_config()
    embeddings = OllamaEmbeddings(model=config["embedding_model"])
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    documents = []
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if file.endswith('.txt'):
            loader = TextLoader(file_path)
        elif file.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        else:
            continue
        try:
            docs = loader.load()
            split_docs = text_splitter.split_documents(docs)
            documents.extend(split_docs)
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
            continue
    
    if documents:
        db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
        db.add_documents(documents)
        return f"Ingested {len(documents)} chunks from {len(os.listdir(folder_path))} files!"
    return "No valid documents found."