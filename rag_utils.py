import os
import json
import warnings
from datetime import datetime

from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma

# Advanced loaders and splitters
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain.retrievers import ParentDocumentRetriever

# Suppress PDF warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="PyPDF2")

# -------------------- File Paths --------------------
BLOCKED_FILE = "blocked_terms.json"
LOGS_FILE = "chat_logs.json"
ADMIN_LOGS_FILE = "admin_logs.json"
CONFIG_FILE = "config.json"


# -------------------- Config --------------------
def load_config():
    if not os.path.exists(CONFIG_FILE):
        default_config = {
            "generation_model": "gemma2:9b",
            "embedding_model": "nomic-embed-text"
        }
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=4)
        return default_config
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


# -------------------- Blocked Terms --------------------
def load_blocked_terms():
    if not os.path.exists(BLOCKED_FILE):
        with open(BLOCKED_FILE, 'w', encoding='utf-8') as f:
            json.dump([], f)
        return []
    with open(BLOCKED_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_blocked_terms(terms):
    with open(BLOCKED_FILE, 'w', encoding='utf-8') as f:
        json.dump(terms, f, indent=4)


def check_blocked(query):
    blocked_terms = load_blocked_terms()
    query_lower = query.lower()
    return any(term.lower() in query_lower for term in blocked_terms)


# -------------------- Logging --------------------
def log_query(query, response, session_id):
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
                if content:
                    logs = json.loads(content)
        except (json.JSONDecodeError, IOError):
            logs = []

    logs.append(log_entry)

    with open(LOGS_FILE, 'w', encoding='utf-8') as f:
        json.dump(logs, f, indent=4)


def log_admin_activity(action, username, details=None):
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
                if content:
                    logs = json.loads(content)
        except (json.JSONDecodeError, IOError):
            logs = []

    logs.append(log_entry)

    with open(ADMIN_LOGS_FILE, 'w', encoding='utf-8') as f:
        json.dump(logs, f, indent=4)


# -------------------- Smarter Chunking --------------------
def get_hierarchical_splitter(embeddings):
    """
    Hierarchical semantic splitter:
    1. Try semantic breakpoints (embedding-based).
    2. Fallback: token-based to avoid cutting tokens.
    3. Last fallback: recursive character splitter.
    """
    try:
        return SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
    except Exception:
        # Fallback hybrid
        return RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " "],
            chunk_size=1000,
            chunk_overlap=150,
            length_function=len,
            is_separator_regex=False
        )


# -------------------- RAG --------------------
def load_rag():
    config = load_config()
    embeddings = OllamaEmbeddings(model=config["embedding_model"])
    db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    llm = OllamaLLM(model=config["generation_model"])

    # Parent-document retriever: children = fine-grained chunks, parents = larger context
    retriever = ParentDocumentRetriever(
        vectorstore=db,
        docstore=db,
        child_splitter=TokenTextSplitter(chunk_size=300, chunk_overlap=50),
        parent_splitter=TokenTextSplitter(chunk_size=1200, chunk_overlap=200),
    )
    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)


def query_rag(query, session_id):
    if check_blocked(query):
        msg = "Sorry, that query is not allowed. Please ask something else."
        log_query(query, msg, session_id)
        return msg
    qa = load_rag()
    try:
        response = qa.invoke({"query": query})["result"]
        log_query(query, response, session_id)
        return response
    except Exception as e:
        err = f"Error querying RAG: {str(e)}"
        log_query(query, err, session_id)
        return err


# -------------------- Document Ingestion --------------------
def ingest_documents(folder_path="./documents"):
    config = load_config()
    embeddings = OllamaEmbeddings(model=config["embedding_model"])
    splitter = get_hierarchical_splitter(embeddings)

    documents = []
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if file.endswith('.txt'):
            loader = TextLoader(file_path)
            docs = loader.load()
        elif file.endswith(('.pdf', '.docx', '.pptx')):
            # Unstructured handles PDFs, Word, PowerPoint, images, etc.
            loader = UnstructuredFileLoader(file_path, mode="elements")
            docs = loader.load()
        else:
            continue

        try:
            split_docs = splitter.split_documents(docs)
            for d in split_docs:
                d.page_content = d.page_content.strip()
            documents.extend(split_docs)
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
            continue

    if documents:
        db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
        db.add_documents(documents)
        return f"Ingested {len(documents)} chunks from {len(os.listdir(folder_path))} files!"
    return "No valid documents found."
