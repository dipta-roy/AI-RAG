# AI-RAG

A minimal but practical implementation of **Retrieval-Augmented Generation (RAG)** in Python.
This project shows how to connect an information retrieval pipeline with a Large Language Model (LLM) so the model can give more accurate, grounded, and context-aware answers instead of hallucinating.

## 🔑 Key Features

* **Document Ingestion** – Import and preprocess external documents into a searchable knowledge base.
* **Context-Aware Retrieval** – Find the most relevant chunks of information for a given query.
* **Augmented Generation** – Use retrieved context to guide the LLM, improving factual accuracy.
* **Admin Interface** – Upload documents, manage ingestion, and monitor logs through a web UI.
* **Modular Design** – Swap or extend components (retriever, embedding model, UI) without breaking the pipeline.

## 📂 Project Structure

```
AI-RAG/
│── documents/
│── app.py              # Main Streamlit app (user + admin UI)
│── gen.py              # Script to generate and hash admin password
│── rag_utils.py        # Utilities for text chunking, embeddings, and RAG pipeline
│── config.json         # App configuration (models, API keys, parameters)
│── credentials.toml    # Stores admin credentials (hashed password required)
│── admin_logs.json     # System logs for admin actions
│── chat_logs.json      # Chat session logs
│── blocked_terms.json  # List of blocked/filtered terms
│── pages/              # Streamlit multipage UI (admin & logs)
	│── adminui.py      # Logic to admin functions and modules
	│── logs.py         # View the logs
│── requirements.txt    # Python dependencies
```

## 🛠️ Tech Stack

* **Python 3.11**
* **Streamlit** – Interactive UI
* **Ollama** – Local model serving
* **Embeddings & LLMs** – Tested with:
  * `gemma3:latest` (LLM for answering)
  * `nomic-embed-text` (for vector embeddings)

## 🚀 Getting Started

### Prerequisites

* Python 3.11 installed
* [Ollama](https://ollama.ai) installed and running

### Installation

1. **Download Ollama Models**

   ```bash
   ollama serve
   ollama pull gemma3:latest
   ollama pull nomic-embed-text
   ```

2. **Clone the repository**

   ```bash
   git clone https://github.com/dipta-roy/AI-RAG.git
   cd AI-RAG
   ```

3. **Set up a virtual environment**

   ```bash
   python -m venv rag_env
   rag_env\Scripts\activate.bat   # Windows
   # or
   source rag_env/bin/activate    # Linux / macOS
   ```

4. **Install dependencies**

   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. **Configure Admin Password**

   * Open `gen.py` in your editor.
   * Replace the default password with your desired one.
   * Run:

     ```bash
     python gen.py
     ```
   * Copy the generated **hashed password** and paste it into `credentials.toml`.

6. **Run the application**

   ```bash
   streamlit run app.py
   ```

## 📊 Usage

### Admin Panel

* Open: `http://<ip>:8501/adminui`
* Login:

  * **Username**: `admin`
  * **Password**: the one you configured in `gen.py`
* Upload documents → Click **Run Ingestion Process**
* Monitor system logs from the panel

### Chat UI

* Open: `http://<ip>:8501/`
* Ask questions directly in the chat interface
* Answers will be generated using retrieved context from your ingested documents

## 📌 Notes

* Ensure Ollama is running in the background before starting the app.
* All credentials are hashed before storage for security.
* Logs (`admin_logs.json` and `chat_logs.json`) are automatically updated as the system runs.
* Higher compute is needed for running this application

