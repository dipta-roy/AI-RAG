import streamlit as st
import os
import json
import toml
import subprocess
import streamlit_authenticator as stauth
from rag_utils import ingest_documents, load_blocked_terms, save_blocked_terms, log_admin_activity

st.set_page_config(page_title="RAG Admin Panel", layout="wide")

# -------------------- Paths --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CRED_FILE = os.path.join(BASE_DIR, "..", "credentials.toml")
DOCUMENTS_DIR = os.path.join(BASE_DIR, "..", "documents")
CONFIG_FILE = os.path.join(BASE_DIR, "..", "config.json")
CHAT_LOGS_FILE = os.path.join(BASE_DIR, "..", "chat_logs.json")
ADMIN_LOGS_FILE = os.path.join(BASE_DIR, "..", "admin_logs.json")
os.makedirs(DOCUMENTS_DIR, exist_ok=True)

# -------------------- Helper: get Ollama models --------------------
def get_installed_ollama_models():
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=True
        )
        lines = result.stdout.strip().split("\n")
        # Skip header line, take first column as model name
        models = [line.split()[0] for line in lines[1:]]
        return models
    except Exception as e:
        print("Error fetching Ollama models:", e)
        return []

# -------------------- Load credentials --------------------
config = toml.load(CRED_FILE)
authenticator = stauth.Authenticate(
    credentials=config["credentials"],
    cookie_name="RAG_Admin",
    key="R@gDREA1",
    cookie_expiry_days=30
)

# -------------------- Login --------------------
authenticator.login(location="main")
auth_status = st.session_state.get("authentication_status", None)
username = st.session_state.get("username", None)
name = st.session_state.get("name", None)

if auth_status is False:
    st.error("Username/password is incorrect")
elif auth_status is None:
    st.warning("Please enter your username and password")
elif auth_status:
    st.success(f"Welcome {name or username}!")
    authenticator.logout(location="main")

    # -------------------- Header --------------------
    st.title("RAG Admin Panel")
    st.markdown("---")

    # View Logs button
    if st.button("View Logs"):
        if os.path.exists(os.path.join(BASE_DIR, "logs.py")):
            st.query_params(page="logs")
        else:
            st.warning("Logs page not found.")

    # View Metrics button (new)
    if st.button("View Metrics"):
        if os.path.exists(os.path.join(BASE_DIR, "metricsui.py")):
            st.query_params(page="metricsui")
        else:
            st.warning("Metrics page not found.")

    # -------------------- Document Upload --------------------
    st.subheader("Upload & Ingest Documents")
    uploaded_files = st.file_uploader(
        "Choose files (TXT or PDF)", accept_multiple_files=True, type=['txt', 'pdf']
    )
    if uploaded_files:
        for file in uploaded_files:
            save_path = os.path.join(DOCUMENTS_DIR, file.name)
            with open(save_path, "wb") as f:
                f.write(file.getbuffer())
        st.success(f"{len(uploaded_files)} document{'s' if len(uploaded_files) > 1 else ''} uploaded")
        log_admin_activity("upload_documents", username, details=f"{len(uploaded_files)} files uploaded")

    # -------------------- Ingestion --------------------
    if st.button("Run Ingestion Process"):
        with st.spinner("Ingesting documents... This may take a minute."):
            result = ingest_documents()
        st.success(result)
        st.balloons()
        log_admin_activity("ingest_documents", username)

    # -------------------- Current Documents --------------------
    st.subheader("Current Documents")
    docs = os.listdir(DOCUMENTS_DIR)
    if docs:
        for doc in docs:
            st.write(f"- {doc}")
    else:
        st.info("No documents yet. Upload some!")

    # -------------------- Blocked Terms --------------------
    st.subheader("Manage Blocked Query Filter")
    blocked_terms = load_blocked_terms()
    current_filter = ", ".join(blocked_terms) if blocked_terms else ""
    new_filter = st.text_area("Blocked Terms", value=current_filter, help="Enter terms like: password, hack, illegal")

    if st.button("Update Filter"):
        if new_filter.strip():
            updated_terms = [term.strip() for term in new_filter.split(",")]
            save_blocked_terms(updated_terms)
            st.success(f"Updated blocked terms: {', '.join(updated_terms)}")
        else:
            save_blocked_terms([])
            st.success("Cleared all blocked terms.")
        log_admin_activity("update_blocked_terms", username)

    # -------------------- Configure Models --------------------
    st.subheader("Configure Models")
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            model_config = json.load(f)
    else:
        model_config = {"generation_model": "gemma2:9b", "embedding_model": "nomic-embed-text"}

    installed_models = get_installed_ollama_models()
    if not installed_models:
        installed_models = ["gemma2:9b"]  # fallback

    current_gen_model = model_config.get("generation_model", installed_models[0])
    if current_gen_model not in installed_models:
        current_gen_model = installed_models[0]

    gen_model = st.selectbox(
        "Generation Model",
        installed_models,
        index=installed_models.index(current_gen_model)
    )

    emb_options = ["nomic-embed-text", "mxbai-embed-large"]
    current_emb_model = model_config.get("embedding_model", emb_options[0])
    if current_emb_model not in emb_options:
        current_emb_model = emb_options[0]

    emb_model = st.selectbox(
        "Embedding Model",
        emb_options,
        index=emb_options.index(current_emb_model)
    )

    if st.button("Update Models"):
        model_config["generation_model"] = gen_model
        model_config["embedding_model"] = emb_model
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(model_config, f, indent=4)
        st.success(f"Updated models: Generation={gen_model}, Embedding={emb_model}")
        log_admin_activity("update_models", username, details=f"Generation: {gen_model}, Embedding: {emb_model}")
