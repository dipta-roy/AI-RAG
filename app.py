import streamlit as st
import uuid
from rag_utils import query_rag

# Set page config (no sidebar layout)
st.set_page_config(page_title="RAG Chat Assistant", page_icon="🤖", layout="wide")

# Hide sidebar, deploy button, and three-dot menu
st.markdown(
    """
    <style>
    /* Hide sidebar and navigation */
    [data-testid="stSidebar"] {display: none !important;}
    [data-testid="stSidebarNav"] {display: none !important;}
    /* Hide entire header, deploy button, and three-dot menu */
    header {display: none !important;}
    [data-testid="stHeader"] {display: none !important;}
    [data-testid="stAppHeader"] {display: none !important;}
    [data-testid="stToolbarActions"] {display: none !important;}
    /* Ensure main content uses full width */
    .main .block-container {max-width: 100%; padding: 1rem;}
    /* Fallback for dynamic elements */
    div[data-testid="stToolbar"] {display: none !important;}
    </style>
    <script>
    // JavaScript fallback to hide header elements
    document.addEventListener("DOMContentLoaded", function() {
        const header = document.querySelector("header");
        if (header) header.style.display = "none";
        const toolbar = document.querySelector('[data-testid="stToolbar"]');
        if (toolbar) toolbar.style.display = "none";
    });
    </script>
    """,
    unsafe_allow_html=True
)

st.title("RAG Chat Assistant")
st.write("Ask questions about your documents!")

# Initialize chat history and session ID
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())  # Unique session ID for logging

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What's your question?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = query_rag(prompt, st.session_state.session_id)
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})