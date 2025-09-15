import streamlit as st
import os
import json
import toml
import streamlit_authenticator as stauth

st.set_page_config(page_title="RAG Admin Logs", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CRED_FILE = os.path.join(BASE_DIR, "..", "credentials.toml")
CHAT_LOGS_FILE = os.path.join(BASE_DIR, "..", "chat_logs.json")
ADMIN_LOGS_FILE = os.path.join(BASE_DIR, "..", "admin_logs.json")

config = toml.load(CRED_FILE)
authenticator = stauth.Authenticate(
    credentials=config["credentials"],
    cookie_name="RAG_Admin",
    key="R@gDREA1",
    cookie_expiry_days=30
)

# Login check
authenticator.login(location="main")
auth_status = st.session_state.get("authentication_status", None)
username = st.session_state.get("username", None)

if auth_status is False:
    st.error("Username/password is incorrect")
elif auth_status is None:
    st.warning("Please enter your username and password")
elif auth_status:
    st.success(f"Welcome {username}!")
    authenticator.logout(location="main")

    st.title("RAG Logs Viewer")
    tabs = st.tabs(["Chat Logs", "Admin Logs"])

    # Chat logs
    with tabs[0]:
        if os.path.exists(CHAT_LOGS_FILE):
            with open(CHAT_LOGS_FILE, 'r', encoding='utf-8') as f:
                logs = json.load(f)
            for entry in reversed(logs[-50:]):
                st.json(entry)
        else:
            st.info("No chat logs found.")

    # Admin logs
    with tabs[1]:
        if os.path.exists(ADMIN_LOGS_FILE):
            with open(ADMIN_LOGS_FILE, 'r', encoding='utf-8') as f:
                logs = json.load(f)
            for entry in reversed(logs[-50:]):
                st.json(entry)
        else:
            st.info("No admin logs found.")
