import os
import json
import streamlit as st
import toml
import streamlit_authenticator as stauth
import pandas as pd
import matplotlib.pyplot as plt

# -------------------- Paths --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CRED_FILE = os.path.join(BASE_DIR, "..", "credentials.toml")
METRICS_FILE = os.path.join(BASE_DIR, "..", "metrics.json")

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

    # -------------------- Dashboard --------------------
    st.title("📊 RAG Metrics Dashboard")
    st.markdown("---")

    if not os.path.exists(METRICS_FILE):
        st.info("No metrics found yet. Run document ingestion first.")
    else:
        with open(METRICS_FILE, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)

        if not metrics_data:
            st.info("Metrics file is empty.")
        else:
            df = pd.DataFrame(metrics_data)

            # Show raw metrics table
            st.subheader("Raw Metrics Data")
            st.dataframe(df)

            # Plot: Chunks over time
            st.subheader("Chunks Created Over Time")
            fig1, ax1 = plt.subplots()
            df.plot(x="timestamp", y="total_chunks", marker="o", ax=ax1)
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig1)

            # Plot: Average vs Max chunk size
            st.subheader("Average & Max Chunk Sizes")
            fig2, ax2 = plt.subplots()
            df.plot(x="timestamp", y=["avg_chunk_size_chars", "max_chunk_size_chars"], marker="o", ax=ax2)
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig2)

            # Summary stats
            st.subheader("Summary")
            st.write(f"**Total ingestion runs:** {len(df)}")
            st.write(f"**Average chunks per run:** {df['total_chunks'].mean():.2f}")
            st.write(f"**Average chunk size (chars):** {df['avg_chunk_size_chars'].mean():.2f}")
