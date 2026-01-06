import os

import streamlit as st
from lib.db import safe_query

from brain.ingestion.ingest_service import IngestConfig

st.set_page_config(page_title="Settings & Health", page_icon="⚙️", layout="wide")

st.title("Settings & Health")

# --- Environment Variables ---
st.subheader("Environment Variables")

env_vars = ["DATABASE_URL", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"]

cols = st.columns(2)
for idx, var in enumerate(env_vars):
    val = os.getenv(var)
    status = "DETECTED ✅" if val else "MISSING ❌"

    with cols[idx % 2]:
        st.metric(var, status)

st.divider()

# --- Brain Config ---
st.subheader("Brain Configuration (Read-Only)")

config = IngestConfig()
c1, c2, c3 = st.columns(3)

with c1:
    st.text_input("Embedding Model", value=config.embedding_model, disabled=True)
with c2:
    st.number_input("Chunk Size", value=config.chunk_tokens, disabled=True)
with c3:
    st.text_input("pgvector", value="Enabled (Assumed)", disabled=True)

st.divider()

# --- Actions ---
st.subheader("System Actions")

if st.button("Run DB Smoke Test"):

    def smoke_test(session):
        # Just try to run a simple query
        from sqlalchemy import text

        session.execute(text("SELECT 1"))
        return True

    try:
        if safe_query(smoke_test):
            st.success("Database Connection OK! ✅")
    except Exception as e:
        st.error(f"Database Connection Failed: {e} ❌")

if st.button("Re-index Vectors (Not Implemented Safe)"):
    st.warning("This operation is dangerous and currently disabled in the console.")
