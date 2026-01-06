import os
import tempfile

import streamlit as st
from lib.workspace import get_active_workspace, get_active_workspace_details

from brain.ingestion.ingest_service import IngestService

st.set_page_config(page_title="Ingest Data", page_icon="📥", layout="wide")

ws_id = get_active_workspace()
ws_details = get_active_workspace_details()

if not ws_id:
    st.error("No active workspace.")
    st.stop()

st.title("Ingest Data")
st.caption(f"Target Workspace: **{ws_details.name}**")

tab1, tab2 = st.tabs(["Upload Document", "Ingest Repository"])

# --- Tab 1: Document Upload ---
with tab1:
    st.header("Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a file", type=["pdf", "txt", "md", "json", "py", "js", "html"]
    )

    doc_tags = st.text_input("Tags (comma separated)", placeholder="e.g. finance, Q1, draft")

    if st.button("Ingest Document"):
        if uploaded_file is not None:
            with st.spinner("Ingesting..."):
                # Save temp file
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=f"_{uploaded_file.name}"
                ) as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    tmp_path = tmp.name

                try:
                    # Initialize service
                    service = IngestService()  # Uses default config

                    # Process tags
                    meta = {}
                    if doc_tags:
                        meta["user_tags"] = [t.strip() for t in doc_tags.split(",")]

                    # Call ingest (we need to pass tenant NAME or ensure context works,
                    # but IngestService takes tenant_name. We should probably update IngestService or lookup name)
                    # The IngestService uses tenant_name to lookup ID.

                    result = service.ingest_document(
                        tmp_path, tenant_name=ws_details.name, metadata=meta
                    )

                    if result.success:
                        st.success("Ingestion Successful!")
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Docs Created", result.documents_created)
                        c2.metric("Chunks", result.chunks_created)
                        c3.metric("Embeddings", result.embeddings_generated)
                        st.success(f"Operation ID: {result.operation_id}")
                    else:
                        st.error("Ingestion Failed")
                        for err in result.errors:
                            st.error(err)

                except Exception as e:
                    st.error(f"System Error: {e}")
                finally:
                    os.unlink(tmp_path)

# --- Tab 2: Repository ---
with tab2:
    st.header("Ingest Repository")
    repo_url = st.text_input("Repository URL (Git)", placeholder="https://github.com/username/repo")
    local_path = st.text_input("Local Path (Optional override)", placeholder="/tmp/cloned_repo")

    if st.button("Ingest Repository"):
        if not repo_url and not local_path:
            st.error("Please provide a URL or local path.")
        else:
            with st.spinner("Ingesting Repository (this may take a while)..."):
                try:
                    service = IngestService()
                    # If local path is provided, use that as the 'path' argument, otherwise we might need to clone first.
                    # Wait, IngestService.ingest_repository expects a PATH on disk. It doesn't seem to do the 'git clone' itself
                    # unless I missed it in `ingest_service.py`. Let's check.
                    # Reading `ingest_service.py`: `repo_path = Path(path).resolve() ... if not repo_path.exists(): ...`
                    # So it expects a local path. It doesn't clone.
                    # For this UI, we might need to assume the user provides a local path OR implement cloning.
                    # Given "Internal Web UI", maybe local path is sufficient or I should do a quick clone.

                    target_path = local_path

                    if not target_path and repo_url:
                        # Attempt to clone to temp dir
                        st.info("Cloning repository...")
                        temp_dir = tempfile.mkdtemp()
                        import subprocess

                        subprocess.run(["git", "clone", repo_url, temp_dir], check=True)
                        target_path = temp_dir

                    if target_path:
                        result = service.ingest_repository(
                            target_path, url=repo_url, tenant_name=ws_details.name
                        )

                        if result.success:
                            st.success("Repository Ingestion Successful!")
                            st.json(result.to_dict())
                        else:
                            st.error("Ingestion Failed")
                            for err in result.errors:
                                st.error(err)

                except Exception as e:
                    st.error(f"Error: {e}")
