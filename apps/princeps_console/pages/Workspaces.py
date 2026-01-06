import streamlit as st
from lib.db import safe_query
from lib.workspace import create_workspace, rename_workspace

from brain.core.models import AgentRun, DocChunk, Document, Tenant

st.set_page_config(page_title="Workspaces", page_icon="🏢", layout="wide")

st.title("Workspaces Management")

# --- Create New Workspace ---
with st.expander("Create New Workspace", expanded=False):
    with st.form("create_ws_form"):
        new_name = st.text_input("Workspace Name")
        new_desc = st.text_area("Description")
        submitted = st.form_submit_button("Create")

        if submitted:
            if not new_name:
                st.error("Name is required.")
            else:
                ws_id, err = create_workspace(new_name, new_desc)
                if err:
                    st.error(f"Error: {err}")
                else:
                    st.success(f"Workspace '{new_name}' created!")
                    st.rerun()

# --- List Workspaces ---
st.subheader("Existing Workspaces")


def get_workspaces_with_stats(session):
    tenants = session.query(Tenant).all()
    stats = []
    for t in tenants:
        doc_count = session.query(Document).filter(Document.tenant_id == t.id).count()
        run_count = session.query(AgentRun).filter(AgentRun.tenant_id == t.id).count()
        chunk_count = session.query(DocChunk).filter(DocChunk.tenant_id == t.id).count()
        stats.append(
            {
                "id": str(t.id),
                "name": t.name,
                "docs": doc_count,
                "chunks": chunk_count,
                "runs": run_count,
                "obj": t,
            }
        )
    return stats


ws_stats = safe_query(get_workspaces_with_stats)

for ws in ws_stats:
    c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 1, 1])

    # Name & Rename
    with c1:
        st.write(f"**{ws['name']}**")
        st.caption(f"ID: {ws['id']}")

    # Stats
    with c2:
        st.metric("Docs", ws["docs"])
    with c3:
        st.metric("Chunks", ws["chunks"])
    with c4:
        st.metric("Runs", ws["runs"])

    # Actions
    with c5:
        with st.popover("Rename"):
            new_rename_val = st.text_input("New Name", value=ws["name"], key=f"rename_{ws['id']}")
            if st.button("Save", key=f"save_{ws['id']}"):
                success, err = rename_workspace(ws["id"], new_rename_val)
                if success:
                    st.success("Renamed!")
                    st.rerun()
                else:
                    st.error(err)

    st.divider()
