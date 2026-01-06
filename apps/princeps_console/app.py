import streamlit as st
from lib.db import get_db_session
from lib.workspace import get_active_workspace, get_active_workspace_details, switch_workspace

from brain.core.models import Tenant

st.set_page_config(
    page_title="Princeps Console",
    page_icon="👑",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for "Gothic Futuristic" look (Dark mode is default in Streamlit config usually, adding some overrides)
st.markdown(
    """
<style>
    /* Add some subtle styling */
    .stApp {
        background-color: #0e1117;
    }
    .metric-card {
        background-color: #262730;
        border: 1px solid #464b5d;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 10px;
    }
</style>
""",
    unsafe_allow_html=True,
)

# --- Sidebar ---
st.sidebar.title("Princeps Console")

# Workspace Selector
active_ws_id = get_active_workspace()
active_ws_details = get_active_workspace_details()

session = get_db_session()
try:
    all_workspaces = session.query(Tenant).all()
    ws_options = {str(ws.id): f"{ws.name} • {str(ws.id)[:8]}" for ws in all_workspaces}
finally:
    session.close()

if active_ws_id:
    selected_ws = st.sidebar.selectbox(
        "Workspace",
        options=list(ws_options.keys()),
        format_func=lambda x: ws_options.get(x, "Unknown"),
        index=list(ws_options.keys()).index(active_ws_id) if active_ws_id in ws_options else 0,
    )

    if selected_ws != active_ws_id:
        if switch_workspace(selected_ws):
            st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.info(f"Active: **{active_ws_details.name if active_ws_details else 'Unknown'}**")

else:
    st.sidebar.warning("No Workspace Active")

st.sidebar.markdown("---")

# Navigation (Handled by Streamlit multipage app structure, but we can add info here)
st.sidebar.markdown("### Navigation")
st.sidebar.page_link("pages/Dashboard.py", label="Dashboard", icon="📊")
st.sidebar.page_link("pages/Run_Task.py", label="Run Task", icon="🚀")
st.sidebar.page_link("pages/Ingest_Data.py", label="Ingest Data", icon="📥")
st.sidebar.page_link("pages/Runs_Logs.py", label="Runs & Logs", icon="📜")
st.sidebar.page_link("pages/Knowledge_Search.py", label="Knowledge Search", icon="🔍")
st.sidebar.page_link("pages/Reports_Gym.py", label="Reports & Gym", icon="🏋️")
st.sidebar.page_link("pages/Workspaces.py", label="Workspaces", icon="🏢")
st.sidebar.page_link("pages/Settings_Health.py", label="Settings & Health", icon="⚙️")

# Main content
st.title("Welcome to Princeps")
st.write("Select a module from the sidebar to begin.")

# Check for uninitialized state
if not active_ws_id:
    st.warning("Please create or select a workspace to continue.")
