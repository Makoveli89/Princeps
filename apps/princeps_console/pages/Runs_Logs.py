import pandas as pd
import streamlit as st
from lib.db import safe_query
from lib.workspace import get_active_workspace

from brain.core.models import AgentRun

st.set_page_config(page_title="Runs & Logs", page_icon="📜", layout="wide")

ws_id = get_active_workspace()
if not ws_id:
    st.error("No active workspace.")
    st.stop()

st.title("Runs & Logs")

# --- Filters ---
with st.expander("Filters", expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        agent_filter = st.text_input("Agent ID")
    with c2:
        status_filter = st.selectbox("Status", ["All", "Success", "Failed"], index=0)
    with c3:
        limit = st.number_input("Limit", min_value=10, max_value=500, value=50)


# --- Query ---
def get_runs(session):
    query = session.query(AgentRun).filter(AgentRun.tenant_id == ws_id)

    if agent_filter:
        query = query.filter(AgentRun.agent_id.ilike(f"%{agent_filter}%"))
    if status_filter == "Success":
        query = query.filter(AgentRun.success)
    elif status_filter == "Failed":
        query = query.filter(not AgentRun.success)

    query = query.order_by(AgentRun.started_at.desc()).limit(limit)
    return query.all()


runs = safe_query(get_runs)

# --- Display List ---
if not runs:
    st.info("No runs found.")
else:
    # Main Table
    data = []
    for r in runs:
        data.append(
            {
                "ID": str(r.id),
                "Time": r.started_at,
                "Agent": r.agent_id,
                "Success": "✅" if r.success else "❌",
                "Duration (ms)": r.duration_ms,
                "Task Preview": (r.task[:80] + "...") if r.task else "",
            }
        )

    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.divider()

    # --- Detail View ---
    st.subheader("Run Details")
    selected_run_id = st.selectbox(
        "Select Run to Inspect", options=[r.id for r in runs], format_func=lambda x: str(x)
    )

    if selected_run_id:

        def get_run_detail(session):
            return session.query(AgentRun).filter(AgentRun.id == selected_run_id).first()

        run_detail = safe_query(get_run_detail)

        if run_detail:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Agent:** {run_detail.agent_id}")
                st.markdown(f"**Status:** {'Success' if run_detail.success else 'Failed'}")
                st.markdown(f"**Duration:** {run_detail.duration_ms} ms")
            with c2:
                st.markdown(f"**Time:** {run_detail.started_at}")
                st.markdown(f"**Task Hash:** `{run_detail.task_hash}`")

            st.markdown("### Task")
            st.code(run_detail.task)

            t1, t2, t3 = st.tabs(["Output / Solution", "Context & Tools", "Errors & Feedback"])

            with t1:
                st.json(run_detail.solution or {})

            with t2:
                st.write("**Tools Used:**")
                st.write(run_detail.tools_used)
                st.write("**Context:**")
                st.json(run_detail.context or {})

            with t3:
                if not run_detail.success:
                    st.error(run_detail.feedback or "Unknown Error")
                else:
                    st.write(run_detail.feedback or "No feedback.")
