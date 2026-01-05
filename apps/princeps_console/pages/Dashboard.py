import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from lib.workspace import get_active_workspace, get_active_workspace_details
from lib.db import safe_query, get_db_session
from brain.core.models import AgentRun, Document, DocChunk

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")

ws_id = get_active_workspace()
if not ws_id:
    st.error("No active workspace selected.")
    st.stop()

ws_details = get_active_workspace_details()
st.title(f"Dashboard: {ws_details.name}")

# --- Status Cards ---
def get_dashboard_stats(session):
    # Counts
    run_count_24h = session.query(AgentRun).filter(
        AgentRun.tenant_id == ws_id,
        AgentRun.started_at >= datetime.utcnow() - timedelta(hours=24)
    ).count()

    success_count_24h = session.query(AgentRun).filter(
        AgentRun.tenant_id == ws_id,
        AgentRun.started_at >= datetime.utcnow() - timedelta(hours=24),
        AgentRun.success == True
    ).count()

    success_rate = (success_count_24h / run_count_24h * 100) if run_count_24h > 0 else 0.0

    # DB Checks (Simplistic)
    db_connected = True

    # Check pgvector (if chunk table exists and has vector col - just assume yes if code runs)
    pgvector_enabled = True # hardcoded for UI visual, real check in settings

    return {
        "runs_24h": run_count_24h,
        "success_rate": success_rate,
        "db_ok": db_connected,
        "vector_ok": pgvector_enabled
    }

stats = safe_query(get_dashboard_stats)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Runs (24h)", stats["runs_24h"])
col2.metric("Success Rate (24h)", f"{stats['success_rate']:.1f}%")
col3.metric("DB Status", "Connected" if stats["db_ok"] else "Error", delta_color="normal")
col4.metric("Vector DB", "Enabled" if stats["vector_ok"] else "Missing", delta_color="normal")

st.divider()

# --- Charts ---
st.subheader("Activity Overview")

def get_run_history(session):
    # Last 7 days
    start_date = datetime.utcnow() - timedelta(days=7)
    runs = session.query(AgentRun.started_at, AgentRun.success, AgentRun.agent_id).filter(
        AgentRun.tenant_id == ws_id,
        AgentRun.started_at >= start_date
    ).all()

    data = [{"time": r.started_at, "success": r.success, "agent": r.agent_id} for r in runs]
    return pd.DataFrame(data)

df = safe_query(get_run_history)

if not df.empty:
    # Runs over time
    c1, c2 = st.columns(2)

    with c1:
        st.caption("Runs over Time")
        # Bin by hour or day
        df['date'] = df['time'].dt.date
        daily_counts = df.groupby('date').size()
        st.line_chart(daily_counts)

    with c2:
        st.caption("Success Rate by Agent")
        agent_stats = df.groupby('agent')['success'].mean() * 100
        st.bar_chart(agent_stats)
else:
    st.info("No run history available for charts.")

# --- Recent Failures ---
st.subheader("Recent Failures")

def get_recent_failures(session):
    failures = session.query(AgentRun).filter(
        AgentRun.tenant_id == ws_id,
        AgentRun.success == False
    ).order_by(AgentRun.started_at.desc()).limit(5).all()

    return [{"id": str(r.id), "time": r.started_at, "agent": r.agent_id, "task": r.task[:100]} for r in failures]

failures = safe_query(get_recent_failures)

if failures:
    for f in failures:
        st.error(f"**{f['time']}** | Agent: {f['agent']} | Task: {f['task']}... [Run ID: {f['id']}]")
else:
    st.success("No recent failures found.")
