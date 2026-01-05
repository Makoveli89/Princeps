import streamlit as st
import asyncio
from lib.workspace import get_active_workspace
from lib.db import safe_query
from lib.dispatcher import run_task_async
from brain.core.models import AgentRun

st.set_page_config(page_title="Reports & Gym", page_icon="🏋️", layout="wide")

ws_id = get_active_workspace()
if not ws_id:
    st.error("No active workspace.")
    st.stop()

st.title("Reports & Gym")

tab1, tab2 = st.tabs(["Metrics", "Gym Suite"])

# --- Metrics ---
with tab1:
    st.header("System Metrics")

    def get_agent_metrics(session):
        # Success rate by agent
        from sqlalchemy import func
        stats = session.query(
            AgentRun.agent_id,
            func.count(AgentRun.id).label('total'),
            func.sum(func.cast(AgentRun.success,  st.session_state.get('int_type', int))).label('success_count'), # Cast bool to int for sum
            func.avg(AgentRun.duration_ms).label('avg_latency')
        ).filter(AgentRun.tenant_id == ws_id).group_by(AgentRun.agent_id).all()

        return stats

    # Check dialect for casting (SQLite vs Postgres)
    # Simple workaround: fetch all and aggregate in python if SQL is tricky across dialects in this quick UI
    def get_metrics_pythonic(session):
        runs = session.query(AgentRun.agent_id, AgentRun.success, AgentRun.duration_ms).filter(AgentRun.tenant_id == ws_id).all()
        return runs

    runs_data = safe_query(get_metrics_pythonic)

    if runs_data:
        import pandas as pd
        df = pd.DataFrame(runs_data, columns=['Agent', 'Success', 'Duration'])

        # Group
        grouped = df.groupby('Agent').agg(
            Total=('Success', 'count'),
            Success=('Success', 'sum'),
            Avg_Latency=('Duration', 'mean')
        ).reset_index()

        grouped['Success_Rate'] = (grouped['Success'] / grouped['Total'] * 100).round(1)
        grouped['Avg_Latency'] = grouped['Avg_Latency'].round(0)

        st.dataframe(
            grouped[['Agent', 'Total', 'Success_Rate', 'Avg_Latency']],
            use_container_width=True,
            column_config={
                "Success_Rate": st.column_config.ProgressColumn(
                    "Success Rate", format="%.1f%%", min_value=0, max_value=100
                ),
                "Avg_Latency": st.column_config.NumberColumn(
                    "Avg Latency (ms)"
                )
            }
        )
    else:
        st.info("No data available.")

# --- Gym Suite ---
with tab2:
    st.header("Gym Suite (Evaluation)")
    st.caption("Run predefined evaluation tasks to assess agent performance.")

    # Define a simple "Golden Set" of questions
    gym_questions = [
        {"id": "G1", "q": "What is the capital of France?", "expect": "Paris"},
        {"id": "G2", "q": "Summarize the concept of 'idempotency'.", "expect": "multiple times"}, # loose keyword match
        {"id": "G3", "q": "Who is the CEO of OpenAI?", "expect": "Altman"}
    ]

    if st.button("Run Gym Suite"):
        results = []
        progress_bar = st.progress(0)

        for idx, item in enumerate(gym_questions):
            with st.spinner(f"Running Test {item['id']}..."):
                # Run task
                res = asyncio.run(run_task_async(item['q'], "default", ws_id))

                # Check pass/fail (very naive string match for now)
                passed = False
                output_str = str(res.output)
                if item['expect'].lower() in output_str.lower():
                    passed = True

                results.append({
                    "Test ID": item['id'],
                    "Question": item['q'],
                    "Passed": "✅" if passed else "❌",
                    "Duration": f"{res.duration_ms:.0f}ms",
                    "Output Preview": output_str[:100] + "..."
                })

            progress_bar.progress((idx + 1) / len(gym_questions))

        st.success("Gym Suite Completed!")

        res_df = pd.DataFrame(results)
        st.dataframe(res_df, use_container_width=True)

        # Summary
        pass_count = len([r for r in results if r["Passed"] == "✅"])
        st.metric("Pass Rate", f"{pass_count}/{len(gym_questions)}")
