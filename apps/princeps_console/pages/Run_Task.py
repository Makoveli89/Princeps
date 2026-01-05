import streamlit as st
import json
import asyncio
from lib.workspace import get_active_workspace, get_active_workspace_details
from lib.dispatcher import run_task_async

st.set_page_config(page_title="Run Task", page_icon="🚀", layout="wide")

ws_id = get_active_workspace()
ws_details = get_active_workspace_details()

if not ws_id:
    st.error("No active workspace.")
    st.stop()

st.title("Run Task")
st.caption(f"Executing in Workspace: **{ws_details.name}** ({ws_id})")

# --- Input Form ---
with st.form("run_task_form"):
    task_prompt = st.text_area("Task Prompt", height=150, help="Describe what you want the agent to do.")

    workflow_options = ["default", "plan_and_execute", "create_plan"]
    workflow_select = st.selectbox("Workflow", options=workflow_options, index=0)

    submitted = st.form_submit_button("Run Task", type="primary")

if submitted:
    if not task_prompt:
        st.error("Please enter a prompt.")
    else:
        status_container = st.container()

        with status_container:
            with st.spinner("Dispatching agent..."):
                try:
                    # Run the task
                    result = asyncio.run(run_task_async(task_prompt, workflow_select, ws_id))

                    if result.success:
                        st.success(f"Task Completed Successfully! (Duration: {result.duration_ms:.2f}ms)")

                        st.subheader("Output")

                        # Try to display output intelligently
                        output_data = result.output

                        # If "answer" or "result" key exists, highlight it
                        if isinstance(output_data, dict):
                            if "answer" in output_data:
                                st.markdown(f"### Answer\n{output_data['answer']}")
                            elif "result" in output_data:
                                st.markdown(f"### Result\n{output_data['result']}")

                            with st.expander("Full Structured Output (JSON)"):
                                st.json(output_data)
                        else:
                            st.write(output_data)

                        st.caption(f"Run ID: `{result.task_id}`")
                        st.link_button("View Logs", f"/Runs_Logs") # Basic link, Streamlit routing is tricky without query params

                    else:
                        st.error("Task Failed")
                        st.error(f"Error: {result.error}")
                        st.json(result.to_dict())

                except Exception as e:
                    st.error(f"System Error: {e}")
