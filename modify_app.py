import sys

with open("app.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

if any("tab_eval, tab_interactive = st.tabs" in line for line in lines):
    print("modify_app.py: app.py is already modified. Skipping.")
    sys.exit(0)

new_lines = []
in_main_area = False

for idx, line in enumerate(lines):
    if line.startswith("# Main area"):
        pass
        
    if "if run_button:" in line and not in_main_area:
        in_main_area = True
        new_lines.append("tab_eval, tab_interactive = st.tabs([\"📊 Evaluation Dashboard\", \"✍️ Interactive User Panel\"])\n\n")
        new_lines.append("with tab_eval:\n")
        new_lines.append("    " + line)
        continue
        
    if in_main_area:
        if line == "\n":
            new_lines.append(line)
        else:
            new_lines.append("    " + line)
    else:
        new_lines.append(line)

# Now add the interactive tab at the end of the file
interactive_code = """
with tab_interactive:
    st.markdown("### ✍️ Test Agent Interactively")
    st.markdown("Draft an email and see how the current agent responds in real-time.")
    
    with st.form("interactive_form"):
        col_s, col_subj = st.columns(2)
        sender_input = col_s.text_input("Sender", value="boss@company.com")
        subject_input = col_subj.text_input("Subject", value="Urgent: Submit Timesheets")
        body_input = st.text_area("Email Body", value="Please submit your timesheets by 5 PM today, otherwise payroll will be delayed.", height=150)
        
        submitted = st.form_submit_button("Submit to Agent")
        
    if submitted:
        # Create an agent instance
        with st.spinner("Agent is thinking..."):
            try:
                interactive_agent = _make_agent(agent_type, seed=seed, llm_config=llm_config)
                
                # Construct observation
                obs = {
                    "email": {
                        "id": "interactive_001",
                        "sender": sender_input,
                        "subject": subject_input,
                        "body": body_input
                    },
                    "history": [],
                    "step": 0,
                    "total_steps": 10,
                    "difficulty": "interactive"
                }
                
                # Get action
                action = interactive_agent.act(obs)
                
                st.success("✅ Analysis Complete")
                
                # Display Results beautifully
                res_col1, res_col2, res_col3 = st.columns(3)
                
                with res_col1:
                    st.markdown("#### 🎯 Predicted Intents")
                    for intent in action.get("intents", []):
                        st.markdown(f"- `{intent}`")
                        
                with res_col2:
                    st.markdown("#### 🚨 Priority")
                    priority = action.get("priority", "N/A")
                    color = "#48bb78" if priority == "low" else "#ecc94b" if priority == "medium" else "#ed8936" if priority == "high" else "#e53e3e"
                    st.markdown(f"**<span style='color:{color}; font-size:1.2rem'>{priority.upper()}</span>**", unsafe_allow_html=True)
                    
                with res_col3:
                    st.markdown("#### ⚡ Action")
                    act = action.get("action", "N/A")
                    st.markdown(f"**`{act.upper()}`**")
                    
                st.markdown("#### 💬 Response Generation")
                st.info(action.get("response", "No response generated."))
                
            except Exception as e:
                st.error(f"Error running agent: {e}")
"""

new_lines.append(interactive_code)

with open("app.py", "w", encoding="utf-8") as f:
    f.writelines(new_lines)
