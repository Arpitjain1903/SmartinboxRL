"""SmartInboxRL — Interactive Streamlit Dashboard.

Provides:
  - Live episode runner with step-by-step visualization
  - Reward component radar charts
  - Episode score history
  - Agent comparison tables

Launch
------
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv

load_dotenv()

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd

from environment.inbox_env import InboxEnv
from agents.random_agent import RandomAgent
from agents.rule_agent import RuleAgent
from agents.llm_agent import LLMAgent

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="SmartInboxRL Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for premium look
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
    }

    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }

    .main-header p {
        font-size: 1rem;
        opacity: 0.9;
        font-weight: 300;
    }

    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        color: white;
    }

    .metric-card .value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .metric-card .label {
        font-size: 0.85rem;
        color: #a0aec0;
        margin-top: 4px;
    }

    .email-card {
        background: #1a1a2e;
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.8rem 0;
        border-left: 4px solid #667eea;
    }

    .step-row {
        background: rgba(26, 26, 46, 0.5);
        border-radius: 8px;
        padding: 0.8rem;
        margin: 0.4rem 0;
        border: 1px solid #2d3748;
    }

    .reward-positive { color: #48bb78; font-weight: 600; }
    .reward-negative { color: #fc8181; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def _make_agent(name: str, seed: int = 42):
    if name == "random":
        return RandomAgent(seed=seed)
    elif name == "rule":
        return RuleAgent()
    elif name == "llm":
        return LLMAgent()
    raise ValueError(f"Unknown agent: {name}")


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown("""
<div class="main-header">
    <h1>🧠 SmartInboxRL</h1>
    <p>Teaching AI to Think Like a Real Inbox Assistant</p>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    agent_type = st.selectbox(
        "Agent Type",
        ["rule", "random", "llm"],
        index=0,
        help="Select the agent to evaluate",
    )

    difficulty = st.selectbox(
        "Difficulty",
        ["all", "easy", "medium", "hard"],
        index=0,
        help="Email difficulty tier",
    )

    num_episodes = st.slider(
        "Episodes",
        min_value=1,
        max_value=50,
        value=5,
        help="Number of episodes to run",
    )

    seed = st.number_input("Seed", value=42, step=1)

    run_button = st.button("▶️ Run Evaluation", type="primary", use_container_width=True)

    st.markdown("---")
    st.markdown("### 📖 About")
    st.markdown(
        "SmartInboxRL evaluates AI agents on realistic inbox management tasks. "
        "Agents must understand emails, decide actions, and generate responses."
    )


# ---------------------------------------------------------------------------
# Main area — run episodes
# ---------------------------------------------------------------------------

if run_button:
    agent = _make_agent(agent_type, seed=seed)
    env = InboxEnv(difficulty=difficulty, seed=seed)

    all_rewards = []
    all_breakdowns = []
    all_step_details = []

    progress = st.progress(0, text="Running episodes...")

    for ep in range(num_episodes):
        agent.reset()
        obs, _ = env.reset()
        ep_steps = []

        while True:
            action = agent.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)

            ep_steps.append({
                "email_id": info.get("email_id", ""),
                "action": action["action"],
                "priority": action["priority"],
                "intents": ", ".join(action["intents"]),
                "response_snippet": action["response"][:80] + ("..." if len(action["response"]) > 80 else ""),
                "reward": round(reward, 4),
                **{f"r_{k}": round(v, 4) for k, v in info.get("reward_breakdown", {}).items()},
            })

            if terminated or truncated:
                break

        summary = env.get_episode_summary()
        all_rewards.append(summary["total_reward"])
        all_breakdowns.append(summary["reward_breakdown"])
        all_step_details.extend(ep_steps)

        progress.progress((ep + 1) / num_episodes, text=f"Episode {ep + 1}/{num_episodes}")

    progress.empty()

    # --- Metrics row ---
    rewards_arr = np.array(all_rewards)

    cols = st.columns(4)
    metrics = [
        ("Mean Reward", f"{rewards_arr.mean():+.4f}"),
        ("Std Dev", f"{rewards_arr.std():.4f}"),
        ("Best Episode", f"{rewards_arr.max():+.4f}"),
        ("Worst Episode", f"{rewards_arr.min():+.4f}"),
    ]
    for col, (label, value) in zip(cols, metrics):
        col.markdown(f"""
        <div class="metric-card">
            <div class="value">{value}</div>
            <div class="label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Charts row ---
    chart_col1, chart_col2 = st.columns(2)

    # Radar chart of reward components
    with chart_col1:
        st.markdown("### 🎯 Reward Component Breakdown")
        avg_breakdown = {}
        for key in ("intent", "priority", "action", "response", "penalty"):
            vals = [b.get(key, 0) for b in all_breakdowns]
            avg_breakdown[key] = np.mean(vals)

        categories = list(avg_breakdown.keys())
        values = list(avg_breakdown.values())

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            fillcolor="rgba(102, 126, 234, 0.3)",
            line=dict(color="#667eea", width=2),
            name="Average",
        ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor="rgba(26, 26, 46, 0.8)",
                radialaxis=dict(visible=True, gridcolor="#2d3748"),
                angularaxis=dict(gridcolor="#2d3748"),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            showlegend=False,
            height=400,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # Episode reward trend
    with chart_col2:
        st.markdown("### 📈 Episode Reward Trend")
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=list(range(1, len(all_rewards) + 1)),
            y=all_rewards,
            mode="lines+markers",
            line=dict(color="#667eea", width=2),
            marker=dict(size=8, color="#764ba2"),
            name="Episode Reward",
        ))
        fig_trend.add_hline(
            y=float(rewards_arr.mean()),
            line_dash="dash",
            line_color="#48bb78",
            annotation_text=f"Mean: {rewards_arr.mean():.4f}",
        )
        fig_trend.update_layout(
            xaxis_title="Episode",
            yaxis_title="Total Reward",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(26, 26, 46, 0.8)",
            font=dict(color="white"),
            xaxis=dict(gridcolor="#2d3748"),
            yaxis=dict(gridcolor="#2d3748"),
            height=400,
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    # --- Step-by-step details ---
    st.markdown("### 📋 Step Details")

    df = pd.DataFrame(all_step_details)
    if not df.empty:
        # Color the reward column
        st.dataframe(
            df.style.applymap(
                lambda v: "color: #48bb78" if isinstance(v, (int, float)) and v > 0
                else "color: #fc8181" if isinstance(v, (int, float)) and v < 0
                else "",
                subset=["reward"],
            ),
            use_container_width=True,
            height=400,
        )

    # --- Reward distribution ---
    st.markdown("### 📊 Reward Distribution")
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=all_rewards,
        nbinsx=20,
        marker_color="#667eea",
        opacity=0.8,
    ))
    fig_hist.update_layout(
        xaxis_title="Episode Reward",
        yaxis_title="Count",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(26, 26, 46, 0.8)",
        font=dict(color="white"),
        xaxis=dict(gridcolor="#2d3748"),
        yaxis=dict(gridcolor="#2d3748"),
        height=300,
    )
    st.plotly_chart(fig_hist, use_container_width=True)

else:
    # Landing state
    st.markdown("### 👈 Configure and press **Run Evaluation** to start")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        #### 🤖 Agents
        - **Rule** — Keyword heuristics
        - **Random** — Uniform random (floor)
        - **LLM** — GPT-4o-mini (API)
        """)

    with col2:
        st.markdown("""
        #### 📧 Difficulty Tiers
        - **Easy** — Clear spam, simple meetings
        - **Medium** — Dual intents, scheduling
        - **Hard** — Multi-intent + noise + ambiguity
        """)

    with col3:
        st.markdown("""
        #### 🎯 Reward System
        - Intent understanding (30%)
        - Priority correctness (20%)
        - Action decision (20%)
        - Response quality (30%)
        """)
