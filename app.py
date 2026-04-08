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

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv

load_dotenv()

from environment.inbox_env import InboxEnv
from agents.random_agent import RandomAgent
from agents.rule_agent import RuleAgent
from agents.llm_agent import LLMAgent
from agents.simple_rl_agent import SimpleRLAgent
from agents.rl_agent import RLAgent

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="SmartInboxRL",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — Obsidian Glass aesthetic
# ---------------------------------------------------------------------------

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

  /* ── Base ── */
  html, body, [class*="css"] { font-family: 'Outfit', sans-serif !important; }

  .stApp {
    background: #050810;
    background-image:
      radial-gradient(ellipse 80% 50% at 20% -10%, rgba(99,102,241,0.18) 0%, transparent 60%),
      radial-gradient(ellipse 60% 40% at 80% 100%, rgba(16,185,129,0.10) 0%, transparent 55%),
      radial-gradient(ellipse 40% 60% at 50% 50%, rgba(236,72,153,0.04) 0%, transparent 70%);
    background-attachment: fixed;
  }

  /* ── Sidebar ── */
  section[data-testid="stSidebar"] {
    background: rgba(8, 10, 20, 0.92) !important;
    border-right: 1px solid rgba(99,102,241,0.15) !important;
    backdrop-filter: blur(20px) !important;
  }
  section[data-testid="stSidebar"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #6366f1, #10b981, #ec4899);
  }

  /* ── Sidebar widgets ── */
  .stSelectbox > div > div,
  .stTextInput > div > div > input,
  .stTextArea > div > div > textarea,
  .stNumberInput > div > div > input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-family: 'Outfit', sans-serif !important;
    transition: border-color 0.2s ease !important;
  }
  .stSelectbox > div > div:focus-within,
  .stTextInput > div > div:focus-within,
  .stTextArea > div > div:focus-within {
    border-color: rgba(99,102,241,0.5) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.08) !important;
  }
  .stSelectbox svg { color: #6366f1 !important; }

  /* ── Labels ── */
  .stSelectbox label,
  .stSlider label,
  .stNumberInput label,
  .stTextInput label,
  .stTextArea label {
    color: rgba(148,163,184,0.8) !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    font-family: 'Space Mono', monospace !important;
  }

  /* ── Slider ── */
  .stSlider [data-baseweb="slider"] div[role="slider"] {
    background: #6366f1 !important;
    border: 2px solid #818cf8 !important;
    box-shadow: 0 0 12px rgba(99,102,241,0.5) !important;
  }
  .stSlider [data-baseweb="slider"] div[data-testid="stThumbValue"] {
    color: #818cf8 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 12px !important;
  }

  /* ── Primary Button ── */
  .stButton > button[kind="primary"],
  .stButton > button[data-testid*="primary"],
  div[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    letter-spacing: 0.03em !important;
    padding: 0.6rem 1.2rem !important;
    box-shadow: 0 4px 20px rgba(99,102,241,0.35), inset 0 1px 0 rgba(255,255,255,0.15) !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
  }
  .stButton > button[kind="primary"]:hover,
  div[data-testid="stFormSubmitButton"] button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 28px rgba(99,102,241,0.45), inset 0 1px 0 rgba(255,255,255,0.15) !important;
  }
  .stButton > button[kind="primary"]:active {
    transform: translateY(0) !important;
  }

  /* ── Secondary buttons ── */
  .stButton > button:not([kind="primary"]) {
    background: rgba(255,255,255,0.05) !important;
    color: #94a3b8 !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
    font-family: 'Outfit', sans-serif !important;
    transition: all 0.2s ease !important;
  }
  .stButton > button:not([kind="primary"]):hover {
    background: rgba(255,255,255,0.09) !important;
    border-color: rgba(99,102,241,0.3) !important;
    color: #e2e8f0 !important;
  }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.02) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 2px !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
  }
  .stTabs [data-baseweb="tab"] {
    font-family: 'Outfit', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    color: rgba(148,163,184,0.7) !important;
    border-radius: 9px !important;
    padding: 8px 20px !important;
    transition: all 0.2s ease !important;
  }
  .stTabs [aria-selected="true"] {
    background: rgba(99,102,241,0.15) !important;
    color: #818cf8 !important;
    border: 1px solid rgba(99,102,241,0.25) !important;
  }
  .stTabs [data-baseweb="tab-highlight"] { display: none !important; }

  /* ── Headings ── */
  h1, h2, h3, h4 {
    font-family: 'Outfit', sans-serif !important;
    color: #f1f5f9 !important;
    font-weight: 600 !important;
  }

  /* ── Dataframe ── */
  .stDataFrame {
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 14px !important;
    overflow: hidden !important;
  }
  .stDataFrame thead tr th {
    background: rgba(99,102,241,0.1) !important;
    color: #818cf8 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 11px !important;
  }

  /* ── Progress bar ── */
  .stProgress > div > div > div {
    background: linear-gradient(90deg, #6366f1, #10b981) !important;
    border-radius: 4px !important;
  }

  /* ── Alert / info ── */
  .stAlert {
    background: rgba(99,102,241,0.08) !important;
    border: 1px solid rgba(99,102,241,0.2) !important;
    border-radius: 12px !important;
    color: #c7d2fe !important;
  }
  .stSuccess {
    background: rgba(16,185,129,0.08) !important;
    border: 1px solid rgba(16,185,129,0.2) !important;
    border-radius: 12px !important;
  }
  .stWarning {
    background: rgba(245,158,11,0.08) !important;
    border: 1px solid rgba(245,158,11,0.2) !important;
    border-radius: 12px !important;
  }
  .stError {
    background: rgba(239,68,68,0.08) !important;
    border: 1px solid rgba(239,68,68,0.2) !important;
    border-radius: 12px !important;
  }

  /* ── Metric cards ── */
  .metric-glass {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 20px 22px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.25s ease, transform 0.25s ease;
  }
  .metric-glass::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.12), transparent);
  }
  .metric-glass:hover {
    border-color: rgba(99,102,241,0.3);
    transform: translateY(-2px);
  }
  .metric-glass .m-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: rgba(148,163,184,0.6);
    font-family: 'Space Mono', monospace;
    margin-bottom: 8px;
  }
  .metric-glass .m-value {
    font-size: 30px;
    font-weight: 700;
    font-family: 'Outfit', sans-serif;
    line-height: 1.1;
    margin-bottom: 4px;
  }
  .metric-glass .m-sub {
    font-size: 11px;
    color: rgba(148,163,184,0.45);
    font-family: 'Space Mono', monospace;
  }
  .m-indigo { color: #818cf8; }
  .m-emerald { color: #34d399; }
  .m-sky { color: #38bdf8; }
  .m-rose { color: #fb7185; }

  /* ── Glow dot accent ── */
  .glow-dot {
    display: inline-block;
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #10b981;
    box-shadow: 0 0 8px #10b981, 0 0 16px rgba(16,185,129,0.4);
    margin-right: 6px;
    vertical-align: middle;
    animation: pulse-dot 2s ease-in-out infinite;
  }
  @keyframes pulse-dot {
    0%, 100% { opacity: 1; box-shadow: 0 0 8px #10b981, 0 0 16px rgba(16,185,129,0.4); }
    50% { opacity: 0.6; box-shadow: 0 0 4px #10b981, 0 0 8px rgba(16,185,129,0.2); }
  }

  /* ── Section headings ── */
  .section-heading {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 24px 0 14px;
  }
  .section-heading .sh-line {
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(99,102,241,0.3), transparent);
  }
  .section-heading .sh-text {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: rgba(129,140,248,0.7);
    font-family: 'Space Mono', monospace;
    white-space: nowrap;
  }

  /* ── Email card for interactive panel ── */
  .email-result-card {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 18px 20px;
    margin-bottom: 12px;
  }
  .email-result-card .er-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: rgba(148,163,184,0.5);
    font-family: 'Space Mono', monospace;
    margin-bottom: 10px;
  }

  /* ── Intent tags ── */
  .intent-tag {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.04em;
    font-family: 'Space Mono', monospace;
    background: rgba(99,102,241,0.15);
    color: #818cf8;
    border: 1px solid rgba(99,102,241,0.25);
    margin: 3px 3px 3px 0;
  }

  /* ── Priority badge ── */
  .prio-low    { background:rgba(16,185,129,0.12);  color:#34d399; border:1px solid rgba(16,185,129,0.25);  }
  .prio-medium { background:rgba(245,158,11,0.12);  color:#fbbf24; border:1px solid rgba(245,158,11,0.25);  }
  .prio-high   { background:rgba(249,115,22,0.12);  color:#fb923c; border:1px solid rgba(249,115,22,0.25);  }
  .prio-critical{ background:rgba(239,68,68,0.12);  color:#f87171; border:1px solid rgba(239,68,68,0.25);   }
  .prio-badge {
    display: inline-block;
    padding: 5px 16px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-family: 'Space Mono', monospace;
  }

  /* ── Action chip ── */
  .action-chip {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 10px;
    font-size: 15px;
    font-weight: 700;
    letter-spacing: 0.06em;
    font-family: 'Outfit', sans-serif;
    background: rgba(56,189,248,0.12);
    color: #38bdf8;
    border: 1px solid rgba(56,189,248,0.25);
  }

  /* ── Response box ── */
  .response-box {
    background: rgba(0,0,0,0.25);
    border: 1px solid rgba(255,255,255,0.07);
    border-left: 3px solid rgba(99,102,241,0.5);
    border-radius: 0 12px 12px 0;
    padding: 16px 18px;
    font-size: 14px;
    line-height: 1.7;
    color: #cbd5e1;
    font-family: 'Outfit', sans-serif;
  }

  /* ── Spinner ── */
  .stSpinner > div { border-top-color: #6366f1 !important; }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width: 5px; height: 5px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.3); border-radius: 10px; }
  ::-webkit-scrollbar-thumb:hover { background: rgba(99,102,241,0.5); }

  /* ── Hide default streamlit chrome ── */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding-top: 1.5rem !important; padding-bottom: 2rem !important; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def _make_agent(name: str, seed: int = 42, llm_config: dict | None = None):
    if name == "random":
        return RandomAgent(seed=seed)
    elif name == "rule":
        return RuleAgent()
    elif name == "llm":
        cfg = llm_config or {}
        return LLMAgent(
            api_base=cfg.get("api_base") or None,
            model=cfg.get("model") or None,
            api_key=cfg.get("api_key") or None,
        )
    elif name == "simple_rl":
        return SimpleRLAgent(seed=seed)
    elif name == "rl_llm":
        cfg = llm_config or {}
        return RLAgent(
            api_base=cfg.get("api_base") or None,
            model=cfg.get("model") or None,
            api_key=cfg.get("api_key") or None,
        )
    raise ValueError(f"Unknown agent: {name}")


# ---------------------------------------------------------------------------
# Plotly shared theme
# ---------------------------------------------------------------------------

PLOT_BG    = "rgba(0,0,0,0)"
PAPER_BG   = "rgba(0,0,0,0)"
GRID_COLOR = "rgba(255,255,255,0.05)"
FONT_COLOR = "#94a3b8"
ACCENT     = "#6366f1"

def _base_layout(**extra):
    return dict(
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(color=FONT_COLOR, family="Outfit, sans-serif", size=12),
        margin=dict(l=8, r=8, t=32, b=8),
        **extra,
    )


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown("""
<div style="
  display:flex; align-items:center; gap:14px;
  padding: 4px 0 22px;
  border-bottom: 1px solid rgba(255,255,255,0.06);
  margin-bottom: 26px;
">
  <div style="
    width:36px; height:36px; border-radius:10px;
    background: linear-gradient(135deg,#6366f1,#4f46e5);
    display:flex; align-items:center; justify-content:center;
    font-size:17px; box-shadow:0 4px 14px rgba(99,102,241,0.4);
  ">✦</div>

  <div>
    <div style="
      font-size:20px; font-weight:700; color:#f1f5f9;
      font-family:'Outfit',sans-serif; letter-spacing:-0.02em; line-height:1.1;
    ">SmartInboxRL</div>
    <div style="font-size:11px; color:rgba(148,163,184,0.5); font-family:'Space Mono',monospace; letter-spacing:0.05em;">
      REINFORCEMENT LEARNING · EMAIL TRIAGE
    </div>
  </div>

  <div style="margin-left:auto; display:flex; gap:8px; align-items:center;">
    <span style="
      padding:4px 12px; border-radius:20px; font-size:11px;
      font-family:'Space Mono',monospace; color:#34d399;
      border:1px solid rgba(52,211,153,0.25);
      background:rgba(52,211,153,0.08);
      letter-spacing:0.04em;
    "><span class="glow-dot"></span>OpenEnv ✓</span>
    <span style="
      padding:4px 12px; border-radius:20px; font-size:11px;
      font-family:'Space Mono',monospace; color:rgba(148,163,184,0.6);
      border:1px solid rgba(255,255,255,0.08);
      background:rgba(255,255,255,0.03);
    ">v1.0.0</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("""
    <div style="padding:4px 0 18px; border-bottom:1px solid rgba(255,255,255,0.06); margin-bottom:18px;">
      <div style="font-size:13px; font-weight:600; color:rgba(148,163,184,0.5);
                  font-family:'Space Mono',monospace; letter-spacing:0.1em; text-transform:uppercase;">
        Configuration
      </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background:rgba(245,158,11,0.1); border:1px solid rgba(245,158,11,0.2); border-radius:8px; padding:10px; margin-bottom:20px;">
      <div style="font-size:11px; font-weight:600; color:#fbbf24; font-family:'Space Mono',monospace; display:flex; align-items:center; gap:6px;">
        <span class="glow-dot" style="background:#fbbf24; box-shadow:0 0 8px #fbbf24;"></span> EVALUATION MODE
      </div>
      <div style="font-size:12px; color:rgba(148,163,184,0.8); margin-top:4px; line-height:1.4;">
        Baseline agents do not learn or save state between runs.
      </div>
    </div>
    """, unsafe_allow_html=True)

    agent_type = st.selectbox(
        "Agent Type",
        ["rule", "random", "llm", "simple_rl", "rl_llm"],
        index=0,
        help="Select the agent to evaluate. simple_rl and rl_llm are trained RL agents.",
    )

    llm_config: dict = {
        "api_key": os.getenv("OPENAI_API_KEY"),
        "api_base": os.getenv("API_BASE_URL"),
        "model": os.getenv("MODEL_NAME")
    }

    difficulty = st.selectbox(
        "Difficulty",
        ["all", "easy", "medium", "hard", "enron"],
        index=0,
    )

    num_episodes = st.slider("Test Batches", min_value=1, max_value=50, value=5, help="Number of email sets to run. More batches provide a more stable average score.")
    seed = st.number_input("Seed", value=42, step=1, help="Ensures exactly repeatable 'random' choices. Change to see different behavior.")

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    run_button = st.button("▶  Run Evaluation", type="primary", width="stretch")

    st.markdown("""
    <div style="margin-top:24px; padding-top:18px; border-top:1px solid rgba(255,255,255,0.05);">
      <div style="font-size:10px; font-weight:600; letter-spacing:0.1em; text-transform:uppercase;
                  color:rgba(148,163,184,0.4); font-family:'Space Mono',monospace; margin-bottom:12px;">
        Reward Weights
      </div>
      <div style="display:flex; flex-direction:column; gap:8px;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <span style="font-size:12px; color:rgba(148,163,184,0.6); font-family:'Outfit',sans-serif;">Intent</span>
          <div style="display:flex; align-items:center; gap:6px;">
            <div style="width:60px; height:4px; border-radius:2px; background:rgba(255,255,255,0.06); overflow:hidden;">
              <div style="width:30%; height:100%; background:#6366f1; border-radius:2px;"></div>
            </div>
            <span style="font-size:11px; color:#818cf8; font-family:'Space Mono',monospace;">30%</span>
          </div>
        </div>
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <span style="font-size:12px; color:rgba(148,163,184,0.6); font-family:'Outfit',sans-serif;">Priority</span>
          <div style="display:flex; align-items:center; gap:6px;">
            <div style="width:60px; height:4px; border-radius:2px; background:rgba(255,255,255,0.06); overflow:hidden;">
              <div style="width:20%; height:100%; background:#10b981; border-radius:2px;"></div>
            </div>
            <span style="font-size:11px; color:#34d399; font-family:'Space Mono',monospace;">20%</span>
          </div>
        </div>
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <span style="font-size:12px; color:rgba(148,163,184,0.6); font-family:'Outfit',sans-serif;">Action</span>
          <div style="display:flex; align-items:center; gap:6px;">
            <div style="width:60px; height:4px; border-radius:2px; background:rgba(255,255,255,0.06); overflow:hidden;">
              <div style="width:20%; height:100%; background:#38bdf8; border-radius:2px;"></div>
            </div>
            <span style="font-size:11px; color:#38bdf8; font-family:'Space Mono',monospace;">20%</span>
          </div>
        </div>
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <span style="font-size:12px; color:rgba(148,163,184,0.6); font-family:'Outfit',sans-serif;">Response</span>
          <div style="display:flex; align-items:center; gap:6px;">
            <div style="width:60px; height:4px; border-radius:2px; background:rgba(255,255,255,0.06); overflow:hidden;">
              <div style="width:30%; height:100%; background:#ec4899; border-radius:2px;"></div>
            </div>
            <span style="font-size:11px; color:#f472b6; font-family:'Space Mono',monospace;">30%</span>
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_eval, tab_interactive, tab_training = st.tabs(["  Evaluation Dashboard  ", "  Interactive Panel  ", "  🧠 Training Mode  "])


# ── Evaluation tab ──────────────────────────────────────────────────────────
with tab_eval:
    if run_button:
        agent = _make_agent(agent_type, seed=seed, llm_config=llm_config)
        env   = InboxEnv(difficulty=difficulty, seed=seed)

        all_step_rewards  = []
        all_total_rewards = []
        all_breakdowns    = []
        all_step_details  = []

        progress = st.progress(0, text="Initialising environment…")

        for ep in range(num_episodes):
            agent.reset()
            obs, _ = env.reset()
            ep_steps = []

            while True:
                obs_for_agent = obs.to_dict() if hasattr(obs, "to_dict") else obs
                action = agent.act(obs_for_agent)
                obs, reward, terminated, truncated, info = env.step(action)

                ep_steps.append({
                    "email_id":        info.get("email_id", ""),
                    "action":          action["action"],
                    "priority":        action["priority"],
                    "intents":         ", ".join(action["intents"]),
                    "response":        action["response"][:80] + ("…" if len(action["response"]) > 80 else ""),
                    "reward":          round(reward, 4),
                    **{f"r_{k}": round(v, 4) for k, v in info.get("reward_breakdown", {}).items()},
                })

                if terminated or truncated:
                    break

            summary = env.get_episode_summary()
            all_step_rewards.append(summary["mean_step_reward"])
            all_total_rewards.append(summary["total_reward"])
            all_breakdowns.append(summary.get("norm_reward_breakdown", summary["reward_breakdown"]))
            all_step_details.extend(ep_steps)

            progress.progress(
                (ep + 1) / num_episodes,
                text=f"Batch {ep + 1} / {num_episodes}  —  reward {summary['mean_step_reward']:.3f}",
            )

        progress.empty()

        rewards_arr = np.array(all_step_rewards)

        # ── Metric cards ────────────────────────────────────────────────────
        st.markdown("""<div class="section-heading"><div class="sh-line"></div>
          <div class="sh-text">Evaluation Metrics</div>
          <div class="sh-line" style="background:linear-gradient(90deg,transparent,rgba(99,102,241,0.3));"></div>
        </div>""", unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        for col, label, value, cls, sub in [
            (m1, "Mean Reward / Step", f"{rewards_arr.mean():.4f}", "m-indigo", "avg across all batches"),
            (m2, "Std Deviation",      f"{rewards_arr.std():.4f}",  "m-sky",    "consistency"),
            (m3, "Best Batch",       f"{rewards_arr.max():.4f}",  "m-emerald","highest single-batch score"),
            (m4, "Worst Batch",      f"{rewards_arr.min():.4f}",  "m-rose",   "lowest single-batch score"),
        ]:
            col.markdown(f"""
            <div class="metric-glass">
              <div class="m-label">{label}</div>
              <div class="m-value {cls}">{value}</div>
              <div class="m-sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

        # ── Charts ──────────────────────────────────────────────────────────
        st.markdown("""<div class="section-heading"><div class="sh-line"></div>
          <div class="sh-text">Visualisations</div>
          <div class="sh-line" style="background:linear-gradient(90deg,transparent,rgba(99,102,241,0.3));"></div>
        </div>""", unsafe_allow_html=True)

        c1, c2 = st.columns(2)

        with c1:
            avg_bd = {}
            for key in ("intent", "priority", "action", "response", "penalty"):
                vals = [b.get(key, 0) for b in all_breakdowns]
                avg_bd[key] = float(np.mean(vals))

            cats = list(avg_bd.keys())
            vals = list(avg_bd.values())
            colors_radar = ["#6366f1","#10b981","#38bdf8","#ec4899","#f59e0b"]

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=cats + [cats[0]],
                fill="toself",
                fillcolor="rgba(99,102,241,0.12)",
                line=dict(color="#6366f1", width=2),
                name="avg",
                marker=dict(size=5, color=colors_radar + [colors_radar[0]]),
            ))
            fig_radar.update_layout(
                **_base_layout(height=360),
                polar=dict(
                    bgcolor="rgba(255,255,255,0.02)",
                    radialaxis=dict(
                        visible=True, range=[0, 1],
                        gridcolor="rgba(255,255,255,0.05)",
                        tickfont=dict(size=9, color="rgba(148,163,184,0.5)"),
                        linecolor="rgba(255,255,255,0.05)",
                    ),
                    angularaxis=dict(
                        gridcolor="rgba(255,255,255,0.05)",
                        linecolor="rgba(255,255,255,0.05)",
                        tickfont=dict(size=11, color="#94a3b8"),
                    ),
                ),
                showlegend=False,
                title=dict(text="Reward Components", font=dict(size=13, color="#94a3b8"), x=0.5),
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        with c2:
            fig_trend = go.Figure()
            x_ep = list(range(1, len(all_step_rewards) + 1))
            mean_v = float(rewards_arr.mean())

            # Filled area
            fig_trend.add_trace(go.Scatter(
                x=x_ep, y=all_step_rewards,
                mode="none", fill="tozeroy",
                fillcolor="rgba(99,102,241,0.07)",
                showlegend=False, hoverinfo="skip",
            ))
            fig_trend.add_trace(go.Scatter(
                x=x_ep, y=all_step_rewards,
                mode="lines+markers",
                line=dict(color="#6366f1", width=2.5, shape="spline", smoothing=0.8),
                marker=dict(size=7, color="#818cf8",
                            line=dict(width=1.5, color="#050810")),
                name="reward",
                hovertemplate="Ep %{x}<br>Reward: %{y:.4f}<extra></extra>",
            ))
            fig_trend.add_hline(
                y=mean_v, line_dash="dot", line_color="rgba(52,211,153,0.5)",
                annotation_text=f"μ = {mean_v:.4f}",
                annotation_font=dict(size=10, color="#34d399"),
            )
            fig_trend.update_layout(
                **_base_layout(height=360),
                xaxis=dict(
                    title="Batch", gridcolor=GRID_COLOR,
                    linecolor="rgba(255,255,255,0.06)", tickfont=dict(size=10),
                ),
                yaxis=dict(
                    title="Mean Reward [0–1]", gridcolor=GRID_COLOR,
                    linecolor="rgba(255,255,255,0.06)", tickfont=dict(size=10),
                    range=[-0.05, 1.05],
                ),
                title=dict(text="Reward Trend", font=dict(size=13, color="#94a3b8"), x=0.5),
                showlegend=False,
            )
            st.plotly_chart(fig_trend, use_container_width=True)

        # ── Step details ────────────────────────────────────────────────────
        st.markdown("""<div class="section-heading"><div class="sh-line"></div>
          <div class="sh-text">Step Details</div>
          <div class="sh-line" style="background:linear-gradient(90deg,transparent,rgba(99,102,241,0.3));"></div>
        </div>""", unsafe_allow_html=True)

        df = pd.DataFrame(all_step_details)
        if not df.empty:
            def _color_reward(val):
                if isinstance(val, (int, float)):
                    if val > 0.6:   return "color:#34d399"
                    elif val > 0.4: return "color:#fbbf24"
                    else:           return "color:#f87171"
                return ""

            st.dataframe(
                df.style.map(_color_reward, subset=["reward"]),
                width="stretch",
                height=360,
            )

        # ── Distribution ────────────────────────────────────────────────────
        st.markdown("""<div class="section-heading"><div class="sh-line"></div>
          <div class="sh-text">Reward Distribution</div>
          <div class="sh-line" style="background:linear-gradient(90deg,transparent,rgba(99,102,241,0.3));"></div>
        </div>""", unsafe_allow_html=True)

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=all_step_rewards,
            nbinsx=min(20, max(5, num_episodes // 2)),
            marker=dict(
                color="rgba(99,102,241,0.6)",
                line=dict(color="rgba(129,140,248,0.8)", width=1),
            ),
            hovertemplate="Range: %{x}<br>Count: %{y}<extra></extra>",
        ))
        fig_hist.update_layout(
            **_base_layout(height=260),
            xaxis=dict(title="Mean Reward per Step", gridcolor=GRID_COLOR,
                       linecolor="rgba(255,255,255,0.06)", tickfont=dict(size=10), range=[-0.05, 1.05]),
            yaxis=dict(title="Batches", gridcolor=GRID_COLOR,
                       linecolor="rgba(255,255,255,0.06)", tickfont=dict(size=10)),
            bargap=0.08,
            title=dict(text="Distribution across test batches", font=dict(size=13, color="#94a3b8"), x=0.5),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    else:
        # ── Empty state ─────────────────────────────────────────────────────
        st.markdown("""
        <div style="
          display:flex; flex-direction:column; align-items:center; justify-content:center;
          padding:60px 20px; text-align:center; gap:16px;
        ">
          <div style="
            width:64px; height:64px; border-radius:18px;
            background:linear-gradient(135deg,rgba(99,102,241,0.15),rgba(79,70,229,0.08));
            border:1px solid rgba(99,102,241,0.2);
            display:flex; align-items:center; justify-content:center; font-size:28px;
          ">✦</div>
          <div style="font-size:22px; font-weight:700; color:#f1f5f9; font-family:'Outfit',sans-serif;">
            Configure and run an evaluation
          </div>
          <div style="font-size:14px; color:rgba(148,163,184,0.55); max-width:400px; line-height:1.7;">
            Choose your agent, difficulty tier, and batch count from the sidebar,
            then press <strong style="color:#818cf8;">Run Evaluation</strong> to begin.
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        g1, g2, g3 = st.columns(3)

        for col, icon, title, lines in [
            (g1, "🤖", "Agents", [
                ("Rule",   "Keyword heuristics", "#818cf8"),
                ("Random", "Uniform baseline",   "#94a3b8"),
                ("LLM",    "GPT-4o-mini / Groq", "#34d399"),
            ]),
            (g2, "📧", "Difficulty Tiers", [
                ("Easy",   "Clear spam, simple meetings",         "#34d399"),
                ("Medium", "Dual intents, scheduling",            "#fbbf24"),
                ("Hard",   "Multi-intent + noise + ambiguity",    "#fb923c"),
                ("Enron",  "Real-world corpus",                   "#f87171"),
            ]),
            (g3, "🎯", "Reward System", [
                ("Intent",   "Understanding 30%",   "#818cf8"),
                ("Priority", "Correctness 20%",     "#34d399"),
                ("Action",   "Decision 20%",        "#38bdf8"),
                ("Response", "Quality 30%",         "#f472b6"),
            ]),
        ]:
            rows_html = "".join(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
                        padding:7px 0;border-bottom:1px solid rgba(255,255,255,0.04);">
              <span style="font-size:13px;color:rgba(148,163,184,0.7);font-family:'Outfit',sans-serif;">{n}</span>
              <span style="font-size:11px;color:{c};font-family:'Space Mono',monospace;">{d}</span>
            </div>
            """ for n, d, c in lines)

            col.markdown(f"""
            <div style="background:rgba(255,255,255,0.025);border:1px solid rgba(255,255,255,0.06);
                        border-radius:14px;padding:18px 20px;">
              <div style="font-size:20px;margin-bottom:8px;">{icon}</div>
              <div style="font-size:13px;font-weight:600;color:#f1f5f9;font-family:'Outfit',sans-serif;
                          margin-bottom:12px;">{title}</div>
              {rows_html}
            </div>
            """, unsafe_allow_html=True)


# ── Interactive tab ──────────────────────────────────────────────────────────
with tab_interactive:
    st.markdown("""
    <div style="margin-bottom:20px;">
      <div style="font-size:20px;font-weight:700;color:#f1f5f9;font-family:'Outfit',sans-serif;
                  margin-bottom:6px;">Test Agent Interactively</div>
      <div style="font-size:13px;color:rgba(148,163,184,0.5);line-height:1.6;">
        Compose any email and see how the selected agent analyses and responds in real-time.
      </div>
    </div>
    """, unsafe_allow_html=True)

    with st.form("interactive_form"):
        fc1, fc2 = st.columns(2)
        sender_input  = fc1.text_input("Sender",  value="boss@company.com")
        subject_input = fc2.text_input("Subject", value="Urgent: Submit Timesheets")

        body_input = st.text_area(
            "Email Body",
            value="Please submit your timesheets by 5 PM today, otherwise payroll will be delayed.",
            height=140,
        )
        submitted = st.form_submit_button("  Submit to Agent  ", type="primary", width="stretch")

    if submitted:
        with st.spinner("Agent analysing…"):
            try:
                interactive_agent = _make_agent(agent_type, seed=seed, llm_config=llm_config)

                obs = {
                    "email": {
                        "id": "interactive_001",
                        "sender": sender_input,
                        "subject": subject_input,
                        "body": body_input,
                    },
                    "history": [],
                    "step": 0,
                    "total_steps": 10,
                    "difficulty": "interactive",
                }

                action = interactive_agent.act(obs)

                st.markdown("""
                <div style="display:flex;align-items:center;gap:8px;margin:16px 0 18px;">
                  <span class="glow-dot"></span>
                  <span style="font-size:13px;font-weight:600;color:#34d399;font-family:'Outfit',sans-serif;">
                    Analysis Complete
                  </span>
                </div>
                """, unsafe_allow_html=True)

                rc1, rc2, rc3 = st.columns(3)

                # Intents
                intents = action.get("intents", [])
                tags_html = "".join(f'<span class="intent-tag">{i}</span>' for i in intents) or \
                            '<span style="color:rgba(148,163,184,0.4);font-size:13px;">none detected</span>'
                rc1.markdown(f"""
                <div class="email-result-card">
                  <div class="er-label">Predicted Intents</div>
                  <div>{tags_html}</div>
                </div>
                """, unsafe_allow_html=True)

                # Priority
                priority = action.get("priority", "N/A")
                rc2.markdown(f"""
                <div class="email-result-card">
                  <div class="er-label">Priority Level</div>
                  <span class="prio-badge prio-{priority}">{priority.upper()}</span>
                </div>
                """, unsafe_allow_html=True)

                # Action
                act_val = action.get("action", "N/A")
                action_icons = {"reply":"↩","ignore":"✕","escalate":"⬆","forward":"→"}
                act_icon = action_icons.get(act_val, "·")
                rc3.markdown(f"""
                <div class="email-result-card">
                  <div class="er-label">Decision</div>
                  <span class="action-chip">{act_icon} {act_val.upper()}</span>
                </div>
                """, unsafe_allow_html=True)

                # Response
                st.markdown("""<div class="section-heading"><div class="sh-line"></div>
                  <div class="sh-text">Generated Response</div>
                  <div class="sh-line" style="background:linear-gradient(90deg,transparent,rgba(99,102,241,0.3));"></div>
                </div>""", unsafe_allow_html=True)

                response_text = action.get("response", "No response generated.")
                st.markdown(f'<div class="response-box">{response_text}</div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error running agent: {e}")


# ── Training Mode tab ────────────────────────────────────────────────────────
with tab_training:
    st.markdown("""
    <div style="margin-bottom:20px;">
      <div style="font-size:20px;font-weight:700;color:#f1f5f9;font-family:'Outfit',sans-serif;
                  margin-bottom:6px;">Training Mode</div>
      <div style="font-size:13px;color:rgba(148,163,184,0.5);line-height:1.6;">
        Train RL agents, visualise learning curves, and compare agent performance.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── SECTION A: Learning Progress Chart ──────────────────────────────
    st.markdown("""<div class="section-heading"><div class="sh-line"></div>
      <div class="sh-text">Learning Progress</div>
      <div class="sh-line" style="background:linear-gradient(90deg,transparent,rgba(99,102,241,0.3));"></div>
    </div>""", unsafe_allow_html=True)

    _training_log_path = Path("data/training_log.json")
    if _training_log_path.exists():
        try:
            with open(_training_log_path, "r") as _f:
                _train_log = json.load(_f)
            if _train_log and isinstance(_train_log, list):
                _episodes_list = [e.get("episode", i) for i, e in enumerate(_train_log)]
                _rewards_list = [e.get("avg_reward", 0) for e in _train_log]
                _epsilon_list = [e.get("epsilon") for e in _train_log]

                from plotly.subplots import make_subplots as _make_subplots

                _has_epsilon = any(e is not None for e in _epsilon_list)

                if _has_epsilon:
                    _fig_lc = _make_subplots(specs=[[{"secondary_y": True}]])
                else:
                    _fig_lc = go.Figure()

                # Reward line
                _fig_lc.add_trace(
                    go.Scatter(
                        x=_episodes_list, y=_rewards_list,
                        mode="lines+markers",
                        name="avg_reward",
                        line=dict(color="#6366f1", width=2.5),
                        marker=dict(size=4, color="#818cf8"),
                    ),
                    secondary_y=False if _has_epsilon else None,
                )

                # Epsilon line (if present)
                if _has_epsilon:
                    _eps_clean = [e if e is not None else 0 for e in _epsilon_list]
                    _fig_lc.add_trace(
                        go.Scatter(
                            x=_episodes_list, y=_eps_clean,
                            mode="lines",
                            name="epsilon",
                            line=dict(color="#f59e0b", width=2, dash="dash"),
                        ),
                        secondary_y=True,
                    )
                    _fig_lc.update_yaxes(
                        title_text="Epsilon", secondary_y=True,
                        gridcolor="rgba(255,255,255,0.03)",
                        range=[0, 1.05],
                    )

                # Baselines
                _fig_lc.add_hline(y=0.48, line_dash="dot", line_color="rgba(148,163,184,0.4)",
                                  annotation_text="random baseline (0.48)",
                                  annotation_font=dict(size=10, color="#94a3b8"))
                _fig_lc.add_hline(y=0.65, line_dash="dot", line_color="rgba(52,211,153,0.4)",
                                  annotation_text="rule agent baseline (0.65)",
                                  annotation_font=dict(size=10, color="#34d399"))

                _fig_lc.update_layout(
                    **_base_layout(height=400),
                    title=dict(text="Agent Learning Curve", font=dict(size=14, color="#94a3b8"), x=0.5),
                    xaxis=dict(title="Episode", gridcolor=GRID_COLOR),
                    yaxis=dict(title="Avg Reward", gridcolor=GRID_COLOR, range=[-0.05, 1.05]),
                    legend=dict(font=dict(color="#94a3b8")),
                )
                if not _has_epsilon:
                    _fig_lc.update_yaxes(gridcolor=GRID_COLOR)

                st.plotly_chart(_fig_lc, use_container_width=True)

                # Metric cards
                _n_metric = min(10, len(_rewards_list) // 2)
                if _n_metric > 0:
                    _first_n = _rewards_list[:_n_metric]
                    _last_n = _rewards_list[-_n_metric:]
                    _first_avg = float(np.mean(_first_n))
                    _last_avg = float(np.mean(_last_n))
                    _improvement = _last_avg - _first_avg

                    _mc1, _mc2, _mc3 = st.columns(3)
                    _mc1.markdown(f"""
                    <div class="metric-glass">
                      <div class="m-label">First {_n_metric} Avg</div>
                      <div class="m-value m-sky">{_first_avg:.4f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    _mc2.markdown(f"""
                    <div class="metric-glass">
                      <div class="m-label">Last {_n_metric} Avg</div>
                      <div class="m-value m-emerald">{_last_avg:.4f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    _delta_color = "m-emerald" if _improvement > 0 else "m-rose"
                    _delta_sign = "+" if _improvement >= 0 else ""
                    _mc3.markdown(f"""
                    <div class="metric-glass">
                      <div class="m-label">Improvement</div>
                      <div class="m-value {_delta_color}">{_delta_sign}{_improvement:.4f}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("Training log is empty.")
        except Exception as _e:
            st.error(f"Error reading training log: {_e}")
    else:
        st.info("No training data yet. Run: `python training/train.py --agent simple_rl --episodes 100`")

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ── SECTION B: Q-Table Heatmap ──────────────────────────────────────
    st.markdown("""<div class="section-heading"><div class="sh-line"></div>
      <div class="sh-text">Q-Table Heatmap (simple_rl)</div>
      <div class="sh-line" style="background:linear-gradient(90deg,transparent,rgba(99,102,241,0.3));"></div>
    </div>""", unsafe_allow_html=True)

    _qt_path = Path("data/q_table.npy")
    _qs_path = Path("data/q_table_states.json")
    if _qt_path.exists() and _qs_path.exists():
        try:
            _q_table = np.load(str(_qt_path))
            with open(_qs_path, "r") as _f:
                _q_states = json.load(_f)

            _action_names = [
                "reply/low", "reply/med", "reply/high", "reply/crit",
                "ignore/low", "ignore/med",
                "esc/high", "esc/crit",
                "fwd/low", "fwd/med", "fwd/high",
            ]

            _state_keys = sorted(_q_states.keys(), key=lambda k: _q_states[k])
            _state_indices = [_q_states[k] for k in _state_keys]

            # Only show rows that exist in the Q-table
            _valid_indices = [i for i in _state_indices if i < _q_table.shape[0]]
            _valid_keys = [k for k, i in zip(_state_keys, _state_indices) if i < _q_table.shape[0]]

            if _valid_indices:
                _q_subset = _q_table[_valid_indices, :len(_action_names)]

                _fig_hm = go.Figure(data=go.Heatmap(
                    z=_q_subset,
                    x=_action_names[:_q_subset.shape[1]],
                    y=_valid_keys,
                    colorscale="Blues",
                    hovertemplate="State: %{y}<br>Action: %{x}<br>Q-value: %{z:.4f}<extra></extra>",
                ))
                _fig_hm.update_layout(
                    **_base_layout(height=max(300, len(_valid_keys) * 25 + 100)),
                    title=dict(text="Q-Table: Learned Action Values", font=dict(size=14, color="#94a3b8"), x=0.5),
                    xaxis=dict(tickfont=dict(size=10), tickangle=45),
                    yaxis=dict(tickfont=dict(size=9)),
                )
                st.plotly_chart(_fig_hm, use_container_width=True)
            else:
                st.info("Q-table loaded but no valid state entries found.")
        except Exception as _e:
            st.error(f"Error loading Q-table: {_e}")
    else:
        st.info("Train simple_rl agent first: `python training/train.py --agent simple_rl --episodes 100`")

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ── SECTION C: Learned Patterns Table (rl_llm) ──────────────────────
    st.markdown("""<div class="section-heading"><div class="sh-line"></div>
      <div class="sh-text">Learned Patterns (rl_llm)</div>
      <div class="sh-line" style="background:linear-gradient(90deg,transparent,rgba(99,102,241,0.3));"></div>
    </div>""", unsafe_allow_html=True)

    _db_path = Path("data/agent_memory.db")
    if _db_path.exists():
        try:
            _conn = sqlite3.connect(str(_db_path))
            _cursor = _conn.cursor()

            _cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='strategy_memory'"
            )
            if _cursor.fetchone():
                _cursor.execute("""
                    SELECT pattern, best_action, best_priority, avg_reward, times_seen
                    FROM strategy_memory ORDER BY avg_reward DESC
                """)
                _rows = _cursor.fetchall()
                if _rows:
                    _df_patterns = pd.DataFrame(
                        _rows,
                        columns=["Pattern", "Best Action", "Best Priority", "Avg Reward", "Times Seen"],
                    )

                    def _color_avg_reward(val):
                        if isinstance(val, (int, float)):
                            if val > 0.7:   return "color:#34d399"
                            elif val > 0.5: return "color:#fbbf24"
                            else:           return "color:#f87171"
                        return ""

                    st.dataframe(
                        _df_patterns.style.map(_color_avg_reward, subset=["Avg Reward"]),
                        width="stretch",
                    )
                else:
                    st.info("No patterns learned yet.")
            else:
                st.info("strategy_memory table not found. Train rl_llm agent first.")

            # Experience count
            _cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='experiences'"
            )
            if _cursor.fetchone():
                _cursor.execute("SELECT COUNT(*) FROM experiences")
                _exp_count = _cursor.fetchone()[0]
                st.markdown(f"""
                <div class="metric-glass" style="max-width:300px;">
                  <div class="m-label">Total Experiences</div>
                  <div class="m-value m-indigo">{_exp_count}</div>
                </div>
                """, unsafe_allow_html=True)

            _conn.close()
        except Exception as _e:
            st.error(f"Error reading agent memory: {_e}")
    else:
        st.info("Train rl_llm agent first: `python training/train.py --agent rl_llm --episodes 50`")

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ── SECTION D: Train Controls ───────────────────────────────────────
    st.markdown("""<div class="section-heading"><div class="sh-line"></div>
      <div class="sh-text">Train Controls</div>
      <div class="sh-line" style="background:linear-gradient(90deg,transparent,rgba(99,102,241,0.3));"></div>
    </div>""", unsafe_allow_html=True)

    _tc1, _tc2, _tc3 = st.columns(3)
    _train_agent_type = _tc1.selectbox(
        "Agent Type", ["simple_rl", "rl_llm"], key="train_agent_type"
    )
    _train_difficulty = _tc2.selectbox(
        "Difficulty", ["all", "easy", "medium", "hard"], key="train_difficulty"
    )
    _train_episodes = _tc3.slider(
        "Episodes", min_value=10, max_value=200, value=50, step=10, key="train_episodes"
    )

    if st.button("▶ Run Training", type="primary"):
        with st.spinner("Training in progress..."):
            try:
                _cmd = [
                    sys.executable, "training/train.py",
                    "--agent", _train_agent_type,
                    "--episodes", str(_train_episodes),
                    "--difficulty", _train_difficulty,
                ]
                _result = subprocess.run(
                    _cmd, capture_output=True, text=True, timeout=300,
                    cwd=str(Path(__file__).resolve().parent),
                )
                st.code(_result.stdout + (_result.stderr or ""), language="text")
                st.rerun()
            except subprocess.TimeoutExpired:
                st.error("Training timed out after 5 minutes.")
            except Exception as _e:
                st.error(f"Training failed: {_e}")