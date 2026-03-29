<div align="center">

# 🧠 SmartInboxRL

### *Teaching AI to Think Like a Real Inbox Assistant*

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29+-FF6F00?style=for-the-badge&logo=openai&logoColor=white)](https://gymnasium.farama.org/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![HuggingFace](https://img.shields.io/badge/🤗_Deploy-Spaces-FFD21E?style=for-the-badge)](https://huggingface.co/spaces)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)

**A realistic RL evaluation benchmark for AI agents navigating real-world email workflows.**

[Overview](#-overview) · [Architecture](#-architecture) · [Quick Start](#-quick-start) · [Reward System](#-reward-system) · [Agents](#-baseline-agents) · [Dashboard](#-interactive-dashboard) · [Deploy](#️-deployment)

</div>

---

## 🎯 Overview

Most AI benchmarks ask:
> *"Can the model answer correctly?"*

**SmartInboxRL asks:**
> *"Can the model behave intelligently in the real world?"*

SmartInboxRL is a **Gymnasium-compatible reinforcement learning environment** that evaluates AI agents on realistic inbox management tasks. Agents must:

1. **Understand** messy, multi-intent emails with noise and ambiguity
2. **Decide** on the correct action (reply / ignore / escalate / forward)
3. **Prioritize** correctly (low / medium / high / critical)
4. **Communicate** with a useful, context-aware response

This mirrors the complete workflow a professional email assistant must execute — not a toy task.

---

## 🧩 What Makes This Different

| Feature | SmartInboxRL | Typical Benchmarks |
|---|---|---|
| Task type | Multi-step decision pipeline | Static Q&A |
| Email realism | Noise injection + Enron-style | Clean synthetic |
| Scoring | Embedding-based semantic | String matching |
| Reward signal | Dense, multi-component | Binary correct/wrong |
| Anti-cheating | Penalties for hacking | None |
| Memory | Interaction history context | Single-turn |
| Score separation | Deliberate variance by difficulty | Ceiling/floor issues |

---

## 🏗️ Architecture

```
SmartInboxRL/
│
├── environment/
│   ├── inbox_env.py          # Core Gymnasium environment (InboxEnv)
│   ├── email_loader.py       # Loads & preprocesses email tasks
│   ├── state.py              # Episode state tracker (history, actions)
│   └── action_space.py       # Discrete action definitions
│
├── rewards/
│   ├── reward_engine.py      # Composite reward (intent + priority + action + response)
│   ├── penalty_system.py     # Anti-cheating penalties
│   └── embedding_scorer.py   # Semantic response scoring via sentence-transformers
│
├── agents/
│   ├── llm_agent.py          # LLM baseline (OpenAI-compatible API)
│   ├── rule_agent.py         # Keyword heuristic baseline
│   └── random_agent.py       # Uniform random baseline
│
├── data/
│   ├── tasks/
│   │   ├── easy_tasks.json   # Clear, unambiguous emails
│   │   ├── medium_tasks.json # Moderate ambiguity
│   │   └── hard_tasks.json   # Multi-intent + noise + conflicting signals
│   └── noise_profiles.json   # Typo/shorthand injection templates
│
├── app.py                    # Streamlit interactive dashboard
├── inference.py              # CLI for running evaluation episodes
├── evaluate.py               # Batch evaluation + metrics export
├── Dockerfile                # HuggingFace Spaces-compatible container
├── docker-compose.yml        # Local development orchestration
└── requirements.txt
```

---

## ⚡ Quick Start

### Local Setup

```bash
# Clone the repository
git clone https://github.com/your-username/SmartInboxRL.git
cd SmartInboxRL

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment variables
cp .env.example .env
# Edit .env — set API_BASE_URL, MODEL_NAME, HF_TOKEN
```
### Run Inference

```bash
# LLM baseline agent — 20 episodes across all difficulties
python inference.py --agent llm --episodes 20 --difficulty all

# Rule-based agent
python inference.py --agent rule --episodes 50 --difficulty medium

# Random baseline (floor performance)
python inference.py --agent random --episodes 100
```

### Launch Dashboard

```bash
streamlit run app.py
```

### Docker

```bash
docker build -t smart-inbox .
docker run -p 7860:7860 \
  -e API_BASE_URL=https://api.openai.com/v1 \
  -e MODEL_NAME=gpt-4o-mini \
  -e HF_TOKEN=your_token \
  smart-inbox
```

---

## 🎯 Reward System

SmartInboxRL uses a **dense, multi-component reward** designed to evaluate *how well* an agent performs, not just *if* it guesses correctly.

### Reward Components

| Component | Weight | Evaluation Method |
|---|---|---|
| 🧠 Intent Understanding | **30%** | Label-match F1 score |
| 🎯 Priority Correctness | **20%** | Exact match |
| ✅ Action Decision | **20%** | Exact match vs. gold label |
| 💬 Response Quality | **30%** | Embedding cosine similarity |

**Total reward ∈ [−1.0, +1.0]** per step.

### Penalties (Anti-Cheating)

| Behavior | Penalty |
|---|---|
| Repeated identical actions | −0.20 per occurrence |
| Ignoring a critical email | −0.50 |
| Trivial / low-effort response | −0.30 |

This ensures agents are **rewarded for thinking better, not guessing right**.

### Semantic Scoring

Response quality is measured using **embedding similarity** (`sentence-transformers/all-MiniLM-L6-v2`), not string matching. This allows:

- Natural language variation in responses
- Flexible phrasing while maintaining semantic correctness
- Robust evaluation of generative outputs

---

## 📧 Task Dataset

### Difficulty Tiers

| Tier | Description | Example Scenario |
|---|---|---|
| 🟢 Easy | Clear, unambiguous signal | Spam detection, meeting confirmation |
| 🟡 Medium | Single ambiguity + moderate noise | Scheduling + follow-up in one email |
| 🔴 Hard | Multi-intent + noise + conflicting signals | Review request + lunch invite + hidden deadline |

### Noise Injection

Emails are perturbed to simulate real-world messiness:

- **Typos** — `recieve`, `teh`, `wiht`
- **Shorthand** — `pls`, `asap`, `lmk`, `fyi`
- **Casing** — `URGENT`, `all caps subject lines`
- **Ambiguity** — `can you check this?`, `get back to me`

### Score Variance by Design

SmartInboxRL deliberately calibrates difficulty to **separate agent performance**:

```
Easy   →  Mean reward > 0.65   (most agents succeed)
Medium →  Mean reward ~ 0.45   (moderate spread)
Hard   →  Mean reward < 0.35   (only strong agents perform well)
Random →  Mean reward < 0.20   (floor)
```

This creates meaningful **performance differentiation** — critical for a real evaluation benchmark.

---

## 🤖 Baseline Agents

### 1. LLM Agent (`agents/llm_agent.py`)

Uses any OpenAI-compatible API. Generates structured outputs via a chain-of-thought prompt:

```
Given this email: {email}
Recent history: {history}

Output:
1. Intents: [list]
2. Priority: low/medium/high/critical
3. Action: reply/ignore/escalate/forward
4. Response: [your reply]
```

### 2. Rule Agent (`agents/rule_agent.py`)

Keyword-heuristic baseline. Useful for establishing a non-ML performance floor:
- Scans for urgency keywords → priority bump
- Detects question marks → reply action
- Checks spam indicators → ignore action

### 3. Random Agent (`agents/random_agent.py`)

Uniform-random action selection. Provides the absolute floor for performance comparison.

---

## 📊 Evaluation Philosophy

SmartInboxRL evaluates agents across **three dimensions**:

```
┌─────────────────────────────────────────────┐
│                                             │
│   Understanding  →  Decision  →  Communication  │
│                                             │
│   "What does      "What should   "What
│    this mean?"      I do?"        should I say?" │
│                                             │
└─────────────────────────────────────────────┘
```

Unlike binary benchmarks, each dimension is scored independently and combined into a single composite reward. This allows **fine-grained capability analysis** — you can identify whether an agent fails at understanding, at decision-making, or at communication.

---

## 🖥️ Interactive Dashboard

`app.py` provides a Streamlit dashboard for:

- **Live episode runner** — Select agent type, difficulty, and watch the agent work step-by-step
- **Reward visualization** — Per-component radar charts and step-by-step reward bars
- **Score history** — Episode-over-episode performance trends
- **Agent comparison** — Side-by-side reward breakdowns across agent types

```bash
streamlit run app.py
```

---

## 🔬 Environment API

```python
import gymnasium as gym
from environment.inbox_env import InboxEnv

env = InboxEnv(difficulty="hard")
obs, info = env.reset()

done = False
while not done:
    action = agent.act(obs)          # agent produces structured action dict
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

print(f"Episode reward: {info['episode_reward']:.3f}")
print(f"Breakdown: {info['reward_breakdown']}")
```

### Observation Space

```python
{
    "email": str,              # current email text (possibly noisy)
    "history": List[dict],     # previous (action, response) pairs
    "step": int,               # current step in episode
    "difficulty": str          # easy / medium / hard
}
```

### Action Space

```python
{
    "intents": List[str],      # detected intent labels
    "priority": str,           # low / medium / high / critical
    "action": str,             # reply / ignore / escalate / forward
    "response": str            # generated response text
}
```

---

## 🗺️ Roadmap

- [x] Project specification and architecture
- [ ] Phase 1 — Gymnasium environment core
- [ ] Phase 2 — Composite reward engine + semantic scorer
- [ ] Phase 3 — Curated task dataset (easy / medium / hard)
- [ ] Phase 4 — Baseline agents (LLM / rule / random)
- [ ] Phase 5 — Inference & batch evaluation CLI
- [ ] Phase 6 — Streamlit interactive dashboard
- [ ] Phase 7 — Docker containerization + HF Spaces deploy
- [ ] Phase 8 — Full documentation

### Future Directions

- 📬 Full Enron dataset integration (streamed, privacy-filtered)
- 🧵 Multi-email thread environments with inbox state
- 📅 Calendar and task management integration
- 🤝 Multi-agent collaborative inbox scenarios
- 🔁 Online RL training loop (PPO / GRPO)

---

## ☁️ Deployment

### Hugging Face Spaces

The included `Dockerfile` is compatible with Hugging Face Spaces.

**Required environment variables:**

| Variable | Description |
|---|---|
| `API_BASE_URL` | OpenAI-compatible endpoint (e.g., `https://api.openai.com/v1`) |
| `MODEL_NAME` | Model identifier (e.g., `gpt-4o-mini`) |
| `HF_TOKEN` | Your Hugging Face token |

```bash
docker build -t smart-inbox .
docker run -p 7860:7860 smart-inbox
```

---

## 📦 Dependencies

```
gymnasium>=0.29.0
numpy>=1.24.0
pandas>=2.0.0
sentence-transformers>=2.7.0
streamlit>=1.35.0
plotly>=5.22.0
openai>=1.30.0
python-dotenv>=1.0.0
```

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch: `git checkout -b feat/your-feature`
3. Follow commit conventions: `type(scope): description`
4. Submit a PR with empirical evidence of correctness

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**SmartInboxRL** — *Because real intelligence handles the messy stuff.*

Built with 🧠 + ☕ | Gymnasium · SentenceTransformers · Streamlit

</div>
