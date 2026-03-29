# SmartInboxRL — Implementation Plan

SmartInboxRL is a realistic RL evaluation benchmark that tests whether AI agents can understand messy emails, make decisions under ambiguity, and generate context-aware responses. The system mirrors a real inbox workflow: **Understand → Decide → Act → Communicate**.

---

## Proposed Changes

### Phase 1 — Project Foundation & Environment Core

#### [NEW] `environment/inbox_env.py`
The core Gymnasium-compatible `InboxEnv` class. Manages episode lifecycle, step execution, observation building, and reward dispatch.

#### [NEW] `environment/email_loader.py`
Loads and preprocesses email samples. Supports:
- Built-in curated dataset (easy/medium/hard tiers)
- Noise injection pipeline (typos, casing, shorthand abbreviations)
- Enron-style email adapter (optional streaming)

#### [NEW] `environment/state.py`
Tracks per-episode state:
- Current email
- Interaction history (for memory/context awareness)
- Step count and action log (for anti-cheating)

#### [NEW] `environment/action_space.py`
Defines the discrete action set: `REPLY`, `IGNORE`, `ESCALATE`, `FORWARD`. Includes action metadata for reward computation.

---

### Phase 2 — Reward System

#### [NEW] `rewards/reward_engine.py`
Computes the composite reward per step:

| Component | Weight | Method |
|---|---|---|
| Intent understanding | 30% | Label-match F1 |
| Priority correctness | 20% | Exact match |
| Action decision | 20% | Exact match |
| Response quality | 30% | Embedding cosine similarity |

#### [NEW] `rewards/penalty_system.py`
Applies structured penalties:
- **Repeated actions**: -0.2 per consecutive duplicate
- **Ignoring critical emails**: -0.5
- **Low-effort / trivial responses**: -0.3 (length + coherence check)

#### [NEW] `rewards/embedding_scorer.py`
Semantic response evaluator using `sentence-transformers` (`all-MiniLM-L6-v2`). Scores generated responses against gold references via cosine similarity.

---

### Phase 3 — Task Dataset

#### [NEW] `data/tasks/easy_tasks.json`
Clear, unambiguous emails. Examples: spam detection, meeting confirmations.

#### [NEW] `data/tasks/medium_tasks.json`
Moderate ambiguity. Examples: scheduling + follow-up in one email, polite refusal needed.

#### [NEW] `data/tasks/hard_tasks.json`
Multi-intent, noisy emails with conflicting signals. Examples: review request + lunch invite + embedded deadline.

#### [NEW] `data/noise_profiles.json`
Noise injection templates: typo sets, abbreviation maps, casing rules.

---

### Phase 4 — Baseline Agent

#### [NEW] `agents/llm_agent.py`
LLM-backed baseline agent. Uses a structured prompt to extract:
1. Intent labels
2. Priority score (low/medium/high/critical)
3. Action decision
4. Response text

Supports OpenAI-compatible API endpoints. Environment variable driven (`API_BASE_URL`, `MODEL_NAME`).

#### [NEW] `agents/random_agent.py`
Uniform-random baseline for sanity checks and floor performance.

#### [NEW] `agents/rule_agent.py`
Simple keyword-heuristic agent. Useful for establishing a non-ML baseline.

---

### Phase 5 — Inference & Evaluation CLI

#### [NEW] `inference.py`
Entry-point CLI for running full evaluation episodes:
```bash
python inference.py --agent llm --episodes 50 --difficulty all
```
Outputs per-episode reward breakdown and summary statistics.

#### [NEW] `evaluate.py`
Batch evaluation script. Produces:
- Mean rewards per component
- Score distribution histograms
- Difficulty-tier performance tables
- JSON results export

---

### Phase 6 — Interactive Dashboard

#### [NEW] `app.py`
Streamlit dashboard providing:
- Live episode runner (select agent + difficulty)
- Step-by-step action + reward visualization
- Radar chart of reward components
- Score history plots across episodes

---

### Phase 7 — Containerization & Deployment

#### [NEW] `Dockerfile`
Multi-stage Docker build:
- Python 3.11 slim base
- Installs dependencies, downloads embedding model at build time
- Exposes Streamlit dashboard on port 7860 (Hugging Face Spaces compatible)

#### [NEW] `docker-compose.yml`
Local orchestration for dev workflow.

#### [NEW] `requirements.txt`
Pinned dependencies:
- `gymnasium`, `numpy`, `pandas`
- `sentence-transformers`
- `streamlit`, `plotly`
- `openai` (for LLM agent)
- `python-dotenv`

---

### Phase 8 — Spec, Docs & README

#### [NEW] `README.md` (project root)
Comprehensive README (see artifact below).

#### [NEW] `.gsd/SPEC.md`
Finalized specification document.

#### [NEW] `.gsd/ROADMAP.md`
Phase-by-phase roadmap tracking.

#### [NEW] `.env.example`
Template for required environment variables.

---

## Open Questions

> [!IMPORTANT]
> **LLM Provider**: Should the baseline agent default to OpenAI API, a local Ollama endpoint, or be fully provider-agnostic via environment variables? Recommendation: env-var driven, defaulting to OpenAI-compatible.

> [!IMPORTANT]
> **Enron Dataset**: Should Phase 1 include real Enron emails (requires download ~1.7GB) or ship with a curated synthetic set only? Recommendation: ship with 150 curated samples; Enron integration as optional Phase 2 extension.

> [!NOTE]
> **Embedding Model**: `all-MiniLM-L6-v2` is 80MB and runs on CPU. If GPU is available, `all-mpnet-base-v2` gives better quality. Plan defaults to MiniLM.

---

## Verification Plan

### Automated Tests
```bash
python -m pytest tests/ -v                        # unit + integration
python inference.py --agent random --episodes 10   # smoke test
python inference.py --agent rule --episodes 10     # baseline check
```

### Docker Build Verification
```bash
docker build -t smart-inbox .
docker run -p 7860:7860 smart-inbox
```

### Dashboard Smoke Test
- Launch Streamlit, run one episode per difficulty tier
- Verify reward breakdown is displayed correctly
- Check no-crash on all three agent types

### Score Distribution Check
- Easy tier: mean reward > 0.65
- Hard tier: mean reward < 0.45 (ensures separation)
- Random agent: mean reward < 0.25

---

## Execution Order (Waves)

| Wave | Phases | Description |
|---|---|---|
| 1 | 1, 3 | Environment core + task data (foundation) |
| 2 | 2 | Reward system (depends on env + data) |
| 3 | 4 | Agents (depend on env + rewards) |
| 4 | 5, 6 | CLI inference + dashboard |
| 5 | 7, 8 | Docker + docs |
