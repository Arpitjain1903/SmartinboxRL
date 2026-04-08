---
title: SmartInboxRL
emoji: 📧
colorFrom: indigo
colorTo: green
sdk: docker
app_port: 7860
tags:
  - openenv
  - email-triage
  - reinforcement-learning
  - nlp
  - gymnasium
short_description: OpenEnv RL environment for email triage.
---

# SmartInboxRL — OpenEnv Email Triage Environment

SmartInboxRL is a Gymnasium-compatible reinforcement learning environment designed to evaluate AI agents on realistic email triage tasks. Triage is a critical first step in productivity workflows, requiring high precision in intent extraction, priority classification, and drafting appropriate responses under noisy real-world conditions.

## 1. Overview & Motivation
As LLM-based agents become more prevalent in enterprise environments, the need for robust evaluation of their triage capabilities grows. Simple RAG-based systems often struggle with:
- **Noisy Inputs**: Emails often contain typos, shorthand, and irrelevant signatures.
- **Multi-Intent Detection**: A single email might contain a request for a meeting, a technical question, and a complaint.
- **Dynamic Context**: Decisions must be made based on previous interaction history within a conversation thread.
- **Structured Output Compliance**: Agents must produce machine-readable actions for downstream automation.

SmartInboxRL addresses these challenges by providing a controlled yet realistic environment for benchmarking RL and LLM agents.

## 2. Environment Description
The environment is structured as a series of episodes where each step presents a new email from a dataset (e.g., Enron or Synthetic) with varying difficulty levels:
- **Easy**: Single-intent, clean text, obvious priority.
- **Medium**: Multiple intents, some noise, ambiguous priority requiring judgment.
- **Hard**: Adversarial noise, conflicting intents, and complex history dependencies.

## 3. Observation Space
The observation is a structured object (Pydantic `EmailObservation`) containing:

| Field | Type | Description |
| :--- | :--- | :--- |
| `email` | `str` | The body text of the current email being triaged. |
| `history` | `List[dict]` | Context from the last N interactions in the same thread. |
| `step` | `int` | Current sequence number in the episode. |
| `difficulty` | `str` | Tier of the current task ("easy", "medium", "hard"). |

## 4. Action Space
Agents must produce a structured action (Pydantic `EmailAction`) with the following fields:

| Field | Type | Description |
| :--- | :--- | :--- |
| `intents` | `List[str]` | Extracted intents (e.g., `meeting_request`, `complaint`). |
| `priority` | `str` | Cleanup level: `low`, `medium`, `high`, `critical`. |
| `action` | `str` | Triage decision: `reply`, `ignore`, `escalate`, `forward`. |
| `response` | `str` | Drafted reply text (if applicable). |

## 5. Reward Function
The environment provides a dense composite reward in the range `[0.0, 1.0]` per step:

| Component | Weight | Description | Measurement |
| :--- | :--- | :--- | :--- |
| **Intent Score** | 30% | How well the agent identified the primary intents. | F1-Score against ground truth labels. |
| **Priority Score** | 20% | Correctness of the classification. | Exact match or partial credit for near-neighbor. |
| **Action Score** | 20% | Appropriateness of the triage decision. | Comparison with expert gold labels. |
| **Response Quality** | 30% | Semantic accuracy of the draft. | Cosine similarity using `all-MiniLM-L6-v2`. |

## 6. Task Descriptions
- **simple_reply (Easy)**: A meeting request that needs a simple "Yes/No" or "I'm available" response.
- **multi_intent_triage (Medium)**: An email containing both a billing question and a feature request, requiring high priority and a detailed response.
- **adversarial_noise (Hard)**: Emails with heavy typos, conversational shorthand, and conflicting instructions designed to test agent robustness.

## 7. Quick Start
### Installation
```bash
pip install -r requirements.txt
```

### Running OpenAI Baseline
1. Set your API Key:
   ```bash
   export OPENAI_API_KEY="sk-..."
   ```
2. Run the baseline evaluation:
   ```bash
   python baseline_openai.py --episodes 3 --difficulty all
   ```

## 8. Docker Setup
Build and run the Streamlit dashboard locally:
```bash
docker build -t smartinboxrl .
docker run -p 7860:7860 --env-file .env smartinboxrl
```

## 9. HF Space
The environment is deployed on Hugging Face Spaces with the `openenv` tag.

**Deployment:** [SmartInboxRL on HF Spaces](https://huggingface.co/spaces/Arpitjain1903/SmartInboxRL)

### Setting up Secrets for Judges
To run the LLM baseline on HF Spaces, configure secrets in **Settings → Secrets**:
| Secret | Value |
|---|---|
| `OPENAI_API_KEY` | Your OpenAI-compatible API key (e.g., Groq, OpenAI) |
| `API_BASE_URL` | API endpoint (default: `https://api.openai.com/v1`) |
| `MODEL_NAME` | Model identifier (default: `gpt-4o-mini`) |

## 10. Baseline Scores
| Difficulty | Model | Mean Episode Reward | Avg Step Reward |
| :--- | :--- | :--- | :--- |
| Easy | Llama-3.1-8b (Groq) | 10.27 | ~0.68 |
| Medium | Llama-3.1-8b (Groq) | 5.82 | ~0.38 |
| Hard | Llama-3.1-8b (Groq) | 5.61 | ~0.37 |

*Note: Results based on 3-episode runs with 15 steps per episode. Reward normalized to [0, 1] per step.*

## 11. Extending the Environment
- **Add Tasks**: Modify `environment/email_loader.py` to include new datasets or synthesis rules.
- **Custom Agents**: Inherit from `BaseAgent` in `agents/base_agent.py` and implement the `act()` method.

## 12. OpenEnv Compliance

SmartInboxRL implements the [OpenEnv](https://openenv.dev) specification:

| Requirement | Status |
|---|---|
| `reset()` → `(observation, info)` | ✅ Returns `EmailObservation` Pydantic model |
| `step(action)` → `(obs, reward, terminated, truncated, info)` | ✅ Gymnasium-compatible |
| `state()` → typed state model | ✅ Returns `EpisodeState` Pydantic model |
| Reward range `[0.0, 1.0]` | ✅ Normalized via `(raw + 1) / 2` |
| `openenv.yaml` with `openenv` tag | ✅ Included |
| Docker support | ✅ `Dockerfile` + pre-cached embedding model |
| HF Spaces deployment | ✅ With `openenv` tag |

## 13. Citation / License
MIT License. Created by Arpit.
