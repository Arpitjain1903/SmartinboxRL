"""SmartInboxRL — Competition Inference Script.

Runs 3 graded tasks (easy / medium / hard) using the OpenAI SDK against an
OpenAI-compatible endpoint.  Emits structured stdout logs in the required
[START] / [STEP] / [END] format for automated evaluation.

Required environment variables
--------------------------------
    API_BASE_URL   The OpenAI-compatible API endpoint URL.
                   e.g. https://api.groq.com/openai/v1
    MODEL_NAME     The model identifier.
                   e.g. llama-3.1-8b-instant
    HF_TOKEN       Your Hugging Face / API key used for authentication.

Usage
-----
    python inference.py

Runtime target: < 20 minutes on 2 vCPU / 8 GB RAM.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Score boundary guard — HARD hackathon requirement
# ---------------------------------------------------------------------------

def _safe_score(x: float) -> float:
    """
    HACKATHON HARD REQUIREMENT:
    Validator REJECTS scores of exactly 0.0 or 1.0.
    Clips every task score to strictly open interval (0.001, 0.999).
    Wrap EVERY return statement that produces a task score.
    """
    return max(0.001, min(0.999, float(x)))

# ---------------------------------------------------------------------------
# Environment configuration (mandatory per competition spec)
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME: str   = os.environ.get("MODEL_NAME", "llama-3.1-8b-instant")
HF_TOKEN: str     = os.environ.get("HF_TOKEN", os.environ.get("OPENAI_API_KEY", ""))

if not HF_TOKEN:
    print(
        "[ERROR] HF_TOKEN (or OPENAI_API_KEY) is not set. "
        "Please set it in your environment or .env file.",
        flush=True,
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# OpenAI client (competition requirement: must use OpenAI SDK)
# ---------------------------------------------------------------------------

from openai import OpenAI

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

# ---------------------------------------------------------------------------
# Environment imports
# ---------------------------------------------------------------------------

from environment.inbox_env import InboxEnv

# ---------------------------------------------------------------------------
# Structured log helpers — MUST match [START] / [STEP] / [END] exactly
# ---------------------------------------------------------------------------


def log_start(task: str, difficulty: str, agent: str = "llm") -> None:
    print(
        f"[START] {json.dumps({'task': task, 'agent': agent, 'difficulty': difficulty})}",
        flush=True,
    )


def log_step(
    step: int,
    email_id: str,
    action: str,
    priority: str,
    intents: list[str],
    reward: float,
    score: float,
    breakdown: dict[str, float],
) -> None:
    print(
        f"[STEP] {json.dumps({'step': step, 'email_id': email_id, 'action': action, 'priority': priority, 'intents': intents, 'reward': round(reward, 4), 'score': round(score, 4), 'breakdown': {k: round(v, 4) for k, v in breakdown.items()}})}",
        flush=True,
    )


def log_end(
    task: str,
    score: float,
    steps: int,
    status: str = "success",
    error: str | None = None,
) -> None:
    payload: dict[str, Any] = {
        "task": task,
        "score": round(score, 4),
        "steps": steps,
        "status": status,
    }
    if error:
        payload["error"] = error
    print(f"[END] {json.dumps(payload)}", flush=True)


# ---------------------------------------------------------------------------
# LLM triage call (OpenAI SDK)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert email triage assistant. "
    "Given an email, respond with ONLY a valid JSON object with these exact keys:\n"
    '{"intents": ["list", "of", "detected", "intents"], '
    '"priority": "low|medium|high|critical", '
    '"action": "reply|ignore|escalate|forward", '
    '"response": "your drafted reply here"}\n'
    "Detect intents from: [meeting_request, complaint, question, urgent, "
    "follow_up, spam, feedback, approval_needed]\n"
    "No explanation. No markdown. Only raw JSON."
)

_VALID_INTENTS   = {"meeting_request", "complaint", "question", "urgent",
                    "follow_up", "spam", "feedback", "approval_needed"}
_VALID_PRIORITIES = {"low", "medium", "high", "critical"}
_VALID_ACTIONS    = {"reply", "ignore", "escalate", "forward"}


def _fallback_action() -> dict[str, Any]:
    """Rule-based fallback when LLM fails."""
    return {
        "intents":  ["question"],
        "priority": "medium",
        "action":   "reply",
        "response": (
            "Thank you for your email. "
            "We have received it and will get back to you shortly."
        ),
    }


def _normalize_action(raw: dict[str, Any]) -> dict[str, Any]:
    """Ensure all action fields are valid; fall back gracefully."""
    intents = [
        i for i in (raw.get("intents") or []) if i in _VALID_INTENTS
    ] or ["question"]
    priority = raw.get("priority", "medium")
    if priority not in _VALID_PRIORITIES:
        priority = "medium"
    action = raw.get("action", "reply")
    if action not in _VALID_ACTIONS:
        action = "reply"
    response = str(raw.get("response") or "Thank you for your message.")
    return {"intents": intents, "priority": priority, "action": action, "response": response}


def call_llm(email_body: str, email_subject: str, history: list[dict]) -> dict[str, Any]:
    """Call LLM via OpenAI SDK with retry on rate-limit."""
    history_str = json.dumps(history[-3:], indent=2) if history else "[]"
    user_prompt = (
        f"SUBJECT: {email_subject}\n\n"
        f"EMAIL CONTENT:\n{email_body}\n\n"
        f"RECENT HISTORY:\n{history_str}\n\n"
        "Provide triage JSON:"
    )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=512,
                response_format={"type": "json_object"},
            )
            raw_content = completion.choices[0].message.content
            raw_dict = json.loads(raw_content)
            return _normalize_action(raw_dict)
        except Exception as exc:
            is_rate_limit = "429" in str(exc) or "rate" in str(exc).lower()
            if is_rate_limit and attempt < max_retries - 1:
                wait = 8 * (attempt + 1)
                time.sleep(wait)
                continue
            # On final failure, use fallback
            return _fallback_action()

    return _fallback_action()


# ---------------------------------------------------------------------------
# Task definitions (3 tasks required by competition spec)
# ---------------------------------------------------------------------------

TASKS = [
    {
        "name":        "simple_reply",
        "difficulty":  "easy",
        "description": "Single-intent emails requiring a polite response",
        "max_steps":   5,
        "seed":        42,
    },
    {
        "name":        "multi_intent_triage",
        "difficulty":  "medium",
        "description": "Multi-intent emails with ambiguous priority",
        "max_steps":   5,
        "seed":        43,
    },
    {
        "name":        "adversarial_noise",
        "difficulty":  "hard",
        "description": "Noisy emails with conflicting intents",
        "max_steps":   5,
        "seed":        44,
    },
]


# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------


def run_task(task_def: dict[str, Any]) -> dict[str, Any]:
    """Run a single graded task and return its results."""
    name       = task_def["name"]
    difficulty = task_def["difficulty"]
    max_steps  = task_def.get("max_steps", 5)
    seed       = task_def.get("seed", 42)

    log_start(task=name, difficulty=difficulty)

    try:
        env = InboxEnv(
            difficulty=difficulty,
            max_steps=max_steps,
            seed=seed,
        )

        obs, _info = env.reset(seed=seed)
        step_count  = 0
        total_score = 0.0

        while True:
            # Extract email fields from observation
            if hasattr(obs, "email"):
                email_body    = obs.email
                email_subject = getattr(obs, "email_subject", "")
                history       = getattr(obs, "history", [])
                email_id      = getattr(obs, "email_id", f"step_{step_count}")
            elif isinstance(obs, dict):
                email_data    = obs.get("email", {})
                email_body    = email_data.get("body", "") if isinstance(email_data, dict) else str(email_data)
                email_subject = email_data.get("subject", "") if isinstance(email_data, dict) else ""
                history       = obs.get("history", [])
                email_id      = email_data.get("id", f"step_{step_count}") if isinstance(email_data, dict) else f"step_{step_count}"
            else:
                break

            if not email_body:
                break

            # Call LLM
            action = call_llm(email_body, email_subject, history)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            step_count  += 1
            reward       = float(reward)
            # Normalize to (0.001, 0.999) — step reward already per-step normalized.
            # _safe_score ensures we never return exactly 0.0 or 1.0.
            step_score   = _safe_score(reward)
            total_score += step_score

            breakdown = info.get("reward_breakdown", {})

            log_step(
                step      = step_count,
                email_id  = str(email_id),
                action    = action["action"],
                priority  = action["priority"],
                intents   = action["intents"],
                reward    = reward,
                score     = step_score,
                breakdown = breakdown,
            )

            if terminated or truncated:
                break

        # Final task score: mean step score clamped to strict open interval.
        # round() alone can snap to 0.0 or 1.0 — _safe_score prevents that.
        final_score = _safe_score(total_score / max(step_count, 1))

        log_end(task=name, score=final_score, steps=step_count, status="success")

        return {"task": name, "score": final_score, "steps": step_count, "status": "success"}

    except Exception as exc:
        error_msg = str(exc)
        # score=0.0 is REJECTED by validator — use minimum safe score instead
        error_score = _safe_score(0.0)  # → 0.001
        log_end(task=name, score=error_score, steps=0, status="error", error=error_msg)
        return {"task": name, "score": error_score, "steps": 0, "status": "error", "error": error_msg}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 60, flush=True)
    print("  SmartInboxRL — Inference / Evaluation", flush=True)
    print(f"  API_BASE_URL : {API_BASE_URL}", flush=True)
    print(f"  MODEL_NAME   : {MODEL_NAME}", flush=True)
    print(f"  Tasks        : {len(TASKS)}", flush=True)
    print("=" * 60, flush=True)

    t_start = time.time()
    all_results: list[dict[str, Any]] = []

    for task_def in TASKS:
        result = run_task(task_def)
        all_results.append(result)

    elapsed = round(time.time() - t_start, 2)

    # Final summary
    scores = [r["score"] for r in all_results]
    mean_score = round(sum(scores) / len(scores), 4) if scores else 0.0

    print("=" * 60, flush=True)
    print("  FINAL RESULTS", flush=True)
    print("=" * 60, flush=True)
    for r in all_results:
        status_icon = "[OK]" if r["status"] == "success" else "[FAIL]"
        print(
            f"  {status_icon}  {r['task']:<25} score={r['score']:.4f}  steps={r['steps']}",
            flush=True,
        )
    print(f"\n  Mean score   : {mean_score:.4f}", flush=True)
    print(f"  Elapsed time : {elapsed}s", flush=True)
    print("=" * 60, flush=True)

    # Save results
    out_path = Path("results") / "inference_results.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "tasks": all_results,
                "mean_score": mean_score,
                "elapsed_seconds": elapsed,
                "model": MODEL_NAME,
                "api_base": API_BASE_URL,
            },
            indent=2,
        )
    )
    print(f"\n  Results saved → {out_path}", flush=True)


if __name__ == "__main__":
    main()
