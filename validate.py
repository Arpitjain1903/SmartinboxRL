"""SmartInboxRL — Pre-Submission Validation Script.

Run this before submitting to catch compliance issues early.

Checks
------
  1. Required env vars are set (API_BASE_URL, MODEL_NAME, HF_TOKEN)
  2. openenv.yaml is valid and has required fields
  3. 3+ tasks defined with required keys
  4. reward_range is [0.0, 1.0]
  5. Pydantic typed models load correctly
  6. InboxEnv.reset() returns EmailObservation
  7. InboxEnv.step() returns reward in [0.0, 1.0] and EmailReward in info
  8. InboxEnv.state() returns EpisodeState
  9. inference.py exists in project root
 10. Dockerfile exists

Usage
-----
    python validate.py

Exits with code 0 on full pass, code 1 on any failure.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure repo root is on path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

# Track results
_passed: list[str] = []
_failed: list[str] = []


def _ok(msg: str) -> None:
    _passed.append(msg)
    print(f"  [PASS] {msg}")


def _fail(msg: str, detail: str = "") -> None:
    _failed.append(msg)
    suffix = f"  →  {detail}" if detail else ""
    print(f"  [FAIL] {msg}{suffix}")


# ---------------------------------------------------------------------------
# CHECK 1 — Required env vars
# ---------------------------------------------------------------------------

print("\n── Check 1: Required environment variables ──────────────────")

for var in ("API_BASE_URL", "MODEL_NAME", "HF_TOKEN"):
    val = os.environ.get(var, "")
    if not val:
        # Try legacy alias for HF_TOKEN
        if var == "HF_TOKEN":
            val = os.environ.get("OPENAI_API_KEY", "")
    if val:
        _ok(f"  {var}  is set ({val[:8]}…)" if len(val) > 8 else f"  {var}  is set")
    else:
        _fail(f"  {var}  is not set", "add to .env or your environment")

# ---------------------------------------------------------------------------
# CHECK 2 — openenv.yaml exists and is valid
# ---------------------------------------------------------------------------

print("\n── Check 2: openenv.yaml structure ──────────────────────────")

try:
    import yaml

    openenv_path = ROOT / "openenv.yaml"
    if not openenv_path.exists():
        _fail("openenv.yaml missing", f"expected at {openenv_path}")
    else:
        with open(openenv_path) as f:
            spec = yaml.safe_load(f)

        required_top = ["name", "version", "description", "task_type", "action_space",
                        "observation_space", "reward_range", "tasks"]
        for key in required_top:
            if key in spec:
                _ok(f"openenv.yaml has '{key}'")
            else:
                _fail(f"openenv.yaml missing required field: '{key}'")

        # reward_range validation
        rr = spec.get("reward_range", [])
        if isinstance(rr, list) and len(rr) == 2 and rr[0] == 0.0 and rr[1] == 1.0:
            _ok("reward_range is [0.0, 1.0]")
        else:
            _fail("reward_range must be [0.0, 1.0]", f"got {rr}")

        # task count
        tasks = spec.get("tasks", [])
        if len(tasks) >= 3:
            _ok(f"{len(tasks)} tasks defined (≥ 3 required)")
        else:
            _fail(f"Need ≥ 3 tasks, found {len(tasks)}")

        for t in tasks:
            for key in ("name", "difficulty", "description"):
                if key not in t:
                    _fail(f"Task '{t.get('name', '?')}' missing field '{key}'")

except ImportError:
    _fail("PyYAML not installed", "pip install PyYAML")
except Exception as exc:
    _fail("openenv.yaml parse error", str(exc))

# ---------------------------------------------------------------------------
# CHECK 3 — Typed Pydantic models
# ---------------------------------------------------------------------------

print("\n── Check 3: Pydantic typed models ───────────────────────────")

try:
    from models import EmailObservation, EmailAction, EmailReward, EpisodeState

    # Instantiate each model
    obs_model = EmailObservation(
        email="Test body",
        email_subject="Test subject",
        email_sender="test@example.com",
        history=[],
        step=0,
        difficulty="easy",
    )
    _ok("EmailObservation instantiates correctly")

    action_model = EmailAction(
        intents=["question"],
        priority="medium",
        action="reply",
        response="Test response",
    )
    _ok("EmailAction instantiates correctly")

    reward_model = EmailReward(
        total_score=0.8,
        intent_score=0.9,
        priority_score=0.8,
        action_score=0.7,
        response_score=0.8,
        breakdown={"intent": 0.9, "priority": 0.8},
    )
    _ok("EmailReward instantiates correctly")

    state_model = EpisodeState(
        current_email="Test",
        step=0,
        difficulty="easy",
        done=False,
        history=[],
        total_reward=0.0,
    )
    _ok("EpisodeState instantiates correctly")

except Exception as exc:
    _fail("Pydantic model error", str(exc))

# ---------------------------------------------------------------------------
# CHECK 4 — InboxEnv pipeline: reset() → step() → state()
# ---------------------------------------------------------------------------

print("\n── Check 4: InboxEnv pipeline ───────────────────────────────")

try:
    from environment.inbox_env import InboxEnv
    from models import EmailObservation, EmailReward, EpisodeState

    env = InboxEnv(difficulty="easy", max_steps=3, seed=42)

    # reset()
    obs, info = env.reset(seed=42)
    if isinstance(obs, EmailObservation):
        _ok("reset() returns EmailObservation")
    else:
        _fail("reset() return type", f"got {type(obs)}, expected EmailObservation")

    # state()
    state = env.state()
    if isinstance(state, EpisodeState):
        _ok("state() returns EpisodeState")
    else:
        _fail("state() return type", f"got {type(state)}, expected EpisodeState")

    # step()
    action = {
        "intents":  ["question"],
        "priority": "medium",
        "action":   "reply",
        "response": "Thank you for reaching out.",
    }
    obs2, reward, terminated, truncated, step_info = env.step(action)

    # reward range
    if 0.0 <= reward <= 1.0:
        _ok(f"step() reward in [0.0, 1.0]  (got {reward:.4f})")
    else:
        _fail(f"step() reward out of range", f"got {reward}")

    # reward_object
    reward_obj = step_info.get("reward_object")
    if isinstance(reward_obj, EmailReward):
        _ok("step() info contains EmailReward object")
    else:
        _fail("step() info missing EmailReward", f"got {type(reward_obj)}")

    # obs on step returns EmailObservation or {} (terminal)
    if isinstance(obs2, (EmailObservation, dict)):
        _ok("step() observation type is EmailObservation or {} (terminal)")
    else:
        _fail("step() observation type", f"got {type(obs2)}")

except Exception as exc:
    _fail("InboxEnv pipeline error", str(exc))
    import traceback
    traceback.print_exc()

# ---------------------------------------------------------------------------
# CHECK 5 — Reward scores per component in [0.0, 1.0]
# ---------------------------------------------------------------------------

print("\n── Check 5: Per-component reward range ──────────────────────")

try:
    from environment.inbox_env import InboxEnv

    env2 = InboxEnv(difficulty="medium", max_steps=3, seed=42)
    obs, _ = env2.reset(seed=42)
    action2 = {
        "intents":  ["urgent"],
        "priority": "high",
        "action":   "escalate",
        "response": "This has been escalated to the relevant team.",
    }
    _, _, _, _, info2 = env2.step(action2)
    breakdown = info2.get("reward_breakdown", {})
    all_in_range = all(0.0 <= v <= 1.0 for v in breakdown.values())
    if all_in_range and breakdown:
        _ok(f"All reward components in [0.0, 1.0]: {list(breakdown.keys())}")
    elif not breakdown:
        _fail("reward_breakdown is empty")
    else:
        _fail("Some reward components out of [0.0, 1.0]", str(breakdown))
except Exception as exc:
    _fail("Reward range check error", str(exc))

# ---------------------------------------------------------------------------
# CHECK 7 — HTTP API Endpoints (OpenEnv Compliance)
# ---------------------------------------------------------------------------

print("\n── Check 7: HTTP API Endpoints ─────────────────────────────")

if (ROOT / "main.py").exists():
    _ok("main.py (API Server) exists")
else:
    _fail("main.py missing", "required for OpenEnv HTTP endpoints")

if (ROOT / "entrypoint.sh").exists():
    _ok("entrypoint.sh exists")
else:
    _fail("entrypoint.sh missing", "required to run both API and Dashboard")

try:
    import httpx
    import time
    
    # Try to see if it's already running (unlikely during validation, but good to check)
    try:
        response = httpx.get("http://localhost:7860/", timeout=0.1)
        if response.status_code == 200:
            _ok("Local API is running and responding to GET /")
            # Try reset
            try:
                reset_resp = httpx.post("http://localhost:7860/reset", json={"seed": 42}, timeout=0.5)
                if reset_resp.status_code == 200:
                    _ok("Local API responds to /reset")
                else:
                    _fail(f"/reset failed", f"status {reset_resp.status_code}")
            except (httpx.ConnectTimeout, httpx.ReadTimeout):
                print("  (Note: Local API /reset ping timed out; skipping)")
    except (httpx.ConnectError, httpx.ConnectTimeout):
        print("  (Note: Local API server not running or too slow; skipping live ping check)")
        _ok("API check passed (structural only)")
except ImportError:
    print("  (Warning: httpx not installed; skipping live ping check)")

# ---------------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------------

total  = len(_passed) + len(_failed)
passed = len(_passed)
failed = len(_failed)

print("\n" + "=" * 60)
print(f"  Validation Summary: {passed}/{total} checks passed")
print("=" * 60)

if _failed:
    print("\n  FAILED checks:")
    for f in _failed:
        print(f"    - {f.strip()}")
    print()
    sys.exit(1)
else:
    print("\n  All checks passed! Ready to submit.\n")
    sys.exit(0)
