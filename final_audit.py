"""SmartInboxRL — Final Pre-Submission Audit.

Covers EVERY item from the competition evaluation criteria and pre-submission checklist.
Run this as the last step before submitting.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

# ═══════════════════════════════════════════════════════════════════════════
# Tracking
# ═══════════════════════════════════════════════════════════════════════════

_results: dict[str, list] = {}
_current_section = ""

def section(name: str):
    global _current_section
    _current_section = name
    _results[name] = []
    print(f"\n{'='*65}")
    print(f"  {name}")
    print(f"{'='*65}")

def ok(msg: str):
    _results[_current_section].append(("PASS", msg))
    print(f"  [PASS] {msg}")

def fail(msg: str, detail: str = ""):
    _results[_current_section].append(("FAIL", msg))
    suffix = f" -> {detail}" if detail else ""
    print(f"  [FAIL] {msg}{suffix}")

def warn(msg: str, detail: str = ""):
    _results[_current_section].append(("WARN", msg))
    suffix = f" -> {detail}" if detail else ""
    print(f"  [WARN] {msg}{suffix}")

def info(msg: str):
    print(f"  [INFO] {msg}")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1: PRE-SUBMISSION CHECKLIST (Disqualification checks)
# ═══════════════════════════════════════════════════════════════════════════

section("1. REQUIRED FILES (Disqualification Check)")
required_files = {
    "inference.py": "Competition inference script",
    "Dockerfile": "Docker build descriptor",
    "openenv.yaml": "OpenEnv specification",
    "requirements.txt": "Python dependencies",
    "main.py": "FastAPI API server",
    "entrypoint.sh": "Dual-service startup script",
    "models.py": "Pydantic typed models",
    "README.md": "Project README with HF metadata",
}
for fname, desc in required_files.items():
    if (ROOT / fname).exists():
        ok(f"{fname} exists ({desc})")
    else:
        fail(f"{fname} MISSING ({desc})")

# ═══════════════════════════════════════════════════════════════════════════

section("2. ENVIRONMENT VARIABLES (Disqualification Check)")
for var in ("API_BASE_URL", "MODEL_NAME", "HF_TOKEN"):
    val = os.environ.get(var, "")
    if not val and var == "HF_TOKEN":
        val = os.environ.get("OPENAI_API_KEY", "")
    if val:
        ok(f"{var} is set ({val[:12]}...)")
    else:
        fail(f"{var} is NOT set")

# ═══════════════════════════════════════════════════════════════════════════

section("3. OPENENV.YAML SPEC COMPLIANCE (Disqualification Check)")
try:
    import yaml
    spec = yaml.safe_load(open(ROOT / "openenv.yaml"))
    required_top = ["name", "version", "description", "task_type", "action_space",
                    "observation_space", "reward_range", "tasks"]
    for k in required_top:
        if k in spec:
            ok(f"openenv.yaml has '{k}'")
        else:
            fail(f"openenv.yaml MISSING field '{k}'")

    rr = spec.get("reward_range", [])
    if isinstance(rr, list) and len(rr) == 2 and rr[0] == 0.0 and rr[1] == 1.0:
        ok("reward_range is [0.0, 1.0]")
    else:
        fail(f"reward_range must be [0.0, 1.0], got {rr}")

    tasks = spec.get("tasks", [])
    if len(tasks) >= 3:
        ok(f"{len(tasks)} tasks defined (>= 3 required)")
    else:
        fail(f"Need >= 3 tasks, found {len(tasks)}")

    for t in tasks:
        for k2 in ("name", "difficulty", "description"):
            if k2 not in t:
                fail(f"Task '{t.get('name','?')}' missing field '{k2}'")
        else:
            ok(f"Task '{t['name']}' has all required fields")

    difficulties = [t["difficulty"] for t in tasks]
    if "easy" in difficulties and "hard" in difficulties:
        ok(f"Task difficulty range covers easy->hard: {difficulties}")
    else:
        warn(f"Difficulty range may be weak: {difficulties}")

except Exception as e:
    fail(f"openenv.yaml parse error: {e}")

# ═══════════════════════════════════════════════════════════════════════════

section("4. PYDANTIC TYPED MODELS (Spec Compliance)")
try:
    from models import EmailObservation, EmailAction, EmailReward, EpisodeState

    obs = EmailObservation(
        email="Test", email_subject="Sub", email_sender="a@b.com",
        history=[], step=0, difficulty="easy"
    )
    ok("EmailObservation instantiates correctly")

    act = EmailAction(
        intents=["question"], priority="medium", action="reply", response="Thanks"
    )
    ok("EmailAction instantiates correctly")

    rew = EmailReward(
        total_score=0.8, intent_score=0.9, priority_score=0.8,
        action_score=0.7, response_score=0.8, breakdown={"intent": 0.9}
    )
    ok("EmailReward instantiates correctly")

    st = EpisodeState(
        current_email="Test", step=0, difficulty="easy",
        done=False, history=[], total_reward=0.0
    )
    ok("EpisodeState instantiates correctly")

    # Check model_dump works (Pydantic v2)
    d = obs.model_dump()
    if isinstance(d, dict):
        ok("EmailObservation.model_dump() returns dict (Pydantic v2)")
    else:
        fail("model_dump() did not return dict")

except Exception as e:
    fail(f"Pydantic model error: {e}")

# ═══════════════════════════════════════════════════════════════════════════

section("5. ENVIRONMENT PIPELINE: reset() -> step() -> state()")
try:
    from environment.inbox_env import InboxEnv
    from models import EmailObservation as EO, EmailReward as ER, EpisodeState as ES

    # --- reset() ---
    env = InboxEnv(difficulty="easy", max_steps=3, seed=42)
    obs, info_r = env.reset(seed=42)
    if isinstance(obs, EO):
        ok("reset() returns EmailObservation")
    else:
        fail(f"reset() returned {type(obs)}, expected EmailObservation")

    if isinstance(info_r, dict):
        ok("reset() info is dict")
    else:
        fail(f"reset() info type: {type(info_r)}")

    # Check observation has content
    if obs.email and len(obs.email) > 10:
        ok(f"Observation has email content ({len(obs.email)} chars)")
    else:
        warn(f"Observation email seems empty/short: '{obs.email[:30]}...'")

    # --- state() ---
    state = env.state()
    if isinstance(state, ES):
        ok("state() returns EpisodeState")
    else:
        fail(f"state() returned {type(state)}")

    # --- step() ---
    action = {
        "intents": ["question"],
        "priority": "medium",
        "action": "reply",
        "response": "Thank you for your email.",
    }
    obs2, reward, terminated, truncated, step_info = env.step(action)

    if 0.0 <= reward <= 1.0:
        ok(f"step() reward in [0.0, 1.0] -> {reward:.4f}")
    else:
        fail(f"step() reward OUT OF RANGE: {reward}")

    robj = step_info.get("reward_object")
    if isinstance(robj, ER):
        ok("step() info contains EmailReward object")
    else:
        fail(f"step() info missing EmailReward, got {type(robj)}")

    bd = step_info.get("reward_breakdown", {})
    if bd:
        all_ok = all(0.0 <= v <= 1.0 for v in bd.values())
        if all_ok:
            ok(f"All reward components in [0.0, 1.0]: {list(bd.keys())}")
        else:
            fail(f"Some components out of range: {bd}")
    else:
        fail("reward_breakdown is empty")

    # --- determinism / reproducibility ---
    env2 = InboxEnv(difficulty="easy", max_steps=3, seed=42)
    obs_a, _ = env2.reset(seed=42)
    env3 = InboxEnv(difficulty="easy", max_steps=3, seed=42)
    obs_b, _ = env3.reset(seed=42)
    if obs_a.email == obs_b.email:
        ok("Environment is deterministic with same seed")
    else:
        warn("Non-deterministic: same seed produced different emails")

except Exception as e:
    fail(f"Environment pipeline error: {e}")
    import traceback; traceback.print_exc()

# ═══════════════════════════════════════════════════════════════════════════

section("6. MULTI-DIFFICULTY GRADING (3+ tasks, rewards in [0,1])")
try:
    from environment.inbox_env import InboxEnv

    for diff in ("easy", "medium", "hard"):
        env_d = InboxEnv(difficulty=diff, max_steps=3, seed=42)
        obs_d, _ = env_d.reset(seed=42)
        act_d = {
            "intents": ["question"],
            "priority": "medium",
            "action": "reply",
            "response": "Thank you for reaching out.",
        }
        _, rew_d, _, _, info_d = env_d.step(act_d)
        bd_d = info_d.get("reward_breakdown", {})
        all_in = all(0.0 <= v <= 1.0 for v in bd_d.values())
        if 0.0 <= rew_d <= 1.0 and all_in:
            ok(f"'{diff}' task: reward={rew_d:.4f}, breakdown OK")
        else:
            fail(f"'{diff}' task: reward={rew_d}, breakdown={bd_d}")
except Exception as e:
    fail(f"Multi-difficulty grading error: {e}")

# ═══════════════════════════════════════════════════════════════════════════

section("7. INFERENCE.PY COMPLIANCE")

# Check inference.py uses OpenAI client
inf_text = (ROOT / "inference.py").read_text(encoding="utf-8")

if "from openai import OpenAI" in inf_text or "import openai" in inf_text:
    ok("inference.py uses OpenAI SDK")
else:
    fail("inference.py does NOT use OpenAI SDK")

if "API_BASE_URL" in inf_text and "MODEL_NAME" in inf_text and "HF_TOKEN" in inf_text:
    ok("inference.py references all 3 mandatory env vars")
else:
    fail("inference.py missing one or more mandatory env vars")

# Check structured logging format
for tag in ("[START]", "[STEP]", "[END]"):
    if tag in inf_text:
        ok(f"inference.py emits '{tag}' log format")
    else:
        fail(f"inference.py MISSING '{tag}' log format")

# Check tasks >= 3
task_matches = re.findall(r'"name":\s*"(\w+)"', inf_text)
if len(task_matches) >= 3:
    ok(f"inference.py defines {len(task_matches)} tasks: {task_matches}")
else:
    fail(f"inference.py has only {len(task_matches)} task definitions")

# Check score clamping
if "max(0.0, min(1.0" in inf_text:
    ok("inference.py clamps scores to [0.0, 1.0]")
else:
    warn("Could not verify score clamping in inference.py")

# ═══════════════════════════════════════════════════════════════════════════

section("8. DOCKERFILE & ENTRYPOINT")

dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
if "requirements.txt" in dockerfile:
    ok("Dockerfile installs requirements.txt")
else:
    fail("Dockerfile does not reference requirements.txt")

if "7860" in dockerfile:
    ok("Dockerfile exposes port 7860")
else:
    fail("Dockerfile missing port 7860")

if "entrypoint.sh" in dockerfile:
    ok("Dockerfile uses entrypoint.sh")
else:
    fail("Dockerfile does not reference entrypoint.sh")

entrypoint = (ROOT / "entrypoint.sh").read_text(encoding="utf-8")
if "streamlit" in entrypoint and "main.py" in entrypoint:
    ok("entrypoint.sh launches both Streamlit + FastAPI")
else:
    fail("entrypoint.sh missing Streamlit or FastAPI launch")

# ═══════════════════════════════════════════════════════════════════════════

section("9. README.md HF SPACE METADATA")

readme = (ROOT / "README.md").read_text(encoding="utf-8")
if readme.startswith("---"):
    ok("README.md has YAML frontmatter")
    if "sdk: docker" in readme:
        ok("README.md specifies sdk: docker")
    else:
        fail("README.md missing 'sdk: docker'")
    if "app_port: 7860" in readme:
        ok("README.md specifies app_port: 7860")
    else:
        fail("README.md missing 'app_port: 7860'")
else:
    fail("README.md missing YAML frontmatter (required for HF Spaces)")

# ═══════════════════════════════════════════════════════════════════════════

section("10. MAIN.PY API ENDPOINTS (OpenEnv HTTP Compliance)")

main_text = (ROOT / "main.py").read_text(encoding="utf-8")
for endpoint in ("/reset", "/step", "/state"):
    if endpoint in main_text:
        ok(f"main.py has '{endpoint}' endpoint")
    else:
        fail(f"main.py MISSING '{endpoint}' endpoint")

if "FastAPI" in main_text:
    ok("main.py uses FastAPI framework")
else:
    fail("main.py does not use FastAPI")

# ═══════════════════════════════════════════════════════════════════════════

section("11. REWARD DESIGN QUALITY (Evaluation Criteria)")

try:
    from rewards.reward_engine import RewardEngine

    re_engine = RewardEngine()
    
    # Check composite reward (not just sparse)
    sample_email = {
        "body": "Hi team, can we schedule a meeting for next week? Also, the server has been down for 2 hours.",
        "subject": "Meeting + Server Issue",
        "sender": "boss@company.com",
        "intents": ["meeting_request", "urgent"],
        "priority": "high",
        "expected_action": "escalate",
    }
    
    sample_action = {
        "intents": ["meeting_request"],
        "priority": "medium",
        "action": "reply",
        "response": "Sure, let me check my calendar.",
    }
    
    reward_val, parts = re_engine.compute(sample_email, sample_action)
    
    if len(parts) >= 3:
        ok(f"Reward has {len(parts)} components (rich, not sparse): {list(parts.keys())}")
    else:
        warn(f"Reward has only {len(parts)} components (may be too sparse)")

    # Check varying signal
    perfect_action = {
        "intents": ["meeting_request", "urgent"],
        "priority": "high",
        "action": "escalate",
        "response": "I'll escalate the server issue immediately and schedule the meeting.",
    }
    rew_perfect, _ = re_engine.compute(sample_email, perfect_action)
    
    bad_action = {
        "intents": ["spam"],
        "priority": "low",
        "action": "ignore",
        "response": "ok",
    }
    rew_bad, _ = re_engine.compute(sample_email, bad_action)
    
    if rew_perfect > rew_bad:
        ok(f"Reward provides varying signal: good={rew_perfect:.3f} > bad={rew_bad:.3f}")
    else:
        warn(f"Reward signal may not discriminate: good={rew_perfect:.3f}, bad={rew_bad:.3f}")

except Exception as e:
    warn(f"Could not test reward engine directly: {e}")

# ═══════════════════════════════════════════════════════════════════════════

section("12. TEST COVERAGE")

test_files = list((ROOT / "tests").glob("test_*.py"))
if len(test_files) >= 2:
    ok(f"{len(test_files)} test files found: {[f.name for f in test_files]}")
else:
    warn(f"Only {len(test_files)} test files — consider adding more")

# ═══════════════════════════════════════════════════════════════════════════

section("13. INFRA RESTRICTIONS")

# Check inference.py has timeout awareness
if "20" in inf_text or "timeout" in inf_text.lower() or "time.time" in inf_text:
    ok("inference.py has time tracking (runtime < 20 min)")
else:
    warn("inference.py may not track runtime")

# Check memory-heavy deps not in requirements
reqs = (ROOT / "requirements.txt").read_text(encoding="utf-8")
heavy_deps = ["torch", "tensorflow", "transformers"]
for dep in heavy_deps:
    if dep in reqs:
        warn(f"Heavy dependency '{dep}' in requirements.txt (8GB RAM limit!)")
    else:
        ok(f"No heavy dep '{dep}' (good for 8GB limit)")

# ═══════════════════════════════════════════════════════════════════════════

section("14. HF SPACE LIVE CHECK")
try:
    import httpx
    space_url = "https://arpit-jain-smartinboxrl.hf.space"
    try:
        resp = httpx.get(space_url, timeout=10.0, follow_redirects=True)
        if resp.status_code == 200:
            ok(f"HF Space responds with 200 OK at {space_url}")
            # Try to check if it's the API or the dashboard
            try:
                body = resp.json()
                if "status" in body:
                    ok(f"API root returns status: '{body.get('status')}'")
            except:
                info("Response is HTML (dashboard or loading page)")
        else:
            warn(f"HF Space returned status {resp.status_code}")
    except httpx.ConnectError:
        warn("Cannot reach HF Space (may still be building)")
    except httpx.ConnectTimeout:
        warn("HF Space connection timed out (may still be building)")
except ImportError:
    warn("httpx not installed, skipping live check")

# ═══════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("  FINAL PRE-SUBMISSION AUDIT REPORT")
print("=" * 65)

total_pass = 0
total_fail = 0
total_warn = 0

for section_name, checks in _results.items():
    passes = sum(1 for s, _ in checks if s == "PASS")
    fails  = sum(1 for s, _ in checks if s == "FAIL")
    warns  = sum(1 for s, _ in checks if s == "WARN")
    total_pass += passes
    total_fail += fails
    total_warn += warns

    icon = "[OK]" if fails == 0 else "[!!]"
    print(f"  {icon}  {section_name}: {passes} pass, {fails} fail, {warns} warn")

print(f"\n  TOTALS: {total_pass} PASS / {total_fail} FAIL / {total_warn} WARN")
print("=" * 65)

if total_fail > 0:
    print("\n  [!!] THERE ARE FAILURES — FIX BEFORE SUBMITTING!")
    print("  Failed checks:")
    for section_name, checks in _results.items():
        for status, msg in checks:
            if status == "FAIL":
                print(f"    - [{section_name}] {msg}")
    sys.exit(1)
elif total_warn > 0:
    print(f"\n  [OK] All critical checks pass! ({total_warn} warnings to review)")
    print("  You are READY TO SUBMIT.")
    sys.exit(0)
else:
    print("\n  [OK] PERFECT SCORE! All checks pass with zero warnings.")
    print("  You are READY TO SUBMIT.")
    sys.exit(0)
