"""
server/app.py — OpenEnv multi-mode deployment server for SmartInboxRL.
Entry point: `server = "server.app:main"` (defined in pyproject.toml)
"""
import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List

app = FastAPI(
    title="SmartInboxRL OpenEnv Server",
    description="Multi-mode deployment server for SmartInboxRL email triage RL environment",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    difficulty: Optional[str] = "easy"
    seed: Optional[int] = None


class StepRequest(BaseModel):
    action: dict


class ResetResponse(BaseModel):
    observation: dict
    info: dict


class StepResponse(BaseModel):
    observation: dict
    reward: float
    terminated: bool
    truncated: bool
    info: dict


# ---------------------------------------------------------------------------
# Environment singleton (lazy init)
# ---------------------------------------------------------------------------

_env = None


def _get_env():
    global _env
    if _env is None:
        from environment.inbox_env import InboxEnv  # imported lazily so server starts fast
        _env = InboxEnv()
    return _env


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "service": "SmartInboxRL"}


@app.post("/reset", response_model=ResetResponse)
def reset(req: ResetRequest):
    try:
        env = _get_env()
        obs, info = env.reset(seed=req.seed, options={"difficulty": req.difficulty})
        return ResetResponse(observation=obs if isinstance(obs, dict) else {"email": str(obs)}, info=info or {})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step", response_model=StepResponse)
def step(req: StepRequest):
    try:
        env = _get_env()
        obs, reward, terminated, truncated, info = env.step(req.action)
        return StepResponse(
            observation=obs if isinstance(obs, dict) else {"email": str(obs)},
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=info or {},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/info")
def env_info():
    return {
        "name": "SmartInboxRL",
        "version": "1.0.0",
        "task_type": "email_triage",
        "action_space": ["reply", "ignore", "escalate", "forward"],
        "difficulty_levels": ["easy", "medium", "hard", "enron"],
        "reward_range": [0.0, 1.0],
    }


# ---------------------------------------------------------------------------
# Entry point — called by `server` script defined in [project.scripts]
# ---------------------------------------------------------------------------

def main(host: str = "0.0.0.0", port: int = 7860):
    """Start the OpenEnv FastAPI server."""
    uvicorn.run(
        "server.app:app",
        host=host,
        port=int(os.environ.get("PORT", port)),
        log_level="info",
    )


if __name__ == "__main__":
    main()
