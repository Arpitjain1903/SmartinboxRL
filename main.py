"""FastAPI Server for OpenEnv Compliance.

Exposes mandatory endpoints:
  - POST /reset
  - POST /step
  - GET  /state
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from environment.inbox_env import InboxEnv
from models import EmailObservation, EmailAction, EmailReward, EpisodeState

app = FastAPI(
    title="SmartInboxRL API",
    description="OpenEnv-compliant API for email triage evaluation.",
    version="1.0.0",
)

# Enable CORS for cross-origin evaluation if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global Environment Instance
# ---------------------------------------------------------------------------

# We initialize with 'easy' by default. Evaluation scripts will call reset with specific seeds.
env = InboxEnv(difficulty="easy", max_steps=15)

# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    """Welcome and health check."""
    return {
        "status": "online",
        "env": "SmartInboxRL",
        "endpoints": ["/reset", "/step", "/state"],
        "docs": "/docs"
    }

@app.post("/reset", response_model=dict)
async def reset_env(params: dict[str, Any] | None = None):
    """
    Reset the environment.
    Accepts optional seed and options.
    Returns: (observation, info)
    """
    seed = params.get("seed") if params else None
    options = params.get("options") if params else None
    
    obs, info = env.reset(seed=seed, options=options)
    
    # Convert Pydantic obs to dict for JSON serialization
    return {
        "observation": obs.model_dump() if hasattr(obs, "model_dump") else obs,
        "info": info
    }

@app.post("/step", response_model=dict)
async def step_env(action: dict[str, Any]):
    """
    Execute a step in the environment.
    Input: EmailAction dictionary
    Returns: (observation, reward, terminated, truncated, info)
    """
    try:
        # Validate action matches EmailAction schema
        # If the action is already a dict that matches EmailAction, InboxEnv handles it.
        obs, reward, terminated, truncated, info = env.step(action)
        
        return {
            "observation": obs.model_dump() if hasattr(obs, "model_dump") else obs,
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "info": info
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/state", response_model=EpisodeState)
async def get_state():
    """
    Get the current episode state.
    """
    try:
        return env.state()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # In Hugging Face Spaces, we run on the port defined by ENV or 7860
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
