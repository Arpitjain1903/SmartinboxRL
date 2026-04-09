
"""Targeted upload to HF Spaces including new API files."""

import os
from huggingface_hub import HfApi
from pathlib import Path
from dotenv import load_dotenv

load_dotenv() # Load variables from .env

# Load token from environment to avoid committing secrets
TOKEN = os.environ.get("HF_TOKEN")
if not TOKEN:
    print("Error: HF_TOKEN environment variable not set.")
    exit(1)
REPO_ID = "arpit-jain/SmartInboxRL"
ROOT = Path(__file__).parent

# Files to specifically include or skip
SKIP_DIRS = {".git", ".venv", "venv", "env", "__pycache__", ".gemini", ".gsd", ".agent", ".cursor", ".vscode"}
SKIP_SUFFIXES = {".pyc", ".db", ".zip", ".npy"}
SKIP_FILES = {".env", "hf_final_deploy.py", "hf_upload.py"}

api = HfApi(token=TOKEN)

files_to_upload = []
for f in ROOT.rglob("*"):
    if not f.is_file(): continue
    skip = False
    for part in f.parts:
        if part in SKIP_DIRS: skip = True; break
    if skip: continue
    if f.suffix in SKIP_SUFFIXES or f.name in SKIP_FILES: continue
    files_to_upload.append(f)

print(f"Uploading {len(files_to_upload)} files...")
for i, f in enumerate(files_to_upload, 1):
    repo_path = f.relative_to(ROOT).as_posix()
    print(f"[{i}/{len(files_to_upload)}] {repo_path} ... ", end="", flush=True)
    api.upload_file(
        path_or_fileobj=str(f),
        path_in_repo=repo_path,
        repo_id=REPO_ID,
        repo_type="space",
        token=TOKEN,
        commit_message=f"feat: add OpenEnv API compliance ({repo_path})"
    )
    print("OK")

print(f"\nDeployment Complete! URL: https://huggingface.co/spaces/{REPO_ID}")
