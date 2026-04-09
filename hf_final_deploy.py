
"""Targeted upload to HF Spaces including new API files."""

import os
import time
from huggingface_hub import HfApi
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env

# Load token from environment to avoid committing secrets
TOKEN = os.environ.get("HF_TOKEN")
if not TOKEN:
    print("Error: HF_TOKEN environment variable not set.")
    exit(1)

REPO_ID = "arpit-jain/SmartInboxRL"
ROOT = Path(__file__).parent
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds (doubles on each retry)

# Files to specifically include or skip
SKIP_DIRS = {".git", ".venv", "venv", "env", "__pycache__", ".gemini", ".gsd", ".agent", ".cursor", ".vscode"}
SKIP_SUFFIXES = {".pyc", ".db", ".zip", ".npy"}
SKIP_FILES = {".env", "hf_final_deploy.py", "hf_upload.py"}

api = HfApi(token=TOKEN)

files_to_upload = []
for f in ROOT.rglob("*"):
    if not f.is_file():
        continue
    skip = False
    for part in f.parts:
        if part in SKIP_DIRS:
            skip = True
            break
    if skip:
        continue
    if f.suffix in SKIP_SUFFIXES or f.name in SKIP_FILES:
        continue
    files_to_upload.append(f)

print(f"Uploading {len(files_to_upload)} files...")
failed_files = []

for i, f in enumerate(files_to_upload, 1):
    repo_path = f.relative_to(ROOT).as_posix()
    print(f"[{i}/{len(files_to_upload)}] {repo_path} ... ", end="", flush=True)

    success = False
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            api.upload_file(
                path_or_fileobj=str(f),
                path_in_repo=repo_path,
                repo_id=REPO_ID,
                repo_type="space",
                token=TOKEN,
                commit_message=f"update: {repo_path}",
            )
            print("OK")
            success = True
            break
        except Exception as e:
            if attempt < MAX_RETRIES:
                wait = RETRY_DELAY * (2 ** (attempt - 1))
                print(f"RETRY ({attempt}/{MAX_RETRIES}) in {wait}s... ", end="", flush=True)
                time.sleep(wait)
            else:
                print(f"FAILED after {MAX_RETRIES} attempts: {type(e).__name__}")
                failed_files.append(repo_path)

if failed_files:
    print(f"\n⚠️  {len(failed_files)} files failed to upload:")
    for ff in failed_files:
        print(f"   - {ff}")
    print("\nRe-run the script to retry only the failed files, or upload them manually.")
else:
    print(f"\nDeployment Complete! URL: https://huggingface.co/spaces/{REPO_ID}")
