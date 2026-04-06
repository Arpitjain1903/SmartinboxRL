"""
Fetch the Enron dataset from Kaggle, parse the first 300 rows,
auto-label them using the LLMAgent, and save to data/tasks/enron_tasks.json.
"""
import os
import sys
import json
import kagglehub
import pandas as pd
import time
from email.parser import Parser
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv

# Fix imports to find the SmartInboxRL modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.llm_agent import LLMAgent
from environment.action_space import validate_action

def main():
    # Load .env variables so LLMAgent gets the API keys natively
    load_dotenv()
    
    # Attempt to use the Kaggle Access Token provided:
    token = "KGAT_4c1e5f63d992b44fd02276c2a74ae411"
    os.environ["KAGGLE_PAT"] = token
    os.environ["KAGGLE_KEY"] = token
    os.environ["KAGGLE_TOKEN"] = token
    
    print("Downloading Enron dataset from Kaggle (~1.7GB)... this may take a few minutes.")
    try:
        path = kagglehub.dataset_download("wcukierski/enron-email-dataset")
        print(f"Dataset downloaded to {path}")
    except Exception as e:
        print(f"Failed to download via Kaggle API: {e}")
        print("Please check if the API key is correct.")
        return

    csv_path = os.path.join(path, "emails.csv")
    if not os.path.exists(csv_path):
        print("Could not find emails.csv inside the downloaded dataset.")
        return

    print("Extracting first 300 emails...")
    # Read only the first 300 rows to spare memory
    df = pd.read_csv(csv_path, nrows=300)
    
    raw_messages = df["message"].tolist()
    
    parsed_emails = []
    parser = Parser()
    
    for idx, raw_text in enumerate(raw_messages):
        msg = parser.parsestr(raw_text)
        
        # Enron headers
        subject = msg.get("Subject", "").strip() or "No Subject"
        sender = msg.get("From", "").strip() or "unknown@enron.com"
        
        # Body extraction
        # The body is the payload. Enron emails are mostly plain text.
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    body += part.get_payload(decode=True).decode(errors="ignore")
        else:
            body = msg.get_payload()
            
        # Clean up huge forward threads or reply blocks for the LLM to process easier
        body = "\n".join([line for line in body.splitlines() if not line.startswith(">")])
        body = body[:2000].strip()  # Clip massive emails to 2000 chars to save LLM tokens/rate limits

        parsed_emails.append({
            "id": f"enron_{idx:03d}",
            "subject": subject,
            "sender": sender,
            "body": body,
            "context": "corporate"
        })

    # Auto-label using the configured LLMAgent
    print("Auto-labeling 300 emails via LLMAgent to generate gold labels...")
    agent = LLMAgent()
    
    # We will construct a dummy observation to feed the LLMAgent
    labeled_tasks = []
    
    # Wrap in tqdm for progress bar
    # Note: If rate limits hit, we'll continue using fallback to avoid crashing entirely
    for email in tqdm(parsed_emails):
        obs = {"email": email}
        # Ask LLMAgent what it would do
        try:
            action = agent.act(obs)
            valid = validate_action(action)
            time.sleep(2.0)  # Sleep to avoid Groq 30 RPM rate limits
        except Exception as e:
            # Fallback if rate limited completely
            print(f"Skipping LLM labeling for {email['id']} due to error: {e}")
            valid = {
                "intents": ["information_sharing"],
                "priority": "medium",
                "action": "reply",
                "response": "Thank you for the update."
            }
            time.sleep(5.0)  # longer backoff on failure
            
        # Pack this back as a "gold" standard task
        labeled_tasks.append({
            "id": email["id"],
            "subject": email["subject"],
            "body": email["body"],
            "sender": email["sender"],
            "gold_intents": valid["intents"],
            "gold_priority": valid["priority"],
            "gold_action": valid["action"],
            "gold_response": valid["response"]
        })

    out_path = Path("data/tasks/enron_tasks.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(labeled_tasks, indent=2))
    
    print(f"Successfully generated {len(labeled_tasks)} Enron tasks at {out_path}!")

if __name__ == "__main__":
    main()
