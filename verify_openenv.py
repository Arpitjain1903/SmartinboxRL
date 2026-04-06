import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

from environment.inbox_env import InboxEnv
from models import EmailObservation, EmailReward, EpisodeState

def test_env():
    print("Initializing InboxEnv...")
    env = InboxEnv(difficulty="easy")
    
    print("Calling reset()...")
    obs, info = env.reset(seed=42)
    
    print("Checking reset() returns...")
    if not isinstance(obs, EmailObservation):
        print(f"FAILED: obs is {type(obs)}, expected EmailObservation")
        return
    print("SUCCESS: obs is EmailObservation")
    
    print("\nCalling state()...")
    state = env.state()
    if not isinstance(state, EpisodeState):
        print(f"FAILED: state is {type(state)}, expected EpisodeState")
        return
    print("SUCCESS: state is EpisodeState")
    print(f"  Step: {state.step}")
    print(f"  Difficulty: {state.difficulty}")
    
    print("\nCalling step()...")
    action = {
        "intents": ["meeting_request"],
        "priority": "medium",
        "action": "reply",
        "response": "I am available tomorrow at 10 AM."
    }
    obs, reward, terminated, truncated, info = env.step(action)
    
    print("Checking step() returns...")
    if not (isinstance(obs, EmailObservation) or obs == {}):
        print(f"FAILED: obs is {type(obs)}, expected EmailObservation or {{}}")
        return
    
    if "reward_object" not in info or not isinstance(info["reward_object"], EmailReward):
        print(f"FAILED: reward_object missing or wrong type in info")
        return
    
    print("SUCCESS: step returns validated")
    print(f"  Reward: {reward}")
    print(f"  Intent Score: {info['reward_object'].intent_score}")
    
    print("\nVerification Successful!")

if __name__ == "__main__":
    try:
        test_env()
    except Exception as e:
        print(f"\nVerification FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
