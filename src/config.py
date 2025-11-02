import os
from dotenv import load_dotenv

load_dotenv()

def get_api_key() -> str:
    key = os.getenv("HOPPR_API_KEY")
    if not key:
        raise RuntimeError("Missing HOPPR_API_KEY (set it in .env).")
    return key
