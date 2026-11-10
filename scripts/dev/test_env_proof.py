from pathlib import Path
import os
from dotenv import dotenv_values
from apify_client import ApifyClient

env_path = Path(".env").resolve()
print(f"Path('.env').exists(): {env_path.exists()}")
if not env_path.exists():
    raise SystemExit("Error: .env missing")

file_keys = dotenv_values(env_path)
for k in ["APIFY_TOKEN", "EXA_API_KEY", "GEMINI_API_KEY"]:
    in_file = k in file_keys and bool(file_keys[k].strip())
    in_os = k in os.environ and bool(os.environ[k].strip())
    src = ".env file" if in_file else ("OS environment" if in_os else "MISSING")
    print(f"Key '{k}': source = {src}")

token = file_keys.get("APIFY_TOKEN") or os.environ.get("APIFY_TOKEN")
client = ApifyClient(token)
user_obj = client.user().get()
print(f"Apify Authenticated Username: {user_obj.username}")
