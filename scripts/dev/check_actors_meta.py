import os
import json
from pathlib import Path
from dotenv import load_dotenv
from apify_client import ApifyClient

load_dotenv(Path(".env").resolve())
token = os.getenv("APIFY_TOKEN")
client = ApifyClient(token)

user = client.user().get()
username = getattr(user, "username", None) or (user.get("username") if isinstance(user, dict) else str(user))
print(f"APIFY User: {username}")

actor_ids = [
    "claude_code_reviewer/lihkg-scraper",
    "bovi/aastocks-scraper",
    "hgservices/naver-stock-discussion-scraper",
    "ardent_fork/daum-news",
    "tri_angle/telegram-scraper",
    "truefetch/telegram-channel-message"
]

def to_dict(obj):
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)

for aid in actor_ids:
    print("-" * 50)
    try:
        actor_res = client.actor(aid).get()
        a = to_dict(actor_res)
        if a:
            print(f"Actor: {aid}")
            print(f"  Owner: {a.get('username')}")
            print(f"  Name: {a.get('name')}")
            print(f"  isPublic: {a.get('isPublic')}")
            print(f"  Pricing: {a.get('pricingModel')}")
            stats = to_dict(a.get("stats", {}))
            print(f"  Stats: totalRuns={stats.get('totalRuns') if isinstance(stats, dict) else stats}")
            example = a.get("exampleRunInput", {})
            print(f"  Example Input: {json.dumps(to_dict(example), ensure_ascii=False)}")
        else:
            print(f"Actor {aid} returned None (Does not exist)")
    except Exception as e:
        print(f"Actor {aid} failed: {e}")
