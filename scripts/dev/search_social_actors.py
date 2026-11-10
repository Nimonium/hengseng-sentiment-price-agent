import os
from pathlib import Path
from dotenv import dotenv_values
from apify_client import ApifyClient

env_vals = dotenv_values(Path(".env").resolve())
client = ApifyClient(env_vals.get("APIFY_TOKEN"))

queries = [
    "futu",
    "moomoo",
    "etnet",
    "aastocks",
    "telegram",
    "blind",
    "daum",
    "lihkg",
    "naver"
]

for q in queries:
    print("=" * 60)
    print(f"SEARCH QUERY: '{q}'")
    try:
        res = client.store().list(search=q, limit=6)
        items = getattr(res, "items", res)
        if not items:
            print("  No actors found.")
        for it in items:
            uname = getattr(it, "username", getattr(it, "user_id", "unknown"))
            name = getattr(it, "name", "")
            title = getattr(it, "title", "")
            pricing = getattr(it, "pricing_model", "")
            desc = (getattr(it, "description", "") or "")[:70]
            print(f"  ID: {uname}/{name}")
            print(f"     Title: {title}")
            print(f"     Pricing: {pricing} | Desc: {desc}")
    except Exception as e:
        print(f"  Error searching for '{q}': {e}")
