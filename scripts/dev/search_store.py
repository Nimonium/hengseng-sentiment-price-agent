from apify_client import ApifyClient
from dotenv import dotenv_values
from pathlib import Path

env_vals = dotenv_values(Path(".env").resolve())
client = ApifyClient(env_vals.get("APIFY_TOKEN"))

search_terms = ["lihkg", "aastocks", "naver", "telegram", "daum", "futu", "moomoo", "korea stock", "hong kong stock"]

for term in search_terms:
    print("=" * 60)
    print(f"Searching Apify Store for: '{term}'")
    try:
        res = client.store().list(search=term, limit=5)
        items = getattr(res, "items", res)
        for it in items:
            username = getattr(it, "username", None) or (it.get("username") if isinstance(it, dict) else "")
            name = getattr(it, "name", None) or (it.get("name") if isinstance(it, dict) else "")
            title = getattr(it, "title", None) or (it.get("title") if isinstance(it, dict) else "")
            pricing = getattr(it, "pricing_infos", None) or getattr(it, "pricingInfos", None) or (it.get("pricingInfos") if isinstance(it, dict) else "")
            print(f"  {username}/{name} - {title}")
    except Exception as e:
        print(f"Error: {e}")

