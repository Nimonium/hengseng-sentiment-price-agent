from apify_client import ApifyClient
from dotenv import dotenv_values
from pathlib import Path
import json

env_vals = dotenv_values(Path(".env").resolve())
client = ApifyClient(env_vals.get("APIFY_TOKEN"))

actors_to_test = [
    "claude_code_reviewer/lihkg-scraper",
    "bovi/aastocks-scraper",
    "hgservices/naver-stock-discussion-scraper",
    "ardent_fork/daum-news",
    "tri_angle/telegram-scraper",
    "truefetch/telegram-channel-message"
]

for act_id in actors_to_test:
    print("=" * 60)
    print("Actor:", act_id)
    try:
        act = client.actor(act_id).get()
        title = getattr(act, "title", "")
        desc = getattr(act, "description", "")
        readme = getattr(act, "readme_summary", "")
        print(f"Title: {title}")
        print(f"Desc: {desc}")
        # Check versions
        versions = client.actor(act_id).versions().list()
        items = getattr(versions, "items", versions)
        for v in items:
            v_num = getattr(v, "version_number", "")
            schema = getattr(v, "input_schema", {})
            if schema:
                print(f"Schema properties (v{v_num}):", list(schema.get("properties", {}).keys()) if isinstance(schema, dict) else "raw")
                if isinstance(schema, dict):
                    print("Full schema:", json.dumps(schema.get("properties", {}), indent=2)[:300])
    except Exception as e:
        print(f"Error: {e}")

