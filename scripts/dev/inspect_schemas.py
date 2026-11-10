from apify_client import ApifyClient
from dotenv import dotenv_values
from pathlib import Path
import json

env_vals = dotenv_values(Path(".env").resolve())
token = env_vals.get("APIFY_TOKEN")
client = ApifyClient(token)

actors_to_check = [
    "claude_code_reviewer/lihkg-scraper",
    "bovi/aastocks-scraper",
    "hgservices/naver-stock-discussion-scraper",
    "tri_angle/telegram-scraper",
    "ardent_fork/daum-news",
]

for act_id in actors_to_check:
    print("=" * 60)
    print("Actor:", act_id)
    act = client.actor(act_id).get()
    
    # Try getting input schema directly from versions
    try:
        versions = client.actor(act_id).versions().list()
        items = versions.items if hasattr(versions, "items") else versions
        for v in items:
            v_num = getattr(v, "version_number", None) or (v.get("versionNumber") if isinstance(v, dict) else "")
            schema = getattr(v, "input_schema", None) or (v.get("inputSchema") if isinstance(v, dict) else None)
            if schema:
                print(f"Version {v_num} Input Schema:")
                props = schema.get("properties", {}) if isinstance(schema, dict) else json.loads(schema).get("properties", {})
                for p_name, p_val in props.items():
                    print(f"  {p_name}: {p_val.get('type')} - {p_val.get('title') or p_val.get('description', '')[:50]}")
                break
    except Exception as e:
        print(f"Could not load versions: {e}")

