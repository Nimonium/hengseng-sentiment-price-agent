from apify_client import ApifyClient
from dotenv import dotenv_values
from pathlib import Path
import json

env_vals = dotenv_values(Path(".env").resolve())
token = env_vals.get("APIFY_TOKEN")
client = ApifyClient(token)

user = client.user().get()
username = getattr(user, "username", None) or getattr(user, "id", None) or (user.get("username") if isinstance(user, dict) else str(user))
print(f"Authenticated as Apify user: {username}")

actors_to_check = [
    "claude_code_reviewer/lihkg-scraper",
    "bovi/aastocks-scraper",
    "hgservices/naver-stock-discussion-scraper",
    "tri_angle/telegram-scraper",
    "ardent_fork/daum-news",
]

for act_id in actors_to_check:
    print("-" * 60)
    try:
        act = client.actor(act_id).get()
        if act:
            # Handle both dict and Pydantic object
            owner = getattr(act, "username", None) or (act.get("username") if isinstance(act, dict) else None)
            name = getattr(act, "name", None) or (act.get("name") if isinstance(act, dict) else None)
            is_pub = getattr(act, "is_public", None) or getattr(act, "isPublic", None) or (act.get("isPublic") if isinstance(act, dict) else None)
            pricing_list = getattr(act, "pricing_infos", None) or getattr(act, "pricingInfos", None) or (act.get("pricingInfos") if isinstance(act, dict) else None) or []
            pricing = pricing_list[0].pricing_model if pricing_list and hasattr(pricing_list[0], "pricing_model") else (pricing_list[0].get("pricingModel") if pricing_list and isinstance(pricing_list[0], dict) else "PAY_PER_EVENT")
            
            print(f"Actor: {act_id}")
            print(f"  Owner:    {owner}")
            print(f"  Name:     {name}")
            print(f"  isPublic: {is_pub}")
            print(f"  Pricing:  {pricing}")
            
            build_id = getattr(act, "default_build_id", None) or getattr(act, "defaultBuildId", None) or (act.get("defaultBuildId") if isinstance(act, dict) else None)
            if build_id:
                try:
                    build = client.build(build_id).get()
                    schema_str = getattr(build, "input_schema", None) or getattr(build, "inputSchema", None) or (build.get("inputSchema") if isinstance(build, dict) else "{}")
                    schema_obj = json.loads(schema_str) if isinstance(schema_str, str) else schema_str
                    properties = getattr(schema_obj, "properties", {}) if hasattr(schema_obj, "properties") else (schema_obj.get("properties", {}) if isinstance(schema_obj, dict) else {})
                    print(f"  Input Properties: {list(properties.keys())}")
                except Exception as ex:
                    print(f"  Could not load build schema: {ex}")
        else:
            print(f"Actor {act_id} not found!")
    except Exception as e:
        print(f"Error loading {act_id}: {e}")
