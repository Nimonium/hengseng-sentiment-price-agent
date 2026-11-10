import os
import json
from pathlib import Path
from dotenv import load_dotenv
from apify_client import ApifyClient

load_dotenv(Path(".env").resolve())
client = ApifyClient(os.getenv("APIFY_TOKEN"))

actor_ids = [
    "claude_code_reviewer/lihkg-scraper",
    "bovi/aastocks-scraper",
    "hgservices/naver-stock-discussion-scraper",
    "ardent_fork/daum-news",
    "tri_angle/telegram-scraper",
    "truefetch/telegram-channel-message"
]

for aid in actor_ids:
    print("=" * 60)
    print(f"ACTOR: {aid}")
    actor = client.actor(aid).get()
    if not actor:
        print("  Not found!")
        continue
    print(f"  Owner: {actor.username}")
    print(f"  Name: {actor.name}")
    print(f"  Is Public: {actor.is_public}")
    pricing_str = "Unknown"
    if actor.pricing_infos:
        pricing_str = actor.pricing_infos[0].pricing_model
    print(f"  Pricing: {pricing_str}")
    
    # Try fetching default build or version to get input_schema
    try:
        build_id = actor.tagged_builds.get("latest") if hasattr(actor, "tagged_builds") and actor.tagged_builds else None
        print(f"  Tagged latest build: {build_id}")
    except Exception as e:
        print(f"  Build info error: {e}")
        
    try:
        # Check versions
        versions = client.actor(aid).versions().list().items
        if versions:
            latest_v = versions[0]
            print(f"  Version: {latest_v.get('versionNumber')}")
            schema = latest_v.get("inputSchema")
            if schema:
                print(f"  Input Schema properties: {list(schema.get('properties', {}).keys())}")
                print(f"  Input Schema required: {schema.get('required', [])}")
            else:
                print("  Input Schema: None in version object")
    except Exception as e:
        print(f"  Version fetch error: {e}")

