import os
import json
import time
from pathlib import Path
from dotenv import dotenv_values
from apify_client import ApifyClient

env_vals = dotenv_values(Path(".env").resolve())
client = ApifyClient(env_vals.get("APIFY_TOKEN"))

def test_actor(actor_id, inp):
    print("=" * 60)
    print(f"Testing Actor: {actor_id}")
    print(f"Input: {json.dumps(inp, ensure_ascii=False)}")
    t0 = time.time()
    try:
        run = client.actor(actor_id).call(run_input=inp)
        dur = round(time.time() - t0, 1)
        run_id = getattr(run, "id", None)
        status = getattr(run, "status", None)
        dataset_id = getattr(run, "default_dataset_id", None)
        items = list(client.dataset(dataset_id).iterate_items(limit=5)) if dataset_id else []
        print(f"Result: {status} in {dur}s | Run ID: {run_id} | Items: {len(items)}")
        if items:
            print(f"Sample item: {list(items[0].keys())}")
            sample_text = str(items[0])[:120]
            print(f"Text snippet: {sample_text}...")
        return run_id, status, len(items)
    except Exception as e:
        dur = round(time.time() - t0, 1)
        print(f"FAILED in {dur}s: {e}")
        return None, "FAILED", 0

# Check schemas
for aid in ["truefetch/telegram-channel-message", "tri_angle/telegram-scraper", "hypebridge/blind-post-scraper"]:
    print("-" * 50)
    try:
        act = client.actor(aid).get()
        print(f"Actor {aid}: owner={act.username}, isPublic={act.is_public}")
        build_id = act.tagged_builds.get("latest").build_id if act.tagged_builds else None
        if build_id:
            b = client.build(build_id).get()
            s = json.loads(b.input_schema) if b.input_schema else {}
            print(f"Properties: {list(s.get('properties', {}).keys())}")
            print(f"Required: {s.get('required', [])}")
    except Exception as e:
        print(f"Error checking {aid}: {e}")

# Now test each with maxItems=5
test_actor("truefetch/telegram-channel-message", {
    "telegram_url": "https://t.me/s/krx_market_news",
    "download_medias": "False",
    "max_results": 5,
    "start_date": "2026-10-30"
})

test_actor("tri_angle/telegram-scraper", {
    "profiles": ["https://t.me/krx_market_news"],
    "collectMessages": True,
    "scrapeLastNDays": 5
})

test_actor("hypebridge/blind-post-scraper", {
    "searchQuery": "삼성전자",
    "maxItems": 5
})
