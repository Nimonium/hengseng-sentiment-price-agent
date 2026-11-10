import os
import json
import time
from pathlib import Path
from dotenv import dotenv_values
from apify_client import ApifyClient

env_vals = dotenv_values(Path(".env").resolve())
token = env_vals.get("APIFY_TOKEN")
client = ApifyClient(token)

test_configs = [
    {
        "actor_id": "claude_code_reviewer/lihkg-scraper",
        "input": {
            "mode": "thread-list",
            "keywords": ["騰訊", "0700"],
            "maxItems": 5,
            "dateFrom": "2026-10-30",
            "proxyConfiguration": {"useApifyProxy": True}
        }
    },
    {
        "actor_id": "bovi/aastocks-scraper",
        "input": {
            "feed": "latest-news",
            "language": "tc",
            "maxResults": 5,
            "proxyConfiguration": {"useApifyProxy": True, "apifyProxyGroups": ["RESIDENTIAL"]}
        }
    },
    {
        "actor_id": "hgservices/naver-stock-discussion-scraper",
        "input": {
            "stocks": ["005930"],
            "dateFrom": "2026-10-30",
            "maxResults": 5
        }
    },
    {
        "actor_id": "ardent_fork/daum-news",
        "input": {
            "sections": ["finance"],
            "maxItems": 5,
            "fetchArticles": False
        }
    }
]

print("Starting test runs (maxItems=5) for Apify actors...")

for cfg in test_configs:
    aid = cfg["actor_id"]
    inp = cfg["input"]
    print("=" * 60)
    print(f"Testing Actor: {aid}")
    print(f"Input: {json.dumps(inp, ensure_ascii=False)}")
    t0 = time.time()
    try:
        # Call actor synchronously
        run = client.actor(aid).call(run_input=inp)
        dur = round(time.time() - t0, 1)
        run_id = getattr(run, "id", None) or (run.get("id") if isinstance(run, dict) else None)
        status = getattr(run, "status", None) or (run.get("status") if isinstance(run, dict) else None)
        dataset_id = getattr(run, "default_dataset_id", None) or (run.get("defaultDatasetId") if isinstance(run, dict) else None)
        usage_usd = getattr(run, "usage_total_usd", 0.0) or (run.get("usageTotalUsd", 0.0) if isinstance(run, dict) else 0.0)
        
        # Get items count
        items = list(client.dataset(dataset_id).iterate_items(limit=10)) if dataset_id else []
        print(f"Run Finished in {dur}s!")
        print(f"  Run ID: {run_id}")
        print(f"  Status: {status}")
        print(f"  Item Count: {len(items)}")
        print(f"  Cost (USD): ${float(usage_usd):.4f}")
        if items:
            sample = items[0]
            print(f"  Sample Item Keys: {list(sample.keys())}")
            sample_title = sample.get("title") or sample.get("headline") or sample.get("text") or str(sample)[:100]
            print(f"  Sample text: {sample_title[:100]}")
    except Exception as e:
        dur = round(time.time() - t0, 1)
        print(f"Run FAILED in {dur}s: {e}")
