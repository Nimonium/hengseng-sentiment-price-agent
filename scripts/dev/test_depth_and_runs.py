import os
import json
import time
from pathlib import Path
from dotenv import dotenv_values
from datetime import datetime, timezone, timedelta
import pandas as pd
from apify_client import ApifyClient

env_vals = dotenv_values(Path(".env").resolve())
client = ApifyClient(env_vals.get("APIFY_TOKEN"))

now = datetime.now(timezone.utc)
start_10d = now - timedelta(days=10)
date_from_str = start_10d.strftime("%Y-%m-%d")

print(f"Testing depth with date_from: {date_from_str} to {now.strftime('%Y-%m-%d')}")

# Test 1: LIHKG search mode with dateFrom="10d"
print("\n--- Testing LIHKG Depth ---")
lihkg_inp = {
    "mode": "search",
    "keywords": ["0700", "騰訊", "9988", "阿里巴巴", "0005", "滙豐", "1299", "友邦", "3690", "美團"],
    "maxItems": 50,
    "dateFrom": "10d",
    "proxyConfiguration": {"useApifyProxy": True}
}
run_lihkg = client.actor("claude_code_reviewer/lihkg-scraper").call(run_input=lihkg_inp)
print(f"LIHKG Run ID: {run_lihkg.id}, status: {run_lihkg.status}")
items_lihkg = list(client.dataset(run_lihkg.default_dataset_id).iterate_items())
print(f"LIHKG Items returned: {len(items_lihkg)}")
if items_lihkg:
    df_li = pd.DataFrame(items_lihkg)
    for c in ["created_at", "last_reply_at"]:
        if c in df_li.columns:
            df_li[c] = pd.to_datetime(df_li[c], errors="coerce", utc=True)
            print(f"LIHKG {c} range: {df_li[c].min()} to {df_li[c].max()}")

# Test 2: AAStocks with maxResults=200
print("\n--- Testing AAStocks Depth ---")
aastocks_inp = {
    "feed": "latest-news",
    "language": "tc",
    "maxResults": 150,
    "proxyConfiguration": {"useApifyProxy": True, "apifyProxyGroups": ["RESIDENTIAL"]}
}
run_aa = client.actor("bovi/aastocks-scraper").call(run_input=aastocks_inp)
print(f"AAStocks Run ID: {run_aa.id}, status: {run_aa.status}")
items_aa = list(client.dataset(run_aa.default_dataset_id).iterate_items())
print(f"AAStocks Items returned: {len(items_aa)}")

# Test 3: Naver discussion with dateFrom and maxResults=200
print("\n--- Testing Naver Discussion Depth ---")
naver_inp = {
    "stocks": ["005930", "000660", "035420", "005380", "035720"],
    "dateFrom": date_from_str,
    "maxResults": 200
}
run_nv = client.actor("hgservices/naver-stock-discussion-scraper").call(run_input=naver_inp)
print(f"Naver Run ID: {run_nv.id}, status: {run_nv.status}")
items_nv = list(client.dataset(run_nv.default_dataset_id).iterate_items())
print(f"Naver Items returned: {len(items_nv)}")
if items_nv:
    df_nv = pd.DataFrame(items_nv)
    if "writtenAt" in df_nv.columns:
        df_nv["writtenAt_dt"] = pd.to_datetime(df_nv["writtenAt"], errors="coerce", utc=True)
        print(f"Naver writtenAt range: {df_nv['writtenAt_dt'].min()} to {df_nv['writtenAt_dt'].max()}")

# Test 4: Daum News with maxItems=100
print("\n--- Testing Daum News Depth ---")
daum_inp = {
    "sections": ["finance"],
    "maxItems": 100,
    "fetchArticles": False
}
run_dm = client.actor("ardent_fork/daum-news").call(run_input=daum_inp)
print(f"Daum Run ID: {run_dm.id}, status: {run_dm.status}")
items_dm = list(client.dataset(run_dm.default_dataset_id).iterate_items())
print(f"Daum Items returned: {len(items_dm)}")

