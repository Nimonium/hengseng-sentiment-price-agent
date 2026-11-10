import os
import json
from pathlib import Path
from dotenv import dotenv_values
from datetime import datetime, timezone, timedelta
from exa_py import Exa

env_vals = dotenv_values(Path(".env").resolve())
exa_key = env_vals.get("EXA_API_KEY")
exa = Exa(api_key=exa_key)

tickers = [
    ("0700.HK", "Tencent", "騰訊"),
    ("9988.HK", "Alibaba", "阿里巴巴"),
    ("0005.HK", "HSBC", "滙豐"),
    ("005930.KS", "Samsung Electronics", "삼성전자"),
    ("000660.KS", "SK Hynix", "SK하이닉스")
]

start_date = (datetime.now(timezone.utc) - timedelta(days=10)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
print(f"Testing Exa searches (start_published_date={start_date})...")

request_count = 0
for symbol, name, local in tickers:
    query = f"{name} {local} stock financial news earnings"
    print(f"\nQuerying: '{query}' for {symbol}...")
    try:
        res = exa.search_and_contents(
            query=query,
            type="neural",
            category="news",
            start_published_date=start_date,
            num_results=3,
            highlights={"num_sentences": 2}
        )
        request_count += 1
        print(f"Results returned: {len(res.results)}")
        for i, item in enumerate(res.results[:3]):
            print(f"  [{i+1}] Title: {item.title}")
            print(f"      URL: {item.url}")
            print(f"      Date: {item.published_date}")
            hl = item.highlights[0] if getattr(item, "highlights", None) else ""
            print(f"      Highlight: {hl[:100]}...")
    except Exception as e:
        print(f"Error querying Exa: {e}")

print(f"\nTotal Exa requests executed: {request_count}")

