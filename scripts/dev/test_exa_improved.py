from pathlib import Path
from dotenv import dotenv_values
from datetime import datetime, timezone, timedelta
from exa_py import Exa

env_vals = dotenv_values(Path(".env").resolve())
exa = Exa(api_key=env_vals.get("EXA_API_KEY"))

window_start = (datetime.now(timezone.utc) - timedelta(days=10)).strftime("%Y-%m-%dT%H:%M:%S.000Z")

test_tickers = [
    ("9988.HK", "Alibaba Group", "阿里巴巴", "9988"),
    ("3690.HK", "Meituan", "美團", "3690"),
    ("035420.KS", "NAVER", "네이버", "035420")
]

for sym, name, local, code in test_tickers:
    query = f"{name} {local} {code} stock news"
    print("=" * 60)
    print(f"Testing Exa query for {sym}: '{query}'")
    try:
        res = exa.search_and_contents(
            query=query,
            type="neural",
            category="news",
            start_published_date=window_start,
            num_results=10,
            highlights={"num_sentences": 2}
        )
        print(f"Returned {len(res.results)} results for {sym}!")
        for r in res.results[:3]:
            print(f"  Title: {r.title}")
            print(f"  Date:  {r.published_date}")
            print(f"  URL:   {r.url}")
    except Exception as e:
        print(f"Error for {sym}: {e}")
