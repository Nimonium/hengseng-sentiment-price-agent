"""
Exa.com News Ingestion with Deduplication, Strict 10-Calendar-Day Window Filtering, and Local Caching.
Calls Exa API when EXA_API_KEY is provided. RSS is NOT substituted unless explicitly permitted by --allow-fallback.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
import urllib.parse
import xml.etree.ElementTree as ET
import pandas as pd
import requests

from src.config import TickerConfig

logger = logging.getLogger(__name__)


class ExaNewsIngestion:
    def __init__(self, cache_dir: Path, api_key: Optional[str] = None, calendar_days: int = 10):
        self.cache_dir = cache_dir / "news"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.api_key = api_key
        self.calendar_days = calendar_days
        self.exa_client = None
        self.request_count = 0
        self.sample_results: List[Dict[str, Any]] = []

        if self.api_key:
            try:
                from exa_py import Exa
                self.exa_client = Exa(api_key=self.api_key)
                logger.info("Initialized Exa client with API key")
            except Exception as e:
                logger.warning(f"Failed to initialize Exa client: {e}")

    def fetch_news(
        self,
        ticker: TickerConfig,
        days_lookback: int = 10,
        use_cache: bool = True,
        max_results: int = 25,
        allow_fallback: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch news for a ticker over the last N days with deduplication and caching.
        Enforces strict 10-calendar-day window filtering.
        """
        clean_name = ticker.symbol.replace(".", "_")
        cache_file = self.cache_dir / f"{clean_name}_news.parquet"
        cache_json = self.cache_dir / f"{clean_name}_news.json"

        # 1. Check local cache
        if use_cache and cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                window_start = datetime.now(timezone.utc) - timedelta(days=days_lookback)
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                df = df[df["timestamp"] >= window_start].copy()
                logger.info(f"Loaded {len(df)} news items from cache for {ticker.symbol}")
                return df
            except Exception as e:
                logger.warning(f"Error loading news cache: {e}")

        # 2. Query Exa or Fallback
        items: List[Dict[str, Any]] = []
        if self.exa_client:
            items = self._fetch_via_exa(ticker, days_lookback, max_results)

        if not items:
            if not self.exa_client and not allow_fallback:
                raise RuntimeError(
                    f"EXA_API_KEY is not configured for {ticker.symbol} and allow_fallback=False. "
                    "RSS fallback is prohibited unless --allow-fallback is specified."
                )
            if allow_fallback:
                logger.info(f"Using live financial news RSS for {ticker.symbol} (allow_fallback=True)")
                items = self._fetch_via_rss(ticker, days_lookback, max_results)

        # 3. Deduplicate
        deduped = self._deduplicate(items)

        if not deduped:
            logger.warning(f"No news items retrieved for {ticker.symbol}")
            return pd.DataFrame(columns=["id", "ticker", "source", "timestamp", "title", "url", "text"])

        df = pd.DataFrame(deduped)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        # 4. Strict 10-calendar-day window filter
        window_start = datetime.now(timezone.utc) - timedelta(days=days_lookback)
        df = df[df["timestamp"] >= window_start].copy()
        df = df.sort_values("timestamp")

        # 5. Cache
        try:
            df.to_parquet(cache_file)
            with open(cache_json, "w", encoding="utf-8") as f:
                json.dump(deduped, f, ensure_ascii=False, indent=2)
            logger.info(f"Cached {len(df)} news items for {ticker.symbol} -> {cache_file}")
        except Exception as e:
            logger.error(f"Failed to cache news for {ticker.symbol}: {e}")

        return df

    def _fetch_via_exa(self, ticker: TickerConfig, days_lookback: int, max_results: int) -> List[Dict[str, Any]]:
        """Perform neural news search using Exa API."""
        query = f"{ticker.name} {ticker.local_name} {ticker.clean_code} stock news"
        start_date = (datetime.now(timezone.utc) - timedelta(days=days_lookback)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        self.request_count += 1
        try:
            logger.info(f"Querying Exa API for '{query}' (Published after: {start_date})...")
            fetch_n = max(15, max_results)
            response = self.exa_client.search_and_contents(
                query=query,
                type="neural",
                category="news",
                start_published_date=start_date,
                num_results=fetch_n,
                highlights={"num_sentences": 2},
            )
            results = []
            for item in response.results:
                pub_date = item.published_date or datetime.now(timezone.utc).isoformat()
                dt_utc = pd.to_datetime(pub_date, utc=True).isoformat()
                hl = item.highlights[0] if getattr(item, "highlights", None) else ""
                res_dict = {
                    "id": f"exa_{item.id}",
                    "ticker": ticker.symbol,
                    "source": "exa_news",
                    "raw_timestamp_field": "published_date",
                    "raw_timestamp_value": item.published_date,
                    "timestamp": dt_utc,
                    "title": item.title or "",
                    "url": item.url or "",
                    "text": hl or item.title or "",
                }
                results.append(res_dict)
                if len(self.sample_results) < 5:
                    self.sample_results.append({
                        "ticker": ticker.symbol,
                        "title": item.title or "",
                        "url": item.url or "",
                        "published_date": pub_date,
                    })

            if not results:
                alt_query = f"{ticker.name} Hong Kong stock" if "HK" in ticker.symbol else f"{ticker.name} Korean stock"
                logger.info(f"Retrying Exa API with alternative query '{alt_query}'...")
                self.request_count += 1
                alt_resp = self.exa_client.search_and_contents(
                    query=alt_query,
                    type="neural",
                    category="news",
                    start_published_date=start_date,
                    num_results=fetch_n,
                    highlights={"num_sentences": 2},
                )
                for item in alt_resp.results:
                    pub_date = item.published_date or datetime.now(timezone.utc).isoformat()
                    dt_utc = pd.to_datetime(pub_date, utc=True).isoformat()
                    hl = item.highlights[0] if getattr(item, "highlights", None) else ""
                    results.append({
                        "id": f"exa_{item.id}",
                        "ticker": ticker.symbol,
                        "source": "exa_news",
                        "raw_timestamp_field": "published_date",
                        "raw_timestamp_value": item.published_date,
                        "timestamp": dt_utc,
                        "title": item.title or "",
                        "url": item.url or "",
                        "text": hl or item.title or "",
                    })
                    if len(self.sample_results) < 5:
                        self.sample_results.append({
                            "ticker": ticker.symbol,
                            "title": item.title or "",
                            "url": item.url or "",
                            "published_date": pub_date,
                        })

            logger.info(f"Exa returned {len(results)} news items for {ticker.symbol}")
            return results
        except Exception as e:
            logger.warning(f"Exa search error for {ticker.symbol}: {e}")
            return []

    def _fetch_via_rss(self, ticker: TickerConfig, days_lookback: int, max_results: int) -> List[Dict[str, Any]]:
        """Fallback live financial news RSS feed."""
        query = f"{ticker.clean_code} {ticker.name} stock news"
        encoded_query = urllib.parse.quote(query)
        rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
        self.request_count += 1
        try:
            resp = requests.get(rss_url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code != 200:
                logger.warning(f"RSS fetch failed ({resp.status_code}) for {ticker.symbol}")
                return []

            root = ET.fromstring(resp.content)
            items = []
            window_start = datetime.now(timezone.utc) - timedelta(days=days_lookback)

            for entry in root.findall(".//item")[:max_results]:
                title = entry.findtext("title", "")
                link = entry.findtext("link", "")
                pub_date_str = entry.findtext("pubDate", "")
                dt = pd.to_datetime(pub_date_str, utc=True) if pub_date_str else datetime.now(timezone.utc)

                if dt < window_start:
                    continue

                items.append({
                    "id": f"rss_{abs(hash(link))}",
                    "ticker": ticker.symbol,
                    "source": "financial_rss",
                    "raw_timestamp_field": "pubDate",
                    "raw_timestamp_value": pub_date_str,
                    "timestamp": dt.isoformat(),
                    "title": title,
                    "url": link,
                    "text": title,
                })
            return items
        except Exception as e:
            logger.warning(f"RSS parsing error for {ticker.symbol}: {e}")
            return []

    def _deduplicate(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate news items based on title similarity and exact URL."""
        seen_urls = set()
        seen_titles = set()
        deduped = []

        for it in items:
            url = it.get("url", "")
            title = it.get("title", "").strip().lower()
            norm_title = "".join(ch for ch in title if ch.isalnum())[:60]

            if url and url in seen_urls:
                continue
            if norm_title and norm_title in seen_titles:
                continue

            if url:
                seen_urls.add(url)
            if norm_title:
                seen_titles.add(norm_title)
            deduped.append(it)

        return deduped
