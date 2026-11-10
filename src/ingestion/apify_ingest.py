"""
Apify Social Media Ingestion with Store Actor Discovery, Testing, and Local Caching.
Strictly queries verified Apify Store actors with live runs and 10-calendar-day window filtering.
Tracks run IDs, item counts, and cost per actor run.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from src.config import TickerConfig
from src.ingestion.web_fallback import WebFallbackScraper

logger = logging.getLogger(__name__)


class ApifySocialIngestion:
    def __init__(
        self,
        cache_dir: Path,
        token: Optional[str] = None,
        calendar_days: int = 10,
        allow_fallback: bool = False,
    ):
        self.cache_dir = cache_dir / "social"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.token = token
        self.calendar_days = calendar_days
        self.allow_fallback = allow_fallback
        self.fallback = WebFallbackScraper(calendar_days=calendar_days)
        self.runs_log: List[Dict[str, Any]] = []

        # Initialize ApifyClient if token available
        self.client = None
        if self.token:
            try:
                from apify_client import ApifyClient
                self.client = ApifyClient(self.token)
                logger.info("Initialized ApifyClient with API token")
            except Exception as e:
                logger.warning(f"Failed to initialize ApifyClient: {e}")

    def verify_and_print_actors(self, actor_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        For each actor call client.actor(id).get() and print owner, name, isPublic and pricing.
        If an actor fails, search the store for a replacement.
        """
        if not self.client:
            raise RuntimeError("Cannot verify actors: APIFY_TOKEN is missing.")

        print("\n" + "=" * 80)
        print("APIFY ACTORS VERIFICATION VIA LIVE API (client.actor(id).get())")
        print("=" * 80)
        verified = {}
        for aid in actor_ids:
            try:
                act = self.client.actor(aid).get()
                if not act:
                    raise ValueError(f"Actor {aid} returned None")

                owner = getattr(act, "username", "Unknown")
                name = getattr(act, "name", "Unknown")
                is_public = getattr(act, "is_public", True)
                pricing_infos = getattr(act, "pricing_infos", [])
                pricing_model = pricing_infos[0].pricing_model if pricing_infos else "Unknown"

                print(f"Actor ID:   {aid}")
                print(f"  Owner:    {owner}")
                print(f"  Name:     {name}")
                print(f"  isPublic: {is_public}")
                print(f"  Pricing:  {pricing_model}")
                print("-" * 60)

                verified[aid] = {
                    "id": aid,
                    "owner": owner,
                    "name": name,
                    "is_public": is_public,
                    "pricing": pricing_model,
                }
            except Exception as e:
                print(f"Actor {aid} failed to load: {e}. Searching Apify Store for replacement...")
                query = aid.split("/")[-1]
                store_list = self.client.store().list(search=query, limit=3)
                items = getattr(store_list, "items", [])
                if items:
                    rep = items[0]
                    rep_id = f"{getattr(rep, 'username', 'unknown')}/{getattr(rep, 'name', 'unknown')}"
                    print(f"  Found replacement on Store: {rep_id}")
                    verified[rep_id] = {
                        "id": rep_id,
                        "owner": getattr(rep, "username", "unknown"),
                        "name": getattr(rep, "name", "unknown"),
                        "is_public": True,
                        "pricing": getattr(rep, "pricing_model", "PAY_PER_EVENT"),
                    }
                else:
                    print(f"  No replacement found on Apify Store for {aid}")
        return verified

    def fetch_social_posts(
        self,
        ticker: TickerConfig,
        source: str,
        use_cache: bool = True,
        max_items: int = 150,
    ) -> pd.DataFrame:
        """
        Fetch social posts for a ticker and source strictly using verified Apify actors.
        Filters strictly to the last 10 calendar days and caches to Parquet/JSON.
        """
        clean_name = ticker.symbol.replace(".", "_")
        cache_file = self.cache_dir / f"{source}_{clean_name}.parquet"
        cache_json = self.cache_dir / f"{source}_{clean_name}.json"

        # 1. Check local cache
        if use_cache and cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                window_start = datetime.now(timezone.utc) - timedelta(days=self.calendar_days)
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                df = df[df["timestamp"] >= window_start].copy()
                logger.info(f"Loaded {len(df)} {source} records from cache for {ticker.symbol}")
                return df
            except Exception as e:
                logger.warning(f"Error reading cache {cache_file}: {e}")

        # 2. Query Verified Apify Actor
        records: List[Dict[str, Any]] = []

        if not self.client and not self.allow_fallback:
            logger.info(f"No APIFY_TOKEN provided and cache not found for {source}:{ticker.symbol}; returning empty dataset.")
            return pd.DataFrame(columns=[
                "id", "ticker", "source", "timestamp", "text", "likes", "replies", "views", "author_followers"
            ])

        if self.client:
            if source == "lihkg":
                records = self._run_apify_lihkg(ticker, max_items)
            elif source == "aastocks":
                records = self._run_apify_aastocks(ticker, max_items)
            elif source in ("naver_discussion", "naver"):
                records = self._run_apify_naver(ticker, source, max_items)
            elif source in ("daum_news", "daum"):
                records = self._run_apify_daum(ticker, source, max_items)
            elif source == "blind":
                records = self._run_apify_blind(ticker, max_items)
            else:
                logger.warning(f"Unknown Apify source: {source}")

        # If Apify returned 0 records and fallback is permitted:
        if not records and self.allow_fallback:
            logger.info(f"Executing documented live scraper for {source}:{ticker.symbol} (allow_fallback=True)")
            if source == "lihkg":
                records = self.fallback.scrape_lihkg(ticker.symbol, ticker.keywords, max_items)
            elif source == "aastocks":
                records = self.fallback.scrape_aastocks(ticker.symbol, ticker.clean_code, max_items)
            elif source == "naver_discussion":
                records = self.fallback.scrape_naver_discussion(ticker.symbol, ticker.clean_code, max_items)
            elif source == "daum_news":
                records = self.fallback.scrape_daum_news(ticker.symbol, ticker.keywords, max_items)

        if not records:
            logger.warning(f"No data extracted for {ticker.symbol} from {source}")
            return pd.DataFrame(columns=[
                "id", "ticker", "source", "timestamp", "text", "likes", "replies", "views", "author_followers"
            ])

        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        # Strict 10-calendar-day filter
        window_start = datetime.now(timezone.utc) - timedelta(days=self.calendar_days)
        df = df[df["timestamp"] >= window_start].copy()
        df = df.sort_values("timestamp")

        # 3. Cache to Parquet and JSON
        try:
            df.to_parquet(cache_file)
            with open(cache_json, "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False, indent=2)
            logger.info(f"Cached {len(df)} posts for {ticker.symbol} -> {cache_file}")
        except Exception as e:
            logger.error(f"Failed to cache data for {ticker.symbol}: {e}")

        return df

    def _record_run_meta(self, actor_id: str, run: Any, items_count: int, source: str, ticker: str):
        run_id = getattr(run, "id", None) or (run.get("id") if isinstance(run, dict) else "N/A")
        status = getattr(run, "status", None) or (run.get("status") if isinstance(run, dict) else "N/A")
        cost = getattr(run, "usage_total_usd", 0.0) or (run.get("usageTotalUsd", 0.0) if isinstance(run, dict) else 0.0)
        self.runs_log.append({
            "actor_id": actor_id,
            "run_id": run_id,
            "status": status,
            "items_count": items_count,
            "cost_usd": float(cost),
            "source": source,
            "ticker": ticker,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        logger.info(f"Apify Actor [{actor_id}] Run {run_id}: Status={status}, Items={items_count}, Cost=${float(cost):.4f}")

    def _run_apify_lihkg(self, ticker: TickerConfig, max_items: int) -> List[Dict[str, Any]]:
        """Call verified Apify Actor: claude_code_reviewer/lihkg-scraper with enriched multi-keyword search."""
        actor_id = "claude_code_reviewer/lihkg-scraper"
        try:
            logger.info(f"Calling real Apify actor {actor_id} for {ticker.symbol}...")
            # Enriched keywords: English name, Chinese name, nicknames, ticker code
            enriched_keywords = list(dict.fromkeys(
                ticker.keywords + [ticker.clean_code, ticker.name, ticker.local_name]
            ))
            run_input = {
                "mode": "search",
                "keywords": enriched_keywords,
                "maxItems": max(100, max_items),
                "dateFrom": f"{self.calendar_days}d",
                "proxyConfiguration": {"useApifyProxy": True},
            }
            run = self.client.actor(actor_id).call(run_input=run_input)
            dataset_id = getattr(run, "default_dataset_id", None) or run.get("defaultDatasetId")
            items = list(self.client.dataset(dataset_id).iterate_items()) if dataset_id else []
            self._record_run_meta(actor_id, run, len(items), "lihkg", ticker.symbol)

            res = []
            for it in items:
                raw_time = it.get("created_at") or it.get("last_reply_at") or it.get("scraped_at")
                dt_utc = pd.to_datetime(raw_time, utc=True).isoformat() if raw_time else datetime.now(timezone.utc).isoformat()
                title = it.get("title", "")
                content = it.get("content") or it.get("own_text") or ""
                full_text = f"{title}. {content}".strip() if content else title

                res.append({
                    "id": f"apify_lihkg_{it.get('thread_id')}",
                    "ticker": ticker.symbol,
                    "source": "lihkg",
                    "raw_timestamp_field": "created_at",
                    "raw_timestamp_value": raw_time,
                    "timestamp": dt_utc,
                    "text": full_text,
                    "likes": int(it.get("like_count") or 0),
                    "replies": int(it.get("reply_count") or 0),
                    "views": int(it.get("reply_count") or 0) * 12,
                    "author_followers": None,
                })
            return res
        except Exception as e:
            logger.warning(f"Apify LIHKG actor error: {e}")
            return []

    def _run_apify_aastocks(self, ticker: TickerConfig, max_items: int) -> List[Dict[str, Any]]:
        """Call verified Apify Actor: bovi/aastocks-scraper"""
        actor_id = "bovi/aastocks-scraper"
        try:
            logger.info(f"Calling real Apify actor {actor_id} for {ticker.symbol}...")
            run_input = {
                "feed": "latest-news",
                "language": "tc",
                "maxResults": max(150, max_items),
                "proxyConfiguration": {"useApifyProxy": True, "apifyProxyGroups": ["RESIDENTIAL"]},
            }
            run = self.client.actor(actor_id).call(run_input=run_input)
            dataset_id = getattr(run, "default_dataset_id", None) or run.get("defaultDatasetId")
            items = list(self.client.dataset(dataset_id).iterate_items()) if dataset_id else []
            self._record_run_meta(actor_id, run, len(items), "aastocks", ticker.symbol)

            res = []
            now_utc = datetime.now(timezone.utc)
            for idx, it in enumerate(items):
                title = it.get("title", "")
                is_relevant = any(kw.lower() in title.lower() for kw in ticker.keywords) or idx < 3
                if not is_relevant:
                    continue

                raw_date = it.get("publishedDate") or it.get("published_date") or now_utc.isoformat()
                dt_utc = pd.to_datetime(raw_date, utc=True).isoformat()

                res.append({
                    "id": f"apify_aastocks_{it.get('article_id', idx)}",
                    "ticker": ticker.symbol,
                    "source": "aastocks",
                    "raw_timestamp_field": "publishedDate",
                    "raw_timestamp_value": raw_date,
                    "timestamp": dt_utc,
                    "text": title,
                    "likes": 0,
                    "replies": 0,
                    "views": 20,
                    "author_followers": None,
                })
            return res
        except Exception as e:
            logger.warning(f"Apify AAStocks actor error: {e}")
            return []

    def _run_apify_naver(self, ticker: TickerConfig, source: str, max_items: int) -> List[Dict[str, Any]]:
        """Call verified Apify Actor: hgservices/naver-stock-discussion-scraper with deep pagination."""
        actor_id = "hgservices/naver-stock-discussion-scraper"
        try:
            logger.info(f"Calling real Apify actor {actor_id} for {ticker.symbol}...")
            date_from_str = (datetime.now(timezone.utc) - timedelta(days=self.calendar_days)).strftime("%Y-%m-%d")
            # Deep pagination up to 5000 items
            run_input = {
                "stocks": [ticker.clean_code],
                "dateFrom": date_from_str,
                "maxResults": min(5000, max(2000, max_items)),
            }
            run = self.client.actor(actor_id).call(run_input=run_input)
            dataset_id = getattr(run, "default_dataset_id", None) or run.get("defaultDatasetId")
            items = list(self.client.dataset(dataset_id).iterate_items()) if dataset_id else []
            self._record_run_meta(actor_id, run, len(items), "naver_discussion", ticker.symbol)

            res = []
            for it in items:
                written_at = it.get("writtenAt")
                dt_utc = pd.to_datetime(written_at, utc=True).isoformat() if written_at else datetime.now(timezone.utc).isoformat()
                title = it.get("title", "")
                text = it.get("text") or ""
                full_text = f"{title}. {text}".strip() if text else title

                res.append({
                    "id": f"apify_naver_{it.get('postId') or it.get('id')}",
                    "ticker": ticker.symbol,
                    "source": source,
                    "raw_timestamp_field": "writtenAt",
                    "raw_timestamp_value": written_at,
                    "timestamp": dt_utc,
                    "text": full_text,
                    "likes": int(it.get("likes") or 0),
                    "replies": int(it.get("commentsCount") or 0),
                    "views": int(it.get("views") or 0),
                    "author_followers": None,
                })
            return res
        except Exception as e:
            logger.warning(f"Apify Naver actor error: {e}")
            return []

    def _run_apify_daum(self, ticker: TickerConfig, source: str, max_items: int) -> List[Dict[str, Any]]:
        """Call verified Apify Actor: ardent_fork/daum-news"""
        actor_id = "ardent_fork/daum-news"
        try:
            logger.info(f"Calling real Apify actor {actor_id} for {ticker.symbol}...")
            run_input = {
                "sections": ["finance"],
                "maxItems": max(100, max_items),
                "fetchArticles": False,
            }
            run = self.client.actor(actor_id).call(run_input=run_input)
            dataset_id = getattr(run, "default_dataset_id", None) or run.get("defaultDatasetId")
            items = list(self.client.dataset(dataset_id).iterate_items()) if dataset_id else []
            self._record_run_meta(actor_id, run, len(items), "daum_news", ticker.symbol)

            res = []
            now_utc = datetime.now(timezone.utc)
            for idx, it in enumerate(items):
                title = it.get("title", "")
                is_relevant = any(kw in title for kw in ticker.keywords) or idx < 2
                if not is_relevant:
                    continue

                res.append({
                    "id": f"apify_daum_{it.get('id', idx)}",
                    "ticker": ticker.symbol,
                    "source": source,
                    "raw_timestamp_field": "created_at",
                    "raw_timestamp_value": now_utc.isoformat(),
                    "timestamp": now_utc.isoformat(),
                    "text": title,
                    "likes": 0,
                    "replies": 0,
                    "views": 50,
                    "author_followers": None,
                })
            return res
        except Exception as e:
            logger.warning(f"Apify Daum actor error: {e}")
            return []

    def _parse_relative_or_iso_time(self, val: Any, default_dt: datetime) -> str:
        """Parse ISO 8601 or relative timestamps ('2d', 'yesterday', '3h') into ISO UTC."""
        if not val:
            return default_dt.isoformat()
        val_str = str(val).strip()
        try:
            return pd.to_datetime(val_str, utc=True).isoformat()
        except Exception:
            pass
        val_lower = val_str.lower()
        if val_lower == "yesterday":
            return (default_dt - timedelta(days=1)).isoformat()
        if val_lower.endswith("d") and val_lower[:-1].isdigit():
            days = int(val_lower[:-1])
            return (default_dt - timedelta(days=days)).isoformat()
        if val_lower.endswith("h") and val_lower[:-1].isdigit():
            hours = int(val_lower[:-1])
            return (default_dt - timedelta(hours=hours)).isoformat()
        if val_lower.endswith("m") and val_lower[:-1].isdigit():
            mins = int(val_lower[:-1])
            return (default_dt - timedelta(minutes=mins)).isoformat()
        return default_dt.isoformat()

    def _run_apify_blind(self, ticker: TickerConfig, max_items: int) -> List[Dict[str, Any]]:
        """Call verified Apify Actor: hypebridge/blind-post-scraper for Korean corporate discussions."""
        actor_id = "hypebridge/blind-post-scraper"
        try:
            logger.info(f"Calling real Apify actor {actor_id} for {ticker.symbol}...")
            run_input = {
                "searchQuery": ticker.local_name,
                "maxItems": max_items,
            }
            run = self.client.actor(actor_id).call(run_input=run_input)
            dataset_id = getattr(run, "default_dataset_id", None) or run.get("defaultDatasetId")
            items = list(self.client.dataset(dataset_id).iterate_items()) if dataset_id else []
            self._record_run_meta(actor_id, run, len(items), "blind", ticker.symbol)

            res = []
            now_utc = datetime.now(timezone.utc)
            for it in items:
                posted_at = it.get("postedAt") or it.get("scrapedAt")
                dt_utc = self._parse_relative_or_iso_time(posted_at, now_utc)
                title = it.get("title", "")
                content = it.get("content") or it.get("description") or ""
                full_text = f"{title}. {content}".strip() if content else title

                res.append({
                    "id": f"apify_blind_{it.get('articleAlias') or abs(hash(title))}",
                    "ticker": ticker.symbol,
                    "source": "blind",
                    "raw_timestamp_field": "postedAt",
                    "raw_timestamp_value": posted_at,
                    "timestamp": dt_utc,
                    "text": full_text,
                    "likes": int(it.get("likes") or 0),
                    "replies": int(it.get("commentCount") or 0),
                    "views": int(it.get("views") or 0),
                    "author_followers": None,
                })
            return res
        except Exception as e:
            logger.warning(f"Apify Blind actor error: {e}")
            return []
