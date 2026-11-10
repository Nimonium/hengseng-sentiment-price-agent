"""
Live documented fallback scrapers for social and financial discussion sources.
Used when Apify token is not provided or when direct scraping is desired.
All items are normalized to UTC timestamps and standard schema.
Enforces strict 10-calendar-day window filtering with exact publication timestamps.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/html, */*",
}


class WebFallbackScraper:
    def __init__(self, timeout: int = 12, calendar_days: int = 10):
        self.timeout = timeout
        self.calendar_days = calendar_days
        self.session = requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)

    @property
    def window_start_utc(self) -> datetime:
        return datetime.now(timezone.utc) - timedelta(days=self.calendar_days)

    # -------------------------------------------------------------
    # HKEX: LIHKG Discussion
    # -------------------------------------------------------------
    def scrape_lihkg(self, ticker: str, keywords: List[str], max_items: int = 80) -> List[Dict[str, Any]]:
        """
        Scrape LIHKG threads by keyword using the public LIHKG REST API.
        Filters strictly for threads published within the last 10 calendar days.
        """
        results: List[Dict[str, Any]] = []
        seen_ids = set()
        window_start = self.window_start_utc

        for kw in keywords[:3]:
            try:
                url = f"https://lihkg.com/api_v2/thread/search?q={requests.utils.quote(kw)}&page=1&count=30"
                resp = self.session.get(url, headers={"Referer": "https://lihkg.com/"}, timeout=self.timeout)
                if resp.status_code == 429:
                    logger.warning(f"LIHKG query '{kw}' returned status 429 (Cloudflare rate limit)")
                    continue
                elif resp.status_code != 200:
                    logger.warning(f"LIHKG query '{kw}' returned status {resp.status_code}")
                    continue

                data = resp.json()
                items = data.get("response", {}).get("items", [])

                for item in items:
                    tid = str(item.get("thread_id"))
                    if tid in seen_ids:
                        continue
                    seen_ids.add(tid)

                    create_time = item.get("create_time", 0)
                    dt_utc = datetime.fromtimestamp(create_time, tz=timezone.utc)

                    # Strict 10-day calendar filter: drop anything older (e.g. 2016-2025 threads)
                    if dt_utc < window_start:
                        continue

                    title = item.get("title", "")
                    likes = int(item.get("like_count", 0))
                    dislikes = int(item.get("dislike_count", 0))
                    replies = int(item.get("reply_count", 0))

                    results.append({
                        "id": f"lihkg_{tid}",
                        "ticker": ticker,
                        "source": "lihkg",
                        "raw_timestamp_field": "create_time",
                        "raw_timestamp_value": create_time,
                        "timestamp": dt_utc.isoformat(),
                        "text": title,
                        "likes": max(0, likes - dislikes),
                        "replies": replies,
                        "views": replies * 12,
                        "author_followers": None,
                    })

                    if len(results) >= max_items:
                        break
            except Exception as e:
                logger.warning(f"LIHKG scraping error for '{kw}': {e}")

        logger.info(f"LIHKG: scraped {len(results)} posts within last {self.calendar_days}d for {ticker}")
        return results

    # -------------------------------------------------------------
    # HKEX: AASTOCKS News & Commentary with Exact Native Timestamps
    # -------------------------------------------------------------
    def scrape_aastocks(self, ticker: str, clean_code: str, max_items: int = 50) -> List[Dict[str, Any]]:
        """
        Scrape AASTOCKS stock analysis and commentary feed with pagination.
        Parses exact native publication timestamps from HTML data-nt and ConvertToLocalTime tags.
        """
        results: List[Dict[str, Any]] = []
        code_padded = clean_code.zfill(5)
        seen_refs = set()
        window_start = self.window_start_utc

        for page in range(1, 4):
            url = f"http://www.aastocks.com/tc/stocks/analysis/stock-aafn/{code_padded}/0/all/{page}"
            try:
                resp = self.session.get(url, timeout=self.timeout)
                if resp.status_code == 200:
                    soup = BeautifulSoup(resp.text, "html.parser")
                    divs = soup.find_all("div", attrs={"ref": re.compile(r"^[A-Z0-9]+$")})

                    for div in divs:
                        ref_id = div.get("ref")
                        if not ref_id or ref_id in seen_refs:
                            continue

                        # Extract title
                        title = ""
                        for a in div.find_all("a"):
                            t = a.get_text(strip=True)
                            if len(t) > len(title):
                                title = t
                        if not title or len(title) < 5:
                            continue

                        # Extract native publication timestamp
                        pub_time = None
                        raw_nt = None
                        vote_div = div.find("div", attrs={"data-nt": True})
                        if vote_div and vote_div.get("data-nt"):
                            raw_nt = vote_div.get("data-nt")
                            try:
                                dt_hkt = datetime.strptime(raw_nt, "%Y%m%d%H%M").replace(
                                    tzinfo=timezone(timedelta(hours=8))
                                )
                                pub_time = dt_hkt.astimezone(timezone.utc)
                            except Exception:
                                pass

                        if not pub_time:
                            m = re.search(r"ConvertToLocalTime\(\{dt:'([^']+)'\}\)", str(div))
                            if m:
                                raw_nt = m.group(1)
                                try:
                                    dt_hkt = datetime.strptime(raw_nt, "%Y/%m/%d %H:%M").replace(
                                        tzinfo=timezone(timedelta(hours=8))
                                    )
                                    pub_time = dt_hkt.astimezone(timezone.utc)
                                except Exception:
                                    pass

                        if not pub_time:
                            continue

                        # Strict 10-day calendar filter
                        if pub_time < window_start:
                            continue

                        seen_refs.add(ref_id)

                        likes = 0
                        rec = div.select_one(".divRecommend .value")
                        if rec and rec.text.strip().isdigit():
                            likes = int(rec.text.strip())

                        results.append({
                            "id": f"aastocks_{ref_id}",
                            "ticker": ticker,
                            "source": "aastocks",
                            "raw_timestamp_field": "data-nt",
                            "raw_timestamp_value": raw_nt,
                            "timestamp": pub_time.isoformat(),
                            "text": title,
                            "likes": likes,
                            "replies": 0,
                            "views": 25,
                            "author_followers": None,
                        })

                        if len(results) >= max_items:
                            break
            except Exception as e:
                logger.warning(f"AASTOCKS scraping error for {ticker}: {e}")

            if len(results) >= max_items:
                break

        logger.info(f"AASTOCKS: scraped {len(results)} items within last {self.calendar_days}d for {ticker}")
        return results

    # -------------------------------------------------------------
    # KRX: Naver Stock Discussion (종목토론실) with Native writtenAt
    # -------------------------------------------------------------
    def scrape_naver_discussion(self, ticker: str, code: str, max_items: int = 180) -> List[Dict[str, Any]]:
        """
        Scrape Naver Finance Community Discussion boards using cursor pagination.
        Uses exact native user post publication timestamp (writtenAt).
        """
        results: List[Dict[str, Any]] = []
        url = "https://stock.naver.com/api/community/discussion/posts/by-item"
        headers = {
            "Referer": f"https://stock.naver.com/domestic/stock/{code}/discussion",
            "User-Agent": DEFAULT_HEADERS["User-Agent"],
        }

        offset = None
        seen_ids = set()
        window_start = self.window_start_utc
        max_pages = max(1, max_items // 60 + 1)
        reached_end_of_window = False

        for page in range(max_pages):
            params = {
                "itemCode": code,
                "discussionType": "domesticStock",
                "pageSize": 60,
                "isHolderOnly": "false",
                "excludesItemNews": "true",
                "isItemNewsOnly": "false",
            }
            if offset:
                params["offset"] = offset

            try:
                resp = self.session.get(url, params=params, headers=headers, timeout=self.timeout)
                if resp.status_code == 200:
                    data = resp.json()
                    posts = data.get("posts", [])
                    if not posts:
                        break

                    for p in posts:
                        pid = str(p.get("id"))
                        if pid in seen_ids:
                            continue
                        seen_ids.add(pid)

                        title = p.get("title", "")
                        body = p.get("contentSwReplacedButImg", "")
                        full_text = f"{title} {body}".strip()

                        written_at_str = p.get("writtenAt")
                        if not written_at_str:
                            continue

                        try:
                            # Parse ISO timestamp in KST (UTC+9)
                            dt_kst = datetime.fromisoformat(written_at_str)
                            if dt_kst.tzinfo is None:
                                dt_kst = dt_kst.replace(tzinfo=timezone(timedelta(hours=9)))
                            dt_utc = dt_kst.astimezone(timezone.utc)
                        except Exception:
                            continue

                        # Strict 10-day filter
                        if dt_utc < window_start:
                            reached_end_of_window = True
                            break

                        likes = int(p.get("recommendCount", 0))
                        dislikes = int(p.get("notRecommendCount", 0))
                        replies = int(p.get("commentCount", 0))
                        views = int(p.get("viewCount", 0))

                        results.append({
                            "id": f"naver_disc_{pid}",
                            "ticker": ticker,
                            "source": "naver_discussion",
                            "raw_timestamp_field": "writtenAt",
                            "raw_timestamp_value": written_at_str,
                            "timestamp": dt_utc.isoformat(),
                            "text": full_text,
                            "likes": max(0, likes - dislikes),
                            "replies": replies,
                            "views": views,
                            "author_followers": None,
                        })

                        if len(results) >= max_items:
                            break

                    if reached_end_of_window or len(results) >= max_items:
                        break

                    offset = data.get("lastOffset")
                    if not offset:
                        break
                else:
                    logger.warning(f"Naver discussion returned status {resp.status_code}")
                    break
            except Exception as e:
                logger.warning(f"Naver discussion scraping error for {ticker}: {e}")
                break

        logger.info(f"Naver Discussion: scraped {len(results)} posts within last {self.calendar_days}d for {ticker}")
        return results

    # -------------------------------------------------------------
    # KRX: Naver Stock Corporate News with Native writtenAt
    # -------------------------------------------------------------
    def scrape_naver_news(self, ticker: str, code: str, max_items: int = 50) -> List[Dict[str, Any]]:
        """
        Scrape official financial news and corporate disclosures from Naver Stock.
        Uses exact native article publication timestamp (writtenAt).
        """
        results: List[Dict[str, Any]] = []
        url = "https://stock.naver.com/api/community/discussion/posts/by-item"
        headers = {
            "Referer": f"https://stock.naver.com/domestic/stock/{code}/discussion",
            "User-Agent": DEFAULT_HEADERS["User-Agent"],
        }

        offset = None
        seen_ids = set()
        window_start = self.window_start_utc

        for page in range(3):
            params = {
                "itemCode": code,
                "discussionType": "domesticStock",
                "pageSize": 30,
                "isHolderOnly": "false",
                "excludesItemNews": "false",
                "isItemNewsOnly": "true",
            }
            if offset:
                params["offset"] = offset

            try:
                resp = self.session.get(url, params=params, headers=headers, timeout=self.timeout)
                if resp.status_code == 200:
                    data = resp.json()
                    posts = data.get("posts", [])
                    if not posts:
                        break

                    for p in posts:
                        pid = str(p.get("id"))
                        if pid in seen_ids:
                            continue
                        seen_ids.add(pid)

                        title = p.get("title", "")
                        body = p.get("contentSwReplacedButImg", "")
                        full_text = f"{title} {body}".strip()

                        written_at_str = p.get("writtenAt")
                        if not written_at_str:
                            continue

                        try:
                            dt_kst = datetime.fromisoformat(written_at_str)
                            if dt_kst.tzinfo is None:
                                dt_kst = dt_kst.replace(tzinfo=timezone(timedelta(hours=9)))
                            dt_utc = dt_kst.astimezone(timezone.utc)
                        except Exception:
                            continue

                        if dt_utc < window_start:
                            continue

                        results.append({
                            "id": f"naver_news_{pid}",
                            "ticker": ticker,
                            "source": "naver_news",
                            "raw_timestamp_field": "writtenAt",
                            "raw_timestamp_value": written_at_str,
                            "timestamp": dt_utc.isoformat(),
                            "text": full_text,
                            "likes": int(p.get("recommendCount", 0)),
                            "replies": int(p.get("commentCount", 0)),
                            "views": int(p.get("viewCount", 0)),
                            "author_followers": None,
                        })

                        if len(results) >= max_items:
                            break

                    offset = data.get("lastOffset")
                    if not offset:
                        break
            except Exception as e:
                logger.warning(f"Naver news scraping error for {ticker}: {e}")
                break

            if len(results) >= max_items:
                break

        logger.info(f"Naver News: scraped {len(results)} news items within last {self.calendar_days}d for {ticker}")
        return results

    # -------------------------------------------------------------
    # KRX Fallback: Daum News with Native URL Timestamps (/v/YYYYMMDDHHMMSS)
    # -------------------------------------------------------------
    def scrape_daum_news(self, ticker: str, keywords: List[str], max_items: int = 50) -> List[Dict[str, Any]]:
        """
        Scrape Daum Finance/News with native publication timestamps parsed from URL regex.
        Strictly enforces 10-calendar-day window; does NOT synthesize timestamps.
        """
        results: List[Dict[str, Any]] = []
        target_kw = keywords[0] if keywords else ticker
        seen_titles = set()
        window_start = self.window_start_utc

        for page in range(1, 3):
            try:
                url = f"https://search.daum.net/search?w=news&q={requests.utils.quote(target_kw)}&sort=recency&p={page}"
                resp = self.session.get(url, timeout=self.timeout)
                if resp.status_code == 200:
                    soup = BeautifulSoup(resp.text, "html.parser")
                    items = soup.select(".c-item-content, .tit-main, .item-title, strong.tit-g")

                    for idx, item in enumerate(items):
                        title = item.get_text(strip=True)
                        if not title or len(title) < 5 or title in seen_titles:
                            continue

                        link = item.select_one("a")
                        pub_time = None
                        raw_match = None
                        if link and link.get("href"):
                            match = re.search(r"/v/(\d{14})", link.get("href"))
                            if match:
                                raw_match = match.group(1)
                                try:
                                    dt_kst = datetime.strptime(raw_match, "%Y%m%d%H%M%S").replace(
                                        tzinfo=timezone(timedelta(hours=9))
                                    )
                                    pub_time = dt_kst.astimezone(timezone.utc)
                                except Exception:
                                    pass

                        # Reject items without verified native timestamp
                        if not pub_time or pub_time < window_start:
                            continue

                        seen_titles.add(title)
                        results.append({
                            "id": f"daum_{ticker}_{page}_{idx}",
                            "ticker": ticker,
                            "source": "daum_news",
                            "raw_timestamp_field": "/v/{14-digits}",
                            "raw_timestamp_value": raw_match,
                            "timestamp": pub_time.isoformat(),
                            "text": title,
                            "likes": 0,
                            "replies": 0,
                            "views": 25,
                            "author_followers": None,
                        })

                        if len(results) >= max_items:
                            break
            except Exception as e:
                logger.warning(f"Daum news scraping error for {ticker}: {e}")

        logger.info(f"Daum News: scraped {len(results)} items within last {self.calendar_days}d for {ticker}")
        return results
