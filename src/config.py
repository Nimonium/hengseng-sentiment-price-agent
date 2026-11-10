"""
Configuration loader and schema definition.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from dotenv import load_dotenv

# Load .env if present
load_dotenv()


@dataclass
class TickerConfig:
    symbol: str
    name: str
    local_name: str
    keywords: List[str]
    code: Optional[str] = None

    @property
    def clean_code(self) -> str:
        """Extract numeric or short ticker code."""
        if self.code:
            return self.code
        return self.symbol.split(".")[0]


@dataclass
class ExchangeConfig:
    name: str
    timezone: str
    market_open: str
    market_close: str
    social_sources: List[str]
    news_sources: List[str] = field(default_factory=list)
    tickers: List[TickerConfig] = field(default_factory=list)


@dataclass
class AppConfig:
    project_name: str
    cache_dir: Path
    processed_dir: Path
    report_dir: Path
    days_lookback: int
    bar_interval: str
    exchanges: Dict[str, ExchangeConfig]
    sentiment_config: Dict[str, Any]
    feature_config: Dict[str, Any]
    model_config: Dict[str, Any]
    apify_token: Optional[str]
    exa_api_key: Optional[str]
    gemini_api_key: Optional[str]

    @property
    def all_tickers(self) -> List[TickerConfig]:
        res = []
        for ex in self.exchanges.values():
            res.extend(ex.tickers)
        return res


def load_config(config_path: str = "config.yaml") -> AppConfig:
    """Load and parse YAML configuration."""
    cfg_file = Path(config_path)
    if not cfg_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(cfg_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    project_data = data.get("project", {})
    data_section = data.get("data", {})

    cache_dir = Path(data_section.get("cache_dir", "data/raw"))
    processed_dir = Path(data_section.get("processed_dir", "data/processed"))
    report_dir = Path(data_section.get("report_dir", "report"))

    cache_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    # Subdirectories for raw data
    (cache_dir / "prices").mkdir(parents=True, exist_ok=True)
    (cache_dir / "social").mkdir(parents=True, exist_ok=True)
    (cache_dir / "news").mkdir(parents=True, exist_ok=True)

    exchanges: Dict[str, ExchangeConfig] = {}
    for ex_key, ex_val in data.get("exchanges", {}).items():
        ticker_objs = []
        for t in ex_val.get("tickers", []):
            ticker_objs.append(
                TickerConfig(
                    symbol=t["symbol"],
                    name=t["name"],
                    local_name=t.get("local_name", t["name"]),
                    keywords=t.get("keywords", []),
                    code=t.get("code", None),
                )
            )

        exchanges[ex_key] = ExchangeConfig(
            name=ex_key,
            timezone=ex_val.get("timezone", "UTC"),
            market_open=ex_val.get("market_open", "09:30"),
            market_close=ex_val.get("market_close", "16:00"),
            social_sources=ex_val.get("social_sources", []),
            news_sources=ex_val.get("news_sources", []),
            tickers=ticker_objs,
        )

    return AppConfig(
        project_name=project_data.get("name", "hengseng-sentiment-agent"),
        cache_dir=cache_dir,
        processed_dir=processed_dir,
        report_dir=report_dir,
        days_lookback=data_section.get("days_lookback", 14),
        bar_interval=data_section.get("bar_interval", "1h"),
        exchanges=exchanges,
        sentiment_config=data.get("sentiment", {}),
        feature_config=data.get("features", {}),
        model_config=data.get("models", {}),
        apify_token=os.getenv("APIFY_TOKEN"),
        exa_api_key=os.getenv("EXA_API_KEY"),
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
    )

