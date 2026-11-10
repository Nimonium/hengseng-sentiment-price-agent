"""
Apify Actor Audit Script.
Queries Apify API via ApifyClient to verify actor existence, owner, and input schema.
"""

from __future__ import annotations

import json
import os
import sys
from apify_client import ApifyClient

ACTORS = [
    "claude_code_reviewer/lihkg-scraper",
    "bovi/aastocks-scraper",
    "hgservices/naver-stock-discussion-scraper",
    "tri_angle/telegram-scraper",
    "ardent_fork/daum-news"
]

def audit_actors():
    token = os.getenv("APIFY_TOKEN")
    client = ApifyClient(token=token)
    print(f"APIFY_TOKEN present in environment: {bool(token)}")
    if not token:
        print("Note: Running unauthenticated public API queries against Apify Store.\n")

    print(f"{'Actor ID':<42} | {'Exists':<7} | {'Owner':<20} | {'Pricing Model':<15} | {'Schema Status'}")
    print("-" * 115)

    for aid in ACTORS:
        try:
            actor = client.actor(aid)
            details = actor.get()
            if details:
                owner = details.username or details.user_id or "Unknown"
                pricing = "N/A"
                if details.pricing_infos:
                    pricing = details.pricing_infos[-1].pricing_model
                
                # Check version schema
                schema_keys = []
                try:
                    ver = actor.version("latest").get()
                    if ver and ver.input_schema:
                        parsed = json.loads(ver.input_schema) if isinstance(ver.input_schema, str) else ver.input_schema
                        schema_keys = list(parsed.get('properties', {}).keys())
                except Exception:
                    pass

                if not schema_keys and details.example_run_input:
                    schema_status = f"Example Input Provided ({details.example_run_input.body[:30]}...)"
                elif schema_keys:
                    schema_status = f"Schema Fields: {schema_keys}"
                else:
                    schema_status = "Default / Internal Schema"

                print(f"{aid:<42} | {'YES':<7} | {owner:<20} | {pricing:<15} | {schema_status}")
            else:
                print(f"{aid:<42} | {'NO':<7} | {'N/A':<20} | {'N/A':<15} | Not Found")
        except Exception as e:
            print(f"{aid:<42} | {'ERROR':<7} | {'N/A':<20} | {'N/A':<15} | {str(e)[:40]}")

if __name__ == "__main__":
    audit_actors()

