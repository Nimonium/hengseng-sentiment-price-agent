import os
import json
from pathlib import Path
from dotenv import dotenv_values
from apify_client import ApifyClient

env_vals = dotenv_values(Path(".env").resolve())
client = ApifyClient(env_vals.get("APIFY_TOKEN"))

actors = [
    "claude_code_reviewer/lihkg-scraper",
    "bovi/aastocks-scraper",
    "hgservices/naver-stock-discussion-scraper",
    "ardent_fork/daum-news",
    "tri_angle/telegram-scraper",
    "truefetch/telegram-channel-message"
]

for aid in actors:
    print("=" * 60)
    print("ACTOR:", aid)
    act = client.actor(aid).get()
    pricing = act.pricing_infos[0].pricing_model if act.pricing_infos else "Unknown"
    print(f"Owner: {act.username} | isPublic: {act.is_public} | Pricing: {pricing}")
    build_id = act.tagged_builds.get("latest").build_id if act.tagged_builds else None
    if build_id:
        b = client.build(build_id).get()
        schema_str = b.input_schema
        if schema_str:
            schema = json.loads(schema_str) if isinstance(schema_str, str) else schema_str
            print("Schema Title:", schema.get("title"))
            print("Properties:", list(schema.get("properties", {}).keys()))
            print("Required:", schema.get("required", []))
            for p, v in schema.get("properties", {}).items():
                p_type = v.get("type", "any")
                title = v.get("title", p)
                default = v.get("default")
                print(f"   {p} ({p_type}): {title} | default={default}")

