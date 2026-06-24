"""Fetch official AI/product RSS feeds and store them for the dashboard's
官方源 tab.

These are S-tier official sources (blogs / changelogs / news) that publish
native RSS. Unlike the Twitter pipeline, these are NOT LLM-scored — they go
straight into a rolling list (data/rss-latest.json) deduped by URL, sorted
newest-first. The dashboard's 官方源 tab reads that file directly.

Run: python fetch_rss.py
"""
import html
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import feedparser

DATA_DIR = Path(__file__).parent / "data"
OUT_FILE = DATA_DIR / "rss-latest.json"
MAX_ITEMS = 400          # rolling cap
SUMMARY_MAX = 280        # chars

# (label, feed_url) — native-RSS official sources only (verified 2026-06-24).
# X & YouTube excluded on purpose (X already covered by the tweet pipeline).
FEEDS = [
    ("OpenAI News",              "https://openai.com/news/rss.xml"),
    ("OpenAI Developer Blog",    "https://developers.openai.com/rss.xml"),
    ("Google DeepMind Blog",     "https://deepmind.google/blog/feed"),
    ("Google AI Blog",           "https://blog.google/technology/ai/rss"),
    ("Google Research Blog",     "https://research.google/blog/rss"),
    ("NVIDIA Blog",              "https://blogs.nvidia.com/feed/"),
    ("NVIDIA Robotics Blog",     "https://blogs.nvidia.com/blog/category/robotics/feed/"),
    ("NVIDIA Robotics Tech Blog","https://developer.nvidia.com/blog/category/robotics/feed"),
    ("Hugging Face Blog",        "https://huggingface.co/blog/feed.xml"),
    ("GitHub Blog",              "https://github.blog/feed"),
    ("Cursor Changelog",         "https://cursor.com/changelog/rss.xml"),
    ("Product Hunt",             "https://www.producthunt.com/feed"),
]


def strip_html(text: str) -> str:
    """Remove tags + collapse whitespace + unescape entities."""
    text = re.sub(r"<[^>]+>", "", text or "")
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def entry_published(entry) -> str | None:
    """Best-effort ISO8601 publish time from a feedparser entry."""
    for key in ("published_parsed", "updated_parsed"):
        t = entry.get(key)
        if t:
            try:
                return datetime(*t[:6], tzinfo=timezone.utc).isoformat()
            except Exception:
                pass
    return None


def fetch_feed(label: str, url: str) -> list[dict]:
    items = []
    try:
        parsed = feedparser.parse(url)
    except Exception as e:  # noqa: BLE001
        print(f"  !! {label}: parse error {type(e).__name__}: {e}")
        return items
    if parsed.bozo and not parsed.entries:
        print(f"  !! {label}: feed error {parsed.get('bozo_exception')}")
        return items
    for e in parsed.entries:
        link = e.get("link") or ""
        if not link:
            continue
        summary = strip_html(e.get("summary") or e.get("description") or "")
        items.append({
            "source": label,
            "title": strip_html(e.get("title") or "(无标题)"),
            "url": link,
            "summary": summary[:SUMMARY_MAX],
            "published_at": entry_published(e),
        })
    print(f"  {label}: {len(items)} items")
    return items


def load_existing() -> dict:
    if OUT_FILE.exists():
        try:
            return json.loads(OUT_FILE.read_text())
        except Exception:
            pass
    return {"items": []}


def sort_key(item: dict):
    # newest first; items without a date sink to the bottom
    return item.get("published_at") or ""


def main() -> int:
    DATA_DIR.mkdir(exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()

    print(f"Fetching {len(FEEDS)} RSS feeds...")
    fresh = []
    for label, url in FEEDS:
        fresh.extend(fetch_feed(label, url))

    existing = load_existing().get("items", [])
    by_url = {it["url"]: it for it in existing}

    new_count = 0
    for it in fresh:
        if it["url"] not in by_url:
            it["first_seen_at"] = now
            by_url[it["url"]] = it
            new_count += 1
        else:
            # refresh title/summary in case of edits, keep first_seen_at
            prev = by_url[it["url"]]
            it["first_seen_at"] = prev.get("first_seen_at", now)
            by_url[it["url"]] = it

    merged = sorted(by_url.values(), key=sort_key, reverse=True)[:MAX_ITEMS]

    result = {
        "updated_at": now,
        "count": len(merged),
        "new_count": new_count,
        "sources": [label for label, _ in FEEDS],
        "items": merged,
    }
    OUT_FILE.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nSaved {len(merged)} items ({new_count} new) to {OUT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
