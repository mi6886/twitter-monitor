"""Fetch official AI/product RSS feeds and store them for the dashboard's
官方源 tab.

These are S-tier official sources (blogs / changelogs / news) that publish
native RSS. Unlike the Twitter pipeline, these are NOT LLM-scored — they go
straight into a rolling list (data/rss-latest.json) deduped by URL, sorted
newest-first. The dashboard's 官方源 tab reads that file directly.

Run: python fetch_rss.py
"""
import concurrent.futures
import html
import json
import re
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import feedparser

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"

DATA_DIR = Path(__file__).parent / "data"
OUT_FILE = DATA_DIR / "rss-latest.json"
MAX_ITEMS = 500          # rolling cap (across RSS + sitemap sources)
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

# Sources with NO native RSS — monitored via sitemap diff. For each, filter the
# sitemap to article URLs by path prefix, take the most-recent N, and fetch each
# NEW page once for its real title/description (slug fallback). Verified
# 2026-06-24. (Luma excluded: its sitemap has no article stream, only product
# pages — its updates only live on X/YouTube, which we don't ingest here.)
# Each: (label, sitemap_url, prefix, exclude, trust_lastmod, limit)
#   exclude       — path substrings to drop (section/listing pages)
#   trust_lastmod — True: use sitemap <lastmod> as the date (pages lack
#                   article:published_time). False: require the page's
#                   article:published_time — this both dates the item AND
#                   auto-skips section pages (which have no published_time).
SITEMAP_SOURCES = [
    ("Anthropic News",          "https://www.anthropic.com/sitemap.xml", "/news/",        [], True, 30),
    ("Anthropic Engineering",   "https://www.anthropic.com/sitemap.xml", "/engineering/", [], True, 30),
    ("Google DeepMind Research","https://deepmind.google/sitemap.xml",   "/research/",    [], True, 30),
    ("Runway",                  "https://runwayml.com/sitemap.xml",      "/news/",        [], False, 40),
    ("Cursor Blog",             "https://cursor.com/sitemap.xml",        "/blog/",        ["/blog/topic/"], False, 100),
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


def http_get(url: str, timeout: int = 15, retries: int = 2) -> str:
    """GET text with small retry — some CDNs intermittently send truncated
    bodies (IncompleteRead) or drop the connection."""
    last = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read().decode("utf-8", "ignore")
        except Exception as e:  # noqa: BLE001
            last = e
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
    raise last


def sitemap_entries(sitemap_url: str, prefix: str) -> list[tuple[str, str | None]]:
    """Return [(url, lastmod_iso|None)] under `prefix`, newest first."""
    try:
        xml = http_get(sitemap_url, timeout=20)
    except Exception as e:  # noqa: BLE001
        print(f"  !! sitemap {sitemap_url}: {type(e).__name__}: {e}")
        return []
    out = []
    for block in re.findall(r"<url>(.*?)</url>", xml, re.DOTALL):
        loc_m = re.search(r"<loc>([^<]+)</loc>", block)
        if not loc_m:
            continue
        loc = loc_m.group(1).strip()
        if prefix not in urlparse(loc).path:
            continue
        lastmod = None
        mod_m = re.search(r"<lastmod>([^<]+)</lastmod>", block)
        if mod_m:
            try:
                lastmod = datetime.fromisoformat(
                    mod_m.group(1).strip().replace("Z", "+00:00")
                ).isoformat()
            except Exception:
                lastmod = None
        out.append((loc, lastmod))
    out.sort(key=lambda x: x[1] or "", reverse=True)
    return out


def slug_title(url: str) -> str:
    slug = urlparse(url).path.rstrip("/").rsplit("/", 1)[-1]
    t = slug.replace("-", " ").replace("_", " ").strip()
    return t.title() if t else "(无标题)"


# Site-name affixes to trim from <title>/og:title — sites put them at the end
# (" — Google DeepMind") or the start ("Runway News | ...").
_SITE_NAMES = r"(google deepmind|anthropic|cursor|runway( news)?|openai)"
_TITLE_SUFFIX = re.compile(r"\s*[|\\—–·-]\s*" + _SITE_NAMES + r"\s*$", re.IGNORECASE)
_TITLE_PREFIX = re.compile(r"^\s*" + _SITE_NAMES + r"\s*[|\\—–·-]\s*", re.IGNORECASE)
# Anthropic reuses one boilerplate og:description on every page — not useful.
_BOILERPLATE = "anthropic is an ai safety and research company"


def _meta(html_text: str, *patterns: str) -> str:
    for p in patterns:
        m = re.search(p, html_text, re.IGNORECASE | re.DOTALL)
        if m:
            return strip_html(m.group(1))
    return ""


def extract_page(html_text: str) -> tuple[str, str, str | None]:
    """Return (title, description, published_iso|None) from a page."""
    title = _meta(
        html_text,
        r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)["\']',
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:title["\']',
        r'<title[^>]*>(.*?)</title>',
    )
    title = _TITLE_PREFIX.sub("", _TITLE_SUFFIX.sub("", title)).strip()
    desc = _meta(
        html_text,
        r'<meta[^>]+property=["\']og:description["\'][^>]+content=["\']([^"\']+)["\']',
        r'<meta[^>]+name=["\']description["\'][^>]+content=["\']([^"\']+)["\']',
    )
    if desc.lower().startswith(_BOILERPLATE):
        desc = ""
    pub = _meta(
        html_text,
        r'<meta[^>]+property=["\']article:published_time["\'][^>]+content=["\']([^"\']+)["\']',
        r'<meta[^>]+property=["\']og:article:published_time["\'][^>]+content=["\']([^"\']+)["\']',
        r'"datePublished"\s*:\s*"([^"]+)"',
    )
    pub_iso = None
    if pub:
        try:
            pub_iso = datetime.fromisoformat(pub.replace("Z", "+00:00")).isoformat()
        except Exception:
            pub_iso = None
    return title, desc, pub_iso


def fetch_sitemap_source(label, sitemap_url, prefix, exclude, trust_lastmod,
                         limit, known_urls) -> list[dict]:
    """Return item dicts for NEW article URLs from one sitemap source."""
    entries = [
        (u, m) for u, m in sitemap_entries(sitemap_url, prefix)
        if not any(x in u for x in exclude)
    ][:limit]
    new_entries = [(u, m) for u, m in entries if u not in known_urls]

    def grab(pair):
        url, lastmod = pair
        title = desc = ""
        page_pub = None
        try:
            title, desc, page_pub = extract_page(http_get(url, timeout=12))
        except Exception:
            pass
        published = lastmod if trust_lastmod else page_pub
        # When we don't trust lastmod, a missing page date means this is a
        # section/listing page (not an article) — drop it.
        if not trust_lastmod and not published:
            return None
        return {
            "source": label,
            "title": (title or slug_title(url))[:200],
            "url": url,
            "summary": desc[:SUMMARY_MAX],
            "published_at": published or lastmod,
        }

    items = []
    if new_entries:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            items = [it for it in pool.map(grab, new_entries) if it]
    print(f"  {label}: {len(entries)} candidates, {len(new_entries)} new, {len(items)} kept")
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

    existing = load_existing().get("items", [])
    by_url = {it["url"]: it for it in existing}
    known_urls = set(by_url)
    new_count = 0

    print(f"Fetching {len(FEEDS)} RSS feeds...")
    fresh = []
    for label, url in FEEDS:
        fresh.extend(fetch_feed(label, url))
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

    print(f"\nChecking {len(SITEMAP_SOURCES)} sitemap sources...")
    for label, sm, prefix, exclude, trust_lastmod, limit in SITEMAP_SOURCES:
        for it in fetch_sitemap_source(label, sm, prefix, exclude,
                                       trust_lastmod, limit, known_urls):
            if it["url"] not in by_url:   # new pages only; never re-fetch known
                it["first_seen_at"] = now
                by_url[it["url"]] = it
                new_count += 1

    merged = sorted(by_url.values(), key=sort_key, reverse=True)[:MAX_ITEMS]

    result = {
        "updated_at": now,
        "count": len(merged),
        "new_count": new_count,
        "sources": [label for label, _ in FEEDS]
                   + [s[0] for s in SITEMAP_SOURCES],
        "items": merged,
    }
    OUT_FILE.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nSaved {len(merged)} items ({new_count} new) to {OUT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
