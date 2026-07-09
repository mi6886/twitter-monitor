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

# Sources with NO native RSS — monitored via sitemap diff.
#
# "New article" is decided by URL-FIRST-APPEARANCE against a per-source
# baseline (data/rss-seen.json), NEVER by <lastmod> recency: sites bulk-touch
# old pages (Anthropic refreshed 3 articles from 2023 on 2026-07-08, and has
# done 17-page bulk refreshes before), so a fresh lastmod on an already-known
# URL means nothing. On the first run for a source, ALL its sitemap URLs are
# recorded as seen (baseline) and nothing is ingested — from then on only
# genuinely new URLs come in.
#
# Each: (label, sitemap_url, prefix, exclude, trust_lastmod, limit)
#   exclude       — path substrings to drop (section/listing pages)
#   trust_lastmod — True: sitemap <lastmod> is an acceptable date fallback
#                   when the page itself yields no date. False: require a
#                   page-extracted date — also auto-skips section pages.
#   (Page-extracted dates — article:published_time, JSON-LD, or a visible
#   "Jul 26, 2023"-style byline — always take priority over lastmod.)
SITEMAP_SOURCES = [
    ("Anthropic News",          "https://www.anthropic.com/sitemap.xml", "/news/",        [], True, 30),
    ("Anthropic Engineering",   "https://www.anthropic.com/sitemap.xml", "/engineering/", [], True, 30),
    ("Google DeepMind Research","https://deepmind.google/sitemap.xml",   "/research/",    [], True, 30),
    ("Runway",                  "https://runwayml.com/sitemap.xml",      "/news/",        [], False, 40),
    ("Cursor Blog",             "https://cursor.com/sitemap.xml",        "/blog/",        ["/blog/topic/"], False, 100),
]
SEEN_FILE = DATA_DIR / "rss-seen.json"


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
    if not pub_iso:
        pub_iso = _visible_date(html_text)
    return title, desc, pub_iso


_MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun",
     "jul", "aug", "sep", "oct", "nov", "dec"])}
_VISIBLE_DATE = re.compile(
    r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+"
    r"(\d{1,2}),\s+(20\d\d)\b")


def _visible_date(html_text: str) -> str | None:
    """First visible byline-style date ("Jul 26, 2023") in the page.
    Anthropic (and similar Next.js sites) show the publish date as text but
    expose no article:published_time meta — this is the only page-level date
    available there. First match = the article's own byline in practice."""
    m = _VISIBLE_DATE.search(html_text)
    if not m:
        return None
    try:
        month = _MONTHS[m.group(1).lower()[:3]]
        return datetime(int(m.group(3)), month, int(m.group(2)),
                        tzinfo=timezone.utc).isoformat()
    except (KeyError, ValueError):
        return None


def fetch_sitemap_source(label, sitemap_url, prefix, exclude, trust_lastmod,
                         limit, seen_urls: set[str]) -> tuple[list[dict], set[str]]:
    """Return (new item dicts, ALL article urls seen in this sitemap).

    New = URL appearing in the sitemap for the first time ever (not in the
    seen baseline). lastmod recency is deliberately ignored for newness —
    sites bulk-refresh lastmod on old pages. First run for a source (empty
    baseline) ingests nothing and just records the baseline.
    """
    all_entries = [
        (u, m) for u, m in sitemap_entries(sitemap_url, prefix)
        if not any(x in u for x in exclude)
    ]
    all_urls = {u for u, _ in all_entries}
    if not all_urls:
        # sitemap fetch failed or empty — don't touch the baseline
        print(f"  {label}: sitemap empty/unreachable, skipping")
        return [], set()

    first_run = not (seen_urls & all_urls)
    if first_run:
        print(f"  {label}: baseline created ({len(all_urls)} urls), 0 ingested")
        return [], all_urls

    new_entries = [(u, m) for u, m in all_entries if u not in seen_urls][:limit]

    def grab(pair):
        url, lastmod = pair
        title = desc = ""
        page_pub = None
        try:
            title, desc, page_pub = extract_page(http_get(url, timeout=12))
        except Exception:
            pass
        published = page_pub or (lastmod if trust_lastmod else None)
        # Without trust_lastmod, no page date means a section/listing page.
        if not published:
            return None
        return {
            "source": label,
            "title": (title or slug_title(url))[:200],
            "url": url,
            "summary": desc[:SUMMARY_MAX],
            "published_at": published,
        }

    items = []
    if new_entries:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            items = [it for it in pool.map(grab, new_entries) if it]
    print(f"  {label}: {len(all_urls)} in sitemap, {len(new_entries)} new urls, {len(items)} ingested")
    return items, all_urls


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
    try:
        seen_map = json.loads(SEEN_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        seen_map = {}
    for label, sm, prefix, exclude, trust_lastmod, limit in SITEMAP_SOURCES:
        seen = set(seen_map.get(label, []))
        items, all_urls = fetch_sitemap_source(label, sm, prefix, exclude,
                                               trust_lastmod, limit, seen)
        for it in items:
            if it["url"] not in by_url:
                it["first_seen_at"] = now
                by_url[it["url"]] = it
                new_count += 1
        if all_urls:
            seen_map[label] = sorted(seen | all_urls)
    SEEN_FILE.write_text(json.dumps(seen_map, ensure_ascii=False, indent=1))

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
