"""Fetch official AI/product RSS feeds and store them for the dashboard's
官方源 tab.

These are S-tier official sources (blogs / changelogs / news) that publish
native RSS. Unlike the Twitter pipeline, these are NOT LLM-scored — they go
straight into a rolling list (data/rss-latest.json) deduped by URL, sorted
newest-first. The dashboard's 官方源 tab reads that file directly.

Run: python fetch_rss.py
"""
import concurrent.futures
import hashlib
import html
import json
import re
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

import feedparser

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"

DATA_DIR = Path(__file__).parent / "data"
OUT_FILE = DATA_DIR / "rss-latest.json"
MAX_ITEMS = 700          # rolling cap (across RSS + YouTube + website sources)
MAX_ITEMS_PER_FEED = 50  # prevent deep podcast archives from crowding the pool
SUMMARY_MAX = 280        # chars

# (label, feed_url) — native-RSS official sources only (verified 2026-08-04).
# X is covered by the tweet pipeline; YouTube feeds are configured separately.
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
    # S-tier robotics media from the source-selection review. Video Friday is
    # published in the IEEE Robotics feed, so it does not need a duplicate feed.
    ("IEEE Spectrum Robotics",    "https://spectrum.ieee.org/feeds/topic/robotics.rss"),
    ("The Robot Report",          "https://www.therobotreport.com/feed/"),
    # S-tier interview/podcast sources. These feeds include episode metadata,
    # transcripts or summaries that the dashboard can display as normal items.
    ("Lex Fridman Podcast",       "https://lexfridman.com/feed/podcast/"),
    ("No Priors Podcast",         "https://feeds.megaphone.fm/nopriors"),
    ("AI + a16z",                 "https://feeds.simplecast.com/Hb_IuXOo"),
    ("Dwarkesh Podcast",          "https://www.dwarkesh.com/feed"),
    ("Latent Space",              "https://www.latent.space/feed"),
    # Product Hunt removed 2026-07-10 per user request (too noisy for 官方源).
]

YOUTUBE_FEED_BASE = "https://www.youtube.com/feeds/videos.xml?channel_id="

# (label, official channel ID, optional relevance filter). Dedicated AI and
# robotics channels are kept whole; broad publisher channels are filtered so
# unrelated corporate/product videos do not crowd the dashboard.
YOUTUBE_FEEDS = [
    ("IEEE Spectrum Robotics YouTube", "UCFQDtftsHGzSh1-TReNT4lA",
     r"\brobot(?:ics?|s)?\b|\bhumanoid\b|\bautomation\b|\bdrone\b|video friday"),
    ("The Robot Report YouTube",       "UCFvPK74I5Hd5eVzZ_5t6FbQ", None),
    ("Figure AI YouTube",              "UCYlq-KmwPjc1DtsGmthFqSQ", None),
    ("Boston Dynamics YouTube",        "UC7vVhkEfw4nOGp8TyDk7RcQ", None),
    ("Unitree YouTube",                "UCsMbp4V8oxzHCMdOUP-3oWw", None),
    ("Agility Robotics YouTube",       "UCN-StetwWuVYf-MU2_NVj4A", None),
    ("1X Technologies YouTube",        "UCoHslVexR2q57wUoCRfdUsg", None),
    ("Tesla AI / Optimus YouTube",     "UC5WjFrtBdufl6CZojX3D8dQ",
     r"\bai\b|\brobot(?:ics?|s)?\b|\boptimus\b|\bautonom(?:y|ous)\b|\bfsd\b|self-driving"),
    ("AgiBot YouTube",                 "UCuKcqTxz_fe1PbrsIAQXr5A", None),
    ("NVIDIA Robotics YouTube",        "UCHuiy8bXnmK5nisYHUd1J5g",
     r"\brobot(?:ics?|s)?\b|\bphysical ai\b|\bisaac\b|\bgr00t\b|\bhumanoid\b|\bjetson\b"),
    ("Lex Fridman YouTube",            "UCSHZKyawb77ixDdsGog4iWA", None),
    ("No Priors YouTube",              "UCSI7h9hydQ40K5MJHnCrQvw", None),
    ("a16z AI YouTube",                "UC9cn0TuPq4dnbTY-CBsm8XA",
     r"\bai\b|\bagent(?:ic|s)?\b|\bllm\b|\bmodels?\b|\binference\b|\bmachine learning\b|\brobot(?:ics?|s)?\b"),
    ("Dwarkesh Podcast YouTube",       "UCXl4i9dYBrFOabk0xGmbkRA", None),
    ("Latent Space YouTube",           "UCxBcwypKK-W3GHd_RZ9FZrQ", None),
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
# Each: (label, sitemap_url, prefix, exclude, trust_lastmod, limit,
#        include_pattern, bootstrap_limit)
#   exclude       — path substrings to drop (section/listing pages)
#   trust_lastmod — True: sitemap <lastmod> is an acceptable date fallback
#                   when the page itself yields no date. False: require a
#                   page-extracted date — also auto-skips section pages.
#   (Page-extracted dates — article:published_time, JSON-LD, or a visible
#   "Jul 26, 2023"-style byline — always take priority over lastmod.)
SITEMAP_SOURCES = [
    ("Anthropic News",          "https://www.anthropic.com/sitemap.xml", "/news/",        [], True, 30, None, 0),
    ("Anthropic Engineering",   "https://www.anthropic.com/sitemap.xml", "/engineering/", [], True, 30, None, 0),
    ("Anthropic Research",      "https://www.anthropic.com/sitemap.xml", "/research/",    [], True, 30, None, 0),
    ("Google DeepMind Research","https://deepmind.google/sitemap.xml",   "/research/",    [], True, 30, None, 0),
    ("Runway",                  "https://runwayml.com/sitemap.xml",      "/news/",        [], False, 40, None, 0),
    ("Runway Research",         "https://runwayml.com/sitemap.xml",      "/research/",
     ["/research/publications", "/research/rna-sessions"], False, 30, None, 0),
    ("Cursor Blog",             "https://cursor.com/sitemap.xml",        "/blog/",        ["/blog/topic/"], False, 100, None, 0),
    # S-tier product/robotics sites without a useful native RSS feed.
    ("Luma Changelog",          "https://lumalabs.ai/changelog/sitemap.xml", "/changelog/", [], True, 40, None, 0),
    ("Figure AI News",          "https://www.figure.ai/sitemap.xml",     "/news/",        [], True, 50, None, 0),
    ("Boston Dynamics Blog",    "https://bostondynamics.com/blog-sitemap.xml", "/blog/",    [], True, 50, None, 0),
    ("Unitree News",            "https://www.unitree.com/sitemap.xml",   "/cn/news/",     [], False, 50, None, 0),
    ("Agility Robotics",        "https://www.agilityrobotics.com/sitemap.xml", "/content/", [], False, 100, None, 0),
    ("AgiBot News",             "https://www.agibot.com/sitemap.xml",    "/article/",     [], False, 100, None, 0),
    # China AI first batch from the source-selection review.
    ("ByteDance Seed Blog",     "https://seed.bytedance.com/sitemap.xml", "/blog/", [], True, 50,
     r"^/blog/[^/]+/?$", 5),
    ("DeepSeek News",           "https://api-docs.deepseek.com/sitemap.xml", "/news/", [], True, 50,
     r"/news/[^/]+/?$", 5),
    ("Vidu Product Updates",    "https://www.vidu.com/sitemap.xml", "/vidu-", [], True, 50,
     r"^/vidu-(?:q\d+(?:-\d+)?|s\d+(?:-\d+)?|claw|agent|studio)/?$", 5),
    ("Qoder Blog",              "https://qoder.com/sitemap.xml", "/zh/blog/", [], True, 80,
     r"^/zh/blog/[^/]+/?$", 5),
]

# Official sites that expose article links on a listing page but have no
# dependable RSS/sitemap article index. Each tuple is:
# (label, listing_url, include_pattern(path+query), keep_query, limit,
#  bootstrap_limit). Patterns deliberately exclude broad SEO/tutorial posts.
LINK_SOURCES = [
    ("Qwen Blog", "https://qwen.ai/blog", r"^/blog\?id=[A-Za-z0-9._-]+$", True, 50, 5),
    ("Kimi Blog", "https://www.kimi.com/blog", r"^/blog/[A-Za-z0-9._-]+/?$", False, 50, 5),
    ("MiniMax Blog", "https://www.minimax.io/blog",
     r"^/(?:blog|news)/[A-Za-z0-9._-]+/?$", False, 50, 5),
    ("Kling AI Updates", "https://kling.ai/blog",
     r"^/(?:release-note/release-notes/[A-Za-z0-9_-]+|blog/kling-ai-(?:introduces|launches|unveils|announces|releases|upgrades)[A-Za-z0-9_-]*)/?$",
     False, 50, 5),
    ("PixVerse Product Updates", "https://pixverse.ai/en/blog",
     r"^/en/blog/(?:pixverse-(?:launches|introduces|announces|updates|evolves|joins|closes|partners|r1|c1|v\d|global|special|game-engine|cli)|[^/]*-now-available-on-pixverse|captain-tsubasa-pixverse|un-ai-for-good[^/]*pixverse)[^/?]*/?$",
     False, 50, 5),
    ("Manus Blog", "https://manus.im/blog",
     r"^/blog/(?:manus-|introducing-|deep-dive-|projects-connectors|year-one|Context-Engineering|elevenlabs-connector|similarweb-manus)[^/?]*/?$",
     False, 80, 5),
]

# These official update hubs do not expose stable article URLs. We retain a
# normalized content hash and emit one dashboard item only when the page
# changes. The first run adds one visible marker so the source is immediately
# represented in the dashboard.
PAGE_SOURCES = [
    ("Z.ai Release Notes", "https://docs.z.ai/release-notes/new-released"),
    ("TRAE Blog", "https://www.trae.ai/blog"),
    ("Coze Changelog", "https://www.coze.com/changelog"),
]
SEEN_FILE = DATA_DIR / "rss-seen.json"
PAGE_STATE_FILE = DATA_DIR / "rss-page-state.json"


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


def fetch_feed(label: str, url: str, source_type: str = "rss") -> list[dict]:
    items = []
    try:
        # Fetch through our retrying HTTP client. Some podcast hosts return an
        # empty response to feedparser's default user agent.
        parsed = None
        for attempt in range(3):
            parsed = feedparser.parse(http_get(url, timeout=20))
            if parsed.entries:
                break
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
    except Exception as e:  # noqa: BLE001
        print(f"  !! {label}: parse error {type(e).__name__}: {e}")
        return items
    if not parsed.entries:
        print(f"  !! {label}: feed error {parsed.get('bozo_exception')}")
        return items
    for e in parsed.entries[:MAX_ITEMS_PER_FEED]:
        link = e.get("link") or ""
        if not link:
            # Podcast feeds may expose only an enclosure URL. It is still a
            # unique, playable destination and prevents valid episodes from
            # being silently dropped.
            link = next((
                item.get("href", "") for item in e.get("links", [])
                if item.get("rel") == "enclosure" and item.get("href")
            ), "")
        if not link:
            continue
        summary = strip_html(e.get("summary") or e.get("description") or "")
        items.append({
            "source": label,
            "source_type": source_type,
            "title": strip_html(e.get("title") or "(无标题)"),
            "url": link,
            "summary": summary[:SUMMARY_MAX],
            "published_at": entry_published(e),
        })
    print(f"  {label}: {len(items)} items")
    return items


def filter_youtube_items(items: list[dict], include_pattern: str | None) -> list[dict]:
    """Keep topic-relevant entries from broad YouTube publisher channels."""
    if not include_pattern:
        return items
    return [
        item for item in items
        if re.search(
            include_pattern,
            f"{item.get('title', '')} {item.get('summary', '')}",
            re.IGNORECASE,
        )
    ]


def http_get(url: str, timeout: int = 15, retries: int = 2) -> str:
    """GET text with small retry — some CDNs intermittently send truncated
    bodies (IncompleteRead) or drop the connection."""
    last = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = r.read()
                if not body.strip():
                    raise ValueError("empty response body")
                return body.decode("utf-8", "ignore")
        except Exception as e:  # noqa: BLE001
            last = e
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
    # A few Akamai-backed sites occasionally stall urllib's HTTP/1.1 chunked
    # reader while curl completes immediately over HTTP/2. GitHub runners ship
    # curl, so use it as a narrow final transport fallback.
    try:
        result = subprocess.run(
            ["curl", "-L", "--fail", "--silent", "--show-error",
             "--max-time", str(timeout), "--user-agent", UA, url],
            capture_output=True,
            timeout=timeout + 5,
        )
        # Some Akamai responses deliver the complete HTML body but never close
        # the connection, so curl exits 28 at max-time. The received body is
        # still useful for listing-link extraction.
        if result.stdout.strip():
            return result.stdout.decode("utf-8", "ignore")
    except Exception:
        pass
    raise last


def _sitemap_documents(sitemap_url: str, depth: int = 0,
                       visited: set[str] | None = None) -> list[str]:
    """Fetch urlset XML documents, following nested sitemap indexes."""
    visited = visited or set()
    if sitemap_url in visited or depth > 3:
        return []
    visited.add(sitemap_url)
    try:
        xml = http_get(sitemap_url, timeout=20)
    except Exception as e:  # noqa: BLE001
        print(f"  !! sitemap {sitemap_url}: {type(e).__name__}: {e}")
        return []

    if "<sitemapindex" not in xml:
        return [xml]

    documents = []
    child_urls = [
        html.unescape(loc.strip())
        for loc in re.findall(r"<loc>([^<]+)</loc>", xml)
    ]
    for child_url in child_urls[:40]:
        documents.extend(_sitemap_documents(child_url, depth + 1, visited))
    return documents


def sitemap_entries(sitemap_url: str, prefix: str) -> list[tuple[str, str | None]]:
    """Return [(url, lastmod_iso|None)] under `prefix`, newest first."""
    out = []
    for xml in _sitemap_documents(sitemap_url):
        for block in re.findall(r"<url>(.*?)</url>", xml, re.DOTALL):
            loc_m = re.search(r"<loc>([^<]+)</loc>", block)
            if not loc_m:
                continue
            loc = html.unescape(loc_m.group(1).strip())
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
_SITE_NAMES = r"(google deepmind|anthropic|cursor|runway( news| research)?|openai)"
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
_NUMERIC_VISIBLE_DATE = re.compile(
    r"(?<!\d)(20\d{2})\s*(?:[-/.]|年)\s*(\d{1,2})\s*"
    r"(?:[-/.]|月)\s*(\d{1,2})\s*日?(?!\d)")


def _visible_date(html_text: str) -> str | None:
    """Return the first English, numeric, or Chinese byline-style date."""
    m = _VISIBLE_DATE.search(html_text)
    if m:
        try:
            month = _MONTHS[m.group(1).lower()[:3]]
            return datetime(int(m.group(3)), month, int(m.group(2)),
                            tzinfo=timezone.utc).isoformat()
        except (KeyError, ValueError):
            return None

    m = _NUMERIC_VISIBLE_DATE.search(html_text)
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                            tzinfo=timezone.utc).isoformat()
        except ValueError:
            return None
    return None


def fetch_sitemap_source(label, sitemap_url, prefix, exclude, trust_lastmod,
                         limit, include_pattern, bootstrap_limit,
                         seen_urls: set[str]) -> tuple[list[dict], set[str]]:
    """Return (new item dicts, ALL article urls seen in this sitemap).

    New = URL appearing in the sitemap for the first time ever (not in the
    seen baseline). lastmod recency is deliberately ignored for newness —
    sites bulk-refresh lastmod on old pages. Established sources baseline on
    first run; newly configured sources can bootstrap a small explicit limit.
    """
    all_entries = [
        (u, m) for u, m in sitemap_entries(sitemap_url, prefix)
        if not any(x in u for x in exclude)
        and (not include_pattern or re.search(include_pattern, urlparse(u).path))
    ]
    all_urls = {u for u, _ in all_entries}
    if not all_urls:
        # sitemap fetch failed or empty — don't touch the baseline
        print(f"  {label}: sitemap empty/unreachable, skipping")
        return [], set()

    first_run = not (seen_urls & all_urls)
    if first_run and not bootstrap_limit:
        print(f"  {label}: baseline created ({len(all_urls)} urls), 0 ingested")
        return [], all_urls
    if first_run:
        new_entries = all_entries[:bootstrap_limit]
        print(f"  {label}: bootstrapping {len(new_entries)}/{len(all_urls)} urls")
    else:
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
            "source_type": "rss",
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


def listing_entries(listing_url: str, include_pattern: str,
                    keep_query: bool) -> list[str]:
    """Extract matching same-site article URLs in document order."""
    try:
        page = http_get(listing_url, timeout=15, retries=0)
    except Exception as e:  # noqa: BLE001
        print(f"  !! listing {listing_url}: {type(e).__name__}: {e}")
        return []

    page = page.replace(r"\/", "/")
    candidates = re.findall(r'href\s*=\s*["\']([^"\']+)', page, re.IGNORECASE)
    # Some SPAs assign routes in inline JavaScript instead of rendering links.
    candidates.extend(re.findall(
        r'["\']((?:https?://[^"\']+|/[^"\'\s<>]+))["\']', page
    ))

    base_host = urlparse(listing_url).netloc.lower().removeprefix("www.")
    seen = set()
    out = []
    for raw in candidates:
        absolute = urljoin(listing_url, html.unescape(raw))
        parsed = urlparse(absolute)
        host = parsed.netloc.lower().removeprefix("www.")
        if parsed.scheme not in {"http", "https"} or host != base_host:
            continue
        match_value = parsed.path + (f"?{parsed.query}" if parsed.query else "")
        if not re.search(include_pattern, match_value):
            continue
        if not keep_query:
            parsed = parsed._replace(query="")
        parsed = parsed._replace(fragment="")
        canonical = parsed.geturl()
        if canonical not in seen:
            seen.add(canonical)
            out.append(canonical)
    return out


def fetch_link_source(label, listing_url, include_pattern, keep_query, limit,
                      bootstrap_limit, seen_urls: set[str]) -> tuple[list[dict], set[str]]:
    """Fetch new article URLs discovered on an official listing page."""
    urls = listing_entries(listing_url, include_pattern, keep_query)
    all_urls = set(urls)
    if not all_urls:
        print(f"  {label}: listing empty/unreachable, skipping")
        return [], set()

    first_run = not (seen_urls & all_urls)
    if first_run:
        new_urls = urls[:bootstrap_limit]
        print(f"  {label}: bootstrapping {len(new_urls)}/{len(all_urls)} urls")
    else:
        new_urls = [url for url in urls if url not in seen_urls][:limit]

    def grab(url):
        title = desc = ""
        published = None
        try:
            title, desc, published = extract_page(http_get(url, timeout=10, retries=1))
        except Exception:
            pass
        return {
            "source": label,
            "source_type": "rss",
            "title": (title or slug_title(url))[:200],
            "url": url,
            "summary": desc[:SUMMARY_MAX],
            "published_at": published or datetime.now(timezone.utc).isoformat(),
        }

    items = []
    if new_urls:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            items = list(pool.map(grab, new_urls))
    print(f"  {label}: {len(all_urls)} on listing, {len(new_urls)} new, {len(items)} ingested")
    return items, all_urls


def page_signature(page: str) -> str:
    """Stable-enough hash for official hubs without individual article URLs."""
    title, desc, _ = extract_page(page)
    visible = re.sub(r"<script\b.*?</script>", "", page,
                     flags=re.IGNORECASE | re.DOTALL)
    visible = re.sub(r"<style\b.*?</style>", "", visible,
                     flags=re.IGNORECASE | re.DOTALL)
    visible = re.sub(r"<!--.*?-->", "", visible, flags=re.DOTALL)
    normalized = f"{title}\n{desc}\n{strip_html(visible)}"
    return hashlib.sha256(normalized.encode()).hexdigest()


def fetch_page_source(label: str, url: str,
                      previous_signature: str | None) -> tuple[list[dict], str | None]:
    """Emit a marker on first setup and one item for each later page change."""
    try:
        page = http_get(url, timeout=15, retries=1)
    except Exception as e:  # noqa: BLE001
        print(f"  !! page {url}: {type(e).__name__}: {e}")
        return [], None
    signature = page_signature(page)
    if signature == previous_signature:
        print(f"  {label}: unchanged")
        return [], signature

    title, desc, _ = extract_page(page)
    first_run = previous_signature is None
    item = {
        "source": label,
        "source_type": "rss",
        "title": title or (f"{label} 已加入监控" if first_run else f"{label} 页面更新"),
        "url": url,
        "summary": (desc or (
            "已加入官网更新监控，后续页面变化会自动记录。" if first_run
            else "官网更新页发生变化，请打开原页面查看最新内容。"
        ))[:SUMMARY_MAX],
        "published_at": datetime.now(timezone.utc).isoformat(),
        "dedupe_key": f"page:{label}:{signature}",
    }
    print(f"  {label}: {'baseline marker' if first_run else 'page changed'}")
    return [item], signature


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


def item_key(item: dict) -> str:
    return item.get("dedupe_key") or item["url"]


def trim_items(items, max_items: int) -> list[dict]:
    """Keep the newest item per represented source, then fill by recency."""
    ordered = sorted(items, key=sort_key, reverse=True)
    if len(ordered) <= max_items:
        return ordered

    source_heads = {}
    for item in ordered:
        source_heads.setdefault(item.get("source", ""), item)
    reserved_keys = {item_key(item) for item in source_heads.values()}
    selected = list(source_heads.values())
    if len(selected) < max_items:
        selected.extend(
            item for item in ordered
            if item_key(item) not in reserved_keys
        )
    return sorted(selected[:max_items], key=sort_key, reverse=True)


def main() -> int:
    DATA_DIR.mkdir(exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()

    existing = load_existing().get("items", [])
    existing_sources = {item.get("source") for item in existing}
    by_key = {item_key(it): it for it in existing}
    new_count = 0

    print(f"Fetching {len(FEEDS)} RSS feeds...")
    fresh = []
    for label, url in FEEDS:
        fresh.extend(fetch_feed(label, url))

    print(f"\nFetching {len(YOUTUBE_FEEDS)} YouTube feeds...")
    for label, channel_id, include_pattern in YOUTUBE_FEEDS:
        items = fetch_feed(label, f"{YOUTUBE_FEED_BASE}{channel_id}", "youtube")
        kept = filter_youtube_items(items, include_pattern)
        if include_pattern:
            print(f"  {label}: kept {len(kept)}/{len(items)} relevant videos")
        fresh.extend(kept)

    for it in fresh:
        key = item_key(it)
        if key not in by_key:
            it["first_seen_at"] = now
            by_key[key] = it
            new_count += 1
        else:
            # refresh title/summary in case of edits, keep first_seen_at
            prev = by_key[key]
            it["first_seen_at"] = prev.get("first_seen_at", now)
            by_key[key] = it

    print(f"\nChecking {len(SITEMAP_SOURCES)} sitemap sources...")
    try:
        seen_map = json.loads(SEEN_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        seen_map = {}
    for source in SITEMAP_SOURCES:
        (label, sm, prefix, exclude, trust_lastmod, limit,
         include_pattern, bootstrap_limit) = source
        seen = set(seen_map.get(label, []))
        if bootstrap_limit and label not in existing_sources:
            seen = set()
        items, all_urls = fetch_sitemap_source(label, sm, prefix, exclude,
                                               trust_lastmod, limit,
                                               include_pattern, bootstrap_limit,
                                               seen)
        for it in items:
            key = item_key(it)
            if key not in by_key:
                it["first_seen_at"] = now
                by_key[key] = it
                new_count += 1
        if all_urls:
            seen_map[label] = sorted(seen | all_urls)

    print(f"\nChecking {len(LINK_SOURCES)} official listing pages...")
    for source in LINK_SOURCES:
        label, listing_url, include_pattern, keep_query, limit, bootstrap_limit = source
        seen = set(seen_map.get(label, [])) if label in existing_sources else set()
        items, all_urls = fetch_link_source(
            label, listing_url, include_pattern, keep_query, limit,
            bootstrap_limit, seen,
        )
        for it in items:
            key = item_key(it)
            if key not in by_key:
                it["first_seen_at"] = now
                by_key[key] = it
                new_count += 1
        if all_urls:
            seen_map[label] = sorted(seen | all_urls)
    SEEN_FILE.write_text(json.dumps(seen_map, ensure_ascii=False, indent=1))

    print(f"\nChecking {len(PAGE_SOURCES)} official update hubs...")
    try:
        page_state = json.loads(PAGE_STATE_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        page_state = {}
    for label, url in PAGE_SOURCES:
        items, signature = fetch_page_source(label, url, page_state.get(label))
        if items:
            # A page hub is one logical source card, not an event archive.
            # Replace its previous marker/update rather than accumulating
            # repeated cards with the same destination URL.
            for key, existing_item in list(by_key.items()):
                if (existing_item.get("source") == label
                        and existing_item.get("dedupe_key", "").startswith("page:")):
                    del by_key[key]
        for it in items:
            key = item_key(it)
            if key not in by_key:
                it["first_seen_at"] = now
                by_key[key] = it
                new_count += 1
        if signature:
            page_state[label] = signature
    PAGE_STATE_FILE.write_text(json.dumps(page_state, ensure_ascii=False, indent=1))

    # Backfill older retained items created before source_type was introduced.
    for item in by_key.values():
        if "source_type" not in item:
            item["source_type"] = (
                "youtube" if item.get("source", "").endswith("YouTube") else "rss"
            )

    merged = trim_items(by_key.values(), MAX_ITEMS)

    result = {
        "updated_at": now,
        "count": len(merged),
        "new_count": new_count,
        "sources": [label for label, _ in FEEDS]
                   + [source[0] for source in YOUTUBE_FEEDS]
                   + [source[0] for source in SITEMAP_SOURCES]
                   + [source[0] for source in LINK_SOURCES]
                   + [source[0] for source in PAGE_SOURCES],
        "items": merged,
    }
    OUT_FILE.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nSaved {len(merged)} items ({new_count} new) to {OUT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
