"""
Fetch AI/tech tweets from Twitter via Apify xtdata/twitter-x-scraper.
Filters for tweets with >= 2000 likes, deduplicates across runs,
and saves raw results as structured JSON.
"""

import json
import os
import sys
import time
import hashlib
from datetime import datetime, timezone, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.parse import urlencode
from urllib.error import HTTPError

import llm_scoring

APIFY_TOKEN = os.environ["APIFY_TOKEN"]
APIFY_BASE = "https://api.apify.com/v2"
ACTOR_ID = "xtdata~twitter-x-scraper"
MIN_FAVES = 2000
MAX_ITEMS_PER_SEARCH = 200
POLL_INTERVAL = 15  # seconds
POLL_TIMEOUT = 300  # 5 minutes per actor

# --- Keyword groups ---
KEYWORD_SEARCHES = [
    '(Claude OR "Claude Code" OR "Opus 4.7" OR Anthropic OR AnthropicAI) min_faves:2000',
    '(OpenAI OR chatgpt OR sama OR "GPT-Image-2") min_faves:2000',
    '(Gemini OR "Google AI Studio" OR Notebooklm) min_faves:2000',
    '("Jensen Huang" OR NVIDIA) min_faves:2000',
    '(Cursor OR Huggingface OR Perplexity OR Antigravity OR Codex) min_faves:2000',
    '(vibecoding OR AIAgent OR "humanoid robot" OR "Humanoid Robots" OR "Embodied AI") min_faves:2000',
    '("DAN KOE" OR "Peter Steinberger" OR OpenClaw OR "Nano banana") min_faves:2000',
    '("open source" (AI OR voice OR "text to speech" OR "image generation" OR "video generation")) min_faves:2000',
]

# --- Monitored accounts (all in one list, auto-split into small batches) ---
ALL_ACCOUNTS = [
    "thedankoe", "paulg", "ycombinator", "petergyang", "lexfridman",
    "moltbook", "higgsfield", "perplexity_ai", "NVIDIAAI", "antigravity",
    "googlechrome", "nvidianewsroom", "AMDRadeon", "Microsoft", "NVIDIAGeForce",
    "nvidia", "gdb", "calinoracation", "skairam", "turtlesoupy",
    "BorisMPower", "dotey", "cursor_ai", "ChatGPTapp", "soraofficialapp",
    "OpenAINewsroom", "DarioAmodei", "lydiahallie", "felixrieseberg", "trq212",
    "bcherny", "AnthropicAI", "claudeai", "zarazhangrui", "karpathy",
    "openclaw", "steipete", "lukas_m_ziegler", "emmanuel_2m", "saniaspeaks_",
    "CyberRobooo", "godofprompt",
    "mikefutia", "ai_bread", "dwarkesh_sp", "PJaccetturo", "OpenAIDevs",
    "billpeeb", "gabriel1", "vista8", "adcock_brett", "reve",
    "skirano", "genel_ai", "tapehead_Lab", "n8n_io", "GoogleAIStudio",
    "ClementDelangue", "XRoboHub", "TheHumanoidHub", "Dr_Singularity", "arjunkhemani",
    "Sumanth_077", "aaditsh", "_avichawla", "startupideaspod", "NotebookLM",
    "GeminiApp", "joshwoodward", "VraserX", "OfficialLoganK", "naval",
    "levie", "GitHubProjects", "starter_story", "levelsio", "deedydas",
    "heyshrutimishra", "DKThomp", "rasbt", "DavidOndrej1", "mims",
    "arena", "SamuelAlbanie",
    "itsPaulAi", "snowmaker", "GoogleDeepMind", "GoogleLabs", "ivanboroja",
    "tranmautritam", "Siron93", "kimmonismus", "dickiebush", "awilkinson",
    "huggingface", "zoink", "gregisenberg", "localhost_4173", "aakashgupta",
    "MrBeast", "Google", "svpino", "venturetwins", "llama_index",
    "aivanlogic", "egeberkina", "antonosika", "Salmaaboukarr", "EHuanglu",
    "prompthero", "adskflowstudio", "AlexanderFYoung", "techhalla", "i_amnajaved",
    "DrJimFan", "op7418", "TechieBySA", "Kimi_Moonshot", "mckaywrigley",
    "freepik", "alex_prompter", "heyrobinai", "ciguleva", "Synthetic_Copy",
    "sama", "OpenAI",
    # Added 2026-04-17: OpenAI Codex/Atlas team
    "AriX", "JamesZmSun",
]
ACCOUNTS_PER_BATCH = 10  # Small batches for reliable results

NOISE_ACCOUNTS = [
    'actufoot_', '_befootball', 'lequipedusoir', 'blue_footy',
    'nongsiii', 'manusiakupu_2', 'mylifegemfourth',
    '_tickettoheaven', 'colorfulstageen', 'claudeluca_', 'claudeclawmark',
]


def apify_request(method, path, data=None):
    """Make an authenticated request to Apify API."""
    url = f"{APIFY_BASE}{path}?token={APIFY_TOKEN}"
    headers = {"Content-Type": "application/json"}
    body = json.dumps(data).encode() if data else None
    req = Request(url, data=body, headers=headers, method=method)
    try:
        with urlopen(req, timeout=60) as resp:
            return json.loads(resp.read())
    except HTTPError as e:
        print(f"  HTTP {e.code}: {e.read().decode()[:200]}", file=sys.stderr)
        raise


def start_actor(search_input):
    """Start an Apify actor run and return the run ID."""
    resp = apify_request("POST", f"/acts/{ACTOR_ID}/runs", search_input)
    return resp["data"]["id"]


def poll_run(run_id):
    """Poll until actor run finishes or times out."""
    elapsed = 0
    while elapsed < POLL_TIMEOUT:
        resp = apify_request("GET", f"/actor-runs/{run_id}")
        status = resp["data"]["status"]
        if status in ("SUCCEEDED", "FAILED", "ABORTED", "TIMED-OUT"):
            return resp["data"]
        time.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL
    print(f"  Run {run_id} timed out after {POLL_TIMEOUT}s", file=sys.stderr)
    return None


def get_dataset_items(dataset_id):
    """Fetch all fields from a dataset (no field restriction)."""
    url = (
        f"{APIFY_BASE}/datasets/{dataset_id}/items"
        f"?token={APIFY_TOKEN}"
        f"&limit=200"
    )
    req = Request(url)
    with urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def is_noise(tweet):
    """Hard blacklist check: tweets from NOISE_ACCOUNTS are dropped without LLM scoring."""
    author = tweet.get("author")
    if isinstance(author, dict):
        screen_name = (author.get("screen_name") or "").lower()
    elif isinstance(author, str):
        screen_name = author.lower()
    else:
        screen_name = ""
    return screen_name in {a.lower() for a in NOISE_ACCOUNTS}


def extract_tweet(tweet):
    """Extract standardized tweet data with all available fields."""
    author = tweet.get("author", {})
    if isinstance(author, dict):
        screen_name = author.get("screen_name", "unknown")
    else:
        screen_name = str(author) if author else "unknown"

    tweet_id = tweet.get("id_str") or tweet.get("id", "")

    url = tweet.get("url", "")
    if not url and screen_name != "unknown" and tweet_id:
        url = f"https://x.com/{screen_name}/status/{tweet_id}"

    # Extract quoted tweet text if present
    quoted_tweet = tweet.get("quoted_tweet") or tweet.get("quoted_status") or {}
    quoted_text = ""
    if isinstance(quoted_tweet, dict):
        quoted_text = quoted_tweet.get("full_text", "")

    # Extract media info if present
    media = tweet.get("media") or tweet.get("extended_entities", {}).get("media") or tweet.get("entities", {}).get("media") or []

    return {
        "tweet_id": str(tweet_id) if tweet_id else "",
        "url": url,
        "screen_name": screen_name,
        "created_at": tweet.get("created_at", ""),
        "full_text": tweet.get("full_text", ""),
        "lang": tweet.get("lang", ""),
        "favorite_count": tweet.get("favorite_count", 0),
        "retweet_count": tweet.get("retweet_count", 0),
        "reply_count": tweet.get("reply_count", 0),
        "view_count": tweet.get("view_count") or (tweet.get("views", {}).get("count") if isinstance(tweet.get("views"), dict) else 0),
        "quoted_tweet_text": quoted_text,
        "media": media,
    }


def build_account_search(accounts):
    """Build search query for a batch of accounts."""
    froms = " OR ".join(f"from:{a}" for a in accounts)
    return f"({froms}) min_faves:2000"


SEEN_URLS_RETENTION_DAYS = 7


def load_seen_ids(path):
    """Load previously seen tweet URLs as {url: date_str} dict.
    Backward-compatible: auto-migrates old list format."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    if isinstance(data, list):
        # Migrate old list format: mark all as 8 days ago so they expire on next save
        expire_date = (datetime.now(timezone.utc) - timedelta(days=8)).strftime("%Y-%m-%d")
        return {url: expire_date for url in data}
    return data  # already dict


def save_seen_ids(path, seen_dict, today_str):
    """Save seen URLs, dropping entries older than SEEN_URLS_RETENTION_DAYS."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=SEEN_URLS_RETENTION_DAYS)).strftime("%Y-%m-%d")
    cleaned = {url: date for url, date in seen_dict.items() if date >= cutoff}
    path.write_text(json.dumps(cleaned, indent=2))


def main():
    observe_mode = os.environ.get("OBSERVE_MODE") == "1"

    if observe_mode:
        output_dir = Path(__file__).parent / "data" / "observe"
    else:
        output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use Beijing time (UTC+8) for both period and date
    beijing_now = datetime.now(timezone.utc) + timedelta(hours=8)
    beijing_hour = beijing_now.hour
    today = beijing_now.strftime("%Y-%m-%d")

    if observe_mode:
        period = beijing_now.strftime("%H%M")  # e.g. "1430"
    else:
        period = "morning" if beijing_hour < 12 else "evening"

    if observe_mode:
        seen_file = Path(__file__).parent / "data" / "observe_seen_urls.json"
    else:
        seen_file = Path(__file__).parent / "data" / "seen_urls.json"

    # Load dedup set
    seen_urls = load_seen_ids(seen_file)

    # Date range: use Twitter's since: operator directly in search query
    since_date = (datetime.now(timezone.utc) - timedelta(hours=28)).strftime("%Y-%m-%d")

    # Build all search inputs
    searches = []

    # Keyword searches - append since: to each query
    for query in KEYWORD_SEARCHES:
        query_with_date = f"{query} since:{since_date}"
        searches.append({
            "searchTerms": [query_with_date],
            "maxItems": MAX_ITEMS_PER_SEARCH,
            "sort": "Top",
        })

    # Account searches (auto-split into small batches of ACCOUNTS_PER_BATCH)
    for i in range(0, len(ALL_ACCOUNTS), ACCOUNTS_PER_BATCH):
        batch = ALL_ACCOUNTS[i:i + ACCOUNTS_PER_BATCH]
        query = build_account_search(batch) + f" since:{since_date}"
        searches.append({
            "searchTerms": [query],
            "maxItems": MAX_ITEMS_PER_SEARCH,
            "sort": "Top",
        })

    # Launch all actors in parallel
    print(f"Launching {len(searches)} actor runs...")
    run_ids = []
    for i, search_input in enumerate(searches):
        print(f"  [{i+1}/{len(searches)}] Starting: {search_input['searchTerms'][0][:60]}...")
        run_id = start_actor(search_input)
        run_ids.append(run_id)
        print(f"    Run ID: {run_id}")

    # Poll all runs
    print(f"\nPolling {len(run_ids)} runs...")
    all_tweets = []
    for i, run_id in enumerate(run_ids):
        print(f"  [{i+1}/{len(run_ids)}] Polling {run_id}...")
        run_data = poll_run(run_id)
        if not run_data:
            print(f"    SKIPPED (timeout)")
            continue
        if run_data["status"] != "SUCCEEDED":
            print(f"    FAILED: {run_data['status']}")
            continue

        dataset_id = run_data.get("defaultDatasetId")
        if not dataset_id:
            print(f"    No dataset")
            continue

        items = get_dataset_items(dataset_id)
        print(f"    Got {len(items)} items")

        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=28)
        for item in items:
            faves = item.get("favorite_count", 0)
            if faves < MIN_FAVES:
                continue
            # Hard date filter: discard tweets older than 28 hours
            created_at = item.get("created_at", "")
            if created_at:
                try:
                    tweet_time = parsedate_to_datetime(created_at)
                    if tweet_time < cutoff_time:
                        continue
                except Exception:
                    pass  # If parse fails, keep the tweet
            if is_noise(item):
                continue
            tweet = extract_tweet(item)
            if tweet["url"] and tweet["url"] not in seen_urls:
                all_tweets.append(tweet)
                seen_urls[tweet["url"]] = today

    # Sort by likes descending
    all_tweets.sort(key=lambda t: t["favorite_count"], reverse=True)

    # --- Save RAW results (all tweets above MIN_FAVES, before AI filter) ---
    raw_result = {
        "date": today,
        "period": period,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "total_tweets": len(all_tweets),
        "tweets": all_tweets,
    }
    raw_file = output_dir / f"raw-{today}-{period}.json"
    raw_file.write_text(json.dumps(raw_result, ensure_ascii=False, indent=2))
    print(f"\nSaved {len(all_tweets)} raw tweets to {raw_file}")

    # --- LLM scoring (replaces regex candidate + final filters) ---
    print(f"\nLLM scoring {len(all_tweets)} tweets via OpenRouter (Haiku 4.5)...")
    scored_tweets = llm_scoring.run_llm_scoring(all_tweets)

    # Sort by total_score descending
    scored_tweets.sort(key=lambda t: t.get("total_score", 0), reverse=True)

    fallback_count = sum(1 for t in scored_tweets if t.get("_fallback"))
    keep_tweets = [t for t in scored_tweets if t.get("verdict") == "keep"]
    drop_tweets = [t for t in scored_tweets if t.get("verdict") == "drop"]

    def _bucket(t):
        s = t.get("total_score", 0)
        if s >= 50:
            return "must_do"
        if s >= 35:
            return "priority"
        if s >= 20:
            return "backup"
        return "drop"

    score_distribution = {">=50": 0, "35-49": 0, "20-34": 0, "<20": 0}
    for t in scored_tweets:
        s = t.get("total_score", 0)
        if s >= 50:
            score_distribution[">=50"] += 1
        elif s >= 35:
            score_distribution["35-49"] += 1
        elif s >= 20:
            score_distribution["20-34"] += 1
        else:
            score_distribution["<20"] += 1

    # --- Save scored-*.json (ALL scored tweets with full breakdown) ---
    scored_result = {
        "date": today,
        "period": period,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "raw_count": len(all_tweets),
        "scored_count": len(scored_tweets),
        "keep_count": len(keep_tweets),
        "drop_count": len(drop_tweets),
        "fallback_count": fallback_count,
        "score_distribution": score_distribution,
        "tweets": scored_tweets,
    }
    scored_file = output_dir / f"scored-{today}-{period}.json"
    scored_file.write_text(json.dumps(scored_result, ensure_ascii=False, indent=2))
    print(f"Saved {len(scored_tweets)} scored tweets to {scored_file}")
    print(f"  Distribution: {score_distribution}")
    if fallback_count:
        print(f"  !! LLM fallback count: {fallback_count}")

    # --- Save final-*.json (tier-grouped, keep only) ---
    tiers = {"must_do": [], "priority": [], "backup": []}
    for t in keep_tweets:
        b = _bucket(t)
        if b in tiers:
            tiers[b].append(t)

    final_result = {
        "date": today,
        "period": period,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "tier_counts": {k: len(v) for k, v in tiers.items()},
        "tiers": tiers,
    }
    final_file = output_dir / f"final-{today}-{period}.json"
    final_file.write_text(json.dumps(final_result, ensure_ascii=False, indent=2))
    print(f"Saved final tiers: must_do={len(tiers['must_do'])} "
          f"priority={len(tiers['priority'])} backup={len(tiers['backup'])}")

    # Update seen URLs (keep last 7 days only)
    save_seen_ids(seen_file, seen_urls, today)


if __name__ == "__main__":
    main()
