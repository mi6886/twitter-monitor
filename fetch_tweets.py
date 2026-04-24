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

    # --- Generate candidate file (first-pass coarse filter on raw) ---
    candidate_tweets = []
    filter_stats = {"monitored_account": 0, "ai_signal": 0,
                    "too_short": 0, "reject_pattern": 0, "no_ai_signal": 0}
    for tweet in all_tweets:
        passed, reason = candidate_filter(tweet)
        filter_stats[reason] = filter_stats.get(reason, 0) + 1
        if passed:
            tweet_with_source = dict(tweet)
            tweet_with_source["candidate_reason"] = reason
            candidate_tweets.append(tweet_with_source)

    candidate_result = {
        "date": today,
        "period": period,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "raw_count": len(all_tweets),
        "candidate_count": len(candidate_tweets),
        "filter_stats": filter_stats,
        "tweets": candidate_tweets,
    }
    candidate_file = output_dir / f"candidate-{today}-{period}.json"
    candidate_file.write_text(json.dumps(candidate_result, ensure_ascii=False, indent=2))
    print(f"Candidate filter: {len(all_tweets)} raw → {len(candidate_tweets)} candidate")
    print(f"  Filter stats: {filter_stats}")
    print(f"Saved {len(candidate_tweets)} candidate tweets to {candidate_file}")

    # --- Generate final + review files (3-way disposition on candidate) ---
    keep_tweets = []
    review_tweets = []
    drop_hard_tweets = []
    disposition_stats = {}
    for tweet in candidate_tweets:
        disposition, reason = final_filter(tweet)
        disposition_stats[reason] = disposition_stats.get(reason, 0) + 1
        tw = dict(tweet)
        tw["disposition"] = disposition
        tw["disposition_reason"] = reason
        if disposition == "keep":
            keep_tweets.append(tw)
        elif disposition == "review":
            review_tweets.append(tw)
        else:
            drop_hard_tweets.append(tw)

    # Save final file (keep only)
    final_result = {
        "date": today,
        "period": period,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "raw_count": len(all_tweets),
        "candidate_count": len(candidate_tweets),
        "final_count": len(keep_tweets),
        "review_count": len(review_tweets),
        "drop_hard_count": len(drop_hard_tweets),
        "disposition_stats": disposition_stats,
        "tweets": keep_tweets,
    }
    final_file = output_dir / f"final-{today}-{period}.json"
    final_file.write_text(json.dumps(final_result, ensure_ascii=False, indent=2))

    # Save review file (review only)
    review_result = {
        "date": today,
        "period": period,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "review_count": len(review_tweets),
        "tweets": review_tweets,
    }
    review_file = output_dir / f"review-{today}-{period}.json"
    review_file.write_text(json.dumps(review_result, ensure_ascii=False, indent=2))

    print(f"Final filter: {len(candidate_tweets)} candidate → "
          f"{len(keep_tweets)} keep / {len(review_tweets)} review / "
          f"{len(drop_hard_tweets)} drop_hard")
    print(f"  Disposition stats: {disposition_stats}")
    print(f"Saved {len(keep_tweets)} final tweets to {final_file}")
    print(f"Saved {len(review_tweets)} review tweets to {review_file}")

    # Update seen URLs (keep last 7 days only)
    save_seen_ids(seen_file, seen_urls, today)


if __name__ == "__main__":
    main()
