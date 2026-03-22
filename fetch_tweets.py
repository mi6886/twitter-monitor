"""
Fetch AI/tech tweets from Twitter via Apify xtdata/twitter-x-scraper.
Filters for tweets with >= 5000 likes, deduplicates across runs,
and saves results as JSON for Claude to process.
"""

import json
import os
import sys
import time
import re
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
MAX_ITEMS_PER_SEARCH = 100
POLL_INTERVAL = 15  # seconds
POLL_TIMEOUT = 300  # 5 minutes per actor

# --- Keyword groups ---
KEYWORD_SEARCHES = [
    '(Claude OR "Claude Code" OR AnthropicAI) min_faves:2000',
    '(OpenAI OR chatgpt OR sama) min_faves:2000',
    '(Gemini OR "Google AI Studio" OR Notebooklm) min_faves:2000',
    '("Jensen Huang" OR NVIDIA) min_faves:2000',
    '(Cursor OR Huggingface OR Perplexity OR Antigravity) min_faves:2000',
    '(vibecoding OR AIAgent OR "humanoid robot" OR "Humanoid Robots" OR "Embodied AI") min_faves:2000',
    '("DAN KOE" OR "Peter Steinberger" OR OpenClaw OR "Nano banana") min_faves:2000',
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
]
ACCOUNTS_PER_BATCH = 10  # Small batches for reliable results

# --- Noise filters ---
NOISE_PATTERNS = [
    r'claude le roy', r'claude mak[eé]l[eé]l[eé]', r'jean[- ]claude van damme',
    r'chuck norris.*claude', r'claude.*chelsea', r'claude.*caicedo',
    r'claude.*maroc', r'claude.*champion', r'claude.*magouille',
    r'geminifourth', r'gemini.*fourth', r'tickettoheaven', r'gmmtv',
    r'seme.*uke', r'uke.*seme', r'gemini.*pond',
    r'mario kart.*antigravity',
    r'virtual singer.*cursor', r'cursor set',
    r'aquarius.*leo.*scorpio',
]

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
    """Fetch items from a dataset with field selection."""
    url = (
        f"{APIFY_BASE}/datasets/{dataset_id}/items"
        f"?token={APIFY_TOKEN}"
        f"&fields=url,full_text,favorite_count,author,created_at"
        f"&limit=100"
    )
    req = Request(url)
    with urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def is_noise(tweet):
    """Check if a tweet matches noise patterns."""
    text = (tweet.get("full_text") or "").lower()
    screen_name = ""
    author = tweet.get("author")
    if isinstance(author, dict):
        screen_name = (author.get("screen_name") or "").lower()
    elif isinstance(author, str):
        screen_name = author.lower()

    if screen_name in [a.lower() for a in NOISE_ACCOUNTS]:
        return True
    for pattern in NOISE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def extract_tweet(tweet):
    """Extract standardized tweet data."""
    author = tweet.get("author", {})
    if isinstance(author, dict):
        screen_name = author.get("screen_name", "unknown")
    else:
        screen_name = str(author) if author else "unknown"

    url = tweet.get("url", "")
    if not url and screen_name != "unknown":
        tweet_id = tweet.get("id_str") or tweet.get("id", "")
        if tweet_id:
            url = f"https://x.com/{screen_name}/status/{tweet_id}"

    return {
        "screen_name": screen_name,
        "full_text": tweet.get("full_text", ""),
        "favorite_count": tweet.get("favorite_count", 0),
        "url": url,
        "created_at": tweet.get("created_at", ""),
    }


def build_account_search(accounts):
    """Build search query for a batch of accounts."""
    froms = " OR ".join(f"from:{a}" for a in accounts)
    return f"({froms}) min_faves:2000"


def load_seen_ids(path):
    """Load previously seen tweet URLs to deduplicate."""
    if path.exists():
        return set(json.loads(path.read_text()))
    return set()


def save_seen_ids(path, seen):
    """Save seen tweet URLs."""
    path.write_text(json.dumps(sorted(seen), indent=2))


def main():
    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(exist_ok=True)

    # Use Beijing time (UTC+8) for period determination
    beijing_hour = (datetime.now(timezone.utc).hour + 8) % 24
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    period = "morning" if beijing_hour < 12 else "evening"
    output_file = output_dir / f"feed-{today}-{period}.json"
    seen_file = output_dir / "seen_urls.json"

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
                seen_urls.add(tweet["url"])

    # Sort by likes descending
    all_tweets.sort(key=lambda t: t["favorite_count"], reverse=True)

    # Save results
    result = {
        "date": today,
        "period": period,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "total_tweets": len(all_tweets),
        "tweets": all_tweets,
    }
    output_file.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nSaved {len(all_tweets)} tweets to {output_file}")

    # Update seen URLs (keep last 2000 to prevent unbounded growth)
    seen_list = sorted(seen_urls)
    if len(seen_list) > 2000:
        seen_list = seen_list[-2000:]
    save_seen_ids(seen_file, seen_list)

    # Also save as latest for easy access
    latest_file = output_dir / f"feed-latest-{period}.json"
    latest_file.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Saved latest to {latest_file}")


if __name__ == "__main__":
    main()
