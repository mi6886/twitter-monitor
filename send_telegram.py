"""
Read the latest final-{date}-{period}.json and send tier-grouped tweets to
Telegram (one message per tweet). Dedup is URL-keyed so partial re-runs don't
double-send.
"""

import json
import html
import os
import re
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]


def send_telegram(text):
    """Send a message via Telegram Bot API (MarkdownV2). Returns parsed JSON."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = json.dumps({
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "MarkdownV2",
        "disable_web_page_preview": True,
    }).encode()
    req = Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except HTTPError as e:
        body = e.read().decode()[:300]
        print(f"Telegram API HTTP {e.code}: {body}", file=sys.stderr)
        raise


def escape_md2(text):
    """Escape special characters for Telegram MarkdownV2."""
    for ch in r'\_*[]()~`>#+-=|{}.!':
        text = text.replace(ch, f'\\{ch}')
    return text


def fmt_likes(n):
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def clean_tweet_text(text):
    """Clean tweet text: unescape HTML entities, remove t.co links, collapse whitespace."""
    text = html.unescape(text)
    text = re.sub(r'https?://t\.co/\S+', '', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()


# --- Dedup (URL-keyed) ---

def load_sent_log(path):
    """Load set of previously sent tweet URLs.

    Backward-compat: prior schema stored short digest hashes (<=16 chars).
    If we detect that, treat as empty and let the new URL-keyed log
    repopulate naturally."""
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, TypeError):
        return set()
    if not isinstance(data, list):
        return set()
    # Old digest format detection: short, no scheme, no tweet path
    if data and all(
        isinstance(x, str) and len(x) <= 16 and "://" not in x
        for x in data
    ):
        return set()
    return {x for x in data if isinstance(x, str)}


def save_sent_log(path, urls):
    """Save sent URLs (bounded to the most recent 500 to avoid unbounded growth)."""
    recent = sorted(urls)[-500:]
    path.write_text(json.dumps(recent, indent=2))


# --- Tiered message rendering ---

TIER_META = [
    ("must_do", "🔥 必做（≥50分）"),
    ("priority", "⭐ 优先（35-49分）"),
    ("backup", "💡 备选（20-34分）"),
]


def format_tweet_message(tweet):
    """Return a MarkdownV2-escaped single-message body for one scored tweet."""
    score = tweet.get("total_score", 0)
    author = f"@{tweet.get('screen_name', '?')}"
    summary = tweet.get("summary") or clean_tweet_text(tweet.get("full_text", ""))[:150]
    angles = tweet.get("angles") or []

    # Score breakdown line
    parts = []
    cat = tweet.get("category")
    if cat:
        parts.append(f"{cat}({tweet.get('category_points', 0)})")
    gap = tweet.get("info_gap")
    if gap and tweet.get("info_gap_points", 0):
        parts.append(f"{gap}({tweet.get('info_gap_points', 0)})")
    signals = tweet.get("viral_signals") or []
    if signals and tweet.get("viral_signals_points", 0):
        parts.append(f"{'·'.join(signals)}(+{tweet.get('viral_signals_points', 0)})")
    emo = tweet.get("emotion")
    if emo and tweet.get("emotion_points", 0):
        parts.append(f"{emo}(+{tweet.get('emotion_points', 0)})")
    if tweet.get("actionability_points", 0):
        parts.append(f"可触达(+{tweet.get('actionability_points', 0)})")
    fit = tweet.get("account_fit")
    if fit and tweet.get("account_bonus", 0):
        parts.append(f"{fit}(+{tweet.get('account_bonus', 0)})")
    breakdown_line = " + ".join(parts) + f" = {score}" if parts else f"总分 {score}"

    lines = [
        f"*\\[{score}分\\] {escape_md2(author)}*",
        "",
        f"📝 {escape_md2(summary)}",
        "",
        f"📊 {escape_md2(breakdown_line)}",
    ]
    if angles:
        lines.append("")
        lines.append("💡 选题切入：")
        for i, a in enumerate(angles, 1):
            lines.append(f"  {i}\\. {escape_md2(a)}")
    url = tweet.get("url", "")
    if url:
        lines.append("")
        lines.append(f"🔗 {escape_md2(url)}")
    return "\n".join(lines)


# --- Main ---

def main():
    observe_mode = os.environ.get("OBSERVE_MODE") == "1"
    data_dir = Path(__file__).parent / ("data/observe" if observe_mode else "data")

    beijing_now = datetime.now(timezone.utc) + timedelta(hours=8)
    today = beijing_now.strftime("%Y-%m-%d")
    period = beijing_now.strftime("%H%M") if observe_mode else (
        "morning" if beijing_now.hour < 12 else "evening"
    )

    final_file = data_dir / f"final-{today}-{period}.json"
    if not final_file.exists():
        print(f"No final file found: {final_file}")
        sys.exit(0)

    data = json.loads(final_file.read_text())
    tiers = data.get("tiers", {})
    total_tweets = sum(len(tiers.get(k, [])) for k, _ in TIER_META)
    if total_tweets == 0:
        print("No tweets in any tier — skipping Telegram.")
        sys.exit(0)

    sent_log_file = data_dir / "telegram_sent.json"
    sent_urls = load_sent_log(sent_log_file) if not observe_mode else set()

    sent_this_run = 0
    for tier_key, header_text in TIER_META:
        tweets = tiers.get(tier_key, [])
        if not tweets:
            continue
        # Header for this tier
        header_body = f"*{escape_md2(header_text)}*  \\({len(tweets)} 条\\)"
        send_telegram(header_body)

        for tw in tweets:
            url = tw.get("url", "")
            if not observe_mode and url and url in sent_urls:
                continue
            body = format_tweet_message(tw)
            if len(body) > 4090:
                body = body[:4080] + "\\.\\.\\."
            result = send_telegram(body)
            if not result.get("ok"):
                print(f"Telegram failed for {url}: {result}", file=sys.stderr)
                sys.exit(1)
            if not observe_mode and url:
                sent_urls.add(url)
            sent_this_run += 1

    if not observe_mode:
        save_sent_log(sent_log_file, sent_urls)

    print(f"Delivered {sent_this_run} tweet messages across {len(TIER_META)} tiers.")


if __name__ == "__main__":
    main()
