"""
Read the latest final-{date}-{period}.json and send Top N tweets to Telegram.
Includes dedup (won't re-send identical content) and verified receipt logging.
"""

import hashlib
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
TOP_N = 15


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


def make_summary(text, max_len=100):
    """Create a one-line summary from cleaned tweet text."""
    text = clean_tweet_text(text)
    text = re.sub(r'^(RT\s+)?@\w+[:\s]*', '', text).strip()
    text = re.sub(r'^\s*(\d+[\.\)]\s*|[•▸►\u2014\u2013-]\s*)', '', text).strip()
    text = re.sub(r'\s+(\d+[\.\)]\s)', ' ', text)
    for sep in ['. ', '! ', '? ', '\u3002', '\uff01', '\uff1f']:
        idx = text.find(sep)
        if 0 < idx <= max_len:
            return text[:idx + len(sep)].strip()
    if len(text) <= max_len:
        return text
    cut = text[:max_len].rsplit(' ', 1)[0]
    cut = re.sub(r'\s+\d+[\.\)]?\s*$', '', cut)
    cut = re.sub(r'\s+[•▸►\u2014\u2013-]\s*$', '', cut)
    return cut + '...'


# --- Dedup ---

def compute_digest(date, period, tweet_urls):
    """Stable digest for a specific date+period+content combination."""
    key = f"{date}|{period}|{'|'.join(sorted(tweet_urls))}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def load_sent_log(path):
    """Load set of previously sent digests."""
    if not path.exists():
        return set()
    try:
        return set(json.loads(path.read_text()))
    except (json.JSONDecodeError, TypeError):
        return set()


def save_sent_log(path, digests):
    """Save sent digests (keep last 50 to avoid unbounded growth)."""
    recent = sorted(digests)[-50:]
    path.write_text(json.dumps(recent, indent=2))


# --- Main ---

def main():
    data_dir = Path(__file__).parent / "data"

    # Determine current period (Beijing time)
    beijing_now = datetime.now(timezone.utc) + timedelta(hours=8)
    today = beijing_now.strftime("%Y-%m-%d")
    period = "morning" if beijing_now.hour < 12 else "evening"

    final_file = data_dir / f"final-{today}-{period}.json"
    if not final_file.exists():
        print(f"No final file found: {final_file}")
        sys.exit(0)

    data = json.loads(final_file.read_text())
    tweets = data.get("tweets", [])

    if not tweets:
        print("No tweets in final file, skipping Telegram.")
        sys.exit(0)

    # --- Dedup check ---
    sent_log_file = data_dir / "telegram_sent.json"
    tweet_urls = [t.get("url", "") for t in tweets]
    digest = compute_digest(today, period, tweet_urls)

    sent_digests = load_sent_log(sent_log_file)
    if digest in sent_digests:
        print(f"SKIP: digest {digest} already sent for {today}-{period}. No duplicate push.")
        sys.exit(0)

    # --- Build message ---
    top = tweets[:TOP_N]
    period_label = "Morning" if period == "morning" else "Evening"
    header = escape_md2(f"AI News | {today} {period_label} | {data.get('final_count', len(tweets))} tweets")

    lines = [f"*{header}*\n"]
    for i, tw in enumerate(top, 1):
        author = escape_md2(f"@{tw.get('screen_name', '?')}")
        likes = escape_md2(fmt_likes(tw.get('favorite_count', 0)))
        summary = escape_md2(make_summary(tw.get("full_text", "")))
        url = tw.get("url", "")

        lines.append(f"{i}\\. {author} \\| {likes} likes")
        lines.append(summary)
        if url:
            lines.append(f"[Link]({url})\n")
        else:
            lines.append("")

    message = "\n".join(lines)

    if len(message) > 4096:
        message = message[:4090] + "\\.\\.\\."

    # --- Send with verified receipt ---
    print(f"Sending {len(top)} tweets to Telegram (digest={digest})...")
    result = send_telegram(message)

    # Verify response
    ok = result.get("ok")
    msg = result.get("result", {})
    message_id = msg.get("message_id")
    chat_id = msg.get("chat", {}).get("id")
    date_ts = msg.get("date")

    print(f"  ok={ok}")
    print(f"  chat_id={chat_id}")
    print(f"  message_id={message_id}")
    print(f"  date={date_ts}")

    if not ok or not message_id:
        print("FATAL: Telegram API did not return ok=true or message_id.", file=sys.stderr)
        print(f"Full response: {json.dumps(result)}", file=sys.stderr)
        sys.exit(1)

    # --- Record digest to prevent re-send ---
    sent_digests.add(digest)
    save_sent_log(sent_log_file, sent_digests)

    print(f"Verified: message delivered (message_id={message_id}).")


if __name__ == "__main__":
    main()
