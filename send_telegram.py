"""
Read the latest final-{date}-{period}.json and send Top N tweets to Telegram.
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
TOP_N = 15


def send_telegram(text):
    """Send a message via Telegram Bot API (MarkdownV2)."""
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
        print(f"Telegram API error {e.code}: {e.read().decode()[:300]}", file=sys.stderr)
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
    # Remove leading RT/reply markers
    text = re.sub(r'^(RT\s+)?@\w+[:\s]*', '', text).strip()
    # Take the first sentence if it fits
    for sep in ['. ', '! ', '? ', '。', '！', '？']:
        idx = text.find(sep)
        if 0 < idx <= max_len:
            return text[:idx + len(sep)].strip()
    # Otherwise truncate at word boundary
    if len(text) <= max_len:
        return text
    cut = text[:max_len].rsplit(' ', 1)[0]
    return cut + '...'


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

    # Telegram message limit is 4096 chars
    if len(message) > 4096:
        message = message[:4090] + "\\.\\.\\."

    print(f"Sending {len(top)} tweets to Telegram...")
    send_telegram(message)
    print("Sent successfully.")


if __name__ == "__main__":
    main()
