"""Stage-1 offline backtest harness for the LLM scoring migration.

Reads past N days of raw-*.json, runs llm_scoring.run_llm_scoring on all tweets,
writes scored-*.json into data/backtest/, and produces a Markdown report with:
- Score distribution
- Category distribution
- Top 20 tweets (Telegram-format preview)
- Bottom 20 keep tweets (score 20-25, threshold stress-test)
- Random 20 drop tweets (false-negative check)
- LLM fallback log

Usage:
  source .env.local && python3 backtest_scoring.py --days 7
"""

import argparse
import json
import random
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path

import llm_scoring

DATA_DIR = Path(__file__).parent / "data"
BACKTEST_DIR = DATA_DIR / "backtest"


def collect_raw_tweets(days: int) -> list[dict]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    cutoff_str = cutoff.strftime("%Y-%m-%d")
    all_tweets: list[dict] = []
    for f in sorted(DATA_DIR.glob("raw-*.json")):
        # Filenames look like raw-2026-04-20-morning.json
        stem = f.stem  # raw-2026-04-20-morning
        try:
            date_part = "-".join(stem.split("-")[1:4])
        except IndexError:
            continue
        if date_part < cutoff_str:
            continue
        payload = json.loads(f.read_text())
        for t in payload.get("tweets", []):
            t["_source_file"] = f.name
            all_tweets.append(t)
    return all_tweets


def format_tweet_preview(t: dict) -> str:
    """Reproduces (roughly) the Telegram message format in Markdown."""
    score = t.get("total_score", 0)
    lines = [
        f"### [{score}分] @{t.get('screen_name', '?')} · {t.get('_source_file', '')}",
        f"📝 {t.get('summary', '') or (t.get('full_text', '') or '')[:150]}",
        "",
        f"📊 category={t.get('category', '?')}({t.get('category_points', 0)})  "
        f"info_gap={t.get('info_gap', '?')}({t.get('info_gap_points', 0)})  "
        f"signals={t.get('viral_signals', [])}(+{t.get('viral_signals_points', 0)})  "
        f"emotion={t.get('emotion', '?')}(+{t.get('emotion_points', 0)})  "
        f"fit={t.get('account_fit', '?')}(+{t.get('account_bonus', 0)})",
        "",
        "💡 angles:",
    ]
    for a in t.get("angles", []):
        lines.append(f"   - {a}")
    url = t.get("url", "")
    if url:
        lines.append(f"🔗 {url}")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=7, help="How many days of raw data to score")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for drop sampling")
    args = parser.parse_args()

    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)

    print(f"Collecting raw tweets from last {args.days} days...")
    raw = collect_raw_tweets(args.days)
    print(f"  {len(raw)} tweets")

    if not raw:
        print("No raw tweets found — nothing to score. Exiting.")
        return

    print("Running LLM scoring...")
    scored = llm_scoring.run_llm_scoring(raw)
    scored.sort(key=lambda t: t.get("total_score", 0), reverse=True)

    # Persist one combined scored file per backtest run
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_file = BACKTEST_DIR / f"scored-backtest-{ts}.json"
    out_file.write_text(json.dumps({
        "generated_at": ts,
        "days": args.days,
        "total": len(scored),
        "tweets": scored,
    }, ensure_ascii=False, indent=2))
    print(f"Saved {out_file}")

    # Report
    keep = [t for t in scored if t.get("verdict") == "keep"]
    drop = [t for t in scored if t.get("verdict") == "drop"]
    fallbacks = [t for t in scored if t.get("_fallback")]

    def bucket(s: int) -> str:
        if s >= 50: return ">=50"
        if s >= 35: return "35-49"
        if s >= 20: return "20-34"
        return "<20"

    dist = Counter(bucket(t.get("total_score", 0)) for t in scored)
    cat_dist = Counter(t.get("category", "?") for t in scored)

    bottom_keep = sorted(
        [t for t in keep if t.get("total_score", 0) <= 25],
        key=lambda t: t.get("total_score", 0),
    )[:20]
    sampled_drop = random.sample(drop, min(20, len(drop)))
    sampled_drop.sort(key=lambda t: -t.get("total_score", 0))

    report_lines = [
        f"# Backtest Report — {ts}",
        f"Days: {args.days}  Total tweets: {len(scored)}",
        f"Keep: {len(keep)}  Drop: {len(drop)}  Fallback: {len(fallbacks)}",
        "",
        "## Score Distribution",
    ]
    for k in [">=50", "35-49", "20-34", "<20"]:
        report_lines.append(f"- {k}: {dist.get(k, 0)}")
    report_lines += ["", "## Category Distribution"]
    for cat, n in cat_dist.most_common():
        report_lines.append(f"- {cat}: {n}")

    report_lines += ["", "## Top 20 Highest-Scored"]
    for t in scored[:20]:
        report_lines.append(format_tweet_preview(t))

    report_lines += ["", "## Bottom 20 Keep (score <= 25, threshold stress-test)"]
    for t in bottom_keep:
        report_lines.append(format_tweet_preview(t))

    report_lines += ["", "## Random 20 Drop Samples (false-negative check)"]
    for t in sampled_drop:
        report_lines.append(format_tweet_preview(t))

    if fallbacks:
        report_lines += ["", "## LLM Fallback Events"]
        for t in fallbacks:
            report_lines.append(f"- {t.get('url', '?')}: {t.get('angles', ['?'])[0]}")

    report_file = BACKTEST_DIR / f"backtest-report-{ts}.md"
    report_file.write_text("\n".join(report_lines))
    print(f"Saved {report_file}")


if __name__ == "__main__":
    main()
