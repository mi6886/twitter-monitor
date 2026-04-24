"""One-off: score a single raw-{date}-{period}.json with the new LLM pipeline.

Usage:
  source .env.local && python3 scripts/score_one_period.py 2026-04-24 morning
"""
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

# Make project root importable when invoked as `python3 scripts/score_one_period.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import llm_scoring


def main():
    if len(sys.argv) != 3:
        print("Usage: score_one_period.py <YYYY-MM-DD> <morning|evening>")
        sys.exit(2)
    date, period = sys.argv[1], sys.argv[2]
    raw_file = Path(f"data/raw-{date}-{period}.json")
    payload = json.loads(raw_file.read_text())
    tweets = payload["tweets"]
    print(f"Scoring {len(tweets)} tweets from {raw_file.name}...", flush=True)

    scored = llm_scoring.run_llm_scoring(tweets)
    scored.sort(key=lambda t: t.get("total_score", 0), reverse=True)

    fallback_count = sum(1 for t in scored if t.get("_fallback"))
    keep = [t for t in scored if t.get("verdict") == "keep"]
    drop = [t for t in scored if t.get("verdict") == "drop"]

    dist = {">=50": 0, "35-49": 0, "20-34": 0, "<20": 0}
    for t in scored:
        s = t.get("total_score", 0)
        if s >= 50: dist[">=50"] += 1
        elif s >= 35: dist["35-49"] += 1
        elif s >= 20: dist["20-34"] += 1
        else: dist["<20"] += 1

    out_scored = {
        "date": date, "period": period,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "raw_count": len(tweets), "scored_count": len(scored),
        "keep_count": len(keep), "drop_count": len(drop),
        "fallback_count": fallback_count, "score_distribution": dist,
        "tweets": scored,
    }
    Path(f"data/scored-{date}-{period}.json").write_text(
        json.dumps(out_scored, ensure_ascii=False, indent=2)
    )

    tiers = {"must_do": [], "priority": [], "backup": []}
    for t in keep:
        s = t.get("total_score", 0)
        if s >= 50: tiers["must_do"].append(t)
        elif s >= 35: tiers["priority"].append(t)
        elif s >= 20: tiers["backup"].append(t)

    out_final = {
        "date": date, "period": period,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "tier_counts": {k: len(v) for k, v in tiers.items()},
        "tiers": tiers,
    }
    # Use a -llm suffix so we don't clobber the regex-era final file
    Path(f"data/final-{date}-{period}-llm.json").write_text(
        json.dumps(out_final, ensure_ascii=False, indent=2)
    )

    print()
    print(f"=== Results for {date} {period} ===")
    print(f"raw: {len(tweets)} | keep: {len(keep)} | drop: {len(drop)} | fallback: {fallback_count}")
    print(f"distribution: {dist}")
    print(f"tier_counts: {out_final['tier_counts']}")


if __name__ == "__main__":
    main()
