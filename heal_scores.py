"""Self-healing re-score.

Each cron run (after the normal fetch+score) scans recent scored-*.json for a
high LLM-fallback rate — the signature of a transient OpenRouter outage that
left scores defaulted to 30/keep — and re-scores those periods from the
preserved raw-*.json using the now-healthy API.

Important: re-scoring needs a working API, so a run that is *itself* mid-outage
cannot fix itself. Healing therefore happens on the NEXT healthy run, cleaning
up the polluted past run (≤ ~12h later). A multi-day outage (credits/key down)
can't be healed at all — the 80%-fallback alert in the workflow is the backstop
for that.

Guards so it never thrashes credits:
  - only files within LOOKBACK_DAYS
  - only files at/above THRESHOLD fallback
  - one cheap API health check; abort entirely if it fails
  - overwrite only when re-scoring actually improves the rate
  - give up on a file after MAX_HEAL_ATTEMPTS (genuinely un-scorable content)
"""
import glob
import json
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import llm_scoring

DATA = Path(__file__).parent / "data"
THRESHOLD = 0.5          # fallback rate that triggers a heal
LOOKBACK_DAYS = 5
MAX_HEAL_ATTEMPTS = 2


def _fallback_rate(d: dict) -> float:
    sc = d.get("scored_count", 0)
    return (d.get("fallback_count", 0) / sc) if sc else 0.0


def _bucket(score: int) -> str:
    if score >= 50:
        return "must_do"
    if score >= 35:
        return "priority"
    if score >= 20:
        return "backup"
    return "drop"


def api_healthy() -> bool:
    try:
        from openai import OpenAI
        c = OpenAI(base_url="https://openrouter.ai/api/v1",
                   api_key=os.environ["OPENROUTER_API_KEY"])
        r = c.chat.completions.create(
            model=llm_scoring.MODEL,
            messages=[{"role": "user", "content": "ok"}],
            max_tokens=5, timeout=20,
        )
        return bool((r.choices[0].message.content or "").strip())
    except Exception as e:  # noqa: BLE001
        print(f"  API health check failed: {type(e).__name__} — aborting heal")
        return False


def heal_one(scored_path: Path, scored: dict) -> None:
    dt, period = scored.get("date"), scored.get("period")
    raw_path = DATA / f"raw-{dt}-{period}.json"
    if not raw_path.exists():
        print(f"  {scored_path.name}: no raw file, cannot heal")
        return
    tweets = json.loads(raw_path.read_text()).get("tweets", [])
    if not tweets:
        return

    old_rate = _fallback_rate(scored)
    attempts = scored.get("_heal_attempts", 0) + 1
    print(f"  {scored_path.name}: re-scoring {len(tweets)} tweets "
          f"(was {old_rate:.0%} fallback, attempt {attempts})")

    rescored = llm_scoring.run_llm_scoring(tweets)
    rescored.sort(key=lambda t: t.get("total_score", 0), reverse=True)
    fb = sum(1 for t in rescored if t.get("_fallback"))
    new_rate = fb / len(rescored) if rescored else 0.0

    if new_rate >= old_rate:
        print(f"    no improvement ({new_rate:.0%}); recording attempt, keeping old data")
        scored["_heal_attempts"] = attempts
        scored_path.write_text(json.dumps(scored, ensure_ascii=False, indent=2))
        return

    keep = [t for t in rescored if t.get("verdict") == "keep"]
    drop = [t for t in rescored if t.get("verdict") == "drop"]
    dist = {">=50": 0, "35-49": 0, "20-34": 0, "<20": 0}
    for t in rescored:
        s = t.get("total_score", 0)
        if s >= 50:
            dist[">=50"] += 1
        elif s >= 35:
            dist["35-49"] += 1
        elif s >= 20:
            dist["20-34"] += 1
        else:
            dist["<20"] += 1

    now = datetime.now(timezone.utc).isoformat()
    scored_path.write_text(json.dumps({
        "date": dt, "period": period,
        "fetched_at": scored.get("fetched_at", now),
        "raw_count": len(tweets),
        "scored_count": len(rescored),
        "keep_count": len(keep),
        "drop_count": len(drop),
        "fallback_count": fb,
        "score_distribution": dist,
        "tweets": rescored,
        "_healed_at": now,
        "_heal_attempts": attempts,
    }, ensure_ascii=False, indent=2))

    tiers = {"must_do": [], "priority": [], "backup": []}
    for t in keep:
        b = _bucket(t.get("total_score", 0))
        if b in tiers:
            tiers[b].append(t)
    (DATA / f"final-{dt}-{period}.json").write_text(json.dumps({
        "date": dt, "period": period, "fetched_at": now,
        "tier_counts": {k: len(v) for k, v in tiers.items()},
        "tiers": tiers,
    }, ensure_ascii=False, indent=2))
    print(f"    healed: {old_rate:.0%} -> {new_rate:.0%} fallback "
          f"(must_do={len(tiers['must_do'])} priority={len(tiers['priority'])} "
          f"backup={len(tiers['backup'])})")


def main() -> int:
    cutoff = date.today() - timedelta(days=LOOKBACK_DAYS)
    candidates = []
    for f in sorted(glob.glob(str(DATA / "scored-*.json"))):
        try:
            d = json.loads(Path(f).read_text())
            fdate = date.fromisoformat(d.get("date", ""))
        except (ValueError, json.JSONDecodeError):
            continue
        if fdate < cutoff:
            continue
        if _fallback_rate(d) >= THRESHOLD and d.get("_heal_attempts", 0) < MAX_HEAL_ATTEMPTS:
            candidates.append((Path(f), d))

    if not candidates:
        print(f"Heal: nothing >= {THRESHOLD:.0%} fallback in last {LOOKBACK_DAYS}d.")
        return 0

    print(f"Heal: {len(candidates)} file(s) >= {THRESHOLD:.0%} fallback:")
    for p, _ in candidates:
        print(f"  - {p.name}")
    if not api_healthy():
        return 0
    for p, d in candidates:
        try:
            heal_one(p, d)
        except Exception as e:  # noqa: BLE001
            print(f"  !! {p.name} heal failed: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
