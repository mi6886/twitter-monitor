# Twitter Monitor - Project Context

## Project Overview
AI/Tech tweet monitoring system that automatically scrapes, filters, and pushes curated AI news to Telegram and a GitHub Pages website. Designed to surface content suitable for AI media account topic selection.

## Architecture
```
Apify (scrape Twitter) → fetch_tweets.py (filter) → data/*.json → GitHub Pages (website) + Telegram (push)
```

Automated via GitHub Actions, runs twice daily (Beijing time 07:00 and 19:00).

## Repository
- **GitHub**: https://github.com/mi6886/twitter-monitor
- **Website**: https://mi6886.github.io/twitter-monitor/
- **Local path**: /Users/elainewang/Downloads/twitter-monitor/

## File Structure

### Core Scripts
| File | Purpose |
|------|---------|
| `fetch_tweets.py` | Main script: Apify scraping + candidate filter + final/review filter + file output |
| `send_telegram.py` | Reads final JSON, formats top 15, sends to Telegram Bot API with delivery receipt |
| `test_filters.py` | Unit tests for filter rules |
| `index.html` | Website frontend (GitHub Pages SPA) |

### Configuration
| File | Purpose |
|------|---------|
| `.github/workflows/fetch-tweets.yml` | GitHub Actions workflow (cron: UTC 23:00 + UTC 11:00) |
| `.claude/launch.json` | Claude Code local preview config |
| `.gitignore` | Git ignore rules |

### Data Files (in `data/`)
| Pattern | Purpose |
|---------|---------|
| `raw-{date}-{period}.json` | Raw scraped tweets (all results from Apify) |
| `candidate-{date}-{period}.json` | After noise/spam removal |
| `final-{date}-{period}.json` | Curated high-value tweets (displayed as "精选") |
| `review-{date}-{period}.json` | Borderline tweets for manual review (displayed as "待审") |
| `seen_urls.json` | URL dedup dict with 7-day TTL |
| `telegram_sent.json` | Content digest log to prevent duplicate Telegram sends |
| `final-2026-01-01-morning.json` | Test sample (raw=0 scenario) |
| `final-2026-01-02-morning.json` | Test sample (candidate=0 scenario) |

### Deprecated (can be deleted)
| Path | Notes |
|------|-------|
| `data/observe/` | Temporary high-freq observation data, no longer in use |
| `data/observe_seen_urls.json` | Observation dedup file, no longer in use |

## Pipeline Flow
```
raw tweets (Apify)
  → candidate_filter() [remove noise, spam, non-AI accounts]
  → final_filter() [classify into keep/review/drop_hard]
    → final-{date}-{period}.json (精选)
    → review-{date}-{period}.json (待审)
  → send_telegram.py [push top 15 final to Telegram]
  → git commit + push [triggers GitHub Pages rebuild]
```

## Filter Rules (final_filter in fetch_tweets.py)

### DROP_HARD (discarded)
- Rule 1: Text < 20 chars → `too_brief`
- Rule 2: Offensive content → `offensive`
- Rule 3: Off-topic (sports/gaming/music/crypto/Linux) → `off_topic`
- Rule 4: Gossip/engagement bait → `gossip_or_bait`

### KEEP (enters 精选)
- Rule 5: Monitored account + 2000+ likes + AI/tech relevant → `monitored_popular`
- Rule 6: Strong info signal (dollar amounts, API, benchmark, tutorial, funding, shutdown) → `info_value`
- Rule 7: Weak info signal (announced/launched/released) + trusted source → `info_value_trusted`

### REVIEW (enters 待审)
- Weak signal from untrusted source → `weak_signal_untrusted`
- Meme/reaction formats → `meme_or_reaction`
- Short text < 50 chars → `brief_mention`
- Monitored account but not AI-related → `monitored_not_ai`
- Fallback: no strong signal → `low_info_content`

## Key Regex/Pattern Groups
- `FINAL_VALUE_STRONG` — strong info signals (keep from any source)
- `FINAL_VALUE_WEAK` — weak news verbs (keep only from trusted sources)
- `FINAL_DROP_PATTERNS` — gossip/bait patterns (drop_hard)
- `FINAL_MEME_PATTERNS` — meme/reaction patterns (review)
- `OFF_TOPIC_PATTERNS` — sports/gaming/music/crypto/Linux (drop_hard)
- `AI_TECH_RELEVANCE_RE` — broad AI/tech word check (only for Rule 5)
- `AI_PRODUCT_RE` — specific AI product names
- `HIGH_SIGNAL_ACCOUNTS` — trusted researcher/company accounts
- `ALL_ACCOUNTS` — 90+ monitored Twitter accounts

## Keyword Searches (Apify)
7 search queries with `min_faves:2000`, each with `maxItems=200`:
- Claude / Claude Code / AnthropicAI
- OpenAI / chatgpt / sama
- Gemini / Google AI Studio / Notebooklm
- Jensen Huang / NVIDIA
- Cursor / Huggingface / Perplexity / Antigravity
- vibecoding / AIAgent / humanoid robot / Embodied AI
- DAN KOE / Peter Steinberger / OpenClaw / Nano banana

## Secrets (GitHub Actions)
- `APIFY_TOKEN` — Apify API access
- `TELEGRAM_BOT_TOKEN` — 8218222040:AAG_InmkpkpLs8kiKaMoa6YcawaIy2YIz-o
- `TELEGRAM_CHAT_ID` — 8640597958

## Website Status Feedback
The website shows different status messages when no cards are displayed:
1. File not found → "该时段暂无数据"
2. raw=0 → "该时段未抓取到数据"
3. candidate=0 → "已抓取 N 条，但清洗后无可用数据"
4. final=0 but review>0 → shows count and suggests switching to 待审
5. Normal → displays tweet cards

## Data Retention
- Historical data is permanently retained in the git repo
- Website can display any historical date
- No auto-deletion of old JSON files
- `seen_urls.json` has 7-day TTL (only affects dedup, not display)

## Schedule
- Morning: UTC 23:00 (Beijing 07:00)
- Evening: UTC 11:00 (Beijing 19:00)
