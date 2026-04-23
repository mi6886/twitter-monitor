# LLM-Based Topic Scoring Migration — Design

**Date**: 2026-04-23
**Author**: Elaine + Claude
**Status**: Approved, pending implementation plan

---

## 1. Context

### Current System
`twitter-monitor` runs on GitHub Actions twice daily (Beijing morning/evening). It:
1. Fetches tweets via Apify using keyword queries + a 142-account watchlist (`min_faves:2000`)
2. Filters with ~250 lines of regex across three stages: noise blacklist → candidate filter → final filter (3-way: keep/review/drop_hard)
3. Pushes `keep` tweets to Telegram as an AI/tech news digest

### Why Replace the Regex Pipeline
- Every new product/keyword (GPT-Image-2, Opus 4.7, Nano Banana) requires manual regex additions
- Regex can't capture content quality / viral potential — only presence of keywords
- The actual goal of the system has shifted: it's no longer a "news digest for a senior engineer" but a **topic discovery tool for a Chinese 小红书 AI content creator**

### New Goal
Score each tweet for **viral potential as content topic** against a validated scoring rubric, output sorted candidates with actionable angle suggestions. The user creates two content styles:
- **Type A · 科技叙事体** — single AI hardware / frontier product deep-dive (AR glasses, flying motorcycles, AI meeting cards, humanoid robots). Driven by 震撼感 + 信息差 + 未来想象.
- **Type B · 效率清单体** — AI software tools collections / reviews / tutorials. Driven by 焦虑感 + 实用价值 + 收藏导向.

### Reference Documents
- `选题评估打分卡_v1.md` — scoring rubric (based on 321 viral vs 32 non-viral posts analysis)
- `科技硬件前沿产品类_爆款文案风格拆解手册.md` — A-type style guide
- `AI软件工具类-爆款文案风格拆解手册.md` — A/B types differentiation

---

## 2. High-Level Architecture

```
Apify (min_faves:2000, 142 accounts + keyword queries)
        ↓
   raw-{date}-{period}.json           [unchanged]
        ↓
   Hard filters (retained):
   - 28-hour time window
   - NOISE_ACCOUNTS blacklist
   - URL dedup (seen_urls.json)
        ↓
   LLM Scoring (NEW)
   - Model: anthropic/claude-haiku-4.5 via OpenRouter
   - Batch: 10 tweets/call, 5 parallel workers
   - Rubric: 選題評估打分卡 → structured JSON
   - Failure policy: 2 retries, then fallback score=30, verdict=keep
        ↓
   scored-{date}-{period}.json        [NEW — all tweets with full scores]
        ↓
   final-{date}-{period}.json         [SEMANTICS CHANGED — tiered by score]
        ↓
   Telegram push (one message per tweet)
   - 🔥 必做 (score ≥ 50)
   - ⭐ 优先 (35-49)
   - 💡 备选 (20-34)
```

### Key Changes
| Component | Change |
|---|---|
| Apify search config (`ALL_ACCOUNTS`, `KEYWORD_SEARCHES`) | Unchanged — still how we fetch tweets |
| `NOISE_ACCOUNTS` blacklist | Retained (hard filter) |
| `MIN_FAVES = 2000` | Retained |
| All judgment regex (`NOISE_PATTERNS`, `AI_SIGNAL_PATTERNS`, `REJECT_PATTERNS`, `FINAL_VALUE_*`, `FINAL_DROP_*`, `FINAL_MEME_*`, `OFF_TOPIC_*`, `AI_TECH_RELEVANCE_RE`, `OFFENSIVE_PATTERNS`, `HIGH_SIGNAL_ACCOUNTS`, `AI_PRODUCT_RE`, `MONITORED_ACCOUNTS_LOWER`) | **Deleted** |
| `candidate_filter()`, `final_filter()` | **Deleted** |
| `candidate-*.json`, `review-*.json` | No longer produced (existing files retained in git history) |
| `final-*.json` semantics | Changed: now score-tiered output with full breakdown |
| `scored-*.json` | New file: all raw-qualified tweets with full score data (including drops), kept for review & debugging |
| `send_telegram.py` | Rewrites rendering to use tiered format |

---

## 3. LLM Prompt (Scoring Rubric)

### Model
`anthropic/claude-haiku-4.5` via OpenRouter (`https://openrouter.ai/api/v1`), OpenAI-compatible SDK.

### System Prompt

````
You score tweets for their viral potential as topics for a Chinese 小红书 AI content account.

=== ACCOUNT CONTEXT ===
The account creates two types of content:
- Type A · 科技叙事体 (tech narrative): single AI hardware / frontier product deep-dive
  Examples: flying motorcycle, AR glasses, AI meeting cards, humanoid robots, bionic tentacles
  Style: 震撼感 + 信息差 + 未来想象（震撼/惊叹/好奇驱动，高分享率）
- Type B · 效率清单体 (efficiency list): AI software tools collection / review / tutorial
  Examples: "5个AI写作神器", "下班后死磕这4个网站", AI 工具合集
  Style: 焦虑感 + 实用价值 + 收藏导向（催促/安利驱动，高收藏率）

=== SCORING RUBRIC (total 0-100) ===

## A. 品类 Category (pick ONE, 0-30 pts)
- 单产品/技术叙事 (30): One specific new product, visuals/demo, single story arc
- 事件速报 (28): "X just released/announced Y", strong timeliness
- AI趋势/观点 (25): Trend analysis / opinion from practitioner
- 多工具/资源合集 (22): "N tools/sites/resources" format
- 资源分享 (20): Recommend site/platform/free resources
- AI/设计小技巧演示 (18): Visual demo of a specific technique
- 效率工具推荐 (17): Single productivity tool recommendation
- AI出图/设计 (16): AI image/design showcase
- AI+行业应用 (15): AI applied to specific industry
- 评测体验 (15): Product review / hands-on
- AI视频制作 (12): AI video generation
- 教程/操作指南 (10): "How to / step-by-step"
- 开发者工具 (8): Dev-only tooling (narrow audience)
- 其他/非AI (0): → verdict = drop

## B. 信息差 Info Novelty (0-25 pts)
- 全球/全网首发 (25): "world's first / never been done"
- 时效首发 (20): "just released / announced today / 深夜突发"
- 海外搬运 (12): Foreign/Reddit/GitHub source, not yet in Chinese circles
- 认知差 (8): "hidden feature / most don't know"
- 已知信息 (0): widely reported

## C. 爆款信号 Viral Signals (each independent, cap total at +30)
- 有人物/团队故事 (+10): specific founder/team narrative
- "终于/第一"型选题 (+10): "finally / first time / 首次 / 首款"
- 反常识/意外性 (+10): unexpected brand move
- 有明确价格 (+8): specific price number

## D. 受众与情绪 Audience & Emotion (pick ONE dominant, -8 to +5)
- 受众=垂直人群 (-8): only appeals to narrow niche (devs only, researchers only)
- 情绪=震撼/惊叹 (+5)
- 情绪=好奇/趣味 (+5)
- 情绪=未来感/想象 (+5)
- 情绪=焦虑/实用 (+3)
- 情绪=白嫖/占便宜 (+3)
- 无明确情绪钩子 (0)

## E. 可触达感 Actionability (0 or +5)
- 多工具/资源合集 + concrete URLs/names → +5
- 资源分享 + concrete URLs/names → +5
- Other categories → 0

## F. 账号匹配 Account Fit (0 or +5)
- A型 match: 单产品/技术叙事 OR 事件速报 about AI hardware/frontier → +5
- B型 match: 多工具/资源合集 OR 资源分享 about AI software → +5
- Neither → 0

=== HARD RULES ===
- Not about AI/tech at all (sports, K-drama, crypto-trading, lifestyle) → category=其他/非AI, verdict=drop
- Same-name confusion (Claude the soccer player, Gemini fan accounts) → verdict=drop
- Pure meme / reaction / engagement bait with no info → verdict=drop

=== VERDICT ===
- total_score ≥ 20 → verdict = "keep"
- total_score <  20 → verdict = "drop"

=== SUMMARY RULES ===
Write `summary` in Chinese, 2-3 sentences, ≤ 150 characters total.
- Lead with the core fact: what product/event/idea
- Include the most important specific numbers/names
- One sentence why it matters
- Neutral tone, for judgment not publishing.

=== ANGLE RULES ===
Output `angles` as an ARRAY of 2-3 different content angles in Chinese.
Each angle ≤ 50 characters, offering a distinct creative cut:
- Different title framing (悬念反转型 / 终于型 / 反常识型 / 第一型)
- Different style (A型叙事 / B型清单)
- Different narrative focus (产品功能 / 团队故事 / 未来影响 / 价格锚点)

=== OUTPUT (JSON only, no markdown) ===
{
  "results": [
    {
      "id": "<tweet_id>",
      "summary": "<≤150字 中文>",
      "category": "<中文品类>",
      "category_points": <int>,
      "info_gap": "<中文等级>",
      "info_gap_points": <int>,
      "viral_signals": ["<信号1>", ...],
      "viral_signals_points": <int>,
      "emotion": "<中文情绪>",
      "emotion_points": <int>,
      "actionability_points": <int>,
      "account_fit": "A型" | "B型" | "不匹配",
      "account_bonus": <int>,
      "total_score": <int, 所有维度之和>,
      "verdict": "keep" | "drop",
      "angles": ["<角度1>", "<角度2>", "<角度3>"]
    }
  ]
}
````

### User Prompt Format (per batch of 10)
```
Tweet 1 (id=<tweet_id>):
@<screen_name>: <full_text>
quoted: <quoted_tweet_text or empty>
likes: <favorite_count>

Tweet 2 (id=...):
...
```

### Prompt Design Rationale
- **No TRUSTED_ACCOUNTS list in prompt**: author identity doesn't factor into viral potential — content does. Accounts are still used for Apify search (fetching) but not for LLM scoring.
- **Zero-shot, no few-shot examples**: Haiku 4.5's instruction-following is strong enough; can add examples later if prompt drifts.
- **Chinese dimension values in output**: user reads Telegram in Chinese; saves mental translation.
- **Hard rules listed first**: cheap short-circuits for obvious drops before scoring work.

---

## 4. Implementation Details

### New Dependency
```python
openai>=1.0  # OpenAI-compatible SDK for OpenRouter
```

### Environment Variables
- Existing: `APIFY_TOKEN`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`
- New: `OPENROUTER_API_KEY` (add to GitHub Secrets)

### `fetch_tweets.py` Changes

**Deletions (~280 lines)**:
- `NOISE_PATTERNS`, `AI_SIGNAL_PATTERNS`, `REJECT_PATTERNS`
- `FINAL_VALUE_STRONG/WEAK`, `FINAL_DROP_PATTERNS`, `FINAL_MEME_PATTERNS`
- `OFF_TOPIC_PATTERNS`, `AI_TECH_RELEVANCE_RE`, `OFFENSIVE_PATTERNS`
- `HIGH_SIGNAL_ACCOUNTS`, `AI_PRODUCT_RE`, `MONITORED_ACCOUNTS_LOWER`
- Regex portion of `is_noise()` (keep blacklist-account check only)
- `candidate_filter()`, `final_filter()` — both fully removed

**Retained**:
- `KEYWORD_SEARCHES`, `ALL_ACCOUNTS` (Apify search config)
- `NOISE_ACCOUNTS` (hard-filter blacklist)
- `MIN_FAVES = 2000`
- `apify_request()`, `start_actor()`, `poll_run()`, `get_dataset_items()`, `extract_tweet()`
- 28-hour time window, URL dedup, `seen_urls.json` handling

**New (~150 lines)**:
```python
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor

SYSTEM_PROMPT = """..."""  # Full prompt from §3

_openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

def build_user_prompt(tweets_batch):
    """Format 10 tweets as LLM input."""

def score_batch(tweets_batch, max_retries=2):
    """Score 10 tweets via LLM with 2 retries, exponential backoff.
    On permanent failure: return fallback dicts with score=30, verdict=keep."""

def fallback_score(tweet, err):
    """Default score on LLM failure — errs toward keep for safety."""

def run_llm_scoring(tweets):
    """Batch into groups of 10, run 5 workers in parallel, merge scores back."""
```

### Output Files

| File | Status | Content |
|---|---|---|
| `raw-{date}-{period}.json` | unchanged | All tweets passing Apify + hard filters |
| `candidate-*.json` | **deprecated** | No longer produced (history retained in git) |
| `review-*.json` | **deprecated** | No longer produced |
| `scored-{date}-{period}.json` | **new** | All scored tweets (keep + drop), full breakdown, sorted by score desc |
| `final-{date}-{period}.json` | **re-semantic'd** | Only `keep` (score ≥ 20), grouped by tier |
| `seen_urls.json` | unchanged | URL dedup across 7 days |
| `telegram_sent.json` | unchanged | Already-sent URLs |

**`scored-*.json` structure**:
```json
{
  "date": "2026-04-23",
  "period": "morning",
  "fetched_at": "...",
  "raw_count": 120,
  "scored_count": 120,
  "keep_count": 45,
  "drop_count": 75,
  "fallback_count": 0,
  "score_distribution": {"≥50": 8, "35-49": 18, "20-34": 19, "<20": 75},
  "tweets": [
    {
      "tweet_id": "...", "url": "...", "screen_name": "...",
      "full_text": "...", "favorite_count": 5234, "created_at": "...",
      "summary": "...", "category": "...", "category_points": 30,
      "info_gap": "...", "info_gap_points": 20,
      "viral_signals": [...], "viral_signals_points": 20,
      "emotion": "...", "emotion_points": 5,
      "actionability_points": 0,
      "account_fit": "A型", "account_bonus": 5,
      "total_score": 80, "verdict": "keep",
      "angles": ["...", "...", "..."]
    }
  ]
}
```

**`final-{date}-{period}.json` structure**:
```json
{
  "date": "...", "period": "...",
  "tier_counts": {"must_do": 8, "priority": 18, "backup": 19},
  "tiers": {
    "must_do": [ /* score≥50, sorted desc */ ],
    "priority": [ /* 35-49 */ ],
    "backup": [ /* 20-34 */ ]
  }
}
```

### `send_telegram.py` Changes

- Read `final-*.json`, iterate tiers in order (must_do → priority → backup)
- Send one section header message per tier (`🔥 必做（≥50分）` etc.)
- Send one message per tweet in that tier, format:

```
[80分] @billpeeb

📝 Anthropic 发布 Claude Code MCP 支持，开发者可把本地 IDE 和
Claude 打通让它直接改代码。24 小时 1000+ 项目集成。

📊 单产品/技术叙事(30) + 时效首发(20) + 人物故事·第一型(+20)
   + 震撼(+5) + A型(+5) = 80

💡 选题切入：
   1. A型·第一型: 让 AI 真正「上手改代码」的第一个工具
   2. A型·反常识型: 不是聊天框了，你的 IDE 成了 AI 的手
   3. 话题聚焦: MCP 生态爆发，国内还没人讲过

🔗 https://x.com/billpeeb/status/...
```

- `telegram_sent.json` dedup logic unchanged (URL-keyed)
- Estimated rewrite: ~70 of ~150 lines

### `test_filters.py` → `test_scoring.py`

Full rewrite:
- Remove all regex-based filter tests
- Add:
  - `score_batch` with mocked OpenRouter responses (success path, malformed JSON, timeout, 429)
  - Retry logic (exponential backoff, max 2 retries)
  - Fallback default score on permanent failure
  - `run_llm_scoring` parallel dispatch
  - JSON schema validation (missing fields → sensible fallback)
  - `is_noise` blacklist-account check (the retained part)
- **Not** testing prompt judgment quality — that's stage-1 backtest work

---

## 5. Testing & Rollout

### Stage 1: Offline Backtest (1-2 days, no production impact)

Write `backtest_scoring.py`:
- Reads past 7 days of `raw-*.json` (~1600 tweets)
- Runs new LLM scorer → `data/backtest/scored-*.json` files
- Generates `backtest_report.md` with:
  - Score distribution histogram
  - Category distribution
  - **Top 20 highest-scored tweets** (Telegram-format preview, for user review)
  - **Bottom 20 `keep` tweets** (score 20-25, check if threshold is right)
  - **Sampled 20 `drop` tweets** (check for false negatives)
  - LLM fallback/failure log

User reviews the report manually. Flags: prompt too loose (Top 20 has garbage) vs too strict (Bottom keep / drop has missed gems). Iterate on prompt and re-run until satisfied.

### Stage 2: Silent Production (3 days)

- Merge PR, GitHub Actions uses new code
- **`send_telegram.py` disabled** (or routed to a test chat_id) during this period
- `scored-*.json` and `final-*.json` committed normally
- User checks `final-*.json` output twice daily, confirms tiering looks reasonable

### Stage 3: Live Push (1-week observation)

- Re-enable Telegram push to user's main chat
- User tracks three categories daily:
  - ❌ Shouldn't-have-pushed (log tweet_id + reason)
  - ❌ Should-have-pushed-but-didn't (find in `scored-*.json`, see why it was low-scored)
  - ✅ Perfect hits (for positive signal)
- At end of week, tune prompt based on feedback

### Verification Gates (before merge)

1. ✅ Unit tests all green
2. ✅ Stage 1 backtest completed, Top 20 / Bottom keep / Sampled drop reviewed
3. ✅ Human eval: ≥70% of Top 20 judged "actually worth doing"
4. ✅ Human eval: ≤2 of 20 sampled drops judged as false negatives

### Rollback Strategy

If Stage 2 or 3 reveals the new system is meaningfully worse:
- `git revert` the merge commit → old regex system restored
- New-format `scored-*.json` / `final-*.json` files stay in git history
- Analyze + redesign

---

## 6. Open Questions / Deferred

- **Prompt tuning loop automation**: currently manual eyeball review. Could add a `backtest_scoring.py --analyze-markings` subcommand that parses user's inline ✅/❌ annotations and reports which scoring dimension is most commonly wrong. Deferred until we see whether manual review is actually a bottleneck.
- **Cost monitoring**: ~250 tweets/day × 0.1 LLM calls × ~2.5k tokens = trivial (~$0.13/day with Haiku 4.5). Not instrumenting yet; OpenRouter dashboard is enough.
- **Prompt caching**: system prompt is ~1500 tokens, will be the same across all calls. OpenRouter supports Anthropic prompt caching — free performance win, should enable on day 1.
- **`index.html` dashboard**: currently reads regex-era `final-*.json`. Will break with the schema change. Must be updated to read the new tier-based schema — scoped into the implementation plan.

---

## 7. References

- `选题评估打分卡_v1.md` (scoring rubric, validated on 353 samples)
- `科技硬件前沿产品类_爆款文案风格拆解手册.md` (A-type style)
- `AI软件工具类-爆款文案风格拆解手册.md` (A/B differentiation)
- Current code: `fetch_tweets.py`, `send_telegram.py`, `test_filters.py`
