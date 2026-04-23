# LLM Topic Scoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the regex-based filter pipeline in `fetch_tweets.py` with an LLM-driven topic-scoring pipeline (Haiku 4.5 via OpenRouter) that outputs tiered Xiaohongshu content candidates.

**Architecture:** Keep hard filters (MIN_FAVES, account blacklist, dedup). Add a new `score_batch` LLM module that processes tweets in batches of 10 with 5-way concurrency. Store full scores in `scored-*.json`, keep-tier output in tier-grouped `final-*.json`. Rewrite `send_telegram.py` to emit one message per tweet across three tiers (🔥/⭐/💡).

**Tech Stack:** Python 3.12, `openai>=1.30` SDK (for OpenRouter's OpenAI-compatible endpoint), `concurrent.futures.ThreadPoolExecutor`, pytest + `unittest.mock` for testing.

**Spec reference:** [docs/superpowers/specs/2026-04-23-llm-topic-scoring-design.md](../specs/2026-04-23-llm-topic-scoring-design.md)

---

## File Structure

**Files created:**
- `requirements.txt` — pin `openai>=1.30.0`
- `llm_scoring.py` — pure LLM scoring module (prompt, client, batch, retry, concurrency)
- `test_scoring.py` — pytest tests for `llm_scoring.py` and retained `is_noise`
- `backtest_scoring.py` — Stage 1 offline backtest harness

**Files modified:**
- `fetch_tweets.py` — delete regex (~280 lines), wire in `llm_scoring`, change output shape
- `send_telegram.py` — full rewrite of the message-building main flow
- `index.html` — read new tier-grouped `final-*.json` schema
- `.github/workflows/fetch-tweets.yml` — add `OPENROUTER_API_KEY` env var, `pip install -r requirements.txt` step, update ai-intel-hub sync to include `scored-*.json` and drop `review-*.json`

**Files deleted:**
- `test_filters.py` (replaced by `test_scoring.py`)

**Files untouched:**
- Apify config inside `fetch_tweets.py` (`ALL_ACCOUNTS`, `KEYWORD_SEARCHES`, `MIN_FAVES`, `NOISE_ACCOUNTS`)
- `data/` pipeline files (raw-*.json, seen_urls.json, telegram_sent.json)
- `AGENTS.md` (will be updated in a separate docs task after the system stabilizes)

---

## Task 1: Add OpenAI dependency and OpenRouter smoke test

**Files:**
- Create: `requirements.txt`
- Create: `scripts/smoke_openrouter.py`

- [ ] **Step 1: Create `requirements.txt`**

Create `/Users/elainewang/Downloads/twitter-monitor/requirements.txt`:
```
openai>=1.30.0
```

- [ ] **Step 2: Install locally**

Run:
```bash
cd /Users/elainewang/Downloads/twitter-monitor
python3 -m pip install -r requirements.txt
```
Expected: installs openai and httpx (transitive).

- [ ] **Step 3: Write smoke test script**

Create `scripts/smoke_openrouter.py`:
```python
"""One-shot smoke test: call OpenRouter Haiku 4.5, confirm JSON round-trip.
Run: OPENROUTER_API_KEY=sk-or-xxx python3 scripts/smoke_openrouter.py
"""
import json
import os
import sys
from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

resp = client.chat.completions.create(
    model="anthropic/claude-haiku-4.5",
    messages=[
        {"role": "system", "content": 'Reply with JSON only: {"ok": true, "model": "<model name>"}'},
        {"role": "user", "content": "ping"},
    ],
    response_format={"type": "json_object"},
    temperature=0,
    timeout=30,
)
content = resp.choices[0].message.content
parsed = json.loads(content)
print("Response:", parsed)
assert parsed.get("ok") is True, f"unexpected response: {parsed}"
print("Smoke test PASSED")
```

- [ ] **Step 4: Run the smoke test**

Run:
```bash
export OPENROUTER_API_KEY=<user's key>
python3 scripts/smoke_openrouter.py
```
Expected: prints `Response: {'ok': True, ...}` then `Smoke test PASSED`.

If the call fails: stop here, report the error to the user. Do not proceed until OpenRouter auth works.

- [ ] **Step 5: Commit**

```bash
git add requirements.txt scripts/smoke_openrouter.py
git commit -m "chore: add openai dep and OpenRouter smoke test"
```

---

## Task 2: Add `OPENROUTER_API_KEY` to the GitHub Actions workflow

**Files:**
- Modify: `.github/workflows/fetch-tweets.yml`

- [ ] **Step 1: Update the workflow**

Edit `.github/workflows/fetch-tweets.yml`.

Add a pip install step after "Set up Python" and before "Fetch tweets from Apify":
```yaml
      - name: Install Python dependencies
        run: python -m pip install -r requirements.txt
```

Add `OPENROUTER_API_KEY` to the "Fetch tweets from Apify" env:
```yaml
      - name: Fetch tweets from Apify
        if: ${{ inputs.skip_fetch != 'true' }}
        env:
          APIFY_TOKEN: ${{ secrets.APIFY_TOKEN }}
          OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
        run: python fetch_tweets.py
```

- [ ] **Step 2: Instruct the user to add the secret**

Print this reminder in plan output (do not automate — requires browser login):
> User must add `OPENROUTER_API_KEY` to GitHub repo secrets: https://github.com/mi6886/twitter-monitor/settings/secrets/actions

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/fetch-tweets.yml
git commit -m "ci: wire OPENROUTER_API_KEY and pip install for LLM scoring"
```

---

## Task 3: Create `llm_scoring.py` skeleton with SYSTEM_PROMPT

**Files:**
- Create: `llm_scoring.py`

- [ ] **Step 1: Create `llm_scoring.py` with the full system prompt**

Create `/Users/elainewang/Downloads/twitter-monitor/llm_scoring.py`:
```python
"""
LLM-based topic scoring for the Twitter monitor pipeline.
Uses Anthropic Haiku 4.5 via OpenRouter (OpenAI-compatible SDK).

Public API:
- SYSTEM_PROMPT: the scoring rubric prompt (1 source of truth)
- build_user_prompt(tweets): format a batch of tweets for the user turn
- score_batch(tweets, max_retries=2): score up to 10 tweets, return list of dicts
- run_llm_scoring(tweets, max_workers=5): score all tweets in parallel batches
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from openai import OpenAI

MODEL = "anthropic/claude-haiku-4.5"
BATCH_SIZE = 10
MAX_WORKERS = 5
REQUEST_TIMEOUT = 45  # seconds per LLM call

SYSTEM_PROMPT = """You score tweets for their viral potential as topics for a Chinese 小红书 AI content account.

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
- total_score >= 20 → verdict = "keep"
- total_score <  20 → verdict = "drop"

=== SUMMARY RULES ===
Write `summary` in Chinese, 2-3 sentences, <= 150 characters total.
- Lead with the core fact: what product/event/idea
- Include the most important specific numbers/names
- One sentence why it matters
- Neutral tone, for judgment not publishing.

=== ANGLE RULES ===
Output `angles` as an ARRAY of 2-3 different content angles in Chinese.
Each angle <= 50 characters, offering a distinct creative cut:
- Different title framing (悬念反转型 / 终于型 / 反常识型 / 第一型)
- Different style (A型叙事 / B型清单)
- Different narrative focus (产品功能 / 团队故事 / 未来影响 / 价格锚点)

=== OUTPUT (JSON only, no markdown) ===
{
  "results": [
    {
      "id": "<tweet_id>",
      "summary": "<<=150字 中文>",
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
      "total_score": <int>,
      "verdict": "keep" | "drop",
      "angles": ["<角度1>", "<角度2>", "<角度3>"]
    }
  ]
}"""


_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
    return _client


def build_user_prompt(tweets_batch: list[dict]) -> str:
    """Format up to BATCH_SIZE tweets as the user turn for a scoring call."""
    lines = []
    for tw in tweets_batch:
        lines.append(f"Tweet (id={tw['tweet_id']}):")
        lines.append(f"@{tw.get('screen_name', 'unknown')}: {tw.get('full_text', '')}")
        if tw.get("quoted_tweet_text"):
            lines.append(f"quoted: {tw['quoted_tweet_text']}")
        lines.append(f"likes: {tw.get('favorite_count', 0)}")
        lines.append("")
    return "\n".join(lines)
```

- [ ] **Step 2: Verify the file imports cleanly**

Run:
```bash
cd /Users/elainewang/Downloads/twitter-monitor
python3 -c "from llm_scoring import SYSTEM_PROMPT, build_user_prompt; print(len(SYSTEM_PROMPT), 'chars')"
```
Expected: prints a number around 3000-4000.

- [ ] **Step 3: Commit**

```bash
git add llm_scoring.py
git commit -m "feat(llm_scoring): add module skeleton with SYSTEM_PROMPT and user prompt builder"
```

---

## Task 4: TDD — `score_batch` success path

**Files:**
- Create: `test_scoring.py`
- Modify: `llm_scoring.py`

- [ ] **Step 1: Write the failing test for the success path**

Create `test_scoring.py`:
```python
"""Unit tests for llm_scoring.py. Run:
  OPENROUTER_API_KEY=dummy python3 -m pytest test_scoring.py -v
"""

import json
import os
from unittest.mock import MagicMock, patch

os.environ.setdefault("OPENROUTER_API_KEY", "dummy")

import llm_scoring


def make_tweet(tweet_id="t1", text="Hello world", screen_name="user", likes=2500, quoted=""):
    return {
        "tweet_id": tweet_id,
        "full_text": text,
        "screen_name": screen_name,
        "favorite_count": likes,
        "quoted_tweet_text": quoted,
    }


def make_llm_response(results):
    """Build a mocked OpenAI ChatCompletion object with JSON content."""
    mock_msg = MagicMock()
    mock_msg.content = json.dumps({"results": results})
    mock_choice = MagicMock()
    mock_choice.message = mock_msg
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    return mock_resp


class TestScoreBatchSuccess:
    def test_parses_single_result(self):
        tweet = make_tweet(tweet_id="t1", text="Anthropic releases Claude 5")
        mocked = make_llm_response([{
            "id": "t1",
            "summary": "Anthropic 发布 Claude 5",
            "category": "事件速报",
            "category_points": 28,
            "info_gap": "时效首发",
            "info_gap_points": 20,
            "viral_signals": ["有人物/团队故事"],
            "viral_signals_points": 10,
            "emotion": "震撼/惊叹",
            "emotion_points": 5,
            "actionability_points": 0,
            "account_fit": "A型",
            "account_bonus": 5,
            "total_score": 68,
            "verdict": "keep",
            "angles": ["A型·第一型", "A型·反常识型", "话题聚焦"],
        }])
        with patch.object(llm_scoring, "_get_client") as gc:
            gc.return_value.chat.completions.create.return_value = mocked
            result = llm_scoring.score_batch([tweet])
        assert len(result) == 1
        assert result[0]["id"] == "t1"
        assert result[0]["total_score"] == 68
        assert result[0]["verdict"] == "keep"
        assert len(result[0]["angles"]) == 3
```

- [ ] **Step 2: Run the test — expect failure (score_batch not defined)**

Run:
```bash
cd /Users/elainewang/Downloads/twitter-monitor
OPENROUTER_API_KEY=dummy python3 -m pytest test_scoring.py::TestScoreBatchSuccess -v
```
Expected: `AttributeError: module 'llm_scoring' has no attribute 'score_batch'`.

- [ ] **Step 3: Implement `score_batch` success path**

Append to `llm_scoring.py`:
```python
def score_batch(tweets_batch: list[dict], max_retries: int = 2) -> list[dict]:
    """Score up to BATCH_SIZE tweets. Returns a list of score dicts.

    On permanent failure (after retries): each tweet gets a fallback dict with
    total_score=30, verdict='keep', and an error-flagged angles entry."""
    client = _get_client()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(tweets_batch)},
        ],
        response_format={"type": "json_object"},
        temperature=0,
        timeout=REQUEST_TIMEOUT,
    )
    parsed = json.loads(resp.choices[0].message.content)
    return parsed["results"]
```

- [ ] **Step 4: Run the test — expect pass**

Run:
```bash
OPENROUTER_API_KEY=dummy python3 -m pytest test_scoring.py::TestScoreBatchSuccess -v
```
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add test_scoring.py llm_scoring.py
git commit -m "feat(llm_scoring): score_batch success path"
```

---

## Task 5: TDD — retry logic with exponential backoff

**Files:**
- Modify: `test_scoring.py`
- Modify: `llm_scoring.py`

- [ ] **Step 1: Write two failing tests — retry-then-succeed, and all-retries-exhausted**

Append to `test_scoring.py`:
```python
class TestScoreBatchRetry:
    def test_retries_on_transient_failure_then_succeeds(self):
        tweet = make_tweet("t1")
        success_resp = make_llm_response([{
            "id": "t1", "summary": "ok", "category": "其他/非AI",
            "category_points": 0, "info_gap": "已知信息", "info_gap_points": 0,
            "viral_signals": [], "viral_signals_points": 0,
            "emotion": "无明确情绪钩子", "emotion_points": 0,
            "actionability_points": 0, "account_fit": "不匹配", "account_bonus": 0,
            "total_score": 0, "verdict": "drop", "angles": [],
        }])
        with patch.object(llm_scoring, "_get_client") as gc, \
             patch("llm_scoring.time.sleep"):
            gc.return_value.chat.completions.create.side_effect = [
                RuntimeError("500"),
                RuntimeError("429"),
                success_resp,
            ]
            result = llm_scoring.score_batch([tweet], max_retries=2)
        assert len(result) == 1
        assert result[0]["id"] == "t1"
        assert result[0]["total_score"] == 0

    def test_fallback_after_retries_exhausted(self):
        tweet = make_tweet("t1", text="some tweet text")
        with patch.object(llm_scoring, "_get_client") as gc, \
             patch("llm_scoring.time.sleep"):
            gc.return_value.chat.completions.create.side_effect = RuntimeError("boom")
            result = llm_scoring.score_batch([tweet], max_retries=2)
        assert len(result) == 1
        assert result[0]["id"] == "t1"
        assert result[0]["total_score"] == 30
        assert result[0]["verdict"] == "keep"
        assert result[0].get("_fallback") is True
        assert "boom" in result[0]["angles"][0]
```

- [ ] **Step 2: Run the tests — expect failure**

Run:
```bash
OPENROUTER_API_KEY=dummy python3 -m pytest test_scoring.py::TestScoreBatchRetry -v
```
Expected: both tests fail — the current `score_batch` doesn't retry or fallback.

- [ ] **Step 3: Replace `score_batch` with the retry + fallback version**

In `llm_scoring.py`, replace the `score_batch` body:
```python
def fallback_score(tweet: dict, err: str) -> dict:
    """Default score dict when the LLM call fails after retries.
    Errs toward keep (score=30) so we don't silently drop tweets on outages."""
    return {
        "id": tweet["tweet_id"],
        "summary": (tweet.get("full_text") or "")[:150],
        "category": "未评分",
        "category_points": 0,
        "info_gap": "未评分",
        "info_gap_points": 0,
        "viral_signals": [],
        "viral_signals_points": 0,
        "emotion": "未评分",
        "emotion_points": 0,
        "actionability_points": 0,
        "account_fit": "不匹配",
        "account_bonus": 0,
        "total_score": 30,
        "verdict": "keep",
        "angles": [f"[LLM fallback: {err[:80]}]"],
        "_fallback": True,
    }


def score_batch(tweets_batch: list[dict], max_retries: int = 2) -> list[dict]:
    """Score up to BATCH_SIZE tweets, with exponential backoff retries.
    On permanent failure: return fallback_score() for each tweet."""
    client = _get_client()
    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(tweets_batch)},
                ],
                response_format={"type": "json_object"},
                temperature=0,
                timeout=REQUEST_TIMEOUT,
            )
            parsed = json.loads(resp.choices[0].message.content)
            return parsed["results"]
        except Exception as e:  # noqa: BLE001 — OpenAI SDK raises many types
            last_err = e
            if attempt < max_retries:
                time.sleep(2 ** attempt)
    err_msg = str(last_err) if last_err else "unknown error"
    return [fallback_score(t, err_msg) for t in tweets_batch]
```

- [ ] **Step 4: Run all tests — expect pass**

Run:
```bash
OPENROUTER_API_KEY=dummy python3 -m pytest test_scoring.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add test_scoring.py llm_scoring.py
git commit -m "feat(llm_scoring): retry with exponential backoff + fallback on permanent failure"
```

---

## Task 6: TDD — concurrent `run_llm_scoring`

**Files:**
- Modify: `test_scoring.py`
- Modify: `llm_scoring.py`

- [ ] **Step 1: Write the failing test for concurrent dispatch**

Append to `test_scoring.py`:
```python
class TestRunLLMScoring:
    def test_splits_into_batches_and_merges(self):
        tweets = [make_tweet(f"t{i}", text=f"tweet {i}") for i in range(25)]

        def fake_score_batch(batch, max_retries=2):
            return [
                {
                    "id": t["tweet_id"], "summary": f"summ {t['tweet_id']}",
                    "category": "事件速报", "category_points": 28,
                    "info_gap": "时效首发", "info_gap_points": 20,
                    "viral_signals": [], "viral_signals_points": 0,
                    "emotion": "震撼/惊叹", "emotion_points": 5,
                    "actionability_points": 0,
                    "account_fit": "A型", "account_bonus": 5,
                    "total_score": 58, "verdict": "keep",
                    "angles": ["a1", "a2"],
                }
                for t in batch
            ]

        with patch("llm_scoring.score_batch", side_effect=fake_score_batch) as mocked:
            scored = llm_scoring.run_llm_scoring(tweets, max_workers=2)

        # 25 tweets -> 3 batches (10 + 10 + 5)
        assert mocked.call_count == 3
        assert len(scored) == 25
        # Each returned dict must merge the ORIGINAL tweet fields + score fields
        for orig, out in zip(tweets, scored):
            assert out["tweet_id"] == orig["tweet_id"]
            assert out["full_text"] == orig["full_text"]
            assert out["total_score"] == 58
            assert out["verdict"] == "keep"

    def test_empty_input_returns_empty(self):
        with patch("llm_scoring.score_batch") as mocked:
            scored = llm_scoring.run_llm_scoring([])
        assert scored == []
        mocked.assert_not_called()
```

- [ ] **Step 2: Run — expect failure**

Run:
```bash
OPENROUTER_API_KEY=dummy python3 -m pytest test_scoring.py::TestRunLLMScoring -v
```
Expected: `AttributeError: module 'llm_scoring' has no attribute 'run_llm_scoring'`.

- [ ] **Step 3: Implement `run_llm_scoring`**

Append to `llm_scoring.py`:
```python
def run_llm_scoring(tweets: list[dict], max_workers: int = MAX_WORKERS) -> list[dict]:
    """Score all tweets in batches of BATCH_SIZE, parallelized across workers.

    Returns a list of dicts where each dict is the original tweet merged with
    the LLM score fields. Order is preserved (matches input order)."""
    if not tweets:
        return []

    batches = [tweets[i:i + BATCH_SIZE] for i in range(0, len(tweets), BATCH_SIZE)]
    all_scores: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for batch_scores in pool.map(score_batch, batches):
            all_scores.extend(batch_scores)

    score_by_id = {s["id"]: s for s in all_scores}
    merged: list[dict] = []
    for t in tweets:
        s = score_by_id.get(t["tweet_id"])
        if s is None:
            s = fallback_score(t, "missing from LLM output")
        merged.append({**t, **s})
    return merged
```

- [ ] **Step 4: Run all tests**

Run:
```bash
OPENROUTER_API_KEY=dummy python3 -m pytest test_scoring.py -v
```
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add test_scoring.py llm_scoring.py
git commit -m "feat(llm_scoring): parallel batch dispatch via ThreadPoolExecutor"
```

---

## Task 7: Strip regex filter constants from `fetch_tweets.py`

**Files:**
- Modify: `fetch_tweets.py`

- [ ] **Step 1: Delete the regex constants block**

In `fetch_tweets.py`, delete these constant blocks (each defined somewhere in lines 74–407):
- `NOISE_PATTERNS` (line 75)
- `AI_SIGNAL_PATTERNS` (line 198)
- `REJECT_PATTERNS` (line 228)
- `MONITORED_ACCOUNTS_LOWER` (line 238)
- `FINAL_VALUE_STRONG` (line 271)
- `FINAL_VALUE_WEAK` (line 295)
- `FINAL_DROP_PATTERNS` (line 307)
- `FINAL_MEME_PATTERNS` (line 319)
- `OFF_TOPIC_PATTERNS` (line 334)
- `HIGH_SIGNAL_ACCOUNTS` (line 352)
- `AI_PRODUCT_RE` (line 369)
- `AI_TECH_RELEVANCE_RE` (line 377)
- `OFFENSIVE_PATTERNS` (line 403)

**Keep:**
- `NOISE_ACCOUNTS` (line 86) — hard blacklist, still needed
- `KEYWORD_SEARCHES`, `ALL_ACCOUNTS`, `ACCOUNTS_PER_BATCH` — Apify config

- [ ] **Step 2: Delete the regex functions**

Delete entirely:
- `candidate_filter` (line 241)
- `final_filter` (line 409)

- [ ] **Step 3: Replace `is_noise` body with blacklist-only check**

Find `is_noise` (currently at line 139) and replace its whole body with:
```python
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
```

- [ ] **Step 4: Remove unused `re` import only if no other uses remain**

Run:
```bash
grep -n "re\." fetch_tweets.py
```
If the only remaining hit is `import re`, remove the import. Otherwise keep it.

- [ ] **Step 5: Verify the module still imports**

Run:
```bash
APIFY_TOKEN=dummy OPENROUTER_API_KEY=dummy python3 -c "import fetch_tweets; print('ok')"
```
Expected: `ok`. If it fails, fix syntax errors before moving on.

- [ ] **Step 6: Commit**

```bash
git add fetch_tweets.py
git commit -m "refactor(fetch_tweets): remove regex filter pipeline"
```

---

## Task 8: Wire `llm_scoring` into `fetch_tweets.main()` and write new output files

**Files:**
- Modify: `fetch_tweets.py`

- [ ] **Step 1: Import the scoring module at the top of `fetch_tweets.py`**

Add near the other imports (below `from urllib.error import HTTPError`):
```python
import llm_scoring
```

- [ ] **Step 2: Replace the "Generate candidate file" and "Generate final + review files" blocks in `main()`**

In `fetch_tweets.py`, find the section that starts with the comment:
```python
    # --- Generate candidate file (first-pass coarse filter on raw) ---
```
and ends just before:
```python
    # Update seen URLs (keep last 7 days only)
    save_seen_ids(seen_file, seen_urls, today)
```

Replace that entire block with:
```python
    # --- LLM scoring (replaces regex candidate + final filters) ---
    print(f"\nLLM scoring {len(all_tweets)} tweets via OpenRouter (Haiku 4.5)...")
    scored_tweets = llm_scoring.run_llm_scoring(all_tweets)

    # Sort by total_score descending
    scored_tweets.sort(key=lambda t: t.get("total_score", 0), reverse=True)

    fallback_count = sum(1 for t in scored_tweets if t.get("_fallback"))
    keep_tweets = [t for t in scored_tweets if t.get("verdict") == "keep"]
    drop_tweets = [t for t in scored_tweets if t.get("verdict") == "drop"]

    def _bucket(t):
        s = t.get("total_score", 0)
        if s >= 50:
            return "must_do"
        if s >= 35:
            return "priority"
        if s >= 20:
            return "backup"
        return "drop"

    score_distribution = {">=50": 0, "35-49": 0, "20-34": 0, "<20": 0}
    for t in scored_tweets:
        s = t.get("total_score", 0)
        if s >= 50:
            score_distribution[">=50"] += 1
        elif s >= 35:
            score_distribution["35-49"] += 1
        elif s >= 20:
            score_distribution["20-34"] += 1
        else:
            score_distribution["<20"] += 1

    # --- Save scored-*.json (ALL scored tweets with full breakdown) ---
    scored_result = {
        "date": today,
        "period": period,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "raw_count": len(all_tweets),
        "scored_count": len(scored_tweets),
        "keep_count": len(keep_tweets),
        "drop_count": len(drop_tweets),
        "fallback_count": fallback_count,
        "score_distribution": score_distribution,
        "tweets": scored_tweets,
    }
    scored_file = output_dir / f"scored-{today}-{period}.json"
    scored_file.write_text(json.dumps(scored_result, ensure_ascii=False, indent=2))
    print(f"Saved {len(scored_tweets)} scored tweets to {scored_file}")
    print(f"  Distribution: {score_distribution}")
    if fallback_count:
        print(f"  !! LLM fallback count: {fallback_count}")

    # --- Save final-*.json (tier-grouped, keep only) ---
    tiers = {"must_do": [], "priority": [], "backup": []}
    for t in keep_tweets:
        b = _bucket(t)
        if b in tiers:
            tiers[b].append(t)

    final_result = {
        "date": today,
        "period": period,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "tier_counts": {k: len(v) for k, v in tiers.items()},
        "tiers": tiers,
    }
    final_file = output_dir / f"final-{today}-{period}.json"
    final_file.write_text(json.dumps(final_result, ensure_ascii=False, indent=2))
    print(f"Saved final tiers: must_do={len(tiers['must_do'])} "
          f"priority={len(tiers['priority'])} backup={len(tiers['backup'])}")
```

- [ ] **Step 3: Verify the module imports and `main()` is syntactically valid**

Run:
```bash
APIFY_TOKEN=dummy OPENROUTER_API_KEY=dummy python3 -c "import fetch_tweets; print(fetch_tweets.main.__doc__ or 'no docstring')"
```
Expected: no exceptions.

- [ ] **Step 4: Commit**

```bash
git add fetch_tweets.py
git commit -m "feat(fetch_tweets): wire LLM scoring into pipeline, emit scored-*.json and tiered final-*.json"
```

---

## Task 9: Update `test_scoring.py` to also cover the retained `is_noise`

**Files:**
- Modify: `test_scoring.py`
- Delete: `test_filters.py`

- [ ] **Step 1: Add is_noise coverage**

Append to `test_scoring.py`:
```python
from fetch_tweets import is_noise, NOISE_ACCOUNTS


class TestIsNoiseBlacklist:
    def test_blacklisted_account_is_noise(self):
        blacklisted = NOISE_ACCOUNTS[0]  # e.g. "actufoot_"
        tweet = {"author": {"screen_name": blacklisted}, "full_text": "any text"}
        assert is_noise(tweet) is True

    def test_non_blacklisted_account_is_not_noise(self):
        tweet = {"author": {"screen_name": "karpathy"}, "full_text": "any text"}
        assert is_noise(tweet) is False

    def test_author_as_string_is_handled(self):
        tweet = {"author": "actufoot_", "full_text": "anything"}
        assert is_noise(tweet) is True

    def test_missing_author_not_noise(self):
        tweet = {"full_text": "anything"}
        assert is_noise(tweet) is False
```

- [ ] **Step 2: Delete the old regex test file**

```bash
rm /Users/elainewang/Downloads/twitter-monitor/test_filters.py
```

- [ ] **Step 3: Run the full test suite**

Run:
```bash
cd /Users/elainewang/Downloads/twitter-monitor
APIFY_TOKEN=dummy OPENROUTER_API_KEY=dummy python3 -m pytest test_scoring.py -v
```
Expected: 9 passed.

- [ ] **Step 4: Commit**

```bash
git add test_scoring.py
git rm test_filters.py
git commit -m "test: replace regex filter tests with scoring + is_noise tests"
```

---

## Task 10: Rewrite `send_telegram.py` — one message per tweet, tiered output

**Files:**
- Modify: `send_telegram.py`

- [ ] **Step 1: Replace the main rendering flow**

Open `send_telegram.py`. Keep these helpers as-is: `send_telegram`, `escape_md2`, `fmt_likes`, `clean_tweet_text`, `load_sent_log`, `save_sent_log`. Delete `make_summary` and `compute_digest` (no longer used).

Replace the `main()` function body with:
```python
TIER_META = [
    ("must_do", "🔥 必做（≥50分）"),
    ("priority", "⭐ 优先（35-49分）"),
    ("backup", "💡 备选（20-34分）"),
]


def format_tweet_message(tier_label_chars, tweet):
    """Return a MarkdownV2-escaped single-message body for one scored tweet."""
    score = tweet.get("total_score", 0)
    author = f"@{tweet.get('screen_name', '?')}"
    summary = tweet.get("summary") or clean_tweet_text(tweet.get("full_text", ""))[:150]
    angles = tweet.get("angles") or []

    # Score breakdown line, e.g. "单产品/技术叙事(30) + 时效首发(20) + ..."
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

    # Dedup by URL (one entry per tweet; avoids re-sending individual tweets
    # we already pushed in a prior run of the same period).
    sent_log_file = data_dir / "telegram_sent.json"
    sent_urls = load_sent_log(sent_log_file) if not observe_mode else set()

    sent_this_run = 0
    for tier_key, header_text in TIER_META:
        tweets = tiers.get(tier_key, [])
        if not tweets:
            continue
        # Header message for this tier
        header_body = f"*{escape_md2(header_text)}*  \\({len(tweets)} 条\\)"
        send_telegram(header_body)

        for tw in tweets:
            url = tw.get("url", "")
            if not observe_mode and url and url in sent_urls:
                continue
            body = format_tweet_message(header_text, tw)
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
```

Also update `load_sent_log` and `save_sent_log` to store a set of URLs instead of digest hashes:
```python
def load_sent_log(path):
    """Load set of previously sent tweet URLs."""
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, TypeError):
        return set()
    # Backward-compat: if the file holds old-style digests (list of short strings),
    # treat as empty — urls will repopulate on next push.
    if isinstance(data, list) and data and all(len(x) <= 16 for x in data):
        return set()
    return set(data) if isinstance(data, list) else set()


def save_sent_log(path, urls):
    """Save sent URLs (bounded to the most recent 500 to avoid unbounded growth)."""
    recent = sorted(urls)[-500:]
    path.write_text(json.dumps(recent, indent=2))
```

- [ ] **Step 2: Verify the module imports**

Run:
```bash
TELEGRAM_BOT_TOKEN=dummy TELEGRAM_CHAT_ID=dummy python3 -c "import send_telegram; print('ok')"
```
Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add send_telegram.py
git commit -m "feat(send_telegram): one message per tweet, tiered output with score breakdown and angles"
```

---

## Task 11: Write `backtest_scoring.py` — Stage 1 offline harness

**Files:**
- Create: `backtest_scoring.py`

- [ ] **Step 1: Create the backtest script**

Create `/Users/elainewang/Downloads/twitter-monitor/backtest_scoring.py`:
```python
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
  OPENROUTER_API_KEY=... python3 backtest_scoring.py --days 7
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

    report_lines += ["", "## Bottom 20 Keep (score ≤ 25, threshold stress-test)"]
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
```

- [ ] **Step 2: Smoke-test the script on a single day (to sanity-check before burning quota on 7 days)**

Run (with real API key):
```bash
cd /Users/elainewang/Downloads/twitter-monitor
OPENROUTER_API_KEY=<real-key> python3 backtest_scoring.py --days 1
```
Expected: writes `data/backtest/scored-backtest-<ts>.json` and `data/backtest/backtest-report-<ts>.md`, takes 1-3 minutes.

If errors: fix before committing.

- [ ] **Step 3: Commit**

```bash
git add backtest_scoring.py
# NOTE: do NOT commit data/backtest/ output files — add to .gitignore
echo "data/backtest/" >> .gitignore
git add .gitignore
git commit -m "feat(backtest): stage-1 offline scoring harness + report"
```

---

## Task 12: HUMAN REVIEW GATE — run the 7-day backtest

**This task is not automated — the user performs it.**

- [ ] **Step 1: Run the full 7-day backtest**

Run:
```bash
OPENROUTER_API_KEY=<real-key> python3 backtest_scoring.py --days 7
```
Expected runtime: ~5-15 minutes.

- [ ] **Step 2: Review the report in an editor**

Open `data/backtest/backtest-report-<ts>.md`.

Check the three review blocks:
1. **Top 20** — are ≥ 70% of these actually "worth making content about"?
2. **Bottom 20 Keep (≤25)** — are most of these close-to-drop borderline? Or are gems hiding here?
3. **Random 20 Drop** — how many are false negatives you'd want to have seen?

- [ ] **Step 3: Decide**

| Result | Next action |
|---|---|
| ≥70% Top-20 hit, ≤2 false negatives in Drop 20 | Proceed to Stage 2 (Task 13) |
| Top-20 hit rate low, or Drop 20 has many false negatives | Tune prompt in `llm_scoring.py`, re-run backtest, iterate |

---

## Task 13: Update GitHub Actions — silent Stage 2 deploy

**Files:**
- Modify: `.github/workflows/fetch-tweets.yml`

- [ ] **Step 1: Add a STAGE 2 guard around the Telegram step**

Edit `.github/workflows/fetch-tweets.yml`. Change the "Send to Telegram" step to be opt-in:
```yaml
      - name: Send to Telegram
        if: ${{ vars.TELEGRAM_ENABLED == 'true' }}
        env:
          TELEGRAM_BOT_TOKEN: ${{ secrets.TELEGRAM_BOT_TOKEN }}
          TELEGRAM_CHAT_ID: ${{ secrets.TELEGRAM_CHAT_ID }}
        run: python send_telegram.py
```

- [ ] **Step 2: Ensure `TELEGRAM_ENABLED` repo variable is `false` for the silent period**

Print reminder:
> User: set `TELEGRAM_ENABLED` to `false` in GitHub Actions variables (NOT secrets):
> https://github.com/mi6886/twitter-monitor/settings/variables/actions

- [ ] **Step 3: Update the ai-intel-hub sync step**

In the same workflow, change the sync step's `cp` lines from:
```yaml
          cp -v data/final-*.json "$DEST/" 2>/dev/null || true
          cp -v data/review-*.json "$DEST/" 2>/dev/null || true
```
to:
```yaml
          cp -v data/final-*.json "$DEST/" 2>/dev/null || true
          cp -v data/scored-*.json "$DEST/" 2>/dev/null || true
```

Also update the inline comment 2 lines above from `# Copy only final-* and review-* JSONs...` to `# Copy only final-* and scored-* JSONs...`.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/fetch-tweets.yml
git commit -m "ci: gate Telegram on TELEGRAM_ENABLED, sync scored-*.json to ai-intel-hub"
```

---

## Task 14: Push and trigger a production run for Stage 2 start

- [ ] **Step 1: Push all commits**

```bash
cd /Users/elainewang/Downloads/twitter-monitor
git push
```

- [ ] **Step 2: Manually trigger the workflow (or wait for next cron)**

User action:
> Open https://github.com/mi6886/twitter-monitor/actions/workflows/fetch-tweets.yml and click "Run workflow" to kick off the first production run under the new system.

- [ ] **Step 3: Verify the first silent run**

After the run finishes, check the committed files in `data/`:
- New: `scored-<date>-<period>.json`
- New: `final-<date>-<period>.json` with `tiers` key (not `tweets` array)
- NOT produced: new `candidate-*.json` / `review-*.json`

If verification fails: diagnose from workflow logs and the committed files. Fix before the next run.

---

## Task 15: Stage 2 soak — user reviews 3 days of silent runs

**This task is not automated — the user performs it.**

- [ ] **Step 1: After each of the next 6 runs (3 days × 2 periods), open `final-<date>-<period>.json`**

Manually spot-check:
- `tier_counts` look reasonable (e.g. 3-10 must_do, 10-25 priority, 15-30 backup)
- `tiers.must_do[0]` really is the most compelling candidate
- No obvious false-positives in `must_do`
- `scored-*.json` has no crazy spikes in `fallback_count`

- [ ] **Step 2: Decide**

| Result | Next action |
|---|---|
| 3 days look good | Proceed to Task 16 (flip `TELEGRAM_ENABLED=true`) |
| Systematic issues | Iterate on prompt, re-deploy |

---

## Task 16: Enable Telegram push (Stage 3)

- [ ] **Step 1: Flip the repo variable**

User action:
> Set `TELEGRAM_ENABLED` to `true` in GitHub Actions variables.

- [ ] **Step 2: Manually trigger a run to see the first real push**

User action:
> Click "Run workflow" in Actions. Wait for it to finish. Check Telegram.

- [ ] **Step 3: Observe for 1 week**

User action (daily):
> For each message batch, note in a scratch file:
> - ❌ Pushed but shouldn't have (tweet_id + why)
> - ❌ Should have been pushed but missing (find in `scored-*.json`, note score)
> - ✅ Perfect hits

At end of week, open a follow-up task to tune the prompt if needed.

---

## Task 17: Update `index.html` dashboard for the new schema

**Files:**
- Modify: `index.html`

- [ ] **Step 1: Inventory what `index.html` currently reads**

Run:
```bash
grep -nE "final-|candidate-|review-|tweets\[" index.html
```
Note which fields it reads from `final-*.json`.

- [ ] **Step 2: Update the fetch+render code**

Change the `final-*.json` consumer to iterate `data.tiers.must_do`, `data.tiers.priority`, `data.tiers.backup` in that order, rendering each as its own section. Display `total_score`, `category`, `angles`, and `summary` prominently.

Remove any references to `review-*.json` / `candidate-*.json` files.

The exact patch depends on the current code; the executing agent should read `index.html` in full and produce a minimal patch that matches the existing styling conventions.

- [ ] **Step 3: Smoke-test locally**

Run:
```bash
python3 -m http.server 8000 --directory /Users/elainewang/Downloads/twitter-monitor
```
Open `http://localhost:8000` in a browser. Verify the latest `final-*.json` renders without console errors.

- [ ] **Step 4: Commit and push**

```bash
git add index.html
git commit -m "feat(dashboard): render tier-grouped final-*.json with score + angles"
git push
```

---

## Task 18: Cleanup — retire stale fields / dead code after 2-week soak

**This task runs AT LEAST 14 days after Task 16 — DO NOT run it until then.**

- [ ] **Step 1: Delete the `scripts/smoke_openrouter.py` one-shot (retain in git history)**

```bash
git rm scripts/smoke_openrouter.py
# Also remove the scripts/ dir if it's empty
rmdir scripts 2>/dev/null || true
git commit -m "chore: remove OpenRouter smoke test (served its purpose)"
```

- [ ] **Step 2: Audit `AGENTS.md` for stale regex-era instructions**

User action:
> Open `AGENTS.md`. Search for regex/filter references. Update to describe the new LLM-scoring architecture.

- [ ] **Step 3: Consider retiring `scored-*.json` (optional)**

If after 2 weeks the file is no longer useful for debugging:
- Change `scored-*.json` to not persist `drop` tweets (only `keep`)
- Or drop the file entirely

This is a tuning decision, not a required step.

---

## Self-Review Checklist

✅ **Spec coverage**
- §2 Architecture: Task 7 (strip regex) + Task 8 (wire LLM) + Task 13 (workflow)
- §3 Prompt: Task 3 (prompt constant)
- §4 Implementation: Tasks 3-10 cover fetch_tweets, send_telegram, tests
- §5 Stage 1 backtest: Tasks 11-12
- §5 Stage 2 silent: Tasks 13-15
- §5 Stage 3 live: Task 16
- §5 Verification gates: embedded in Task 12 (≥70% top-20) and Task 15 (soak)
- §6 index.html: Task 17
- §6 `AGENTS.md`: Task 18

✅ **Type/name consistency**
- `score_batch`, `run_llm_scoring`, `fallback_score`, `SYSTEM_PROMPT`, `build_user_prompt` — all used consistently across tasks 3-10
- `scored-*.json`, `final-*.json` file names consistent
- `tiers.must_do/priority/backup` used in both fetch_tweets (Task 8) and send_telegram (Task 10) and index.html (Task 17)

✅ **No placeholders**
- Only "this task is manual — user performs it" (Tasks 12, 15, 16) — acceptable, each has explicit user actions
- Task 17 Step 2 says "executing agent should read index.html in full and produce a minimal patch" — this is appropriate because we don't want to prescribe a patch against unknown current HTML; the agent can read the file and apply a minimal change following existing conventions. Not a TBD.

✅ **Commit cadence**: every task ends with a commit.
