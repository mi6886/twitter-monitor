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

=== OUTPUT (JSON only, no markdown fences) ===
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


import re as _re


def _strip_code_fence(text: str) -> str:
    """Strip ```json ... ``` or ``` ... ``` wrappers that Anthropic via
    OpenRouter sometimes adds even when response_format=json_object is set."""
    text = text.strip()
    m = _re.match(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", text, _re.DOTALL)
    return m.group(1).strip() if m else text


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

    Failure modes (in order of preference):
    1. LLM call succeeds + JSON parses → return parsed results
    2. LLM call fails or JSON malformed, retries available → backoff + retry
    3. All retries exhausted, batch size > 1 → subdivide into single-tweet
       calls (with max_retries=1 each) so one bad tweet doesn't tank the batch
    4. Single-tweet batch still failing → return fallback_score()
    """
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
            content = resp.choices[0].message.content
            parsed = json.loads(_strip_code_fence(content))
            return parsed["results"]
        except Exception as e:  # noqa: BLE001 — OpenAI SDK raises many types
            last_err = e
            if attempt < max_retries:
                time.sleep(2 ** attempt)

    # All batch attempts exhausted. Subdivide if we have multiple tweets so
    # a single problematic tweet doesn't poison the whole batch.
    if len(tweets_batch) > 1:
        results: list[dict] = []
        for tweet in tweets_batch:
            results.extend(score_batch([tweet], max_retries=1))
        return results

    # Single tweet truly cannot be scored — final fallback.
    err_msg = str(last_err) if last_err else "unknown error"
    return [fallback_score(t, err_msg) for t in tweets_batch]


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
