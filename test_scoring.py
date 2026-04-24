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


def make_llm_response(results, fenced: bool = False):
    """Build a mocked OpenAI ChatCompletion object with JSON content.
    If fenced=True, wrap the JSON in ```json ... ``` fences to simulate what
    Anthropic-via-OpenRouter actually returns."""
    payload = json.dumps({"results": results}, ensure_ascii=False)
    if fenced:
        payload = f"```json\n{payload}\n```"
    mock_msg = MagicMock()
    mock_msg.content = payload
    mock_choice = MagicMock()
    mock_choice.message = mock_msg
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    return mock_resp


def make_score_result(tweet_id="t1", total_score=68, verdict="keep"):
    return {
        "id": tweet_id,
        "summary": "测试概要",
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
        "total_score": total_score,
        "verdict": verdict,
        "angles": ["A型·第一型", "A型·反常识型", "话题聚焦"],
    }


class TestScoreBatchSuccess:
    def test_parses_clean_json_response(self):
        tweet = make_tweet(tweet_id="t1", text="Anthropic releases Claude 5")
        mocked = make_llm_response([make_score_result("t1")])
        with patch.object(llm_scoring, "_get_client") as gc:
            gc.return_value.chat.completions.create.return_value = mocked
            result = llm_scoring.score_batch([tweet])
        assert len(result) == 1
        assert result[0]["id"] == "t1"
        assert result[0]["total_score"] == 68
        assert result[0]["verdict"] == "keep"
        assert len(result[0]["angles"]) == 3

    def test_parses_markdown_fenced_response(self):
        """OpenRouter+Anthropic sometimes returns ```json ... ``` fences even
        when response_format=json_object is set. Parser must strip them."""
        tweet = make_tweet(tweet_id="t2", text="Another tweet")
        mocked = make_llm_response([make_score_result("t2")], fenced=True)
        with patch.object(llm_scoring, "_get_client") as gc:
            gc.return_value.chat.completions.create.return_value = mocked
            result = llm_scoring.score_batch([tweet])
        assert len(result) == 1
        assert result[0]["id"] == "t2"
        assert result[0]["total_score"] == 68
