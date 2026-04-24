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


class TestScoreBatchRetry:
    def test_retries_on_transient_failure_then_succeeds(self):
        tweet = make_tweet("t1")
        success_resp = make_llm_response([make_score_result("t1", total_score=0, verdict="drop")])
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
        tweet = {"author": NOISE_ACCOUNTS[0], "full_text": "anything"}
        assert is_noise(tweet) is True

    def test_missing_author_not_noise(self):
        tweet = {"full_text": "anything"}
        assert is_noise(tweet) is False
