"""
Unit tests for final_filter and candidate_filter.
Run: APIFY_TOKEN=dummy python3 -m pytest test_filters.py -v
  or: APIFY_TOKEN=dummy python3 test_filters.py
"""

import os
os.environ.setdefault("APIFY_TOKEN", "dummy")

from fetch_tweets import final_filter, candidate_filter


def make_tweet(text, screen_name="testuser", likes=5000, quoted=""):
    return {
        "full_text": text,
        "screen_name": screen_name,
        "favorite_count": likes,
        "quoted_tweet_text": quoted,
    }


# ============================================================
# final_filter tests
# ============================================================

class TestFinalKeep:
    """Tweets that SHOULD land in final (keep)."""

    def test_product_launch(self):
        tw = make_tweet(
            "Anthropic just launched Claude 4.0 with 1M context window. "
            "New API endpoints available today.",
            screen_name="AnthropicAI",
        )
        disp, reason = final_filter(tw)
        assert disp == "keep", f"Expected keep, got {disp} ({reason})"

    def test_funding_news(self):
        tw = make_tweet(
            "OpenAI raises $6.6 billion at $157 billion valuation, "
            "the largest private funding round ever.",
        )
        disp, reason = final_filter(tw)
        assert disp == "keep", f"Expected keep, got {disp} ({reason})"

    def test_sora_shutdown(self):
        tw = make_tweet(
            "OpenAI is shutting down its AI video platform Sora "
            "after less than six months.",
        )
        disp, reason = final_filter(tw)
        assert disp == "keep", f"Expected keep, got {disp} ({reason})"

    def test_tutorial(self):
        tw = make_tweet(
            "How to build a RAG pipeline with Claude and LlamaIndex: "
            "step-by-step tutorial with code examples.",
        )
        disp, reason = final_filter(tw)
        assert disp == "keep", f"Expected keep, got {disp} ({reason})"

    def test_high_signal_source(self):
        tw = make_tweet(
            "We're seeing fascinating patterns in how people use Claude "
            "differently as they gain more experience with the tool.",
            screen_name="AnthropicAI",
        )
        disp, reason = final_filter(tw)
        assert disp == "keep", f"Expected keep, got {disp} ({reason})"

    def test_benchmark(self):
        tw = make_tweet(
            "Claude Opus scores 92.3% on GPQA Diamond, beating GPT-4o "
            "by 8 points. Full benchmark comparison here.",
        )
        disp, reason = final_filter(tw)
        assert disp == "keep", f"Expected keep, got {disp} ({reason})"


class TestFinalReview:
    """Tweets that should land in review, NOT final."""

    def test_entertainment_repost_animetrendsla(self):
        """@AnimetrendsLA style: entertainment/media account reposting AI news."""
        tw = make_tweet(
            "OpenAI has officially announced a partnership with Disney "
            "for AI-generated animation. This is huge for the anime industry!",
            screen_name="AnimetrendsLA",
        )
        disp, reason = final_filter(tw)
        assert disp == "review", f"Expected review, got {disp} ({reason})"

    def test_sports_gossip_with_ai_mention(self):
        tw = make_tweet(
            "BREAKING: Real Madrid nutritionist caught using free ChatGPT "
            "for player diet plans instead of professional tools.",
            screen_name="theMadridZone",
        )
        disp, reason = final_filter(tw)
        assert disp == "review", f"Expected review, got {disp} ({reason})"

    def test_pure_reaction(self):
        tw = make_tweet(
            "Claude is insane. Just absolutely insane. I can't even.",
        )
        disp, reason = final_filter(tw)
        assert disp == "review", f"Expected review, got {disp} ({reason})"

    def test_engagement_bait(self):
        tw = make_tweet(
            "Steal these 100 Claude tips from me right now! "
            "Comment 'Claude' to get the PDF in your inbox.",
        )
        disp, reason = final_filter(tw)
        assert disp == "review", f"Expected review, got {disp} ({reason})"

    def test_meme_format(self):
        tw = make_tweet("me when ChatGPT gives me the wrong answer for the 5th time today")
        disp, reason = final_filter(tw)
        assert disp == "review", f"Expected review, got {disp} ({reason})"

    def test_commentary_recap(self):
        tw = make_tweet(
            "Do you understand what happened today? Disney pulled out of "
            "OpenAI, Sora is dead, and Claude launched computer use.",
        )
        disp, reason = final_filter(tw)
        assert disp == "review", f"Expected review, got {disp} ({reason})"

    def test_music_gemini(self):
        tw = make_tweet(
            "Dangerous - GEMINI MUSIC VIDEO RELEASE 27.03.2026 "
            "YouTube: RISER MUSIC #DangerousGEMINI #GEMINI_NT",
            screen_name="RiserMusic",
        )
        disp, reason = final_filter(tw)
        assert disp == "review", f"Expected review, got {disp} ({reason})"

    def test_linux_offtopic(self):
        tw = make_tweet(
            "Nobara OS is what Linux should have been. Built on Fedora, "
            "gaming-first distro. Claude recommended it to me.",
        )
        disp, reason = final_filter(tw)
        assert disp == "review", f"Expected review, got {disp} ({reason})"

    def test_crypto_with_ai_mention(self):
        tw = make_tweet(
            "New Solana memecoin called $CLAUDE just launched. "
            "Already 500% up. This is the next big token.",
        )
        disp, reason = final_filter(tw)
        assert disp == "review", f"Expected review, got {disp} ({reason})"


class TestFinalDropHard:
    """Tweets that should be dropped entirely."""

    def test_too_short(self):
        tw = make_tweet("lol nice")
        disp, reason = final_filter(tw)
        assert disp == "drop_hard", f"Expected drop_hard, got {disp} ({reason})"

    def test_no_ai_name(self):
        tw = make_tweet(
            "Just shipped a new feature for our SaaS dashboard. "
            "Really proud of the team effort on this one.",
        )
        disp, reason = final_filter(tw)
        assert disp == "drop_hard", f"Expected drop_hard, got {disp} ({reason})"

    def test_offensive(self):
        tw = make_tweet("kill all those chatgpt users lmao")
        disp, reason = final_filter(tw)
        assert disp == "drop_hard", f"Expected drop_hard, got {disp} ({reason})"


# ============================================================
# Run directly
# ============================================================

if __name__ == "__main__":
    import sys

    classes = [TestFinalKeep, TestFinalReview, TestFinalDropHard]
    passed = 0
    failed = 0

    for cls in classes:
        print(f"\n{'=' * 50}")
        print(f"  {cls.__name__}")
        print(f"{'=' * 50}")
        obj = cls()
        for name in sorted(dir(obj)):
            if not name.startswith("test_"):
                continue
            method = getattr(obj, name)
            try:
                method()
                print(f"  PASS  {name}")
                passed += 1
            except AssertionError as e:
                print(f"  FAIL  {name}: {e}")
                failed += 1

    print(f"\n{'=' * 50}")
    print(f"  {passed} passed, {failed} failed")
    print(f"{'=' * 50}")
    sys.exit(1 if failed else 0)
