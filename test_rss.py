"""Offline tests for RSS source configuration and page date extraction."""
import re
import unittest
from unittest.mock import MagicMock, patch

import fetch_rss


class VisibleDateTests(unittest.TestCase):
    def test_english_date(self):
        self.assertEqual(
            fetch_rss._visible_date("Published June 30, 2026"),
            "2026-06-30T00:00:00+00:00",
        )

    def test_slash_separated_date(self):
        self.assertEqual(
            fetch_rss._visible_date("发布日期 2022/5/27"),
            "2022-05-27T00:00:00+00:00",
        )

    def test_chinese_date(self):
        self.assertEqual(
            fetch_rss._visible_date("2026年7月18日"),
            "2026-07-18T00:00:00+00:00",
        )

    def test_invalid_date(self):
        self.assertIsNone(fetch_rss._visible_date("2026年13月40日"))


class HttpGetTests(unittest.TestCase):
    def test_retries_empty_success_response(self):
        empty = MagicMock()
        empty.__enter__.return_value.read.return_value = b""
        valid = MagicMock()
        valid.__enter__.return_value.read.return_value = b"<rss />"

        with patch("fetch_rss.urllib.request.urlopen", side_effect=[empty, valid]), \
             patch("fetch_rss.time.sleep"):
            self.assertEqual(fetch_rss.http_get("https://example.com/feed"), "<rss />")

    def test_feed_retries_when_parsed_response_has_no_entries(self):
        empty_feed = "<rss><channel><title>Empty</title></channel></rss>"
        valid_feed = (
            "<rss><channel><title>Test</title><item><title>New item</title>"
            "<link>https://example.com/new</link></item></channel></rss>"
        )
        with patch("fetch_rss.http_get", side_effect=[empty_feed, valid_feed]), \
             patch("fetch_rss.time.sleep"):
            items = fetch_rss.fetch_feed("Test", "https://example.com/feed")

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["url"], "https://example.com/new")

    def test_feed_uses_podcast_enclosure_when_link_is_missing(self):
        podcast_feed = (
            "<rss><channel><title>Podcast</title><item><title>Episode</title>"
            "<enclosure url=\"https://cdn.example.com/episode.mp3\" "
            "type=\"audio/mpeg\" /></item></channel></rss>"
        )
        with patch("fetch_rss.http_get", return_value=podcast_feed):
            items = fetch_rss.fetch_feed("Podcast", "https://example.com/feed")

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["url"], "https://cdn.example.com/episode.mp3")

    def test_feed_records_source_type(self):
        feed = (
            "<rss><channel><title>Video</title><item><title>Episode</title>"
            "<link>https://youtube.com/watch?v=test</link></item></channel></rss>"
        )
        with patch("fetch_rss.http_get", return_value=feed):
            items = fetch_rss.fetch_feed(
                "Test YouTube", "https://example.com/feed", "youtube"
            )

        self.assertEqual(items[0]["source_type"], "youtube")


class SourceConfigurationTests(unittest.TestCase):
    def test_per_feed_limit_is_bounded(self):
        self.assertGreater(fetch_rss.MAX_ITEMS_PER_FEED, 0)
        self.assertLessEqual(fetch_rss.MAX_ITEMS_PER_FEED, fetch_rss.MAX_ITEMS)

    def test_required_native_feeds_are_configured(self):
        labels = {label for label, _ in fetch_rss.FEEDS}
        self.assertTrue({
            "IEEE Spectrum Robotics",
            "The Robot Report",
            "Lex Fridman Podcast",
            "No Priors Podcast",
            "AI + a16z",
            "Dwarkesh Podcast",
            "Latent Space",
        }.issubset(labels))

    def test_required_sitemap_sources_are_configured(self):
        labels = {source[0] for source in fetch_rss.SITEMAP_SOURCES}
        self.assertTrue({
            "Luma Changelog",
            "Figure AI News",
            "Boston Dynamics Blog",
            "Unitree News",
            "Agility Robotics",
            "AgiBot News",
        }.issubset(labels))

    def test_required_youtube_sources_are_configured(self):
        labels = {source[0] for source in fetch_rss.YOUTUBE_FEEDS}
        self.assertEqual(labels, {
            "IEEE Spectrum Robotics YouTube",
            "The Robot Report YouTube",
            "Figure AI YouTube",
            "Boston Dynamics YouTube",
            "Unitree YouTube",
            "Agility Robotics YouTube",
            "1X Technologies YouTube",
            "Tesla AI / Optimus YouTube",
            "AgiBot YouTube",
            "NVIDIA Robotics YouTube",
            "Lex Fridman YouTube",
            "No Priors YouTube",
            "a16z AI YouTube",
            "Dwarkesh Podcast YouTube",
            "Latent Space YouTube",
        })

    def test_youtube_channel_ids_and_labels_are_unique(self):
        labels = [source[0] for source in fetch_rss.YOUTUBE_FEEDS]
        channel_ids = [source[1] for source in fetch_rss.YOUTUBE_FEEDS]
        self.assertEqual(len(labels), len(set(labels)))
        self.assertEqual(len(channel_ids), len(set(channel_ids)))
        self.assertTrue(all(
            re.fullmatch(r"UC[A-Za-z0-9_-]{22}", channel_id)
            for channel_id in channel_ids
        ))

    def test_broad_youtube_channel_filter(self):
        items = [
            {"title": "Atlas humanoid robot update", "summary": ""},
            {"title": "A new electric car paint color", "summary": ""},
        ]
        kept = fetch_rss.filter_youtube_items(items, r"\brobot\b|\bhumanoid\b")
        self.assertEqual(kept, [items[0]])
        self.assertIs(fetch_rss.filter_youtube_items(items, None), items)

    def test_source_labels_and_urls_are_unique(self):
        labels = [label for label, _ in fetch_rss.FEEDS]
        urls = [url for _, url in fetch_rss.FEEDS]
        self.assertEqual(len(labels), len(set(labels)))
        self.assertEqual(len(urls), len(set(urls)))


if __name__ == "__main__":
    unittest.main()
