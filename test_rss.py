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

    def test_curl_fallback_after_urllib_failures(self):
        completed = MagicMock(stdout=b"<rss />")
        with patch("fetch_rss.urllib.request.urlopen", side_effect=OSError("down")), \
             patch("fetch_rss.time.sleep"), \
             patch("fetch_rss.subprocess.run", return_value=completed) as run:
            body = fetch_rss.http_get("https://example.com/feed", retries=0)

        self.assertEqual(body, "<rss />")
        run.assert_called_once()


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

    def test_china_ai_first_batch_is_fully_configured(self):
        expected = {
            "Qwen Blog",
            "ByteDance Seed Blog",
            "DeepSeek News",
            "Kimi Blog",
            "Z.ai Release Notes",
            "MiniMax Blog",
            "Kling AI Updates",
            "Vidu Product Updates",
            "PixVerse Product Updates",
            "TRAE Blog",
            "Qoder Blog",
            "Coze Changelog",
            "Manus Blog",
        }
        configured = (
            {source[0] for source in fetch_rss.SITEMAP_SOURCES}
            | {source[0] for source in fetch_rss.LINK_SOURCES}
            | {source[0] for source in fetch_rss.PAGE_SOURCES}
        )
        self.assertTrue(expected.issubset(configured))

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


class WebsiteDiscoveryTests(unittest.TestCase):
    def test_nested_sitemap_index_is_followed(self):
        root = (
            "<sitemapindex><sitemap><loc>https://example.com/articles.xml</loc>"
            "</sitemap></sitemapindex>"
        )
        child = (
            "<urlset><url><loc>https://example.com/blog/new-model</loc>"
            "<lastmod>2026-08-01</lastmod></url></urlset>"
        )

        def fake_get(url, timeout=20):
            return root if url.endswith("sitemap.xml") else child

        with patch("fetch_rss.http_get", side_effect=fake_get):
            entries = fetch_rss.sitemap_entries(
                "https://example.com/sitemap.xml", "/blog/"
            )

        self.assertEqual(entries, [
            ("https://example.com/blog/new-model", "2026-08-01T00:00:00"),
        ])

    def test_listing_extractor_handles_spa_routes_and_query_policy(self):
        page = """
        <a href='/blog/product-launch?tab=all'>Launch</a>
        <a href='/blog/how-to-write-prompts'>Guide</a>
        <script>button.href = '/blog?id=qwen3.8';</script>
        """
        with patch("fetch_rss.http_get", return_value=page):
            product_urls = fetch_rss.listing_entries(
                "https://example.com/blog",
                r"^/blog/product-[A-Za-z0-9_-]+(?:\?[^#]+)?$",
                keep_query=False,
            )
            qwen_urls = fetch_rss.listing_entries(
                "https://example.com/blog",
                r"^/blog\?id=[A-Za-z0-9._-]+$",
                keep_query=True,
            )

        self.assertEqual(product_urls, ["https://example.com/blog/product-launch"])
        self.assertEqual(qwen_urls, ["https://example.com/blog?id=qwen3.8"])

    def test_page_signature_ignores_nonce_and_comments(self):
        first = '<html nonce="random-a"><!-- built now --><h1>Update</h1></html>'
        second = '<html nonce="random-b"><!-- built later --><h1>Update</h1></html>'
        self.assertEqual(
            fetch_rss.page_signature(first),
            fetch_rss.page_signature(second),
        )

    def test_trim_items_keeps_each_represented_source(self):
        items = [
            {"source": "Busy", "url": f"https://busy/{i}",
             "published_at": f"2026-08-{10 - i:02d}T00:00:00+00:00"}
            for i in range(4)
        ]
        items.append({
            "source": "Quiet",
            "url": "https://quiet/important",
            "published_at": "2025-01-01T00:00:00+00:00",
        })

        trimmed = fetch_rss.trim_items(items, 3)

        self.assertEqual(len(trimmed), 3)
        self.assertEqual({item["source"] for item in trimmed}, {"Busy", "Quiet"})


if __name__ == "__main__":
    unittest.main()
