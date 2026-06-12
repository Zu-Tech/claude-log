from datetime import datetime, timezone
import unittest

from cclog.analytics import filter_summaries_by_range, get_model_cost


class TimeRangeFilterTests(unittest.TestCase):
    def test_filters_sessions_started_in_recent_window(self):
        now = datetime(2026, 6, 11, 12, 0, tzinfo=timezone.utc)
        summaries = [
            {"session_id": "recent", "started_at": "2026-06-01T09:00:00+00:00"},
            {"session_id": "old", "started_at": "2026-04-01T09:00:00+00:00"},
            {"session_id": "missing"},
        ]

        filtered = filter_summaries_by_range(summaries, "30d", now=now)

        self.assertEqual([s["session_id"] for s in filtered], ["recent"])

    def test_all_time_keeps_all_summaries(self):
        summaries = [{"session_id": "a"}, {"session_id": "b"}]

        self.assertEqual(filter_summaries_by_range(summaries, "all"), summaries)

    def test_invalid_range_falls_back_to_all_time(self):
        summaries = [{"session_id": "a"}, {"session_id": "b"}]

        self.assertEqual(filter_summaries_by_range(summaries, "invalid-range"), summaries)


class ModelCostTests(unittest.TestCase):
    def test_fable_models_use_fable_api_pricing(self):
        self.assertEqual(
            get_model_cost("claude-fable-5-20260610"),
            {"input": 10.0, "cache_create": 12.50, "cache_read": 1.0, "output": 50.0},
        )


if __name__ == "__main__":
    unittest.main()
