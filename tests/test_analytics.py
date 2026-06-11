from datetime import datetime, timezone
import unittest

from cclog.analytics import filter_summaries_by_range


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


if __name__ == "__main__":
    unittest.main()
