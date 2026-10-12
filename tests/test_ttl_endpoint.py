import os
import unittest


class TestTTL(unittest.TestCase):
    def test_ttl_env_parsing(self):
        # Just ensure env var can be parsed (sanity)
        os.environ["SENTINEL_MEMORY_TTL_HOURS"] = "2"
        self.assertEqual(int(os.getenv("SENTINEL_MEMORY_TTL_HOURS")), 2)
