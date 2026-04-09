import unittest
from sentinel.core.auth import is_admin

class TestAuth(unittest.TestCase):
    def test_is_admin_header(self):
        class Req:
            headers = {"X-Role": "admin"}
            query_params = {}
        self.assertTrue(is_admin(Req()))

    def test_is_not_admin(self):
        class Req:
            headers = {"X-Role": "user"}
            query_params = type('QP', (), {'get': lambda self, k: None})()
        self.assertFalse(is_admin(Req()))
