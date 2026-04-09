import unittest
import json
import os
from sentinel.app.config import AUDIT_LOG_PATH
from sentinel.core.audit import log_audit

class TestAuditLog(unittest.TestCase):
    def test_write_audit_log_entry(self):
        # Write an audit event and verify the last line is valid JSON with actor
        if os.path.exists(AUDIT_LOG_PATH):
            os.remove(AUDIT_LOG_PATH)
        log_audit("tester", "memory_export", "phase4 test")
        self.assertTrue(os.path.exists(AUDIT_LOG_PATH))
        with open(AUDIT_LOG_PATH, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        self.assertGreater(len(lines), 0)
        last = json.loads(lines[-1])
        self.assertIn('actor', last)
        self.assertEqual(last['actor'], 'tester')
