import unittest

def test_placeholder():
    pass


class TestPluginHealth(unittest.TestCase):
    def test_default_health_true(self):
        from sentinel.core.plugin_system import PluginBase
        p = PluginBase()
        self.assertTrue(p.health())
