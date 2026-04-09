import unittest

class TestVolumeControl(unittest.TestCase):
    def test_volume_up(self):
        from sentinel.commands.volume_control import VolumeControl
        vc = VolumeControl()
        out = vc.execute("volume up", None)
        self.assertIn("volume", out.lower())

    def test_set_volume(self):
        from sentinel.commands.volume_control import VolumeControl
        vc = VolumeControl()
        out = vc.execute("set volume to 75 percent", None)
        self.assertIn("75", out)
