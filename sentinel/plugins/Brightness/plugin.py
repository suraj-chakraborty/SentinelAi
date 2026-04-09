from sentinel.commands.brightness import Brightness as CoreBrightness


class Brightness(CoreBrightness):
    def __init__(self, orchestrator=None):
        super().__init__(orchestrator)
    def on_load(self):
        pass
    def on_unload(self):
        pass
