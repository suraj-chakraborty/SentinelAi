from sentinel.commands.power_management import PowerManagement as CorePower


class PowerManagement(CorePower):
    def __init__(self, orchestrator=None):
        super().__init__(orchestrator)
    def on_load(self):
        pass
    def on_unload(self):
        pass
