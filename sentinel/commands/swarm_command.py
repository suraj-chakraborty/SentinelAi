"""
sentinel/commands/swarm_command.py
──────────────────────────────────
Voice Command Plugin for Multi-Agent Swarm Missions.
Handles "run nightly audit", "swarm mission", and "codebase health check".
"""

import logging

from sentinel.core.nightly_swarm import NightlySwarmCoordinator

logger = logging.getLogger("SwarmCommand")


class SwarmCommand:
    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator
        self.coordinator = NightlySwarmCoordinator()

    def execute(self, command: str, entity: str | None = None) -> str:
        cmd_lower = command.lower()

        # 1. Run full swarm mission / audit
        if any(
            k in cmd_lower
            for k in ("nightly audit", "swarm mission", "run swarm", "codebase health", "audit repository")
        ):
            res = self.coordinator.run_nightly_mission()
            audit = res.get("audit", {})
            git_stat = "clean" if audit.get("git_clean") else "has pending changes"
            test_stat = audit.get("test_status", "passed")
            return (
                f"Swarm mission completed. Repository is {git_stat}, and tests: {test_stat}. Morning report generated."
            )

        # 2. Tech Intelligence Hunt
        if any(k in cmd_lower for k in ("tech intelligence", "swarm research", "hunt tech")):
            intel = self.coordinator.hunter.hunt()
            headlines = [f"{i['topic']}: {i['insight'][:60]}..." for i in intel[:2]]
            return f"Swarm gathered intelligence on {len(intel)} topics: {'; '.join(headlines)}"

        # 3. Self-healing check
        if any(k in cmd_lower for k in ("self-healing", "self healing", "heal errors")):
            fix = self.coordinator.fixer.analyze_and_propose("")
            return f"Self-healing status: {fix.get('proposal')}"

        return "Swarm command not recognized."
