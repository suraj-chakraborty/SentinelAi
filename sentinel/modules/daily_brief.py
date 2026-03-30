from datetime import datetime
import logging

class DailyBriefModule:
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.logger = logging.getLogger("DailyBrief")

    def generate_brief(self):
        """Generates a comprehensive daily brief for the user."""
        try:
            # 1. Calendar Events
            meets = self.orchestrator.calendar_module.get_upcoming_meets(hours=12)
            meet_summary = f"You have {len(meets)} meetings scheduled for today." if meets else "No meetings scheduled for today."
            
            # 2. System Health
            cpu = self.orchestrator.system_monitor.get_cpu_temp()
            health = f"System health is optimal (CPU at {cpu}°C)." if cpu and cpu < 70 else "System ran a bit hot recently."
            
            # 3. Document Summary
            # Scan downloads for the last 24 hours
            recent_files = self.orchestrator.file_intel.scan_directory(
                self.orchestrator.file_intel.knowledge_base.db_path, # Using DB path as a proxy for scanned docs
                hours=24
            )
            file_info = f"I've indexed {len(recent_files)} new documents since yesterday." if recent_files else ""

            # 4. Habit Insight
            # Query RAG for patterns
            habit_insight = self.orchestrator.knowledge_base.query_knowledge("What are my usual habits?")
            
            brief = f"Good morning! {meet_summary} {health} {file_info} {habit_insight}"
            return brief
        except Exception as e:
            self.logger.error(f"Error generating brief: {e}")
            return "Good morning! I was unable to compile your full brief, but I'm ready for your commands."
