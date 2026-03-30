import time
import threading
import logging
import psutil
from datetime import datetime
try:
    import pygetwindow as gw
except ImportError:
    gw = None

class HabitLearnerModule:
    def __init__(self, knowledge_base):
        self.knowledge_base = knowledge_base
        self.logger = logging.getLogger("HabitLearner")
        self.habits = {} # {app_name: {time_slot: count}}
        self.is_running = False
        self.current_app = None
        self.start_time = None

    def start_monitoring(self):
        """Starts monitoring foreground window habits in a background thread."""
        if self.is_running:
            return
        self.is_running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        self.logger.info("Habit monitoring started.")

    def _monitor_loop(self):
        while self.is_running:
            try:
                if gw:
                    win = gw.getActiveWindow()
                    if win and win.title:
                        app_name = win.title.split('-')[-1].strip()
                        self._record_event(app_name)
            except Exception as e:
                self.logger.error(f"Habit monitor error: {e}")
            time.sleep(60) # Check every minute

    def _record_event(self, app_name):
        now = datetime.now()
        hour = now.hour
        slot = f"{hour}:00"
        
        # Simple local tracking
        if app_name not in self.habits:
            self.habits[app_name] = {}
        self.habits[app_name][slot] = self.habits[app_name].get(slot, 0) + 1
        
        # If we see a strong pattern (e.g., app used 5 times at this hour)
        if self.habits[app_name][slot] >= 5:
            pattern_msg = f"User frequently uses {app_name} around {slot}."
            # Save to RAG for long-term reasoning
            self.knowledge_base.add_information(
                pattern_msg, 
                metadata={"source": "habit_learner", "type": "pattern"}
            )
            # Reset count to avoid spamming knowledge base
            self.habits[app_name][slot] = 0

    def get_learned_insights(self, query):
        """Asks the knowledge base for learned habits based on a query."""
        return self.knowledge_base.query_knowledge(f"User habits and patterns related to: {query}")
