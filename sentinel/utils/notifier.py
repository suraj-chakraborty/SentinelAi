from win10toast import ToastNotifier
from queue import Queue
import threading
import logging
import os

class WindowsNotifier:
    def __init__(self):
        self.toaster = ToastNotifier()
        self.logger = logging.getLogger("WindowsNotifier")
        self.icon_path = os.path.join(os.path.dirname(__file__), "..", "..", "assets", "icon.ico")
        if not os.path.exists(self.icon_path):
            self.icon_path = None
        self._queue = Queue()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def _run(self):
        while True:
            title, message, duration = self._queue.get()
            try:
                self.toaster.show_toast(
                    title,
                    message,
                    icon_path=self.icon_path,
                    duration=duration,
                    threaded=False
                )
            except Exception as e:
                self.logger.error(f"Error showing notification: {e}")

    def show_notification(self, title, message, duration=5):
        """Shows a Windows toast notification in a separate thread."""
        self._queue.put((title, message, duration))
