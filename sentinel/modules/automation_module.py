import pyautogui
import logging
import time

class AutomationModule:
    def __init__(self):
        # Fail-safe to move mouse to corner to abort
        pyautogui.FAILSAFE = True
        self.logger = logging.getLogger("AutomationModule")

    def move_and_click(self, x, y, button='left'):
        """Moves the mouse and clicks."""
        try:
            self.logger.info(f"Moving to ({x}, {y}) and clicking {button}.")
            pyautogui.moveTo(x, y, duration=0.5)
            pyautogui.click(button=button)
            return True, f"Clicked at ({x}, {y})."
        except Exception as e:
            self.logger.error(f"Click error: {e}")
            return False, f"Error: {e}"

    def type_text(self, text, interval=0.1):
        """Types text with an interval between keystrokes."""
        try:
            self.logger.info(f"Typing: {text[:20]}...")
            pyautogui.write(text, interval=interval)
            return True, "Typed text."
        except Exception as e:
            self.logger.error(f"Typing error: {e}")
            return False, f"Error: {e}"

    def press_key(self, key):
        """Presses a specific key (e.g., 'enter', 'tab')."""
        try:
            pyautogui.press(key)
            return True, f"Pressed {key}."
        except Exception as e:
            self.logger.error(f"Key press error: {e}")
            return False, f"Error: {e}"

    def hotkey(self, *keys):
        """Performs a hotkey combination (e.g., 'ctrl', 'c')."""
        try:
            pyautogui.hotkey(*keys)
            return True, f"Pressed hotkey: {'+'.join(keys)}."
        except Exception as e:
            self.logger.error(f"Hotkey error: {e}")
            return False, f"Error: {e}"

    def get_screen_info(self):
        """Returns the current screen size and mouse position."""
        size = pyautogui.size()
        pos = pyautogui.position()
        return f"Screen size: {size.width}x{size.height}, Mouse at: ({pos.x}, {pos.y})"

    def execute_gui_task(self, task_description, llm_callback):
        """Translates a natural language GUI task into PyAutoGUI actions."""
        # This function would use the LLM to generate a sequence of PyAutoGUI commands.
        # Example: "Open Notepad, type Hello, and save it."
        prompt = f"Translate the following GUI task into a series of PyAutoGUI function calls as a Python list of strings. Each string should be a valid Python call to an instance named 'automation'.\nTask: {task_description}\nExample: [\"automation.move_and_click(100, 200)\", \"automation.type_text('Hello')\"]"
        response = llm_callback(prompt)
        # Parse the list and execute. (Simple implementation for now)
        return f"Executing GUI task: {task_description}. I will attempt to follow the sequence: {response}"
