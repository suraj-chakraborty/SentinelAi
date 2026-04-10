"""
sentinel/memory/screen_capture.py
─────────────────────────────────
Screen capture module for semantic screen memory.
Captures screenshots and extracts text via OCR.
"""

import os
import time
import logging
from datetime import datetime
from typing import Optional, Tuple
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import pytesseract
    PYTESSERACT_OK = True
except ImportError:
    PYTESSERACT_OK = False
    logging.warning("pytesseract not available - OCR disabled")

from sentinel.app.config import APPDATA_DIR

logger = logging.getLogger(__name__)

SCREENSHOTS_DIR = os.path.join(APPDATA_DIR, "screenshots")
Path(SCREENSHOTS_DIR).mkdir(parents=True, exist_ok=True)


class ScreenCapture:
    def __init__(self, save_screenshots: bool = True):
        self.save_screenshots = save_screenshots
        self._screenshot_count = 0

    def capture_full_screen(self) -> Optional[np.ndarray]:
        try:
            import pyautogui
            screenshot = pyautogui.screenshot()
            return np.array(screenshot)
        except Exception as e:
            logger.error(f"Failed to capture screen: {e}")
            return None

    def capture_window(self, window_title: str) -> Optional[np.ndarray]:
        try:
            import pygetwindow as gw
            windows = gw.getWindowsWithTitle(window_title)
            if not windows:
                logger.warning(f"Window not found: {window_title}")
                return None
            
            win = windows[0]
            left, top = win.left, win.top
            width, height = win.width, win.height
            
            import pyautogui
            screenshot = pyautogui.screenshot(region=(left, top, width, height))
            return np.array(screenshot)
        except Exception as e:
            logger.error(f"Failed to capture window '{window_title}': {e}")
            return None

    def save_screenshot(self, image: np.ndarray, prefix: str = "screen") -> Optional[str]:
        if not self.save_screenshots:
            return None
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{prefix}_{timestamp}_{self._screenshot_count:04d}.png"
            filepath = os.path.join(SCREENSHOTS_DIR, filename)
            
            Image.fromarray(image).save(filepath)
            self._screenshot_count += 1
            logger.info(f"Screenshot saved: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to save screenshot: {e}")
            return None

    def extract_text(self, image: np.ndarray, lang: str = "eng") -> str:
        if not PYTESSERACT_OK:
            logger.warning("OCR not available - pytesseract not installed")
            return ""
        try:
            pil_image = Image.fromarray(image)
            text = pytesseract.image_to_string(pil_image, lang=lang)
            return text.strip()
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return ""

    def capture_and_index(self) -> Optional[Tuple[str, str, Optional[str]]]:
        image = self.capture_full_screen()
        if image is None:
            return None
        
        text = self.extract_text(image)
        if not text:
            logger.warning("No text extracted from screenshot")
            return None
        
        filepath = self.save_screenshot(image)
        return text, filepath

    def get_screenshot_count(self) -> int:
        return self._screenshot_count