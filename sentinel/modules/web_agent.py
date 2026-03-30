from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import logging
import time

class WebAgentModule:
    def __init__(self, headless=False):
        self.headless = headless
        self.driver = None
        self.logger = logging.getLogger("WebAgentModule")

    def init_driver(self):
        """Initializes the Chrome WebDriver with options."""
        options = Options()
        if self.headless:
            options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--start-maximized")

        try:
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=options)
            return True, "Driver initialized."
        except Exception as e:
            self.logger.error(f"Error initializing driver: {e}")
            return False, f"Error initializing driver: {e}"

    def quit_driver(self):
        """Quits the driver session."""
        if self.driver:
            self.driver.quit()
            self.driver = None

    def search_and_navigate(self, query, url="https://www.google.com"):
        """Searches Google and navigates to the first result."""
        if not self.driver:
            self.init_driver()
        
        try:
            self.driver.get(url)
            search_box = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.NAME, "q"))
            )
            search_box.send_keys(query)
            search_box.submit()
            
            # Wait for results and click first one
            first_result = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "h3"))
            )
            first_result.click()
            return True, f"Navigated to first result for: {query}"
        except Exception as e:
            self.logger.error(f"Error in search and navigate: {e}")
            return False, f"Error: {e}"

    def book_ticket_generic(self, task_description):
        """A generic method that uses LLM-driven actions to book a ticket."""
        # This would ideally use a more advanced agentic approach (like Browser-use or similar)
        # For now, it provides a foundation for the assistant to perform web tasks.
        self.logger.info(f"Booking ticket task: {task_description}")
        # Implementation of autonomous booking would go here.
        return f"Starting web task for: {task_description}. I will search and try to navigate through the booking process."
