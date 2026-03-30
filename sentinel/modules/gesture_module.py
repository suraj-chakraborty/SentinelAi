import cv2
import mediapipe as mp
import pyautogui
import threading
import logging
import time
import numpy as np

class GestureModule:
    def __init__(self):
        self.logger = logging.getLogger("GestureModule")
        self.is_running = False
        self.screen_width, self.screen_height = pyautogui.size()
        self.cap = None
        
        # Delayed initialization to catch errors
        try:
            import mediapipe as mp
            # Try multiple import paths for solutions
            if hasattr(mp, 'solutions'):
                self.mp_hands = mp.solutions.hands
                self.mp_draw = mp.solutions.drawing_utils
            else:
                import mediapipe.python.solutions.hands as mp_hands
                import mediapipe.python.solutions.drawing_utils as mp_draw
                self.mp_hands = mp_hands
                self.mp_draw = mp_draw
                
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
            self.available = True
        except Exception as e:
            self.logger.error(f"Failed to initialize Mediapipe: {e}")
            self.available = False
            self.mp_hands = None
            self.hands = None
            self.mp_draw = None

    def start(self):
        """Starts the gesture control in a separate thread."""
        if not self.available:
            self.logger.error("Gesture control unavailable due to library error.")
            return False
        if self.is_running:
            return True
        self.is_running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        self.logger.info("Advanced Gesture control started.")
        return True

    def stop(self):
        """Stops the gesture control."""
        self.is_running = False
        if self.cap:
            self.cap.release()
        self.logger.info("Gesture control stopped.")

    def _get_distance(self, p1, p2):
        """Calculates Euclidean distance between two landmarks."""
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    def _run(self):
        self.cap = cv2.VideoCapture(0)
        prev_x, prev_y = 0, 0
        smoothing = 5
        last_click_time = 0
        scroll_threshold = 0.05
        
        while self.is_running:
            success, img = self.cap.read()
            if not success:
                continue
            
            img = cv2.flip(img, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.hands.process(img_rgb)
            
            if results.multi_hand_landmarks:
                for hand_lms in results.multi_hand_landmarks:
                    # Landmarks mapping
                    thumb_tip = hand_lms.landmark[4]
                    index_tip = hand_lms.landmark[8]
                    middle_tip = hand_lms.landmark[12]
                    ring_tip = hand_lms.landmark[16]
                    pinky_tip = hand_lms.landmark[20]
                    
                    # PIP joints for finger state
                    index_pip = hand_lms.landmark[6]
                    middle_pip = hand_lms.landmark[10]
                    ring_pip = hand_lms.landmark[14]
                    pinky_pip = hand_lms.landmark[18]

                    # 1. MOUSE MOVEMENT (Index finger up)
                    h, w, c = img.shape
                    screen_x = int(index_tip.x * self.screen_width)
                    screen_y = int(index_tip.y * self.screen_height)
                    
                    curr_x = prev_x + (screen_x - prev_x) / smoothing
                    curr_y = prev_y + (screen_y - prev_y) / smoothing
                    pyautogui.moveTo(curr_x, curr_y)
                    prev_x, prev_y = curr_x, curr_y

                    # 2. LEFT CLICK (Index + Thumb Pinch)
                    dist_thumb_index = self._get_distance(thumb_tip, index_tip)
                    if dist_thumb_index < 0.04:
                        if time.time() - last_click_time > 0.3:
                            self.logger.info("Pinch detected - Left Click.")
                            pyautogui.click()
                            last_click_time = time.time()

                    # 3. RIGHT CLICK (Middle + Thumb Pinch)
                    dist_thumb_middle = self._get_distance(thumb_tip, middle_tip)
                    if dist_thumb_middle < 0.04:
                        if time.time() - last_click_time > 0.3:
                            self.logger.info("Pinch detected - Right Click.")
                            pyautogui.rightClick()
                            last_click_time = time.time()

                    # 4. SCROLLING (Three fingers up: Index, Middle, Ring)
                    index_up = index_tip.y < index_pip.y
                    middle_up = middle_tip.y < middle_pip.y
                    ring_up = ring_tip.y < ring_pip.y
                    pinky_up = pinky_tip.y < pinky_pip.y
                    
                    if index_up and middle_up and ring_up and not pinky_up:
                        # Scroll based on vertical movement of index finger
                        if prev_y - screen_y > 10:
                            pyautogui.scroll(150) # Scroll Up
                        elif screen_y - prev_y > 10:
                            pyautogui.scroll(-150) # Scroll Down

                    # 5. ENTER / OPEN (Fist: All fingers down)
                    tips = [index_tip, middle_tip, ring_tip, pinky_tip]
                    pips = [index_pip, middle_pip, ring_pip, pinky_pip]
                    closed_fingers = 0
                    for t, p in zip(tips, pips):
                        if t.y > p.y:
                            closed_fingers += 1
                    
                    if closed_fingers == 4 and dist_thumb_index > 0.05: # Fist but not a click pinch
                        if time.time() - last_click_time > 0.8:
                            self.logger.info("Fist detected - pressing Enter.")
                            pyautogui.press('enter')
                            last_click_time = time.time()

                    # 6. TASK VIEW (Four fingers up)
                    if index_up and middle_up and ring_up and pinky_up:
                        if time.time() - last_click_time > 1.0:
                            self.logger.info("Four fingers detected - Win+Tab.")
                            pyautogui.hotkey('win', 'tab')
                            last_click_time = time.time()
            
            time.sleep(0.01)

        if self.cap:
            self.cap.release()
