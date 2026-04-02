"""
sentinel/modules/lora_collector.py
───────────────────────────────────
Phase 4: Local Model Fine-Tuning Pipeline.
Listens to conversation flows and saves high-quality instruction-response pairs
into a `.jsonl` file formatted for Unsloth/HuggingFace LoRA fine-tuning.
"""

import os
import json
import logging
import threading

logger = logging.getLogger("LoRACollector")

class LoRACollector:
    def __init__(self, filename="sentinel_dataset.jsonl"):
        self.output_dir = os.path.join(os.path.expanduser("~"), "AppData", "Roaming", "SentinelAi", "training_data")
        os.makedirs(self.output_dir, exist_ok=True)
        self.filename = os.path.join(self.output_dir, filename)
        self._lock = threading.Lock()
        
    def log_interaction(self, instruction: str, response: str, system_prompt: str = "You are Sentinel, an advanced AI."):
        """Saves a single interaction in Alpaca/ChatML JSONL format."""
        if not instruction or not response:
            return
            
        # We only want to log substantial, high-quality interactions.
        if len(instruction) < 10 or len(response) < 20:
            return
            
        data = {
            "instruction": instruction.strip(),
            "input": "",
            "output": response.strip(),
            "system": system_prompt
        }
        
        with self._lock:
            try:
                with open(self.filename, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(data) + "\n")
            except Exception as e:
                logger.error(f"Failed to log LoRA training data: {e}")
                
    def get_dataset_size(self) -> int:
        """Returns the number of training examples collected."""
        if not os.path.exists(self.filename):
            return 0
        try:
            with open(self.filename, 'r', encoding='utf-8') as f:
                return sum(1 for _ in f)
        except Exception:
            return 0
