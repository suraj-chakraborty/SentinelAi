"""
sentinel/modules/bci_module.py
──────────────────────────────────
Sci-Fi Tier 3: Brain-Computer Interface (BCI).
Listens to a Lab Streaming Layer (LSL) stream which is the standard protocol for
commercial EEG headsets like Muse, Emotiv, or OpenBCI.
"""

import threading
import time
import logging

try:
    from pylsl import resolve_stream, StreamInlet
    _LSL_AVAILABLE = True
except ImportError:
    _LSL_AVAILABLE = False

logger = logging.getLogger("BCIModule")

class BCIModule:
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.inlet = None
        self._running = False
        self._thread = None
        self.consecutive_stress_spikes = 0

    def start(self):
        if not _LSL_AVAILABLE:
            logger.warning("pylsl not installed. BCI module unavailable.")
            return "Brain-Computer Interface disabled. Install pylsl and connect an EEG headset."
        
        if self._running:
            return "BCI is already running."
            
        self._running = True
        self._thread = threading.Thread(target=self._bci_loop, daemon=True)
        self._thread.start()
        return "Checking for EEG brainwave streams via LSL..."

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        return "BCI module stopped."

    def _bci_loop(self):
        logger.info("Resolving LSL EEG streams...")
        try:
            streams = resolve_stream('type', 'EEG')
            if not streams:
                logger.warning("No EEG stream found on the network.")
                self._running = False
                return
                
            self.inlet = StreamInlet(streams[0])
            logger.info("Successfully connected to Brainwave Stream.")
        except Exception as e:
            logger.error(f"Failed to connect to LSL stream: {e}")
            self._running = False
            return

        while self._running:
            try:
                # Sample chunk of data (mock logic for thresholds)
                chunk, timestamps = self.inlet.pull_chunk(timeout=1.0, max_samples=100)
                if timestamps:
                    # In a real scenario: calculate Power Spectral Density for Alpha/Beta bands
                    # Here we mock a basic "Beta wave (stress)" heuristic
                    avg_signal = sum(sum(sample) for sample in chunk) / (len(chunk) * len(chunk[0]))
                    
                    if avg_signal > 1.5:  # Mock high-stress threshold
                        self.consecutive_stress_spikes += 1
                        if self.consecutive_stress_spikes >= 5:
                            logger.info("BCI: High stress detected in Beta waves.")
                            self._trigger_stress_protocol()
                            self.consecutive_stress_spikes = 0
                            time.sleep(60) # Cooldown
                    else:
                        self.consecutive_stress_spikes = 0
            except Exception as e:
                logger.error(f"BCI read error: {e}")
            time.sleep(0.5)

    def _trigger_stress_protocol(self):
        """Autonomously reacts to user brainwave stress."""
        if self.orchestrator and self.orchestrator.notifier:
            self.orchestrator.notifier.show_notification(
                "BCI Alert", 
                "High stress detected. Applying soothing protocols..."
            )
            # Send a command to the orchestrator to lower volume, play lofi, or dim screen
            self.orchestrator.run_command("turn on low focus mode and play soothing music")
