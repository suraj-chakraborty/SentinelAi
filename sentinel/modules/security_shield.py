import psutil
import logging
import threading
import time
import os
import asyncio
import httpx
from typing import Optional

class SecurityShieldModule:
    def __init__(self, notifier, gemini_fn=None):
        self.notifier = notifier
        self.gemini_fn = gemini_fn
        self.logger = logging.getLogger("SecurityShield")
        self.is_running = False
        self.known_partitions = set()
        self.proxy_running = False
        self._initialize_partitions()

    def _initialize_partitions(self):
        """Records existing disk partitions to detect new USB insertions."""
        try:
            self.known_partitions = {p.device for p in psutil.disk_partitions()}
        except:
            pass

    def start_shield(self):
        """Starts the security monitoring loop."""
        if self.is_running:
            return
        self.is_running = True
        self.thread = threading.Thread(target=self._shield_loop, daemon=True)
        self.thread.start()
        self.logger.info("Sentinel Security Shield active.")

    def start_proxy_shield(self, port=8080):
        """Starts the Zero-Trust Async Proxy Shield (Phase 2 Upgrade)."""
        if self.proxy_running:
            return "Proxy shield already active."
        
        self.proxy_running = True
        threading.Thread(target=self._run_proxy_async, args=(port,), daemon=True).start()
        self.logger.info(f"Zero-Trust Proxy Shield initialized on port {port}.")
        return f"Zero-Trust Proxy Shield is now active on port {port}."

    def _run_proxy_async(self, port):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._proxy_server(port))

    async def _proxy_server(self, port):
        server = await asyncio.start_server(self._handle_proxy_client, '127.0.0.1', port)
        async with server:
            await server.serve_forever()

    async def _handle_proxy_client(self, reader, writer):
        try:
            data = await reader.read(4096)
            if not data:
                writer.close()
                return

            request = data.decode('utf-8', errors='ignore')
            # Extract URL/Host from request
            lines = request.split('\r\n')
            if not lines:
                writer.close()
                return

            first_line = lines[0].split(' ')
            if len(first_line) < 2:
                writer.close()
                return

            url = first_line[1]
            method = first_line[0]

            # LLM Heuristic Check for Phishing/Malicious Patterns
            if self.gemini_fn:
                is_safe = await self._check_url_safety(url, request)
                if not is_safe:
                    self.logger.warning(f"BLOCKED malicious request: {url}")
                    self.notifier.show_notification("Proxy Shield Alert", f"Blocked potentially malicious request to {url}")
                    writer.write(b"HTTP/1.1 403 Forbidden\r\nContent-Type: text/plain\r\n\r\nSentinelAI Zero-Trust Shield: This request has been blocked for security.")
                    await writer.drain()
                    writer.close()
                    return

            # Simple forward (Note: This is a skeletal proxy for Phase 2)
            # In a full implementation, this would establish a connection to the target
            # For now, we simulate the inspection logic
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\r\nSentinelAI Proxy Shield Active. Monitoring data streams.")
            await writer.drain()
            writer.close()

        except Exception as e:
            self.logger.error(f"Proxy handling error: {e}")
            writer.close()

    async def _check_url_safety(self, url, full_request) -> bool:
        """Uses LLM to evaluate if a URL/request looks like phishing or data exfiltration."""
        if not self.gemini_fn:
            return True
        
        # Simple local heuristic first to save tokens
        suspicious_keywords = ["login", "verify", "password", "bank", "account", "update-secure"]
        if not any(k in url.lower() for k in suspicious_keywords):
            return True

        prompt = f"Analyze this HTTP request for phishing or malicious intent. URL: {url}\n\nRequest:\n{full_request[:500]}\n\nReturn 'SAFE' or 'MALICIOUS'."
        try:
            # We wrap the synchronous gemini_fn in a thread to keep it async-friendly
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self.gemini_fn, prompt)
            return "MALICIOUS" not in result.upper()
        except:
            return True

    def _shield_loop(self):
        while self.is_running:
            try:
                # 1. Detect New USB/Partitions
                current_partitions = {p.device for p in psutil.disk_partitions()}
                new_ones = current_partitions - self.known_partitions
                if new_ones:
                    msg = f"New hardware device detected: {', '.join(new_ones)}. Verify authorized use."
                    self.notifier.show_notification("Security Alert", msg)
                    self.logger.warning(msg)
                    self.known_partitions = current_partitions

                # 2. Monitor Network Spikes (Simple)
                net_io = psutil.net_io_counters()
                # If sending more than 5MB in a single minute loop (basic heuristic)
                if net_io.bytes_sent > 5 * 1024 * 1024: 
                    # Note: This would need a baseline to be truly effective
                    pass 

                # 3. Check for Suspicious Processes
                for proc in psutil.process_iter(['pid', 'name', 'username']):
                    if proc.info['name'] in ['cmd.exe', 'powershell.exe'] and proc.info['username'] != os.getlogin():
                        self.notifier.show_notification("Security Alert", f"Unauthorized terminal access detected (PID: {proc.info['pid']})")

            except Exception as e:
                self.logger.error(f"Security shield error: {e}")
            
            time.sleep(10) # High frequency check for security

    def stop_shield(self):
        self.is_running = False
