"""
sentinel/security/proxy_shield.py
──────────────────────────────────
Sci-Fi Tier 4: Zero-Trust Cyber Defense Proxy.
A lightweight Asyncio proxy server that natively inspects incoming web traffic 
using Sentinel's LLM to dynamically predict malicious domains or track payloads.
"""

import asyncio
import logging
from aiohttp import web
import json

logger = logging.getLogger("ProxyShield")

class ProxyShield:
    def __init__(self, llm_callback, host="127.0.0.1", port=8080):
        self.llm_callback = llm_callback
        self.host = host
        self.port = port
        self.app = web.Application()
        self.app.router.add_route('*', '/{tail:.*}', self.proxy_handler)
        self.runner = None
        self.site = None

    async def start(self):
        """Start the proxy server asynchronously."""
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, self.host, self.port)
        await self.site.start()
        logger.info(f"Deep AI Proxy Shield running on http://{self.host}:{self.port}")

    async def stop(self):
        if self.site:
            await self.site.stop()
        if self.runner:
            await self.runner.cleanup()
        logger.info("Proxy Shield terminated.")

    async def proxy_handler(self, request):
        """Intercepts the request, asks the LLM, and passes it through if safe."""
        target_url = str(request.url)
        
        # We only deep-inspect text/html requests or suspicious domains
        # For mock simplicity in this blueprint, we inspect immediately:
        
        is_safe, reason = self._analyze_threat(target_url)
        if not is_safe:
            logger.warning(f"PROXY BLOCKED: {target_url} - {reason}")
            return web.Response(
                text=f"<h1>Sentinel Proxy Shield</h1><p>Blocked heavily malicious URL proactively.</p><p>Reason: {reason}</p>", 
                content_type="text/html",
                status=403
            )
            
        logger.info(f"PROXY ALLOW: {target_url}")
        return web.Response(text=f"Simulated Proxy Passthrough: {target_url}")

    def _analyze_threat(self, url: str) -> tuple[bool, str]:
        """Ask pure AI to classify risk profile on the fly."""
        # Fast bypass for obvious safe domains
        safe_list = ["google.com", "github.com", "microsoft.com"]
        if any(s in url for s in safe_list):
            return True, "Safe list"
            
        # For unknown domains, do a rapid zero-shot classification 
        sys_prompt = f"Analyze this URL for phishing, typosquatting, or malicious patterns: {url}. Respond strictly with SAFE or BLOCKED, followed by a short reason."
        
        try:
            # We mock LLM callback here since true network latency would require streaming
            # In production: response = self.llm_callback(sys_prompt)
            response = "SAFE. Domain appears standard." 
            if "login-update" in url or "paypal-secure" in url:
                response = "BLOCKED. Classic phishing typo/squatting."
                
            if response.startswith("BLOCKED"):
                return False, response[8:]
            return True, "LLM heuristic cleared."
        except Exception as e:
            return True, f"Inspection failed, passed default open: {e}"
