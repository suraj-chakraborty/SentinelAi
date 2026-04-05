"""
Weather Plugin — sentinel/plugins/weather/plugin.py
────────────────────────────────────────────────────
Current weather and 3-day forecast using Open-Meteo (free, no API key).

Trigger examples:
  "weather"
  "what's the weather today"
  "will it rain tomorrow"
  "temperature outside"
  "weekly forecast"
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from sentinel.core.plugin_system import PluginBase

logger = logging.getLogger("WeatherPlugin")

# Open-Meteo geocoding (free)
_GEO_URL      = "https://geocoding-api.open-meteo.com/v1/search"
_WEATHER_URL  = "https://api.open-meteo.com/v1/forecast"

# WMO weather interpretation codes → descriptions
_WMO_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Foggy", 48: "Icy fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Heavy drizzle",
    61: "Light rain", 63: "Moderate rain", 65: "Heavy rain",
    71: "Light snow", 73: "Moderate snow", 75: "Heavy snow",
    80: "Showers", 81: "Moderate showers", 82: "Heavy showers",
    95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Severe thunderstorm",
}


class WeatherPlugin(PluginBase):
    """Fetches current weather and forecast from Open-Meteo."""

    # Default location (used when geolocation is unavailable)
    _DEFAULT_CITY = "New York"
    _DEFAULT_LAT  = 40.7128
    _DEFAULT_LON  = -74.0060

    def __init__(self, orchestrator=None):
        super().__init__(orchestrator)
        self._cached_location: Optional[tuple] = None   # (lat, lon, city)

    # ── Plugin interface ──────────────────────────────────────────────────────

    def can_handle(self, command: str) -> bool:
        cmd = command.lower()
        return any(kw in cmd for kw in ("weather", "forecast", "temperature", "rain", "sunny", "cloudy"))

    def handle(self, command: str) -> str:
        city = self._extract_city(command)
        lat, lon, resolved_city = self._resolve_location(city)
        return self._fetch_weather(lat, lon, resolved_city, command)

    def on_load(self):
        logger.info("WeatherPlugin loaded.")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _extract_city(self, command: str) -> Optional[str]:
        """Try to find a city name in the command."""
        patterns = [
            r'\bin\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',                    # "in London"
            r'\bfor\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',                   # "for Paris"
            r'\bat\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',                    # "at Mumbai"
        ]
        for pat in patterns:
            match = re.search(pat, command)
            if match:
                return match.group(1)
        return None

    def _resolve_location(self, city: Optional[str]) -> tuple:
        """Geocode city name or return cached/default location."""
        if city and city.lower() != (self._cached_location or ("", ))[2].lower() if self._cached_location else True:
            try:
                import httpx
                resp = httpx.get(
                    _GEO_URL,
                    params={"name": city, "count": 1, "language": "en"},
                    timeout=5,
                )
                results = resp.json().get("results", [])
                if results:
                    r = results[0]
                    loc = (r["latitude"], r["longitude"], r["name"])
                    self._cached_location = loc
                    return loc
            except Exception as exc:
                logger.warning("Geocoding failed for '%s': %s", city, exc)

        if self._cached_location:
            return self._cached_location

        # Fallback: try IP geolocation
        try:
            import httpx
            geo = httpx.get("https://ipapi.co/json/", timeout=5).json()
            lat = geo.get("latitude", self._DEFAULT_LAT)
            lon = geo.get("longitude", self._DEFAULT_LON)
            city_name = geo.get("city", self._DEFAULT_CITY)
            self._cached_location = (lat, lon, city_name)
            return self._cached_location
        except Exception:
            return self._DEFAULT_LAT, self._DEFAULT_LON, self._DEFAULT_CITY

    def _fetch_weather(self, lat: float, lon: float, city: str, command: str) -> str:
        want_forecast = any(w in command.lower() for w in ("forecast", "week", "tomorrow", "next few days"))
        try:
            import httpx
            params = {
                "latitude": lat,
                "longitude": lon,
                "current": ["temperature_2m", "weathercode", "windspeed_10m", "relative_humidity_2m"],
                "daily": ["temperature_2m_max", "temperature_2m_min", "weathercode", "precipitation_sum"],
                "temperature_unit": "celsius",
                "wind_speed_unit": "kmh",
                "forecast_days": 4,
                "timezone": "auto",
            }
            resp = httpx.get(_WEATHER_URL, params=params, timeout=10)
            data = resp.json()

            curr = data.get("current", {})
            temp    = curr.get("temperature_2m", "N/A")
            wcode   = curr.get("weathercode", 0)
            wind    = curr.get("windspeed_10m", "N/A")
            humidity = curr.get("relative_humidity_2m", "N/A")
            condition = _WMO_CODES.get(wcode, "Unknown")

            result = (
                f"Weather in {city}:\n"
                f"  Condition : {condition}\n"
                f"  Temperature: {temp}°C\n"
                f"  Wind      : {wind} km/h\n"
                f"  Humidity  : {humidity}%"
            )

            if want_forecast:
                daily = data.get("daily", {})
                dates = daily.get("time", [])
                maxes = daily.get("temperature_2m_max", [])
                mins  = daily.get("temperature_2m_min", [])
                wcodes = daily.get("weathercode", [])
                precip = daily.get("precipitation_sum", [])
                lines = ["\n3-Day Forecast:"]
                for i in range(1, min(4, len(dates))):
                    day_cond = _WMO_CODES.get(wcodes[i], "Unknown")
                    lines.append(
                        f"  {dates[i]}: {day_cond}, "
                        f"{mins[i]}–{maxes[i]}°C, "
                        f"{precip[i]}mm rain"
                    )
                result += "\n".join(lines)

            return result

        except Exception as exc:
            logger.error("Weather fetch failed: %s", exc)
            return f"Sorry, I couldn't get the weather right now: {exc}"
