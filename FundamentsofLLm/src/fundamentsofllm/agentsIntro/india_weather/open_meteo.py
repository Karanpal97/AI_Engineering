import requests
import json
from pydantic import BaseModel, Field
from typing import Optional


# ── Pydantic response models ────────────────────────────────────────────────

class WeatherResult(BaseModel):
    """Successful weather result for an Indian city"""
    city: str = Field(..., description="The city name as provided by the user")
    temperature_c: float = Field(..., description="Current temperature in Celsius")
    feels_like_c: float = Field(..., description="Feels-like temperature in Celsius")
    humidity_percent: int = Field(..., description="Relative humidity as a percentage")
    wind_speed_kmh: float = Field(..., description="Wind speed in kilometres per hour")
    weather_condition: str = Field(..., description="Human-readable weather condition description")
    is_day: bool = Field(..., description="True if it is currently daytime at the location")


class WeatherError(BaseModel):
    """Error result returned when weather data cannot be fetched"""
    error: str = Field(..., description="The error message explaining what went wrong")


# ── WMO weather-code → description mapping ──────────────────────────────────

WMO_CODES: dict[int, str] = {
    0:  "Clear sky",
    1:  "Mainly clear",
    2:  "Partly cloudy",
    3:  "Overcast",
    45: "Foggy",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snowfall",
    73: "Moderate snowfall",
    75: "Heavy snowfall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


def _geocode_indian_city(city_name: str) -> tuple[float, float]:
    """
    Use the Open-Meteo geocoding API to resolve an Indian city name to
    (latitude, longitude).  Raises ValueError if not found.
    """
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {
        "name": city_name,
        "count": 5,           # fetch a few candidates so we can filter India
        "language": "en",
        "format": "json",
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    results = data.get("results", [])
    if not results:
        raise ValueError(f"City '{city_name}' not found. Please check the spelling.")

    # Prefer results inside India (country_code == "IN")
    for r in results:
        if r.get("country_code", "").upper() == "IN":
            return r["latitude"], r["longitude"]

    # Fall back to the first result if none are tagged India
    # (handles union territories / smaller places with missing codes)
    first = results[0]
    return first["latitude"], first["longitude"]


def get_weather(city_name: str) -> str:
    """
    Fetch the current weather for an Indian city.

    Parameters
    ----------
    city_name : str
        The name of the Indian city (Hindi or English spelling accepted).

    Returns
    -------
    str
        JSON-serialised WeatherResult or WeatherError.
    """
    try:
        lat, lon = _geocode_indian_city(city_name)

        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": [
                "temperature_2m",
                "apparent_temperature",
                "relative_humidity_2m",
                "wind_speed_10m",
                "weather_code",
                "is_day",
            ],
            "timezone": "Asia/Kolkata",   # IST for all Indian cities
            "forecast_days": 1,
        }
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        current = data["current"]
        wmo_code = int(current["weather_code"])
        condition = WMO_CODES.get(wmo_code, f"Unknown (code {wmo_code})")

        result = WeatherResult(
            city=city_name,
            temperature_c=round(current["temperature_2m"], 1),
            feels_like_c=round(current["apparent_temperature"], 1),
            humidity_percent=int(current["relative_humidity_2m"]),
            wind_speed_kmh=round(current["wind_speed_10m"], 1),
            weather_condition=condition,
            is_day=bool(current["is_day"]),
        )
        return result.model_dump_json()

    except Exception as exc:
        error = WeatherError(error=str(exc))
        return error.model_dump_json()


# ── Quick smoke-test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    for city in ["Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Kolkata"]:
        print(f"\n{city}:")
        print(get_weather(city))
