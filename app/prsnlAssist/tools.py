import os
import httpx
import re
from typing import Optional, Dict

# ─── Query classification ────────────────────────────────────────────────────

REALTIME_PATTERNS = {
    "flight": [
        r"\bflight(s)?\b", r"\bfly\b", r"\bairfare\b", r"\bplane ticket",
        r"\bcheapest.*to\b", r"\bbook.*flight",
    ],
    "weather": [
        r"\bweather\b", r"\btemperature\b", r"\brain(ing)?\b", r"\bforecast\b",
        r"\bhumidity\b", r"\bwind\b",
    ],
    "stock": [
        r"\bstock price\b", r"\bshare price\b", r"\b(NYSE|NASDAQ|BSE|NSE)\b",
        r"\btrading at\b", r"\bmarket cap\b",
    ],
    "news": [
        r"\blatest news\b", r"\bbreaking\b", r"\btoday'?s news\b", r"\brecent(ly)?\b",
        r"\bwhat happened\b",
    ],
    "price": [
        r"\bprice of\b", r"\bhow much (is|does|cost)\b", r"\bcost of\b",
        r"\bcheapest\b", r"\bexpensive\b",
    ],
}


def classify_query(query: str) -> Optional[str]:
    """Return query category if it needs real-time data, else None."""
    q = query.lower()
    for category, patterns in REALTIME_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, q):
                return category
    return None


# ─── SerpApi calls ───────────────────────────────────────────────────────────

SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")
SERPAPI_BASE = "https://serpapi.com/search"


async def _serpapi(params: Dict) -> Dict:
    params["api_key"] = SERPAPI_KEY
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(SERPAPI_BASE, params=params)
        resp.raise_for_status()
        return resp.json()


async def fetch_flight_data(query: str) -> Optional[str]:
    """Parse origin/dest from query and hit Google Flights via SerpApi."""
    # Very simple extraction — LLM will have already parsed the intent
    data = await _serpapi({
        "engine": "google_flights",
        "q": query,
        "hl": "en",
        "currency": "USD",
    })
    best = None
    if "best_flights" in data and data["best_flights"]:
        f = data["best_flights"][0]
        price = f.get("price")
        airline = f.get("flights", [{}])[0].get("airline", "Unknown airline")
        duration = f.get("total_duration")
        best = f"${price} on {airline}"
        if duration:
            best += f", {duration // 60}h {duration % 60}m total"
    elif "other_flights" in data and data["other_flights"]:
        f = data["other_flights"][0]
        price = f.get("price")
        airline = f.get("flights", [{}])[0].get("airline", "Unknown airline")
        best = f"${price} on {airline}"
    return best


async def fetch_web_snippet(query: str) -> Optional[str]:
    """Generic Google search — returns top organic snippet."""
    data = await _serpapi({"engine": "google", "q": query, "num": 3})
    results = data.get("organic_results", [])
    if not results:
        return None
    snippets = [r.get("snippet", "") for r in results[:2] if r.get("snippet")]
    return " | ".join(snippets) if snippets else None


async def get_realtime_context(query: str, category: str) -> Optional[str]:
    """Dispatch to the right fetcher based on category."""
    if not SERPAPI_KEY:
        return None
    try:
        if category == "flight":
            result = await fetch_flight_data(query)
            return f"[Live flight data] {result}" if result else None
        else:
            result = await fetch_web_snippet(query)
            return f"[Live web data] {result}" if result else None
    except Exception as e:
        return None  # Degrade gracefully — LLM will answer from training data
