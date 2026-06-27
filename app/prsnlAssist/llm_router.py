import os
import httpx
import asyncio
import logging
from typing import List, Dict, Optional, Tuple
from ..config import get_settings

settings = get_settings()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider registry — order = priority
# All except HuggingFace use the OpenAI-compatible chat completions endpoint
# ---------------------------------------------------------------------------
PROVIDERS = [
    {
        "name": "groq",
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "key_env": settings.groq_api_key,
        "model": "llama-3.3-70b-versatile",
        "openai_compat": True,
    },
    {
        "name": "sambanova",
        "url": "https://api.sambanova.ai/v1/chat/completions",
        "key_env": settings.sambanova_api_key,
        "model": "Meta-Llama-3.3-70B-Instruct",
        "openai_compat": True,
    },
    {
        "name": "openrouter",
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "key_env": settings.openrouter_api_key,
        "model": "meta-llama/llama-3.3-70b-instruct:free",
        "openai_compat": True,
        "extra_headers": {
            "HTTP-Referer": os.getenv("APP_URL", "https://aria.app"),
            "X-Title": "Aria",
        },
    },
    {
        "name": "huggingface",
        "url": "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3/v1/chat/completions",
        "key_env": settings.huggingface_api_key,
        "model": "mistralai/Mistral-7B-Instruct-v0.3",
        "openai_compat": True,  # HF also supports OAI-compat now
    },
]


async def _call_provider(
    provider: Dict,
    messages: List[Dict],
    max_tokens: int,
    timeout: float = 20.0,
) -> str:
    key = os.getenv(provider["key_env"], "")
    if not key:
        raise ValueError(f"No API key for {provider['name']}")

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    headers.update(provider.get("extra_headers", {}))

    payload = {
        "model": provider["model"],
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(provider["url"], json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()


async def call_llm(
    messages: List[Dict],
    max_tokens: int = 512,
    preferred_provider: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Try each provider in priority order.
    Returns (reply_text, provider_name_used).
    Raises RuntimeError if all fail.
    """
    ordered = PROVIDERS[:]
    if preferred_provider:
        ordered = sorted(ordered, key=lambda p: 0 if p["name"] == preferred_provider else 1)

    last_error = None
    for provider in ordered:
        try:
            logger.info(f"Trying LLM provider: {provider['name']}")
            reply = await _call_provider(provider, messages, max_tokens)
            logger.info(f"Success with: {provider['name']}")
            return reply, provider["name"]
        except Exception as e:
            logger.warning(f"Provider {provider['name']} failed: {e}")
            last_error = e
            await asyncio.sleep(0.3)  # small back-off before next try

    raise RuntimeError(f"All LLM providers failed. Last error: {last_error}")


def get_provider_status() -> List[Dict]:
    """Return which providers have keys configured."""
    return [
        {
            "name": p["name"],
            "configured": bool(os.getenv(p["key_env"], "")),
            "model": p["model"],
        }
        for p in PROVIDERS
    ]
