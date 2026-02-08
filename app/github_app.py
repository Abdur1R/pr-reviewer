import time
import jwt
import httpx
from pathlib import Path

from .config import get_settings

GITHUB_API_BASE = "https://api.github.com"

def _load_private_key() -> str:
    settings = get_settings()
    return Path(settings.github_app_private_key_path).read_text()

def generate_app_jwt() -> str:
    settings = get_settings()
    now = int(time.time())
    payload = {"iat": now-60, "exp": now+600, "iss": settings.github_app_id}
    private_key = _load_private_key()
    return jwt.encode(payload, private_key, algorithm="RS256")

async def get_installation_access_token(installation_id: int) -> str:
    jwt_token = generate_app_jwt()
    url = f"{GITHUB_API_BASE}/app/installations/{installation_id}/access_tokens"
    async with httpx.AsyncClient() as client:
        r = await client.post(url, headers={
            "Authorization": f"Bearer {jwt_token}",
            "Accept": "application/vnd.github+json"
        })
        r.raise_for_status()
        return r.json()["token"]

async def github_request(method: str, url: str, token: str, **kwargs):
    headers = kwargs.pop("headers", {})
    headers.update({
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    })
    async with httpx.AsyncClient() as client:
        r = await client.request(method, url, headers=headers, **kwargs)
        r.raise_for_status()
        return r
