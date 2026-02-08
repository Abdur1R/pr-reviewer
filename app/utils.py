import hmac, hashlib
from .config import get_settings

def verify_github_signature(header: str, body: bytes) -> bool:
    if not header or not header.startswith("sha256="):
        return False
    signature = header.split("=")[1]
    secret = get_settings().github_webhook_secret.encode()
    digest = hmac.new(secret, body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(digest, signature)
