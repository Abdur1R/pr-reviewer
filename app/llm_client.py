# # app/llm_client.py
# import httpx


# class LocalLLMClient:
#     def __init__(self, base_url: str, model: str):
#         self.base_url = base_url.rstrip("/")
#         self.model = model

#     async def review(self, prompt: str) -> dict:
#         payload = {
#             "format": "json",
#             "model": self.model,
#             "prompt": prompt,
#             "stream": False,
#             "options": {
#                 "temperature": 0.0,
#                 "num_predict": 800,
#             },
#         }

#         async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
#             resp = await client.post(
#                 f"{self.base_url}/api/generate",
#                 json=payload,
#             )
#             resp.raise_for_status()

#         # LLM MUST return JSON
#         return resp.json()

# llm_client.py
import httpx
from typing import Dict, Any

class LocalLLMClient:
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=90.0)

    async def review(self, prompt: str) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "top_p": 0.9,
                "top_k": 40,
            },
            # Very important for structured output
            "format": "json"
        }

        resp = await self.client.post("/api/generate", json=payload)
        resp.raise_for_status()
        result = resp.json()

        return {
            "model": result.get("model"),
            "response": result.get("response", ""),
            "done": result.get("done", False),
            # ... other fields if needed
        }