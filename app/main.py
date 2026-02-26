import asyncio
import hashlib
import hmac
import json
import logging
import time
from typing import Any, Dict

import httpx
import jwt
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from app.db import get_installations_collection, get_users_collection
from app.models import InstallationResponse, InstallationSettingsRequest, InstallationUpdateRequest
from .llm_client import LocalLLMClient
from .diff_parser import parse_diff
from .rules import run_rules
import requests

from .config import get_settings
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

# from .db import get_installations_collection, get_users_collection
# from .models import InstallationResponse, InstallationSettingsRequest, InstallationUpdateRequest

router = APIRouter(prefix="/api")

# ==========================
# Settings & Logging
# ==========================


settings = get_settings()

API_URL = settings.hf_api_url
HEADERS = {"Authorization": f"Bearer {settings.hf_token}"}

def query_hf(prompt):

    payload = {
        "model": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 512,
        "temperature": 0.2
    }

    response = requests.post(API_URL, headers=HEADERS, json=payload)

    response.raise_for_status()

    data = response.json()
    return data["choices"][0]["message"]["content"]

def query_groq(prompt: str):

    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {settings.groq_api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 1024
    }

    response = requests.post(url, headers=headers, json=payload)

    response.raise_for_status()

    return response.json()["choices"][0]["message"]["content"]


# print(query_hf("Review this code diff: ..."))

# Initialize local LLM client
llm = LocalLLMClient(
    base_url="http://localhost:11434",
    model="llama3.1:8b",          # ← change here
)

print("llm: ", llm)


logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger("pr-guardian")

# FastAPI app
app = FastAPI(title="PR Guardian AI Webhook")

# CORS CONFIGURATION
origins = [
    settings.cors_orgins,  # from env variable
    "http://localhost:8080",
    "http://127.0.0.1:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================
# Helpers
# ==========================

def verify_github_signature(
    body: bytes,
    signature_header: str | None,
    secret: str,
) -> None:
    """
    Verify X-Hub-Signature-256 from GitHub webhook.

    """
    if not signature_header:
        logger.warning("Missing X-Hub-Signature-256 header - skipping verification in DEV mode")
        return

    try:
        sha_name, signature = signature_header.split("=")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid signature format")

    if sha_name != "sha256":
        raise HTTPException(status_code=400, detail="Unsupported hash algorithm")

    mac = hmac.new(secret.encode("utf-8"), msg=body, digestmod=hashlib.sha256)
    expected = mac.hexdigest()

    if not hmac.compare_digest(expected, signature):
        logger.warning("Invalid webhook signature")
        raise HTTPException(status_code=401, detail="Invalid signature")


def generate_app_jwt() -> str:
    """
    Generate JWT for GitHub App using RS256.
    """
    app_id = settings.github_app_id
    private_key_path = settings.github_private_key_path

    with open(private_key_path, "r", encoding="utf-8") as f:
        private_key = f.read()

    now = int(time.time())
    payload = {
        "iat": now - 60,
        "exp": now + 9 * 60,
        "iss": app_id,
    }

    token = jwt.encode(payload, private_key, algorithm="RS256")
    return token


async def create_installation_token(installation_id: int) -> str:
    """
    Create an installation access token for a given installation_id.
    """
    app_jwt = generate_app_jwt()

    url = f"https://api.github.com/app/installations/{installation_id}/access_tokens"
    headers = {
        "Authorization": f"Bearer {app_jwt}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "PR-Guardian-AI",
    }

    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.post(url, headers=headers)
        logger.info(f"Installation token status: {resp.status_code}")
        resp.raise_for_status()
        data = resp.json()
        return data["token"]


async def fetch_pr_diff(diff_url: str, token: str) -> str:
    """
    Fetch PR diff text.
    """
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3.diff",
        "User-Agent": "PR-Guardian-AI",
    }

    logger.info(f"Fetching diff from: {diff_url}")

    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.get(diff_url, headers=headers)
        logger.info(f"First diff fetch status: {resp.status_code}")

        if resp.status_code in (301, 302, 303, 307, 308):
            redirect_url = resp.headers.get("Location")
            logger.info(f"Redirecting to: {redirect_url}")
            if not redirect_url:
                resp.raise_for_status()

            resp = await client.get(redirect_url, headers=headers)
            logger.info(f"Second diff fetch status: {resp.status_code}")

        resp.raise_for_status()
        return resp.text


async def post_pr_comment(comments_url: str, token: str, body: str) -> None:
    """
    Post a comment on the PR using the issue comments URL.
    """
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "PR-Guardian-AI",
    }

    payload = {"body": body}

    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.post(comments_url, headers=headers, json=payload)
        logger.info(f"Post comment status: {resp.status_code}")
        resp.raise_for_status()


# async def review_diff_with_ai(diff_text: str, pr_title: str, pr_body: str | None) -> str:
#     """
#     Send the diff to OpenAI and get a review comment.
#     """
#     max_chars = 16000
#     short_diff = diff_text[:max_chars]

#     system_prompt = (
#         "You are an expert senior code reviewer. "
#         "Given a Git diff, you will provide a concise review:\n"
#         "- Point out potential bugs, security risks, and performance issues.\n"
#         "- Suggest improvements and best practices.\n"
#         "- If everything looks good, say that explicitly.\n"
#         "- Answer in English and use Markdown with bullet points."
#     )

#     user_prompt = f"""
# Pull Request Title: {pr_title}

# Pull Request Description:
# {pr_body or "(no description)"}

# Git Diff:
# {short_diff}
# """

#     def _call_openai() -> str:
#         resp = openai_client.chat.completions.create(
#             model="gpt-4.1-mini",
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": user_prompt},
#             ],
#             temperature=0.2,
#             max_tokens=700,
#         )
#         return resp.choices[0].message.content.strip()

#     review_text = await asyncio.to_thread(_call_openai)
#     return review_text

async def post_inline_comment(
    owner: str,
    repo: str,
    pull_number: int,
    token: str,
    commit_id: str,
    path: str,
    line: int,
    body: str,
):
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pull_number}/comments"

    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "PR-Guardian-AI",
    }
    print("commit_id, line",commit_id,line)
    payload = {
        "body": body,
        "commit_id": commit_id,
        "path": path,
        "line": line,
        "side": "RIGHT",
    }

    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.post(url, headers=headers, json=payload)
        print("response from posting inline comment: ", resp.text)
        resp.raise_for_status()

def generate_jwt():

    with open(settings.github_private_key_path, "r") as f:
        private_key = f.read()

    payload = {
        "iat": int(time.time()),
        "exp": int(time.time()) + 600,
        "iss": settings.github_app_id,
    }

    encoded_jwt = jwt.encode(
        payload,
        private_key,
        algorithm="RS256"
    )

    return encoded_jwt


def generate_installation_token(installation_id: int):

    jwt_token = generate_jwt()
    
    print("Generated JWT: ", jwt_token)

    url = f"https://api.github.com/app/installations/{installation_id}/access_tokens"
    
    print("Token URL: ", url)
    headers = {
        "Authorization": f"Bearer {jwt_token}",
        "Accept": "application/vnd.github+json"
    }

    response = requests.post(url, headers=headers)

    response.raise_for_status()

    data = response.json()
    
    print('data: ', data)

    return data["token"]


# # ==========================
# # Routes
# # ==========================

@app.get("/")
async def root():
    return {"status": "ok", "app": "PR Guardian AI"}


@app.post("/webhook")
async def webhook(
    request: Request,
    x_github_event: str = Header(None, alias="X-GitHub-Event"),
    x_hub_signature_256: str = Header(None, alias="X-Hub-Signature-256"),
):
    raw_body = await request.body()

    verify_github_signature(raw_body, x_hub_signature_256, settings.github_webhook_secret)

    try:
        payload: Dict[str, Any] = json.loads(raw_body.decode("utf-8"))
    except json.JSONDecodeError:
        logger.exception("Invalid JSON payload")
        raise HTTPException(status_code=400, detail="Invalid JSON")

    logger.info("=" * 30)
    logger.info(">>> Webhook received")
    logger.info(f">>> Event: {x_github_event}")

    # 1) Ping
    if x_github_event == "ping":
        return JSONResponse({"msg": "pong"})

    # 2) Installation
    if x_github_event == "installation":
        logger.info(f"Installation payload: {payload.get('action')}")
        return JSONResponse({"msg": "installation event ok"})

    # 3) Pull Request
    if x_github_event == "pull_request":
        action = payload.get("action")
        logger.info(f">>> Action: {action}")

        if action not in {"opened", "synchronize", "reopened"}:
            logger.info("Ignoring PR action: %s", action)
            return JSONResponse({"msg": f"ignored action {action}"})

        pr = payload.get("pull_request", {})
        comments_url = pr.get("comments_url")
        diff_url = pr.get("diff_url")
        pr_title = pr.get("title", "")
        pr_body = pr.get("body", "")

        logger.info(f">>> PR comments_url: {comments_url}")
        logger.info(f">>> diff_url: {diff_url}")

        installation = payload.get("installation") or {}
        installation_id = installation.get("id")
        if not installation_id:
            logger.error("No installation id in payload")
            raise HTTPException(status_code=400, detail="Missing installation id")

        logger.info(f">>> Installation ID: {installation_id}")
        logger.info(">>> Creating installation access token...")

        try:
            gh_token = await create_installation_token(installation_id)
            logger.info(">>> Installation token created.")
        except Exception as e:
            logger.exception("Failed to create installation token")
            raise HTTPException(status_code=500, detail="Failed to create installation token") from e

        # Fetch diff
        try:
            diff_text = await fetch_pr_diff(diff_url, gh_token)
        except Exception as e:
            logger.exception("Failed to fetch PR diff")
            raise HTTPException(status_code=500, detail="Failed to fetch PR diff") from e

        # Review with AI
        try:
            repo_info = payload["repository"]
            owner = repo_info["owner"]["login"]
            repo = repo_info["name"]
            pull_number = pr["number"]
            commit_id = pr["head"]["sha"]

            # 1) Parse diff
            changes = parse_diff(diff_text)

            # 2) Run static rules
            rule_comments = run_rules(changes)

            # 3) Build LLM prompt
            # prompt = {
            #     "task": "code_review",
            #     "format": {
            #         "comments": [
            #             {"path": "string", "line": 0, "comment": "string"}
            #         ],
            #         "summary": "string"
            #     },
            #     "rules": [
            #         "Return ONLY JSON",
            #         "No explanations",
            #         "No markdown",
            #         "No text outside JSON",
            #         "Start with { and end with }"
            #     ],
            #     "changes": changes[:10],
            # }
            
            instruction = """
            You are a senior software engineer performing pull request reviews.

            You MUST output ONLY valid JSON.

            Output format:
            {
            "comments": [
                {"path": "...", "line": 0, "comment": "..."}
            ],
            "summary": "..."
            }

            Rules:
            - Do not explain anything
            - Do not describe the input
            - Do not repeat instructions
            - Only comment if something is wrong or improvable
            - If nothing is wrong return:
            {"comments": [], "summary": "No issues found"}
            """

            example = """
            Example:

            Input:
            [{"path":"app.py","line":10,"content":"password='123'"}]

            Output:
            {
            "comments":[
            {"path":"app.py","line":10,"comment":"Hardcoded password detected"}
            ],
            "summary":"Security issue found"
            }

            Now do the real review:
            """

            payload = instruction + example + json.dumps(changes[:10])


            logger.info(">>> Calling LLM with %d changes", len(changes[:10]))
            
            logger.info("llm payload:\n%s \n%s", payload,json.dumps(payload, indent=2))
            
            # 4) Ask local LLM
            # llm_result = await llm.review(json.dumps(payload))
            
            installations = get_installations_collection()
            doc = installations.find_one({"installationId": installation_id}, {"_id": 0})
            if not doc:
                logger.warning(f"No installation config found for {installation_id}")
                selected_ai = "huggingface"  # default fallback
            else:
                selected_ai = doc.get("reviewer", {}).get("selectedAI", "huggingface")

            logger.info(f"Selected AI for review: {selected_ai}")
            
            if(selected_ai == "huggingface"):
                # Ask llm via Hugging Face API
                llm_result=query_hf(payload)
            # elif(selected_ai == "groq"):
            else:                
                # Ask llm via Groq API
                llm_result=query_groq(payload)

            logger.info(">>> LLM response: %s", llm_result)

            raw = llm_result
            
            logger.info(">>> LLM RAW TEXT:\n%s", raw)

            try:
                llm_data = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("LLM returned non-JSON, using fallback.")
                llm_data = {
                    "comments": [],
                    "summary": raw[:500]  # show text as summary
                }

            # 5) Merge comments
            all_comments = rule_comments + llm_data.get("comments", [])

            # 6) Post inline comments
            for c in all_comments:
                await post_inline_comment(
                    owner=owner,
                    repo=repo,
                    pull_number=pull_number,
                    token=gh_token,
                    commit_id=commit_id,
                    path=c["path"],
                    line=c["line"],
                    body=c["comment"],
                )

            # 7) Post summary
            summary = llm_data.get("summary", "Automated review completed.")
            await post_pr_comment(
            comments_url,
            gh_token,
            f"🤖 **PR Guardian AI Summary**\n\n{summary}",
            )

        except Exception as e:
            logger.exception("Failed to post PR comment")
            raise HTTPException(status_code=500, detail="Failed to post PR comment") from e

        return JSONResponse({"msg": "AI review posted"})

    logger.info(f"Unhandled event: {x_github_event}")
    return JSONResponse({"msg": f"unhandled event {x_github_event}"})


@app.post("/installations/{installation_id}/settings")
async def save_installation_settings(
    installation_id: int,
    body: InstallationSettingsRequest,
):
    users = get_users_collection()
    installations = get_installations_collection()
    now = datetime.now(timezone.utc)

    # Upsert user
    users.update_one(
        {"userId": body.user_id,
        "installationId": installation_id,
         },
        {"$set": {"userId": body.user_id,
                "installationId": installation_id,
                  "updatedAt": now}, "$setOnInsert": {"createdAt": now}},
        upsert=True,
    )

    # Build installation document
    doc = {
        "installationId": installation_id,
        "userId": body.user_id,
        "selectedRepos": [r.model_dump(by_alias=True) for r in body.selected_repos],
        "reviewer": body.reviewer.model_dump(by_alias=True),
        "updatedAt": now,
    }

    installations.update_one(
        {"installationId": installation_id},
        {"$set": doc, "$setOnInsert": {"createdAt": now}},
        upsert=True,
    )

    return {"success": True, "installationId": installation_id}


@app.put("/installations/{installation_id}/settings")
async def update_installation_settings(
    installation_id: int,
    body: InstallationUpdateRequest,
):
    installations = get_installations_collection()

    existing = installations.find_one({"installationId": installation_id})
    if not existing:
        raise HTTPException(status_code=404, detail="Installation not found")

    now = datetime.now(timezone.utc)
    update_fields: dict = {"updatedAt": now}

    if body.selected_repos is not None:
        update_fields["selectedRepos"] = [r.model_dump(by_alias=True) for r in body.selected_repos]

    if body.reviewer is not None:
        update_fields["reviewer"] = body.reviewer.model_dump(by_alias=True)

    installations.update_one(
        {"installationId": installation_id},
        {"$set": update_fields},
    )

    updated = installations.find_one({"installationId": installation_id}, {"_id": 0})
    return InstallationResponse(**updated)


@app.get("/installations/{installation_id}")
async def get_installation(installation_id: int):
    installations = get_installations_collection()
    doc = installations.find_one({"installationId": installation_id}, {"_id": 0})

    if not doc:
        return {"exists": False}

    return InstallationResponse(**doc)


@app.get("/users/{user_id}/installations")
async def get_user_installations(user_id: str):
    installations = get_installations_collection()
    docs = list(installations.find({"userId": user_id}, {"_id": 0}))

    return [InstallationResponse(**d) for d in docs]

# get all the selected repos for an installation
@app.get("/api/repos/{installation_id}")
async def get_repos(installation_id: int):
    try:
        token = generate_installation_token(installation_id)
        print("Generated token: ", token)

        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json"
        }

        response = requests.get(
            "https://api.github.com/installation/repositories",
            headers=headers
        )
    except Exception as e:
        logger.exception("Failed to fetch repositories")
        return {"error": "Failed to fetch repositories"}
    return response.json()

# For userID
@app.get("/installations/{installation_id}/info")
async def get_installation_info(installation_id: int):

    jwt_token = generate_app_jwt()  # your existing JWT generator

    headers = {
        "Authorization": f"Bearer {jwt_token}",
        "Accept": "application/vnd.github+json"
    }

    response = requests.get(
        f"https://api.github.com/app/installations/{installation_id}",
        headers=headers
    )

    response.raise_for_status()

    return response.json().get("account", {}).get("login", "unknown")