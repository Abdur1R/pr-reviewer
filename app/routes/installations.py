from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from ..db import get_installations_collection, get_users_collection
from ..models import InstallationResponse, InstallationSettingsRequest, InstallationUpdateRequest

router = APIRouter(prefix="/api")


@router.post("/installations/{installation_id}/settings")
async def save_installation_settings(
    installation_id: int,
    body: InstallationSettingsRequest,
):
    users = get_users_collection()
    installations = get_installations_collection()
    now = datetime.now(timezone.utc)

    # Upsert user
    users.update_one(
        {"userId": body.user_id},
        {"$set": {"userId": body.user_id, "updatedAt": now}, "$setOnInsert": {"createdAt": now}},
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


@router.put("/installations/{installation_id}/settings")
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


@router.get("/installations/{installation_id}")
async def get_installation(installation_id: int):
    installations = get_installations_collection()
    doc = installations.find_one({"installationId": installation_id}, {"_id": 0})

    if not doc:
        return {"exists": False}

    return InstallationResponse(**doc)


@router.get("/users/{user_id}/installations")
async def get_user_installations(user_id: str):
    installations = get_installations_collection()
    docs = list(installations.find({"userId": user_id}, {"_id": 0}))

    return [InstallationResponse(**d) for d in docs]
