from pymongo import MongoClient
from pymongo.collection import Collection

from .config import get_settings

settings = get_settings()

_client: MongoClient = MongoClient(settings.mongodb_uri)
_db = _client[settings.mongodb_db]


def get_users_collection() -> Collection:
    return _db["users"]


def get_installations_collection() -> Collection:
    return _db["installations"]