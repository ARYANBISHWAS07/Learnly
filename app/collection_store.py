"""
Collection storage service using DynamoDB.
Manages collections and chat sessions.
"""
import asyncio
import os
from datetime import datetime
from typing import List, Optional
import uuid

from app.database import (
    append_to_list,
    delete_item,
    get_by_id,
    get_scan_attr,
    put_item,
    scan_table,
    update_item,
)
from app.redis_client import cache_delete_sync, publish_event_sync

# Chat sessions share the quiz_sessions table with quiz sessions; entity_type tells them apart.
CHAT_SESSION_ENTITY = "chat_session"


def _is_chat_session(item: Optional[dict]) -> bool:
    # Sessions created before entity_type existed have no type and are chat sessions.
    return bool(item) and item.get("entity_type", CHAT_SESSION_ENTITY) == CHAT_SESSION_ENTITY


def vector_collection_name(collection: dict) -> Optional[str]:
    """Qdrant collection holding this collection's embeddings (fixed at creation, survives renames)."""
    return collection.get("vector_collection_name") or collection.get("name")


async def create_collection(collection_data: dict) -> dict:
    collection_id = collection_data.get("id") or collection_data.get("PK") or str(uuid.uuid4())
    document = {
        "PK": collection_id,
        "id": collection_id,
        **collection_data,
    }
    await asyncio.to_thread(put_item, "collections", document)
    return document


async def get_collection_by_id(collection_id: str) -> Optional[dict]:
    return await asyncio.to_thread(get_by_id, "collections", collection_id)


async def get_collection_by_name(name: str) -> Optional[dict]:
    items = await asyncio.to_thread(
        scan_table,
        "collections",
        get_scan_attr("name").eq(name),
        1,
    )
    return items[0] if items else None


async def update_collection(collection_id: str, updates: dict) -> Optional[dict]:
    updates["updated_at"] = datetime.utcnow().isoformat()
    return await asyncio.to_thread(update_item, "collections", {"PK": collection_id}, updates)


def update_collection_sync(collection_id: str, updates: dict) -> bool:
    updates["updated_at"] = datetime.utcnow().isoformat()
    try:
        collection = update_item("collections", {"PK": collection_id}, updates)
        keys = ["collections:all", f"collection:id:{collection_id}"]
        if collection and collection.get("name"):
            keys.append(f"collection:name:{collection['name']}")
        if collection and collection.get("user_id"):
            keys.append(f"collections:user:{collection['user_id']}")
        cache_delete_sync(*keys)
        publish_event_sync("collections.updated", {"collection_id": collection_id, "updates": updates})
        return collection is not None
    except Exception as exc:
        print(f"DynamoDB sync update error: {exc}")
        return False


async def get_all_collections(user_id: Optional[str] = None, limit: int = 100) -> List[dict]:
    global_view = os.getenv("COLLECTIONS_GLOBAL", "true").strip().lower() in {"1", "true", "yes"}
    filter_expression = None
    if user_id and not global_view:
        filter_expression = get_scan_attr("user_id").eq(user_id)

    collections = await asyncio.to_thread(scan_table, "collections", filter_expression, limit)
    collections.sort(key=lambda item: item.get("created_at", ""), reverse=True)
    return collections[:limit]


async def delete_collection(collection_id: str) -> bool:
    return await asyncio.to_thread(delete_item, "collections", {"PK": collection_id})


async def create_chat_session(collection_id: str, session_data: dict) -> dict:
    session_id = session_data.get("id") or session_data.get("PK") or str(uuid.uuid4())
    item = {
        "PK": session_id,
        "id": session_id,
        "entity_type": CHAT_SESSION_ENTITY,
        **session_data,
    }
    await asyncio.to_thread(put_item, "quiz_sessions", item)
    return item


async def get_chat_session(collection_id: str, session_id: str) -> Optional[dict]:
    session = await asyncio.to_thread(get_by_id, "quiz_sessions", session_id)
    if _is_chat_session(session) and session.get("collection_id") == collection_id:
        return session
    return None


async def get_all_chat_sessions(collection_id: str) -> List[dict]:
    sessions = await asyncio.to_thread(
        scan_table,
        "quiz_sessions",
        get_scan_attr("collection_id").eq(collection_id),
    )
    sessions = [session for session in sessions if _is_chat_session(session)]
    sessions.sort(key=lambda item: item.get("created_at", ""))
    return sessions


async def add_message_to_session(collection_id: str, session_id: str, message_data: dict) -> bool:
    updated = await asyncio.to_thread(
        append_to_list,
        "quiz_sessions",
        {"PK": session_id},
        "messages",
        message_data,
        {"updated_at": datetime.utcnow().isoformat()},
    )
    return updated is not None and updated.get("collection_id") == collection_id


async def update_session_title(collection_id: str, session_id: str, title: str) -> bool:
    updated = await asyncio.to_thread(
        update_item,
        "quiz_sessions",
        {"PK": session_id},
        {"title": title, "updated_at": datetime.utcnow().isoformat()},
    )
    return updated is not None and updated.get("collection_id") == collection_id


async def delete_chat_session(collection_id: str, session_id: str) -> bool:
    session = await get_chat_session(collection_id, session_id)
    if not session:
        return False
    return await asyncio.to_thread(delete_item, "quiz_sessions", {"PK": session_id})
