"""
Chat routes for collection-based conversations.
Uses Socket.IO for real-time messaging and HTTP endpoints for REST API.
"""
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
import uuid
import asyncio
import re

from app import collection_store
from app import vector_store
from app import llm_service
from app.routes.scrape import get_temp_website_session


router = APIRouter(prefix="/chat", tags=["chat"])
WEBSITE_CONTEXT_LIMIT = 12000
CHAT_HISTORY_LIMIT = 8
FOLLOW_UP_PATTERNS = [
    r"\bmore info\b",
    r"\btell me more\b",
    r"\bmore details\b",
    r"\belaborate\b",
    r"\bcontinue\b",
    r"\bexpand\b",
    r"\bcan you explain more\b",
]

class CreateSessionRequest(BaseModel):
    collection_id: str
    title: Optional[str] = None


class SendMessageRequest(BaseModel):
    message: str
    top_k: int = 3


class MessageResponse(BaseModel):
    id: str
    role: str
    content: str
    sources: List[dict] = Field(default_factory=list)
    created_at: str


class SessionResponse(BaseModel):
    id: str
    collection_id: str
    title: Optional[str]
    messages: List[dict] = Field(default_factory=list)
    created_at: str
    updated_at: str


def _is_follow_up_message(message: str) -> bool:
    text = (message or "").strip().lower()
    if not text:
        return False
    return any(re.search(pattern, text) for pattern in FOLLOW_UP_PATTERNS)


def _infer_topic_from_session(messages: List[dict]) -> Optional[str]:
    """
    Infer the active topic from the latest non-follow-up user message.
    """
    for msg in reversed(messages or []):
        if msg.get("role") != "user":
            continue
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        if _is_follow_up_message(content):
            continue
        return content
    return None


def _resolve_retrieval_query(message: str, session_messages: List[dict]) -> str:
    """
    Resolve ambiguous follow-ups to the previous topic for vector retrieval.
    """
    if not _is_follow_up_message(message):
        return message
    topic = _infer_topic_from_session(session_messages)
    if not topic:
        return message
    return f"{message.strip()} about: {topic}"


@router.post("/sessions", status_code=status.HTTP_201_CREATED)
async def create_session(req: CreateSessionRequest):
    """Create a new chat session for a collection."""
    # Verify collection exists
    collection = await collection_store.get_collection_by_id(req.collection_id)
    if not collection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Collection not found"
        )
    
    session_id = str(uuid.uuid4())
    session_data = {
        "id": session_id,
        "collection_id": req.collection_id,
        "title": req.title or "New Chat",
        "messages": [],
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat()
    }
    
    await collection_store.create_chat_session(req.collection_id, session_data)
    
    return session_data


@router.get("/sessions/{collection_id}")
async def get_sessions(collection_id: str):
    """Get all chat sessions for a collection."""
    collection = await collection_store.get_collection_by_id(collection_id)
    if not collection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Collection not found"
        )
    
    sessions = await collection_store.get_all_chat_sessions(collection_id)
    return {"collection_id": collection_id, "sessions": sessions}


@router.get("/session/{collection_id}/{session_id}")
async def get_session(collection_id: str, session_id: str):
    """Get a specific chat session with all messages."""
    session = await collection_store.get_chat_session(collection_id, session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
    return session


@router.delete("/session/{collection_id}/{session_id}")
async def delete_session(collection_id: str, session_id: str):
    """Delete a chat session."""
    deleted = await collection_store.delete_chat_session(collection_id, session_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
    return {"message": "Session deleted", "session_id": session_id}


@router.post("/message/{collection_id}/{session_id}")
async def send_message(
    collection_id: str,
    session_id: str,
    req: SendMessageRequest
):
    """
    Send a message and get AI response (REST API).
    For real-time updates, use Socket.IO instead.
    """
    collection = await collection_store.get_collection_by_id(collection_id)
    if not collection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Collection not found"
        )

    session = await collection_store.get_chat_session(collection_id, session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
    collection_name = collection_store.vector_collection_name(collection)
    prior_messages = session.get("messages", [])
    user_message_id = str(uuid.uuid4())
    user_message = {
        "id": user_message_id,
        "role": "user",
        "content": req.message,
        "sources": [],
        "created_at": datetime.utcnow().isoformat()
    }
    await collection_store.add_message_to_session(collection_id, session_id, user_message)
    
    try:
        retrieval_query = _resolve_retrieval_query(req.message, prior_messages)
        retrieval_top_k = max(req.top_k, 6)

        context, sources = await asyncio.to_thread(
            vector_store.get_context_and_sources,
            retrieval_query,
            retrieval_top_k,
            collection_name,
            0.4,
        )
        if not sources:
            # Retry without threshold so we can still answer when embeddings are sparse.
            context, sources = await asyncio.to_thread(
                vector_store.get_context_and_sources,
                retrieval_query,
                retrieval_top_k,
                collection_name,
            )
        history_for_llm = [
            {"role": m.get("role"), "content": m.get("content", "")}
            for m in prior_messages[-CHAT_HISTORY_LIMIT:]
            if m.get("role") in {"user", "assistant"} and m.get("content")
        ]
        history_for_llm.append({"role": "user", "content": req.message})

        answer = await asyncio.to_thread(
            llm_service.chat_with_history,
            messages=history_for_llm,
            context=context,
            system_prompt=(
                f"{llm_service.DEFAULT_SYSTEM_PROMPT}\n\n"
                "Use conversation history to resolve references like 'more info' or 'that'. "
                "When asked for more information, add new details from context and avoid repeating prior text."
            ),
        )
        

        assistant_message_id = str(uuid.uuid4())
        assistant_message = {
            "id": assistant_message_id,
            "role": "assistant",
            "content": answer,
            "sources": [
                {
                    "url": s.get("url", ""),
                    "title": s.get("title", ""),
                    "similarity": s.get("similarity", 0.0),
                }
                for s in sources
            ],
            "created_at": datetime.utcnow().isoformat()
        }
        await collection_store.add_message_to_session(collection_id, session_id, assistant_message)
        
        # Emit via Socket.IO if available
        try:
            from app.socketio_manager import sio
            room = f"chat_{collection_id}_{session_id}"
            await sio.emit('new_message', {
                'session_id': session_id,
                'message': assistant_message
            }, room=room)
        except Exception:
            pass  # Socket.IO not available
        
        return {
            "user_message": user_message,
            "assistant_message": assistant_message
        }
        
    except ValueError as e:
        # AI service not configured
        return {
            "user_message": user_message,
            "assistant_message": None,
            "error": str(e)
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate response: {str(e)}"
        )


@router.get("/history/{collection_id}/{session_id}")
async def get_chat_history(collection_id: str, session_id: str):
    """Get full chat history for a session."""
    session = await collection_store.get_chat_session(collection_id, session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
    return {
        "session_id": session_id,
        "collection_id": collection_id,
        "messages": session.get("messages", [])
    }



@router.get("/website/history/{session_id}")
async def get_website_chat_history(session_id: str):
    session = get_temp_website_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Website session not found",
        )
    return {
        "session_id": session_id,
        "url": session.get("url"),
        "messages": session.get("messages", []),
    }
