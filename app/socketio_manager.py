"""
Socket.IO Manager for real-time updates.
Handles WebSocket connections for job/collection processing updates and chat.
Fixed connection handling with proper ASGI integration.
"""
import socketio
from typing import Optional, Dict
import asyncio

sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins='*',
    ping_timeout=60,
    ping_interval=25,
    logger=False,
    engineio_logger=False,
    transports=['websocket', 'polling']
)

socket_app = socketio.ASGIApp(
    sio,
    socketio_path='socket.io'
)

# Store connected clients by room
connected_clients: Dict[str, set] = {}

# Store the main event loop reference
_main_loop: Optional[asyncio.AbstractEventLoop] = None


def set_main_loop(loop: asyncio.AbstractEventLoop):
    """Set the main event loop reference for background tasks."""
    global _main_loop
    _main_loop = loop


def get_main_loop() -> Optional[asyncio.AbstractEventLoop]:
    """Get the main event loop reference."""
    return _main_loop

@sio.event
async def connect(sid, environ, auth=None):
    """Handle client connection."""
    global _main_loop
    if _main_loop is None:
        _main_loop = asyncio.get_event_loop()
    print(f"🔌 Client connected: {sid}")
    await sio.emit('connected', {'sid': sid, 'status': 'connected'}, to=sid)


@sio.event
async def disconnect(sid):
    """Handle client disconnection."""
    print(f"🔌 Client disconnected: {sid}")
    for room, clients in list(connected_clients.items()):
        clients.discard(sid)
        if not clients:
            del connected_clients[room]


def _track_join(room: str, sid: str) -> None:
    clients = connected_clients.setdefault(room, set())
    clients.add(sid)


def _track_leave(room: str, sid: str) -> None:
    clients = connected_clients.get(room)
    if not clients:
        return
    clients.discard(sid)
    if not clients:
        connected_clients.pop(room, None)

@sio.event
async def ping(sid):
    """Handle ping from client."""
    await sio.emit('pong', {'timestamp': asyncio.get_event_loop().time()}, to=sid)

@sio.event
async def join_job(sid, data):
    """Join a room to receive updates for a specific job."""
    job_id = data.get('job_id') if isinstance(data, dict) else data
    if job_id:
        room = f"job_{job_id}"
        await sio.enter_room(sid, room)
        _track_join(room, sid)
        print(f"Client {sid} joined job room: {job_id}")
        await sio.emit('joined_job', {'job_id': job_id, 'room': room}, to=sid)


@sio.event
async def leave_job(sid, data):
    """Leave a job room."""
    job_id = data.get('job_id') if isinstance(data, dict) else data
    if job_id:
        room = f"job_{job_id}"
        await sio.leave_room(sid, room)
        _track_leave(room, sid)
        print(f"Client {sid} left job room: {job_id}")


@sio.event
async def join_collection(sid, data):
    """Join a room to receive updates for a specific collection."""
    collection_id = data.get('collection_id') if isinstance(data, dict) else data
    if collection_id:
        room = f"collection_{collection_id}"
        await sio.enter_room(sid, room)
        _track_join(room, sid)
        print(f"Client {sid} joined collection room: {collection_id}")
        await sio.emit('joined_collection', {'collection_id': collection_id, 'room': room}, to=sid)


@sio.event
async def leave_collection(sid, data):
    """Leave a collection room."""
    collection_id = data.get('collection_id') if isinstance(data, dict) else data
    if collection_id:
        room = f"collection_{collection_id}"
        await sio.leave_room(sid, room)
        _track_leave(room, sid)
        print(f"Client {sid} left collection room: {collection_id}")

@sio.event
async def join_chat(sid, data):
    """Join a chat room for a collection."""
    collection_id = data.get('collection_id')
    session_id = data.get('session_id')
    if collection_id:
        room = f"chat_{collection_id}_{session_id}" if session_id else f"chat_{collection_id}"
        await sio.enter_room(sid, room)
        _track_join(room, sid)
        print(f"Client {sid} joined chat room: {room}")
        await sio.emit('joined_chat', {
            'collection_id': collection_id,
            'session_id': session_id,
            'room': room
        }, to=sid)


@sio.event
async def leave_chat(sid, data):
    """Leave a chat room."""
    collection_id = data.get('collection_id')
    session_id = data.get('session_id')
    if collection_id:
        room = f"chat_{collection_id}_{session_id}" if session_id else f"chat_{collection_id}"
        await sio.leave_room(sid, room)
        _track_leave(room, sid)
        print(f"Client {sid} left chat room: {room}")


# Message handlers for chat
@sio.event
async def send_message(sid, data):
    """
    Handle incoming chat message via Socket.IO.
    Process the message and send AI response.
    """
    from app import vector_store
    from app import llm_service
    from app import collection_store
    from datetime import datetime
    import uuid
    
    collection_id = data.get('collection_id')
    session_id = data.get('session_id')
    message = data.get('message')
    top_k = data.get('top_k', 3)
    
    if not collection_id or not message:
        await sio.emit('error', {'message': 'collection_id and message are required'}, to=sid)
        return
    
    # Get collection
    collection = await collection_store.get_collection_by_id(collection_id)
    if not collection:
        await sio.emit('error', {'message': 'Collection not found'}, to=sid)
        return
    
    collection_name = collection_store.vector_collection_name(collection)
    
    # Create or get session
    if not session_id:
        session_id = str(uuid.uuid4())
        session_data = {
            "id": session_id,
            "collection_id": collection_id,
            "title": message[:50] + "..." if len(message) > 50 else message,
            "messages": [],
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        await collection_store.create_chat_session(collection_id, session_data)
        await sio.emit('session_created', {'session_id': session_id}, to=sid)
    elif not await collection_store.get_chat_session(collection_id, session_id):
        await sio.emit('error', {'message': 'Session not found'}, to=sid)
        return
    
    user_message_id = str(uuid.uuid4())
    user_message = {
        "id": user_message_id,
        "role": "user",
        "content": message,
        "sources": [],
        "created_at": datetime.utcnow().isoformat()
    }
    
    # Save user message
    await collection_store.add_message_to_session(collection_id, session_id, user_message)
    
    # Broadcast user message to chat room; make sure the sender is in it
    # (a newly created session has no members yet).
    room = f"chat_{collection_id}_{session_id}"
    await sio.enter_room(sid, room)
    _track_join(room, sid)
    await sio.emit('new_message', {
        'session_id': session_id,
        'message': user_message
    }, room=room)
    
    # Emit typing indicator
    await sio.emit('typing', {'session_id': session_id}, room=room)
    
    try:
        # Get context and sources from vector store without blocking the event loop
        context, sources = await asyncio.to_thread(
            vector_store.get_context_and_sources,
            message,
            top_k,
            collection_name,
        )

        # Generate AI response without blocking the event loop
        answer = await asyncio.to_thread(
            llm_service.generate_chat_response,
            question=message,
            context=context,
        )
        
        # Create assistant message
        assistant_message_id = str(uuid.uuid4())
        assistant_message = {
            "id": assistant_message_id,
            "role": "assistant",
            "content": answer,
            "sources": [
                {"url": s.get("url", ""), "title": s.get("title", ""), "similarity": s.get("similarity", 0.0)}
                for s in sources
            ],
            "created_at": datetime.utcnow().isoformat()
        }
        
        # Save assistant message
        await collection_store.add_message_to_session(collection_id, session_id, assistant_message)
        
        # Broadcast assistant message
        await sio.emit('new_message', {
            'session_id': session_id,
            'message': assistant_message
        }, room=room)
        
    except Exception as e:
        # Send error message
        await sio.emit('chat_error', {
            'session_id': session_id,
            'message': f"Failed to generate response: {str(e)}"
        }, to=sid)
    
    finally:
        # Stop typing indicator
        await sio.emit('typing_stopped', {'session_id': session_id}, room=room)


# Async utility functions to emit events
async def emit_job_update(job_id: str, data: dict):
    """Emit job status update to all clients watching this job."""
    room = f"job_{job_id}"
    print(f"Emitting job_update to room {room}: {data}")
    await sio.emit('job_update', {
        'job_id': job_id,
        **data
    }, room=room)


async def emit_collection_update(collection_id: str, data: dict):
    """Emit collection status update to all clients watching this collection."""
    room = f"collection_{collection_id}"
    await sio.emit('collection_update', {
        'collection_id': collection_id,
        **data
    }, room=room)


async def emit_progress(job_id: str, data: dict):
    """Emit progress update for a job."""
    room = f"job_{job_id}"
    print(f"Emitting progress to room {room}: {data.get('percentage')}%")
    await sio.emit('progress', {
        'job_id': job_id,
        **data
    }, room=room)


# Sync wrappers using the stored main event loop
def emit_job_update_sync(job_id: str, data: dict):
    """
    Synchronous version for background tasks.
    Uses the stored main event loop reference.
    """
    loop = get_main_loop()
    if loop is not None and loop.is_running():
        future = asyncio.run_coroutine_threadsafe(
            emit_job_update(job_id, data),
            loop
        )
        try:
            # Wait for the result with a timeout
            future.result(timeout=5.0)
        except Exception as e:
            print(f"Failed to emit job update: {e}")
    else:
        print(f"Warning: Main event loop not available for job_update emit")


def emit_collection_update_sync(collection_id: str, data: dict):
    """
    Synchronous version for background tasks.
    """
    loop = get_main_loop()
    if loop is not None and loop.is_running():
        future = asyncio.run_coroutine_threadsafe(
            emit_collection_update(collection_id, data),
            loop
        )
        try:
            future.result(timeout=5.0)
        except Exception as e:
            print(f"Failed to emit collection update: {e}")
    else:
        print(f"Warning: Main event loop not available for collection_update emit")


def emit_progress_sync(job_id: str, current: int, total: int, message: str, stage: str = "processing"):
    """
    Emit progress update synchronously.
    
    Args:
        job_id: The job ID
        current: Current progress count
        total: Total count
        message: Progress message
        stage: Current stage (crawling, chunking, embedding, storing)
    """
    percentage = int((current / total) * 100) if total > 0 else 0
    
    data = {
        'current': current,
        'total': total,
        'percentage': percentage,
        'message': message,
        'stage': stage
    }
    
    loop = get_main_loop()
    if loop is not None and loop.is_running():
        future = asyncio.run_coroutine_threadsafe(
            emit_progress(job_id, data),
            loop
        )
        try:
            future.result(timeout=5.0)
        except Exception as e:
            print(f"Failed to emit progress: {e}")
    else:
        print(f"Warning: Main event loop not available for progress emit (job_id: {job_id})")
