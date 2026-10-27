from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from contextlib import asynccontextmanager
import asyncio

from app.crawler import crawl_sync
from app import vector_store
from app.database import connect_to_dynamodb, close_dynamodb_connection
from app.redis_client import connect_redis, close_redis
from app.routes import users_router, chat_router, collections_router, quizzes_router, scrape_router
from app.socketio_manager import socket_app, sio, set_main_loop
# from extension.main import app as extension_app

import uuid


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Connect to DynamoDB
    await connect_to_dynamodb()
    await connect_redis()
    
    # Capture the main event loop for background tasks
    loop = asyncio.get_event_loop()
    set_main_loop(loop)
    print(f"Main event loop captured for Socket.IO emissions")
    
    yield
    
    # Shutdown: Close connections
    await close_dynamodb_connection()
    await close_redis()


app = FastAPI(
    title="Website Knowledge Base API",
    description="Crawl websites and turn them into searchable knowledge bases",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(users_router)
app.include_router(chat_router)
app.include_router(collections_router)
app.include_router(quizzes_router)
app.include_router(scrape_router)


# Mount Socket.IO app at /socket.io path
app.mount("/", socket_app)
