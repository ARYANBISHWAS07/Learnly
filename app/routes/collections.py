"""
Collection routes for managing scraped website collections.
"""
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, status
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
import uuid

from app.crawler import crawl_sync
from app import vector_store
from app import job_store
from app import collection_store
from app import learning_node_summary_store
from app import chunk_store
from app.socketio_manager import emit_job_update_sync, emit_collection_update_sync, emit_progress_sync
from app.crawler import handle_chunk
from app.llm_service import generate_learning_path
from app.models.collection import GenerateLearningPathRequest, LearningPathResponse
import asyncio
from app.redis_client import cache_get_json, cache_set_json, cache_delete, publish_event

router = APIRouter(prefix="/collections", tags=["collections"])

# Crawl logs live inside the job item; DynamoDB items are capped at 400KB.
CRAWL_LOG_STORE_LIMIT = 1000


class CreateCollectionRequest(BaseModel):
    name: str
    url: str
    max_pages: int = 0  # 0 = unlimited
    max_depth: int = 10
    user_id: Optional[str] = None


class UpdateCollectionRequest(BaseModel):
    name: Optional[str] = None


def run_collection_scrape_job(job_id: str, collection_id: str, req: CreateCollectionRequest):
    """Background task to scrape and process a collection with real-time progress."""
    try:
        # Update status to crawling
        job_store.update_job_sync(job_id, {"status": "crawling"})
        collection_store.update_collection_sync(collection_id, {"status": "crawling"})
        
        # Emit Socket.IO update
        emit_job_update_sync(job_id, {"status": "crawling", "message": "Starting to crawl website..."})
        emit_collection_update_sync(collection_id, {"status": "crawling"})
        
        # Progress callback for crawling
        def crawl_progress(current: int, total: int, message: str):
            emit_progress_sync(job_id, current, total, message, stage="crawling")
        
        # Run crawler and capture route-level crawl report
        crawl_report = crawl_sync(
            req.url,
            req.max_pages,
            req.max_depth,
            progress_callback=crawl_progress,
            include_report=True,
        )

        if isinstance(crawl_report, dict):
            data = crawl_report.get("pages", [])
            crawled_routes = crawl_report.get("crawled_routes", [])
            not_crawled_routes = crawl_report.get("not_crawled_routes", [])
            crawl_logs = crawl_report.get("crawl_logs", [])
            crawl_summary = crawl_report.get("summary", {})
        else:
            data = crawl_report
            crawled_routes = [page.get("url", "") for page in data if page.get("url")]
            not_crawled_routes = []
            crawl_logs = []
            crawl_summary = {}

        if len(crawl_logs) > CRAWL_LOG_STORE_LIMIT:
            crawl_logs = crawl_logs[-CRAWL_LOG_STORE_LIMIT:]

        routes_crawled_count = len(crawled_routes)
        routes_not_crawled_count = len(not_crawled_routes)
        crawl_logs_count = len(crawl_logs)
        
        # Update status to embedding
        job_store.update_job_sync(job_id, {
            "status": "embedding",
            "pages_crawled": len(data),
            "routes_crawled": crawled_routes,
            "routes_not_crawled": not_crawled_routes,
            "routes_crawled_count": routes_crawled_count,
            "routes_not_crawled_count": routes_not_crawled_count,
            "crawl_logs": crawl_logs,
            "crawl_logs_count": crawl_logs_count,
            "crawl_summary": crawl_summary,
        })
        collection_store.update_collection_sync(collection_id, {
            "status": "embedding",
            "routes_crawled_count": routes_crawled_count,
            "routes_not_crawled_count": routes_not_crawled_count,
        })
        
        emit_job_update_sync(job_id, {
            "status": "embedding",
            "pages_crawled": len(data),
            "routes_crawled_count": routes_crawled_count,
            "routes_not_crawled_count": routes_not_crawled_count,
            "crawl_logs_count": crawl_logs_count,
            "message": (
                f"Crawled {routes_crawled_count} route(s). "
                f"Not crawled: {routes_not_crawled_count}. Creating embeddings..."
            ),
        })
        emit_collection_update_sync(collection_id, {
            "status": "embedding",
            "pages_count": len(data),
            "routes_crawled_count": routes_crawled_count,
            "routes_not_crawled_count": routes_not_crawled_count,
        })
        
        # Progress callback for embedding
        def embed_progress(current: int, total: int, message: str):
            emit_progress_sync(job_id, current, total, message, stage="embedding")
        
        # Store in vector database using collection name with progress and creation of questionnaire
        pages_stored = vector_store.store_pages(data, req.name, progress_callback=embed_progress, chunk_callback=lambda chunk, metadata:
        handle_chunk(chunk, metadata, collection_id))

        # Update job as completed
        job_store.update_job_sync(job_id, {
            "status": "completed",
            "pages_stored": pages_stored,
            "collection_name": req.name,
            "results_count": len(data),
            "routes_crawled": crawled_routes,
            "routes_not_crawled": not_crawled_routes,
            "routes_crawled_count": routes_crawled_count,
            "routes_not_crawled_count": routes_not_crawled_count,
            "crawl_logs": crawl_logs,
            "crawl_logs_count": crawl_logs_count,
            "crawl_summary": crawl_summary,
        })
        
        # Update collection as completed
        collection_store.update_collection_sync(collection_id, {
            "status": "completed",
            "pages_count": pages_stored,
            "routes_crawled_count": routes_crawled_count,
            "routes_not_crawled_count": routes_not_crawled_count,
        })
        
        emit_job_update_sync(job_id, {
            "status": "completed",
            "pages_stored": pages_stored,
            "routes_crawled_count": routes_crawled_count,
            "routes_not_crawled_count": routes_not_crawled_count,
            "crawl_logs_count": crawl_logs_count,
            "percentage": 100,
            "message": f"Successfully stored {pages_stored} chunks. Ready to chat!"
        })
        emit_collection_update_sync(collection_id, {
            "status": "completed",
            "pages_count": pages_stored,
            "routes_crawled_count": routes_crawled_count,
            "routes_not_crawled_count": routes_not_crawled_count,
        })
        
    except Exception as e:
        error_msg = str(e)
        job_store.update_job_sync(job_id, {
            "status": "error",
            "error": error_msg
        })
        collection_store.update_collection_sync(collection_id, {
            "status": "error"
        })
        
        emit_job_update_sync(job_id, {"status": "error", "error": error_msg})
        emit_collection_update_sync(collection_id, {"status": "error", "error": error_msg})


@router.post("/", status_code=status.HTTP_201_CREATED)
async def create_collection(
    req: CreateCollectionRequest,
    background_tasks: BackgroundTasks
):
    """
    Create a new collection and start scraping the website.
    Returns both collection and job IDs for tracking.
    """
    # Check if collection name already exists
    existing = await collection_store.get_collection_by_name(req.name)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Collection '{req.name}' already exists"
        )
    
    collection_id = str(uuid.uuid4())
    job_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    
    # Create collection
    collection_data = {
        "id": collection_id,
        "name": req.name,
        "vector_collection_name": req.name,
        "url": req.url,
        "status": "pending",
        "pages_count": 0,
        "routes_crawled_count": 0,
        "routes_not_crawled_count": 0,
        "job_id": job_id,
        "user_id": req.user_id,
        "created_at": now,
        "updated_at": now
    }
    await collection_store.create_collection(collection_data)
    await cache_delete(
        "collections:all",
        f"collection:name:{req.name}",
        f"collections:user:{req.user_id}",
    )
    await publish_event("collections.updated", {"collection_id": collection_id, "updates": {"status": "pending"}})
    
    # Create associated job
    job_data = {
        "id": job_id,
        "url": req.url,
        "max_pages": req.max_pages,
        "max_depth": req.max_depth,
        "collection_name": req.name,
        "collection_id": collection_id,
        "user_id": req.user_id,
        "status": "pending",
        "pages_stored": 0,
        "routes_crawled": [],
        "routes_not_crawled": [],
        "routes_crawled_count": 0,
        "routes_not_crawled_count": 0,
        "crawl_logs": [],
        "crawl_logs_count": 0,
        "created_at": now,
        "updated_at": now
    }
    await job_store.create_job(job_data)
    
    # Start background task
    background_tasks.add_task(run_collection_scrape_job, job_id, collection_id, req)
    
    return {
        "collection_id": collection_id,
        "job_id": job_id,
        "name": req.name,
        "url": req.url,
        "status": "pending",
        "message": "Collection created and scraping started. Use Socket.IO to receive real-time updates."
    }


@router.get("/")
async def list_collections(user_id: Optional[str] = None):
    """List all collections, optionally filtered by user."""
    cache_key = f"collections:user:{user_id}" if user_id else "collections:all"
    cached = await cache_get_json(cache_key)
    if cached:
        return cached

    collections = await collection_store.get_all_collections(user_id)
    payload = {"collections": collections, "count": len(collections)}
    await cache_set_json(cache_key, payload, ttl_seconds=60)
    return payload


@router.get("/{collection_id}")
async def get_collection(collection_id: str):
    """Get a specific collection with its details."""
    cache_key = f"collection:id:{collection_id}"
    cached = await cache_get_json(cache_key)
    if cached:
        return cached

    collection = await collection_store.get_collection_by_id(collection_id)
    if not collection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Collection not found"
        )
    
    # Get vector store stats
    try:
        stats = await asyncio.to_thread(
            vector_store.get_collection_stats,
            collection_store.vector_collection_name(collection),
        )
        collection["vector_stats"] = stats
    except Exception:
        collection["vector_stats"] = None
    
    await cache_set_json(cache_key, collection, ttl_seconds=60)
    return collection


@router.get("/name/{name}")
async def get_collection_by_name(name: str):
    """Get a collection by its name."""
    cache_key = f"collection:name:{name}"
    cached = await cache_get_json(cache_key)
    if cached:
        return cached

    collection = await collection_store.get_collection_by_name(name)
    if not collection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Collection not found"
        )
    await cache_set_json(cache_key, collection, ttl_seconds=60)
    return collection


@router.patch("/{collection_id}")
async def update_collection(collection_id: str, req: UpdateCollectionRequest):
    """Update collection metadata."""
    updates = {k: v for k, v in req.model_dump().items() if v is not None}
    
    if not updates:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No updates provided"
        )
    
    existing = await collection_store.get_collection_by_id(collection_id)
    if not existing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Collection not found"
        )

    if "name" in updates and updates["name"] != existing.get("name"):
        same_name = await collection_store.get_collection_by_name(updates["name"])
        if same_name and same_name.get("id") != collection_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Collection '{updates['name']}' already exists"
            )
        # Embeddings are stored under the original name; keep pointing at them after a rename.
        updates["vector_collection_name"] = collection_store.vector_collection_name(existing)

    collection = await collection_store.update_collection(collection_id, updates)
    if not collection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Collection not found"
        )
    await cache_delete(
        "collections:all",
        f"collection:id:{collection_id}",
        f"collection:name:{existing.get('name')}",
        f"collection:name:{collection.get('name')}",
        f"collections:user:{collection.get('user_id')}",
    )
    await publish_event("collections.updated", {"collection_id": collection_id, "updates": updates})
    return collection


@router.delete("/{collection_id}")
async def delete_collection(collection_id: str):
    """Delete a collection and its associated data."""
    collection = await collection_store.get_collection_by_id(collection_id)
    if not collection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Collection not found"
        )
    
    # Delete from vector store
    try:
        await asyncio.to_thread(vector_store.delete_collection, collection_store.vector_collection_name(collection))
    except Exception as exc:
        print(f"Failed to delete vector collection for {collection_id}: {exc}")

    # Delete derived data so it doesn't linger (e.g. quiz questions for a deleted collection)
    await asyncio.to_thread(chunk_store.delete_collection_questionnaires, collection_id)
    await learning_node_summary_store.replace_collection_node_summaries(collection_id, [])
    
    # Delete associated job
    if collection.get("job_id"):
        await job_store.delete_job(collection.get("job_id"))
    
    # Delete collection
    await collection_store.delete_collection(collection_id)
    await cache_delete(
        "collections:all",
        f"collection:id:{collection_id}",
        f"collection:name:{collection.get('name')}",
        f"collections:user:{collection.get('user_id')}",
    )
    await publish_event("collections.updated", {"collection_id": collection_id, "deleted": True})
    
    return {"message": "Collection deleted", "collection_id": collection_id}


@router.get("/{collection_id}/job")
async def get_collection_job(collection_id: str):
    """Get the scraping job associated with a collection."""
    collection = await collection_store.get_collection_by_id(collection_id)
    if not collection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Collection not found"
        )
    
    job_id = collection.get("job_id")
    if not job_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No job associated with this collection"
        )
    
    job = await job_store.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )
    
    return job


@router.get("/{collection_id}/routes")
async def get_collection_routes(collection_id: str):
    """Get crawled and not-crawled routes for a collection."""
    collection = await collection_store.get_collection_by_id(collection_id)
    if not collection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Collection not found"
        )

    job_id = collection.get("job_id")
    if not job_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No job associated with this collection"
        )

    job = await job_store.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )

    routes_crawled = job.get("routes_crawled", [])
    routes_not_crawled = job.get("routes_not_crawled", [])

    return {
        "collection_id": collection_id,
        "job_id": job_id,
        "routes_crawled_count": job.get("routes_crawled_count", len(routes_crawled)),
        "routes_not_crawled_count": job.get("routes_not_crawled_count", len(routes_not_crawled)),
        "routes_crawled": routes_crawled,
        "routes_not_crawled": routes_not_crawled,
        "crawl_summary": job.get("crawl_summary", {}),
    }


@router.get("/{collection_id}/crawl-logs")
async def get_collection_crawl_logs(
    collection_id: str,
    limit: int = Query(200, ge=1, le=5000),
    offset: int = Query(0, ge=0),
):
    """Get crawl logs for a collection job (including crawled and skipped endpoints)."""
    collection = await collection_store.get_collection_by_id(collection_id)
    if not collection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Collection not found"
        )

    job_id = collection.get("job_id")
    if not job_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No job associated with this collection"
        )

    job = await job_store.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )

    logs = job.get("crawl_logs", [])
    sliced_logs = logs[offset:offset + limit]

    return {
        "collection_id": collection_id,
        "job_id": job_id,
        "crawl_logs_count": job.get("crawl_logs_count", len(logs)),
        "offset": offset,
        "limit": limit,
        "logs": sliced_logs,
    }


@router.post("/learning-path", response_model=LearningPathResponse)
async def generate_learning_path_from_urls(req: GenerateLearningPathRequest):
    """
    Generate a structured learning path from a list of URLs.
    
    Takes a list of URLs (typically from a sitemap) and uses AI to:
    - Identify main learning topics
    - Group related URLs into concepts/modules
    - Infer prerequisite relationships
    - Organize topics from beginner to advanced
    
    Returns a directed acyclic graph (DAG) representing the learning path.
    """
    if not req.urls:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="URLs list cannot be empty"
        )
    
    if len(req.urls) > 500:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 500 URLs allowed per request"
        )
    
    try:
        result = await asyncio.to_thread(generate_learning_path, req.urls)
        return result
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate learning path: {str(e)}"
        )


@router.post("/{collection_id}/learning-path", response_model=LearningPathResponse)
async def generate_learning_path_for_collection(collection_id: str):
    """
    Generate a structured learning path from a collection's crawled URLs.
    
    Uses the URLs that were crawled during the collection's scraping job
    to generate a learning path.
    """
    collection = await collection_store.get_collection_by_id(collection_id)
    if not collection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Collection not found"
        )
    
    job_id = collection.get("job_id")
    if not job_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No job associated with this collection"
        )
    
    job = await job_store.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )
    
    routes_crawled = job.get("routes_crawled", [])
    if not routes_crawled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No crawled URLs found for this collection"
        )
    
    try:
        result = await asyncio.to_thread(generate_learning_path, routes_crawled)
        await learning_node_summary_store.replace_collection_node_summaries(
            collection_id=collection_id,
            nodes=result.get("nodes", []),
        )
        await collection_store.update_collection(collection_id, {
            "learning_path_graph": result,
            "learning_path_generated_at": datetime.utcnow().isoformat(),
        })
        await cache_delete(
            "collections:all",
            f"collection:id:{collection_id}",
            f"collection:name:{collection.get('name')}",
        )
        await publish_event(
            "collections.updated",
            {
                "collection_id": collection_id,
                "updates": {
                    "learning_path_generated_at": datetime.utcnow().isoformat(),
                },
            },
        )
        return result
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate learning path: {str(e)}"
        )


@router.get("/{collection_id}/learning-path", response_model=LearningPathResponse)
async def get_learning_path_for_collection(collection_id: str):
    """
    Get the latest generated learning path graph for a collection.
    """
    collection = await collection_store.get_collection_by_id(collection_id)
    if not collection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Collection not found"
        )

    learning_path_graph = collection.get("learning_path_graph")
    if not learning_path_graph:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Learning path not generated yet. Call POST /collections/{collection_id}/learning-path first."
        )

    return learning_path_graph


@router.get("/{collection_id}/learning-path/summaries")
async def get_learning_path_summaries_for_collection(collection_id: str):
    """Get persisted learning-node summaries for a collection."""
    collection = await collection_store.get_collection_by_id(collection_id)
    if not collection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Collection not found"
        )

    summaries = await learning_node_summary_store.get_collection_node_summaries(collection_id)
    return {
        "collection_id": collection_id,
        "count": len(summaries),
        "summaries": summaries,
    }


@router.get("/{collection_id}/learning-path/nodes/{node_id}/summary")
async def get_learning_path_node_summary(collection_id: str, node_id: str):
    """
    Get the summary for a single learning-path node.
    Use this endpoint when a user clicks a node in the frontend.
    """
    collection = await collection_store.get_collection_by_id(collection_id)
    if not collection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Collection not found"
        )

    summary_doc = await learning_node_summary_store.get_collection_node_summary(collection_id, node_id)
    if summary_doc:
        return {
            "collection_id": collection_id,
            "node_id": node_id,
            "node_label": summary_doc.get("node_label", ""),
            "summary": summary_doc.get("summary", ""),
            "difficulty": summary_doc.get("difficulty"),
            "module": summary_doc.get("module"),
            "type": summary_doc.get("node_type"),
        }

    # Fallback: try stored graph if summary collection has not been populated yet
    graph = collection.get("learning_path_graph", {})
    for node in graph.get("nodes", []):
        if node.get("id") == node_id and node.get("summary"):
            return {
                "collection_id": collection_id,
                "node_id": node_id,
                "node_label": node.get("label", ""),
                "summary": node.get("summary", ""),
                "difficulty": node.get("difficulty"),
                "module": node.get("module"),
                "type": node.get("type"),
            }

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Summary not found for node '{node_id}'"
    )
