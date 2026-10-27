"""
Vector Store using LangChain with Qdrant.
Handles document chunking, embedding, and semantic search.
"""
import hashlib
import os
import re
import uuid
from typing import Any, Callable, Dict, List, Optional
from functools import lru_cache

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app import llm_service

load_dotenv()

# Qdrant Configuration
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_PORT = os.getenv("QDRANT_PORT", "6333")
if not QDRANT_URL:
    if QDRANT_HOST:
        QDRANT_URL = f"http://{QDRANT_HOST}:{QDRANT_PORT}"
    else:
        QDRANT_URL = "http://localhost:6333"

QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_DEFAULT_COLLECTION = os.getenv("QDRANT_COLLECTION", "scraped_data")
QDRANT_TIMEOUT_SECONDS = float(os.getenv("QDRANT_TIMEOUT_SECONDS", "60"))

# Lazy initialization of Qdrant client to defer connection until needed
_qdrant_client = None

def get_qdrant_client():
    """Get or initialize the Qdrant client (lazy initialization)."""
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=QDRANT_TIMEOUT_SECONDS,
            check_compatibility=False,
        )
    return _qdrant_client

# Alias for backward compatibility
def qdrant_client():
    return get_qdrant_client()

# LangChain embeddings using HuggingFace
embeddings_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

EMBEDDING_DIMENSION = len(embeddings_model.embed_query("dimension probe"))

# LangChain text splitter for chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=10000,  # ~10KB chunks
    chunk_overlap=500,
    length_function=len,
    separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
    is_separator_regex=False,
)


@lru_cache(maxsize=1024)
def _safe_collection_name(collection_name: Optional[str]) -> str:
    """Map app collection names to Qdrant-compatible names deterministically."""
    raw_name = (collection_name or QDRANT_DEFAULT_COLLECTION).strip()
    if not raw_name:
        raw_name = QDRANT_DEFAULT_COLLECTION

    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", raw_name)
    sanitized = sanitized[:255].strip("_")
    if sanitized == raw_name and len(sanitized) <= 255:
        return sanitized

    prefix = sanitized[:100] if sanitized else "collection"
    digest = hashlib.sha256(raw_name.encode()).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _collection_exists_in_qdrant(collection_name: str) -> bool:
    """Version-safe check for collection existence."""
    try:
        return get_qdrant_client().collection_exists(collection_name=collection_name)
    except AttributeError:
        try:
            get_qdrant_client().get_collection(collection_name=collection_name)
            return True
        except Exception:
            return False
    except Exception:
        return False


def _ensure_collection(collection_name: Optional[str] = None) -> str:
    name = _safe_collection_name(collection_name)
    if not _collection_exists_in_qdrant(name):
        get_qdrant_client().create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=EMBEDDING_DIMENSION,
                distance=Distance.COSINE,
            ),
        )
    return name


# Ensure the default collection exists at startup, but do not crash if Qdrant
# is still starting or unreachable.
try:
    _ensure_collection()
except Exception as exc:
    print(f"Warning: Qdrant not reachable at startup ({QDRANT_URL}): {exc}")


def chunk_text(text: str) -> List[str]:
    """
    Split text into chunks using LangChain's RecursiveCharacterTextSplitter.
    """
    if not text or not text.strip():
        return []

    chunks = text_splitter.split_text(text)
    return [chunk for chunk in chunks if chunk.strip()]


def store_pages(
    pages: List[Dict[str, Any]],
    collection_name: str = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    chunk_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> int:
    """
    Store crawled pages as embeddings in the vector database.
    Uses LangChain for chunking and embedding.
    """
    if not pages:
        return 0

    qdrant_collection = _ensure_collection(collection_name)

    # Filter out duplicate URLs and empty content
    seen_urls = set()
    unique_pages = []
    for page in pages:
        if page["url"] not in seen_urls and page.get("content"):
            seen_urls.add(page["url"])
            unique_pages.append(page)

    if not unique_pages:
        return 0

    total_pages = len(unique_pages)

    # Chunk all pages and prepare for storage
    all_texts: List[str] = []
    all_metadatas: List[Dict[str, Any]] = []
    all_ids: List[str] = []

    for page_idx, page in enumerate(unique_pages):
        content = page["content"]
        chunks = chunk_text(content)

        if progress_callback:
            progress_callback(
                page_idx + 1,
                total_pages,
                f"Chunking page {page_idx + 1}/{total_pages}: {(page.get('title') or page['url'])[:50]}...",
            )

        for chunk_idx, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            metadata = {
                "url": page["url"],
                "title": page.get("title") or "",
                "chunk_index": chunk_idx,
                "total_chunks": len(chunks),
            }
            if chunk_callback:
                chunk_callback(chunk, metadata)
            all_texts.append(chunk)
            all_metadatas.append(metadata)
            chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{page['url']}#{chunk_idx}"))
            all_ids.append(chunk_id)

    if not all_texts:
        return 0

    total_chunks = len(all_texts)

    if progress_callback:
        progress_callback(0, total_chunks, f"Generating embeddings for {total_chunks} chunks...")

    # Generate embeddings in batches
    embedding_batch_size = 50
    all_embeddings = []
    for i in range(0, total_chunks, embedding_batch_size):
        end = min(i + embedding_batch_size, total_chunks)
        batch_texts = all_texts[i:end]

        if progress_callback:
            progress_callback(
                end,
                total_chunks,
                f"Embedding chunks {i + 1}-{end} of {total_chunks}...",
            )

        batch_embeddings = embeddings_model.embed_documents(batch_texts)
        all_embeddings.extend(batch_embeddings)

    if progress_callback:
        progress_callback(0, total_chunks, f"Storing {total_chunks} chunks in vector database...")

    # Upsert in batches
    upsert_batch_size = 100
    for i in range(0, total_chunks, upsert_batch_size):
        end = min(i + upsert_batch_size, total_chunks)

        if progress_callback:
            progress_callback(
                end,
                total_chunks,
                f"Storing chunks {i + 1}-{end} of {total_chunks}...",
            )

        points = []
        for j in range(i, end):
            payload = dict(all_metadatas[j])
            payload["content"] = all_texts[j]
            points.append(
                PointStruct(
                    id=all_ids[j],
                    vector=all_embeddings[j],
                    payload=payload,
                )
            )

        get_qdrant_client().upsert(
            collection_name=qdrant_collection,
            points=points,
            wait=True,
        )

    print(f"Stored {total_chunks} chunks from {len(unique_pages)} pages in Qdrant")
    return total_chunks


def search(
    query: str,
    top_k: int = 5,
    collection_name: str = None,
    min_similarity: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Search for relevant content based on semantic similarity.
    Uses LangChain embeddings for query encoding.
    """
    if not query or not query.strip():
        return []

    qdrant_collection = _ensure_collection(collection_name)

    print(f"Searching for: {query} (top_k={top_k})")
    query_embedding = embeddings_model.embed_query(query)

    if hasattr(get_qdrant_client(), "search"):
        # Older qdrant-client API.
        search_results = get_qdrant_client().search(
            collection_name=qdrant_collection,
            query_vector=query_embedding,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
    else:
        # Newer qdrant-client API (query_points).
        query_response = get_qdrant_client().query_points(
            collection_name=qdrant_collection,
            query=query_embedding,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
        search_results = getattr(query_response, "points", [])

    formatted_results = []
    for point in search_results or []:
        payload = point.payload or {}
        score = float(point.score or 0.0)
        if min_similarity is not None and score < min_similarity:
            continue
        formatted_results.append(
            {
                "content": payload.get("content", ""),
                "url": payload.get("url", ""),
                "title": payload.get("title", ""),
                "similarity": score,
            }
        )

    return formatted_results


def _build_context_from_results(results: List[Dict[str, Any]]) -> str:
    if not results:
        return "No relevant information found."

    context_parts = []
    for i, result in enumerate(results, 1):
        content = result.get("content", "")
        content_preview = content[:1500] + "..." if len(content) > 1500 else content
        context_parts.append(
            f"[Source {i}: {result.get('title', '')}]\n"
            f"URL: {result.get('url', '')}\n"
            f"Content: {content_preview}\n"
        )

    return "\n---\n".join(context_parts)


def get_context_for_question(query: str, top_k: int = 3, collection_name: str = None) -> str:
    """
    Get relevant context for answering a question.
    Formats the search results into a context string for LLM.
    """
    results = search(query, top_k=top_k, collection_name=collection_name)
    return _build_context_from_results(results)


def get_context_and_sources(
    query: str,
    top_k: int = 3,
    collection_name: str = None,
    min_similarity: Optional[float] = None,
) -> tuple[str, List[Dict[str, Any]]]:
    """
    Retrieve sources once and build the context string from them.
    """
    results = search(
        query,
        top_k=top_k,
        collection_name=collection_name,
        min_similarity=min_similarity,
    )
    return _build_context_from_results(results), results


def clear_collection(collection_name: str = None):
    """Clear all documents from a collection."""
    name = _safe_collection_name(collection_name)

    if _collection_exists_in_qdrant(name):
        get_qdrant_client().delete_collection(collection_name=name)

    get_qdrant_client().create_collection(
        collection_name=name,
        vectors_config=VectorParams(
            size=EMBEDDING_DIMENSION,
            distance=Distance.COSINE,
        ),
    )


def delete_collection(collection_name: str = None):
    """Delete a collection and all of its documents."""
    name = _safe_collection_name(collection_name)
    if _collection_exists_in_qdrant(name):
        get_qdrant_client().delete_collection(collection_name=name)


def get_collection_stats(collection_name: str = None) -> Dict[str, Any]:
    """Get statistics about the collection."""
    name = _ensure_collection(collection_name)
    count_result = get_qdrant_client().count(collection_name=name, exact=True)
    return {
        "name": collection_name or QDRANT_DEFAULT_COLLECTION,
        "count": count_result.count,
    }


def has_embeddings(collection_name: str = None) -> bool:
    """Check if a collection has any embeddings stored."""
    try:
        name = _ensure_collection(collection_name)
        count_result = get_qdrant_client().count(collection_name=name, exact=True)
        return count_result.count > 0
    except Exception:
        return False


def collection_exists(collection_name: str) -> bool:
    """Check if a collection exists and has content."""
    if not collection_name:
        return False

    try:
        name = _safe_collection_name(collection_name)
        if not _collection_exists_in_qdrant(name):
            return False
        count_result = get_qdrant_client().count(collection_name=name, exact=True)
        return count_result.count > 0
    except Exception:
        return False


def get_collection_urls(collection_name: str = None, limit: int = 100) -> List[str]:
    """Get all unique URLs stored in a collection."""
    name = _ensure_collection(collection_name)

    try:
        points, _ = get_qdrant_client().scroll(
            collection_name=name,
            limit=limit,
            with_payload=["url"],
            with_vectors=False,
        )
        urls = set()
        for point in points:
            payload = point.payload or {}
            if "url" in payload and payload["url"]:
                urls.add(payload["url"])
        return list(urls)
    except Exception:
        return []


def url_is_embedded(url: str, collection_name: str = None) -> bool:
    """Check if a specific URL has been embedded."""
    if not url:
        return False

    name = _ensure_collection(collection_name)

    try:
        filter_query = Filter(
            must=[
                FieldCondition(
                    key="url",
                    match=MatchValue(value=url),
                )
            ]
        )

        try:
            points, _ = get_qdrant_client().scroll(
                collection_name=name,
                scroll_filter=filter_query,
                limit=1,
                with_payload=False,
                with_vectors=False,
            )
        except TypeError:
            # Compatibility for older qdrant-client versions.
            points, _ = get_qdrant_client().scroll(
                collection_name=name,
                query_filter=filter_query,
                limit=1,
                with_payload=False,
                with_vectors=False,
            )
        return len(points) > 0
    except Exception:
        return False
