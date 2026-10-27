from datetime import datetime
import uuid

from app.database import delete_item, get_scan_attr, put_item, scan_table


def save_chunk_questionnaire(
    collection_id: str,
    questionnaire: dict,
    metadata: dict,
    chunk_text: str = "",
) -> bool:
    item_id = str(uuid.uuid4())
    document = {
        "PK": item_id,
        "id": item_id,
        "collection_id": collection_id,
        "questionnaire": questionnaire.get("questions", []) if isinstance(questionnaire, dict) else [],
        "metadata": metadata,
        "chunk_excerpt": (chunk_text or "").strip()[:1200],
        "created_at": datetime.utcnow().isoformat(),
    }

    put_item("chunk_question", document)
    return True


def delete_collection_questionnaires(collection_id: str) -> int:
    items = scan_table("chunk_question", get_scan_attr("collection_id").eq(collection_id))
    for item in items:
        delete_item("chunk_question", {"PK": item["PK"]})
    return len(items)
