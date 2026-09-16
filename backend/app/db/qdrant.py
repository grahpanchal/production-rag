import logging

from qdrant_client import QdrantClient

from qdrant_client.models import (
    Distance,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
    FilterSelector,
    PayloadSchemaType,
)

from app.config import settings

logger = logging.getLogger(__name__)


# --------------------------------------------------
# Qdrant client
# --------------------------------------------------

client = QdrantClient(
    url=settings.qdrant_url,
    timeout=10.0,
)


# --------------------------------------------------
# Create collection
# --------------------------------------------------


def create_collection():

    try:

        collections = client.get_collections()

        existing = [collection.name for collection in collections.collections]

        if settings.qdrant_collection not in existing:

            client.create_collection(
                collection_name=settings.qdrant_collection,
                vectors_config=VectorParams(
                    size=384,
                    distance=Distance.COSINE,
                ),
            )

        # Payload index for file_id filtering

        client.create_payload_index(
            collection_name=settings.qdrant_collection,
            field_name="file_id",
            field_schema=PayloadSchemaType.KEYWORD,
        )

        # Payload index for filename filtering

        client.create_payload_index(
            collection_name=settings.qdrant_collection,
            field_name="filename",
            field_schema=PayloadSchemaType.KEYWORD,
        )

    except Exception:

        logger.exception("Qdrant startup initialization failed")

        raise


# --------------------------------------------------
# Delete document chunks
# --------------------------------------------------


def delete_document_chunks(file_id: str):
    """
    Delete all Qdrant chunks belonging to one document.
    """

    try:

        client.delete(
            collection_name=settings.qdrant_collection,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="file_id",
                            match=MatchValue(value=file_id),
                        )
                    ]
                )
            ),
            wait=True,
        )

    except Exception:

        logger.exception(
            "Qdrant document deletion failed: file_id=%s",
            file_id,
        )

        raise
