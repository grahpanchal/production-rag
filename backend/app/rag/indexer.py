import uuid

from qdrant_client.models import PointStruct

from app.db.qdrant import client
from app.config import settings
from app.rag.chunker import chunk_pages
from app.rag.embeddings import create_embeddings


def index_document(file_id: str, filename: str, pages: list[dict]):
    # Create chunks across the entire document
    all_chunks = chunk_pages(pages=pages, chunk_size=300, overlap=50)

    if not all_chunks:
        return 0

    # Extract text for embedding
    texts = [chunk["text"] for chunk in all_chunks]

    # Generate local embeddings
    embeddings = create_embeddings(texts)

    points = []

    for chunk, embedding in zip(all_chunks, embeddings):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "file_id": file_id,
                    "filename": filename,
                    "page_start": chunk["page_start"],
                    "page_end": chunk["page_end"],
                    "text": chunk["text"],
                },
            )
        )

    # Store in Qdrant
    client.upsert(collection_name=settings.qdrant_collection, points=points, wait=True)

    return len(points)
