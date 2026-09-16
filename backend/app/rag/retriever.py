import logging
import time

from qdrant_client.models import Filter, FieldCondition, MatchValue

from app.db.qdrant import client
from app.config import settings
from app.rag.embeddings import create_embeddings
from app.rag.reranker import rerank_documents

logger = logging.getLogger(__name__)

MIN_RERANK_SCORE = 0.0


def search_documents(
    query: str,
    top_k: int = 5,
    filename: str | None = None,
    file_id: str | None = None,
):
    # --------------------------------------------------
    # 1. Create query embedding
    # --------------------------------------------------

    embedding_start = time.perf_counter()

    query_embedding = create_embeddings([query])[0]

    embedding_duration = time.perf_counter() - embedding_start

    # --------------------------------------------------
    # 2. Build filters dynamically
    # --------------------------------------------------

    must_conditions = []

    if filename:
        must_conditions.append(
            FieldCondition(
                key="filename",
                match=MatchValue(value=filename),
            )
        )

    if file_id:
        must_conditions.append(
            FieldCondition(
                key="file_id",
                match=MatchValue(value=file_id),
            )
        )

    query_filter = None

    if must_conditions:
        query_filter = Filter(must=must_conditions)

    # --------------------------------------------------
    # 3. Dense retrieval from Qdrant
    # --------------------------------------------------

    qdrant_start = time.perf_counter()

    candidate_results = client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_embedding,
        query_filter=query_filter,
        limit=20,
        with_payload=True,
    )

    qdrant_duration = time.perf_counter() - qdrant_start

    # --------------------------------------------------
    # 4. Rerank candidates
    # --------------------------------------------------

    reranker_start = time.perf_counter()

    reranked_results = rerank_documents(
        query=query,
        documents=candidate_results.points[:10],
        top_k=5,
    )

    reranker_duration = time.perf_counter() - reranker_start

    # --------------------------------------------------
    # 5. Remove irrelevant results
    # --------------------------------------------------

    filtered_results = [
        result
        for result in reranked_results
        if result["rerank_score"] >= MIN_RERANK_SCORE
    ]

    # --------------------------------------------------
    # 6. Log performance
    # --------------------------------------------------

    total_duration = embedding_duration + qdrant_duration + reranker_duration

    logger.info(
        "RAG retrieval timing: "
        "embedding=%.3fs "
        "qdrant=%.3fs "
        "reranker=%.3fs "
        "total=%.3fs "
        "candidates=%s "
        "results=%s",
        embedding_duration,
        qdrant_duration,
        reranker_duration,
        total_duration,
        len(candidate_results.points),
        len(filtered_results),
    )

    return filtered_results
