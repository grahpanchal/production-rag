from rank_bm25 import BM25Okapi

from app.db.qdrant import client
from app.config import settings
from app.rag.embeddings import create_embeddings


def tokenize(text: str):
    return text.lower().split()


def get_all_documents():
    """
    Retrieve all indexed chunks from Qdrant.
    """

    documents = []

    offset = None

    while True:

        records, offset = client.scroll(
            collection_name=settings.qdrant_collection,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        documents.extend(records)

        if offset is None:
            break

    return documents


def dense_search(query: str, documents: list, top_k: int = 10):
    """
    Perform dense vector search against Qdrant.
    """

    query_embedding = create_embeddings([query])[0]

    results = client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_embedding,
        limit=top_k,
        with_payload=True,
    )

    return results.points


def hybrid_search(query: str, top_k: int = 5, dense_k: int = 10):
    """
    Combine dense semantic retrieval with BM25 keyword retrieval.
    """

    # ---------------------------------------
    # 1. Get all indexed chunks
    # ---------------------------------------

    all_documents = get_all_documents()

    if not all_documents:
        return []

    # ---------------------------------------
    # 2. BM25 keyword search
    # ---------------------------------------

    corpus = [tokenize(document.payload.get("text", "")) for document in all_documents]

    bm25 = BM25Okapi(corpus)

    query_tokens = tokenize(query)

    bm25_scores = bm25.get_scores(query_tokens)

    # ---------------------------------------
    # 3. Dense search
    # ---------------------------------------

    dense_results = dense_search(query=query, documents=all_documents, top_k=dense_k)

    # ---------------------------------------
    # 4. Normalize BM25 scores
    # ---------------------------------------

    max_bm25 = max(bm25_scores)

    if max_bm25 > 0:

        normalized_bm25 = [score / max_bm25 for score in bm25_scores]

    else:

        normalized_bm25 = [0.0 for _ in bm25_scores]

    # ---------------------------------------
    # 5. Create combined score map
    # ---------------------------------------

    combined_scores = {}

    # Dense contribution
    for document in dense_results:

        point_id = str(document.id)

        dense_score = float(document.score)

        combined_scores[point_id] = {
            "document": document,
            "dense_score": dense_score,
            "bm25_score": 0.0,
        }

    # BM25 contribution
    for index, document in enumerate(all_documents):

        point_id = str(document.id)

        bm25_score = normalized_bm25[index]

        if point_id not in combined_scores:

            combined_scores[point_id] = {
                "document": document,
                "dense_score": 0.0,
                "bm25_score": bm25_score,
            }

        else:

            combined_scores[point_id]["bm25_score"] = bm25_score

    # ---------------------------------------
    # 6. Weighted score fusion
    # ---------------------------------------

    results = []

    for item in combined_scores.values():

        dense_score = item["dense_score"]
        bm25_score = item["bm25_score"]

        hybrid_score = 0.7 * dense_score + 0.3 * bm25_score

        results.append(
            {
                "document": item["document"],
                "dense_score": dense_score,
                "bm25_score": bm25_score,
                "hybrid_score": hybrid_score,
            }
        )

    # ---------------------------------------
    # 7. Sort by hybrid score
    # ---------------------------------------

    results.sort(key=lambda item: item["hybrid_score"], reverse=True)

    return results[:top_k]
