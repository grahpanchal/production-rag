from app.config import settings
from app.db.qdrant import client
from app.rag.embeddings import create_embeddings
from app.rag.reranker import rerank_documents

from eval.chunk_id_ground_truth import CHUNK_ID_GROUND_TRUTH

DENSE_CANDIDATES = 20
TOP_K = 5


def dense_search(query: str, limit: int = DENSE_CANDIDATES):

    query_embedding = create_embeddings([query])[0]

    results = client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_embedding,
        limit=limit,
        with_payload=True,
    )

    return results.points


def get_relevant_rank(results, relevant_ids):

    relevant_ids = set(relevant_ids)

    for rank, document in enumerate(results, start=1):
        if str(document.id) in relevant_ids:
            return rank

    return None


def get_document_text(document):

    if not document.payload:
        return ""

    return document.payload.get("text", "")


def compare():

    dense_hits = 0
    rerank_hits = 0

    dense_rr = []
    rerank_rr = []

    print("\nDENSE vs RERANKER DETAILED COMPARISON")
    print("=" * 100)

    for index, test in enumerate(CHUNK_ID_GROUND_TRUTH, start=1):

        query = test["query"]
        relevant_ids = set(test["relevant_ids"])

        dense_results = dense_search(
            query=query,
            limit=DENSE_CANDIDATES,
        )

        dense_top5 = dense_results[:TOP_K]

        reranked = rerank_documents(
            query=query,
            documents=dense_results,
            top_k=DENSE_CANDIDATES,
        )

        rerank_top5 = [item["document"] for item in reranked[:TOP_K]]

        dense_rank = get_relevant_rank(
            dense_top5,
            relevant_ids,
        )

        rerank_rank = get_relevant_rank(
            rerank_top5,
            relevant_ids,
        )

        # Dense metrics
        if dense_rank:
            dense_hits += 1
            dense_rr.append(1 / dense_rank)
        else:
            dense_rr.append(0)

        # Reranker metrics
        if rerank_rank:
            rerank_hits += 1
            rerank_rr.append(1 / rerank_rank)
        else:
            rerank_rr.append(0)

        print(f"\n[{index}] {query}")
        print("-" * 100)

        print(
            f"Dense Top-5:    "
            f"{'PASS' if dense_rank else 'FAIL'} "
            f"(rank={dense_rank})"
        )

        print(
            f"Reranker Top-5: "
            f"{'PASS' if rerank_rank else 'FAIL'} "
            f"(rank={rerank_rank})"
        )

        print("\nDense Top-5:")

        for rank, document in enumerate(dense_top5, start=1):

            marker = " <-- RELEVANT" if str(document.id) in relevant_ids else ""

            dense_score = getattr(
                document,
                "score",
                None,
            )

            print(
                f"{rank}. "
                f"ID={document.id} "
                f"DenseScore={dense_score:.4f}"
                f"{marker}"
            )

        print("\nReranker Top-5:")

        for rank, item in enumerate(reranked[:TOP_K], start=1):

            document = item["document"]
            rerank_score = item["rerank_score"]

            marker = " <-- RELEVANT" if str(document.id) in relevant_ids else ""

            dense_score = getattr(
                document,
                "score",
                None,
            )

            print(
                f"{rank}. "
                f"ID={document.id} "
                f"DenseScore={dense_score:.4f} "
                f"RerankScore={rerank_score:.4f}"
                f"{marker}"
            )

    total = len(CHUNK_ID_GROUND_TRUTH)

    dense_recall = dense_hits / total
    dense_mrr = sum(dense_rr) / total

    rerank_recall = rerank_hits / total
    rerank_mrr = sum(rerank_rr) / total

    print("\n" + "=" * 100)
    print("FINAL RESULTS")
    print("=" * 100)

    print(f"Dense Recall@5:    {dense_recall:.2%}")

    print(f"Dense MRR@5:       {dense_mrr:.2%}")

    print(f"Reranker Recall@5: {rerank_recall:.2%}")

    print(f"Reranker MRR@5:    {rerank_mrr:.2%}")

    print("\nDifference:")

    print(f"Recall change: " f"{(rerank_recall - dense_recall):+.2%}")

    print(f"MRR change:    " f"{(rerank_mrr - dense_mrr):+.2%}")


if __name__ == "__main__":
    compare()
