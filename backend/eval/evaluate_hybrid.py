from app.rag.embeddings import create_embeddings
from app.db.qdrant import client
from app.config import settings

from app.rag.hybrid_retriever import hybrid_search

from eval.test_queries import TEST_QUERIES


def true_dense_search(query: str, top_k: int = 5):
    """
    Pure dense retrieval.
    No reranker.
    """

    query_embedding = create_embeddings([query])[0]

    results = client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_embedding,
        limit=top_k,
        with_payload=True,
    )

    return results.points


def get_filename(document):
    return document.payload.get("filename")


def find_relevant_rank(results, expected_filename):
    for rank, result in enumerate(results, start=1):

        if isinstance(result, dict):
            document = result["document"]
        else:
            document = result

        filename = get_filename(document)

        if filename == expected_filename:
            return rank

    return None


def calculate_metrics(ranks):

    total = len(ranks)

    recall = sum(rank is not None for rank in ranks) / total

    reciprocal_ranks = []

    for rank in ranks:

        if rank is not None:
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0)

    mrr = sum(reciprocal_ranks) / total

    return recall, mrr


def main():

    dense_ranks = []
    hybrid_ranks = []

    print("\nTRUE DENSE VS HYBRID RETRIEVAL EVALUATION")

    print("=" * 80)

    for index, test in enumerate(TEST_QUERIES, start=1):

        query = test["query"]

        expected_filename = test["expected_filename"]

        # ---------------------------------------
        # TRUE DENSE SEARCH
        # ---------------------------------------

        dense_results = true_dense_search(query=query, top_k=5)

        dense_rank = find_relevant_rank(dense_results, expected_filename)

        dense_ranks.append(dense_rank)

        # ---------------------------------------
        # HYBRID SEARCH
        # ---------------------------------------

        hybrid_results = hybrid_search(query=query, top_k=5)

        hybrid_rank = find_relevant_rank(hybrid_results, expected_filename)

        hybrid_ranks.append(hybrid_rank)

        # ---------------------------------------
        # Status
        # ---------------------------------------

        dense_status = f"PASS (rank={dense_rank})" if dense_rank is not None else "FAIL"

        hybrid_status = (
            f"PASS (rank={hybrid_rank})" if hybrid_rank is not None else "FAIL"
        )

        print(f"\n[{index}] {query}")

        print(f"Dense:  {dense_status}")

        print(f"Hybrid: {hybrid_status}")

    # -------------------------------------------
    # Metrics
    # -------------------------------------------

    dense_recall, dense_mrr = calculate_metrics(dense_ranks)

    hybrid_recall, hybrid_mrr = calculate_metrics(hybrid_ranks)

    # -------------------------------------------
    # Final report
    # -------------------------------------------

    print("\n" + "=" * 80)

    print(f"Dense Recall@5:  " f"{dense_recall:.2%}")

    print(f"Dense MRR@5:     " f"{dense_mrr:.2%}")

    print(f"Hybrid Recall@5: " f"{hybrid_recall:.2%}")

    print(f"Hybrid MRR@5:    " f"{hybrid_mrr:.2%}")

    print("\n" + "-" * 80)

    recall_change = hybrid_recall - dense_recall

    mrr_change = hybrid_mrr - dense_mrr

    print(f"Recall change: " f"{recall_change:+.2%}")

    print(f"MRR change:    " f"{mrr_change:+.2%}")


if __name__ == "__main__":
    main()
