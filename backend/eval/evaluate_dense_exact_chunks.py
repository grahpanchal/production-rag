from app.db.qdrant import client
from app.config import settings
from app.rag.embeddings import create_embeddings
from eval.chunk_id_ground_truth import CHUNK_ID_GROUND_TRUTH

TOP_K = 5


def evaluate():
    total_queries = len(CHUNK_ID_GROUND_TRUTH)

    recall_hits = 0
    reciprocal_ranks = []

    print()
    print("DENSE-ONLY EXACT CHUNK-ID RETRIEVAL EVALUATION")
    print("=" * 80)

    for index, item in enumerate(CHUNK_ID_GROUND_TRUTH, start=1):
        query = item["query"]
        relevant_ids = set(item["relevant_ids"])

        # Create embedding for the query
        query_embedding = create_embeddings([query])[0]

        # Direct Qdrant search - NO RERANKER
        results = client.query_points(
            collection_name=settings.qdrant_collection,
            query=query_embedding,
            limit=TOP_K,
            with_payload=True,
        )

        retrieved_ids = [point.id for point in results.points]

        first_relevant_rank = None

        for rank, point_id in enumerate(retrieved_ids, start=1):
            if point_id in relevant_ids:
                first_relevant_rank = rank
                break

        if first_relevant_rank is not None:
            recall_hits += 1
            reciprocal_rank = 1 / first_relevant_rank
            reciprocal_ranks.append(reciprocal_rank)

            status = "PASS"
        else:
            reciprocal_rank = 0.0
            reciprocal_ranks.append(0.0)

            status = "FAIL"

        print()
        print(f"[{index}] {status}")
        print(f"Query: {query}")
        print(f"Relevant IDs: {list(relevant_ids)}")
        print(f"Retrieved IDs: {retrieved_ids}")
        print(
            f"First relevant rank: "
            f"{first_relevant_rank if first_relevant_rank else 'Not found in Top 5'}"
        )
        print(f"Reciprocal Rank: {reciprocal_rank:.2f}")

    recall_at_5 = (recall_hits / total_queries) * 100
    mrr_at_5 = (sum(reciprocal_ranks) / total_queries) * 100

    print()
    print("=" * 80)
    print(f"Exact Chunk Recall@5: {recall_at_5:.2f}%")
    print(f"Exact Chunk MRR@5:    {mrr_at_5:.2f}%")
    print(f"Passed:               {recall_hits}/{total_queries}")


if __name__ == "__main__":
    evaluate()
