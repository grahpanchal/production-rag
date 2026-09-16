import time

from app.rag.embeddings import create_embeddings
from app.rag.reranker import rerank_documents
from app.db.qdrant import client
from app.config import settings

from eval.chunk_id_ground_truth import CHUNK_ID_GROUND_TRUTH


def retrieve_dense(query: str, limit: int = 20):
    query_embedding = create_embeddings([query])[0]

    results = client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_embedding,
        limit=limit,
        with_payload=True,
    )

    return results.points


def evaluate_reranker_candidates(candidate_k: int):

    recall_scores = []
    reciprocal_ranks = []
    reranker_times = []

    print()
    print("=" * 80)
    print(f"RERANKER CANDIDATES = {candidate_k}")
    print("=" * 80)

    for index, test_case in enumerate(CHUNK_ID_GROUND_TRUTH, start=1):

        query = test_case["query"]
        relevant_ids = set(test_case["relevant_ids"])

        # ----------------------------------------------
        # Dense retrieval always gets 20 candidates
        # ----------------------------------------------

        dense_results = retrieve_dense(
            query=query,
            limit=20,
        )

        # ----------------------------------------------
        # IMPORTANT:
        # Only send candidate_k documents to reranker
        # ----------------------------------------------

        reranker_documents = dense_results[:candidate_k]

        start_time = time.perf_counter()

        reranked_results = rerank_documents(
            query=query,
            documents=reranker_documents,
            top_k=5,
        )

        reranker_time = time.perf_counter() - start_time

        reranker_times.append(reranker_time)

        # ----------------------------------------------
        # Evaluate final Top-5
        # ----------------------------------------------

        retrieved_ids = [str(result["document"].id) for result in reranked_results[:5]]

        found = any(chunk_id in relevant_ids for chunk_id in retrieved_ids)

        if found:

            recall_scores.append(1)

            first_relevant_rank = next(
                rank
                for rank, chunk_id in enumerate(
                    retrieved_ids,
                    start=1,
                )
                if chunk_id in relevant_ids
            )

            reciprocal_rank = 1 / first_relevant_rank

        else:

            recall_scores.append(0)
            reciprocal_rank = 0

        reciprocal_ranks.append(reciprocal_rank)

        print(
            f"[{index}] "
            f"{'PASS' if found else 'FAIL'} | "
            f"candidates={candidate_k} | "
            f"rank={first_relevant_rank if found else '-'} | "
            f"time={reranker_time:.3f}s"
        )

    recall = sum(recall_scores) / len(recall_scores)
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    avg_time = sum(reranker_times) / len(reranker_times)

    return recall, mrr, avg_time


def main():

    print()
    print("TRUE RERANKER CANDIDATE SIZE EXPERIMENT")
    print("=" * 80)

    # Current production configuration
    recall_20, mrr_20, time_20 = evaluate_reranker_candidates(20)

    # Experimental configuration
    recall_10, mrr_10, time_10 = evaluate_reranker_candidates(10)

    print()
    print("=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)

    print()
    print(f"{'Metric':<25}" f"{'K=20':>12}" f"{'K=10':>12}")

    print("-" * 50)

    print(f"{'Recall@5':<25}" f"{recall_20 * 100:>11.2f}%" f"{recall_10 * 100:>11.2f}%")

    print(f"{'MRR@5':<25}" f"{mrr_20 * 100:>11.2f}%" f"{mrr_10 * 100:>11.2f}%")

    print(f"{'Avg reranker time':<25}" f"{time_20:>11.3f}s" f"{time_10:>11.3f}s")

    print()
    print("Recall change:", f"{(recall_10 - recall_20) * 100:+.2f}%")

    print("MRR change:", f"{(mrr_10 - mrr_20) * 100:+.2f}%")

    print("Latency change:", f"{(time_10 - time_20):+.3f}s")


if __name__ == "__main__":
    main()
    
# import time

# from app.rag.embeddings import create_embeddings
# from app.rag.reranker import rerank_documents
# from app.db.qdrant import client
# from app.config import settings

# from eval.chunk_id_ground_truth import CHUNK_ID_GROUND_TRUTH


# def retrieve_dense(query: str, limit: int = 20):
#     query_embedding = create_embeddings([query])[0]

#     results = client.query_points(
#         collection_name=settings.qdrant_collection,
#         query=query_embedding,
#         limit=limit,
#         with_payload=True,
#     )

#     return results.points


# def evaluate_reranker_k(reranker_k: int):

#     recall_scores = []
#     reciprocal_ranks = []

#     print()
#     print("=" * 80)
#     print(f"RERANKER K = {reranker_k}")
#     print("=" * 80)

#     for index, test_case in enumerate(CHUNK_ID_GROUND_TRUTH, start=1):

#         query = test_case["query"]
#         relevant_ids = set(test_case["relevant_ids"])

#         # Dense retrieval
#         dense_results = retrieve_dense(
#             query=query,
#             limit=20,
#         )

#         # Reranking
#         start_time = time.perf_counter()

#         reranked_results = rerank_documents(
#             query=query,
#             documents=dense_results,
#             top_k=reranker_k,
#         )

#         reranker_time = time.perf_counter() - start_time

#         # Evaluate final Top-5
#         final_results = reranked_results[:5]

#         retrieved_ids = [str(result["document"].id) for result in final_results]

#         found = any(chunk_id in relevant_ids for chunk_id in retrieved_ids)

#         if found:
#             recall_scores.append(1)

#             first_relevant_rank = next(
#                 rank
#                 for rank, chunk_id in enumerate(
#                     retrieved_ids,
#                     start=1,
#                 )
#                 if chunk_id in relevant_ids
#             )

#             reciprocal_rank = 1 / first_relevant_rank

#         else:
#             recall_scores.append(0)
#             reciprocal_rank = 0

#         reciprocal_ranks.append(reciprocal_rank)

#         print(
#             f"[{index}] "
#             f"{'PASS' if found else 'FAIL'} | "
#             f"reranker_k={reranker_k} | "
#             f"rank={first_relevant_rank if found else '-'} | "
#             f"time={reranker_time:.3f}s"
#         )

#     recall = sum(recall_scores) / len(recall_scores)
#     mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)

#     return recall, mrr


# def main():

#     print()
#     print("RERANKER CANDIDATE SIZE EXPERIMENT")
#     print("=" * 80)

#     # Current production setup
#     recall_20, mrr_20 = evaluate_reranker_k(20)

#     # Experimental setup
#     recall_10, mrr_10 = evaluate_reranker_k(10)

#     print()
#     print("=" * 80)
#     print("FINAL COMPARISON")
#     print("=" * 80)

#     print()
#     print(f"{'Metric':<20} {'K=20':>10} {'K=10':>10}")
#     print("-" * 45)

#     print(f"{'Recall@5':<20} " f"{recall_20 * 100:>9.2f}% " f"{recall_10 * 100:>9.2f}%")

#     print(f"{'MRR@5':<20} " f"{mrr_20 * 100:>9.2f}% " f"{mrr_10 * 100:>9.2f}%")

#     print()
#     print("Recall change:", f"{(recall_10 - recall_20) * 100:+.2f}%")
#     print("MRR change:", f"{(mrr_10 - mrr_20) * 100:+.2f}%")


# if __name__ == "__main__":
#     main()
