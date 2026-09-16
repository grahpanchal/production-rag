from app.config import settings
from app.db.qdrant import client
from app.rag.embeddings import create_embeddings
from app.rag.reranker import rerank_documents

QUERY = "How do I create a MySQL index?"


def inspect_reranker():

    query_embedding = create_embeddings([QUERY])[0]

    dense_results = client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_embedding,
        limit=10,
        with_payload=True,
    )

    print("\nRERANKER INSPECTION")
    print("=" * 80)

    reranked_results = rerank_documents(
        query=QUERY, documents=dense_results.points, top_k=10
    )

    for rank, result in enumerate(reranked_results, start=1):

        document = result["document"]
        payload = document.payload

        print(f"\nRerank: {rank}")
        print(f"Point ID: {document.id}")
        print(f"Filename: {payload['filename']}")
        print(f"Pages: " f"{payload['page_start']}-" f"{payload['page_end']}")

        print(f"Dense score: " f"{document.score:.4f}")

        print(f"Rerank score: " f"{result['rerank_score']:.4f}")

        text = payload["text"].replace("\n", " ")

        if len(text) > 350:
            text = text[:350] + "..."

        print(f"Text: {text}")


if __name__ == "__main__":
    inspect_reranker()
