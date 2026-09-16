from app.config import settings
from app.db.qdrant import client
from app.rag.embeddings import create_embeddings

QUERY = "How do I create a MySQL index?"


def inspect_dense():

    query_embedding = create_embeddings([QUERY])[0]

    results = client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_embedding,
        limit=10,
        with_payload=True,
    )

    print("\nDENSE RETRIEVAL — TOP 10")
    print("=" * 80)

    for rank, result in enumerate(results.points, start=1):

        payload = result.payload

        print(f"\nRank: {rank}")
        print(f"Point ID: {result.id}")
        print(f"Filename: {payload['filename']}")
        print(f"Pages: " f"{payload['page_start']}-" f"{payload['page_end']}")
        print(f"Qdrant score: " f"{result.score:.4f}")

        text = payload["text"].replace("\n", " ")

        if len(text) > 300:
            text = text[:300] + "..."

        print(f"Text: {text}")


if __name__ == "__main__":
    inspect_dense()
