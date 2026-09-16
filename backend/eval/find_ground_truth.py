from app.db.qdrant import client
from app.config import settings

SEARCH_TERMS = [
    "git checkout -b",
    "git switch",
    "git merge",
    "INSERT INTO",
    "TRIGGER",
    "CREATE INDEX",
    "PRIMARY KEY",
]


def find_chunks():
    points = []
    offset = None

    while True:
        batch, next_offset = client.scroll(
            collection_name=settings.qdrant_collection,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        points.extend(batch)

        if next_offset is None:
            break

        offset = next_offset

    print("\nGROUND TRUTH CHUNK SEARCH")
    print("=" * 80)

    for term in SEARCH_TERMS:

        print(f"\n{'=' * 80}")
        print(f"SEARCH TERM: {term}")
        print(f"{'=' * 80}")

        found = False

        for point in points:

            payload = point.payload
            text = payload.get("text", "")

            if term.lower() in text.lower():

                found = True

                preview = text.replace("\n", " ")

                if len(preview) > 300:
                    preview = preview[:300] + "..."

                print(f"\nPoint ID: {point.id}")
                print(f"Filename: {payload.get('filename')}")
                print(
                    f"Pages: "
                    f"{payload.get('page_start')}-"
                    f"{payload.get('page_end')}"
                )
                print(f"Text: {preview}")

        if not found:
            print("No matching chunk found.")


if __name__ == "__main__":
    find_chunks()
