import re
import tiktoken

encoding = tiktoken.get_encoding("cl100k_base")


def chunk_pages(pages: list[dict], chunk_size: int = 300, overlap: int = 50):
    """
    Create chunks across the entire document.

    Chunks can span multiple PDF pages.
    Each chunk keeps track of its starting and ending page.
    """

    token_items = []

    for page in pages:
        page_number = page["page"]
        text = page["text"]

        # Normalize whitespace
        text = re.sub(r"\r\n?", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = text.strip()

        if not text:
            continue

        tokens = encoding.encode(text)

        for token in tokens:
            token_items.append((token, page_number))

    chunks = []

    start = 0

    while start < len(token_items):
        end = min(start + chunk_size, len(token_items))

        chunk_items = token_items[start:end]

        token_ids = [item[0] for item in chunk_items]

        chunk_text = encoding.decode(token_ids).strip()

        if chunk_text:
            page_numbers = [item[1] for item in chunk_items]

            chunks.append(
                {
                    "text": chunk_text,
                    "page_start": min(page_numbers),
                    "page_end": max(page_numbers),
                }
            )

        if end >= len(token_items):
            break

        start = end - overlap

    return chunks
