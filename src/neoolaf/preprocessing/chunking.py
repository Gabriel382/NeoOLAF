from __future__ import annotations

from typing import List

from neoolaf.domain.documents import DocumentChunk


def chunk_text(text: str, chunk_size: int = 1500, overlap: int = 200) -> List[DocumentChunk]:
    """
    Split a plain-text string into overlapping fixed-size chunks.

    Each chunk is wrapped in a :class:`DocumentChunk` with character-level
    offsets and a zero-padded identifier. The last chunk may be shorter than
    ``chunk_size`` if the remaining text does not fill a full window.

    Args:
        text:
            The input string to split.
        chunk_size:
            Maximum number of characters per chunk.
        overlap:
            Number of characters carried over from the end of one chunk
            into the start of the next.

    Returns:
        Ordered list of :class:`DocumentChunk` objects covering the full text.
    """
    chunks: List[DocumentChunk] = []
    start = 0
    idx = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk_text_value = text[start:end]

        chunks.append(
            DocumentChunk(
                chunk_id=f"chunk_{idx:04d}",
                text=chunk_text_value,
                start_char=start,
                end_char=end,
            )
        )

        if end >= n:
            break

        start = max(0, end - overlap)
        idx += 1

    return chunks