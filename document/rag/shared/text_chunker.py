import re
from typing import List


def split_text_into_chunks(
    text: str, chunk_size: int = 500, chunk_overlap: int = 50
) -> List[str]:
    """Split long text into overlapping chunks for RAG indexing."""
    if not text:
        return []

    paragraphs = re.split(r"\n\s*\n", text)
    chunks: List[str] = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(para) > chunk_size:
            sentences = re.split(r"([。！？.!?])", para)
            combined_sentences = []
            for i in range(0, len(sentences) - 1, 2):
                combined_sentences.append(sentences[i] + sentences[i + 1])
            if len(sentences) % 2 != 0 and sentences[-1]:
                combined_sentences.append(sentences[-1])

            for sentence in combined_sentences:
                if len(current_chunk) + len(sentence) <= chunk_size:
                    current_chunk += " " + sentence if current_chunk else sentence
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    overlap_start = max(0, len(current_chunk) - chunk_overlap)
                    current_chunk = (
                        current_chunk[overlap_start:] + " " + sentence
                        if current_chunk
                        else sentence
                    )
                    while len(current_chunk) > chunk_size:
                        chunks.append(current_chunk[:chunk_size].strip())
                        current_chunk = current_chunk[chunk_size - chunk_overlap :]
        else:
            if len(current_chunk) + len(para) <= chunk_size:
                current_chunk += "\n\n" + para if current_chunk else para
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                overlap_start = max(0, len(current_chunk) - chunk_overlap)
                current_chunk = (
                    current_chunk[overlap_start:] + "\n\n" + para
                    if current_chunk
                    else para
                )

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks
