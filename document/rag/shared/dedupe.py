"""去重：精确哈希去重 + MinHash / embedding 语义去重。"""

import hashlib
import math
from hashlib import md5
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


def _chunk_hash(chunk_text: str) -> str:
    return md5(chunk_text.encode("utf-8")).hexdigest()


def _hash_shingle(shingle: str, seed: int) -> int:
    h = hashlib.sha256(f"{seed}:{shingle}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big", signed=False)


def _shingle_set(text: str, k: int = 3) -> set:
    from document.rag.shared.data_cleaner import tokenize_text

    tokens = tokenize_text(text)
    if len(tokens) < k:
        return set(tokens)
    return set(" ".join(tokens[i : i + k]) for i in range(len(tokens) - k + 1))


def _minhash_signature(shingles: set, num_hashes: int = 64) -> Tuple[int, ...]:
    if not shingles:
        return tuple([0] * num_hashes)
    signature = []
    for seed in range(num_hashes):
        min_hash = min(_hash_shingle(shingle, seed) for shingle in shingles)
        signature.append(min_hash)
    return tuple(signature)


def _minhash_similarity(sig_a: Tuple[int, ...], sig_b: Tuple[int, ...]) -> float:
    if not sig_a or not sig_b or len(sig_a) != len(sig_b):
        return 0.0
    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / len(sig_a)


def _vector_norm(vector: List[float]) -> float:
    return math.sqrt(sum(v * v for v in vector))


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm = _vector_norm(a) * _vector_norm(b)
    return dot / norm if norm else 0.0


def semantic_dedupe(
    chunks: Iterable[Dict[str, Any]],
    key_field: str = "chunk_text",
    threshold: float = 0.85,
    embedding_fn: Optional[Callable[[str], List[float]]] = None,
    num_hashes: int = 64,
    min_shingle_size: int = 3,
) -> List[Dict[str, Any]]:
    """Semantic dedupe via embeddings or MinHash."""
    seen_signatures: List[Tuple[int, ...]] = []
    seen_embeddings: List[List[float]] = []
    out: List[Dict[str, Any]] = []

    for item in chunks:
        if not isinstance(item, dict):
            continue
        text = str(item.get(key_field, "") or "")
        if embedding_fn is not None:
            emb = embedding_fn(text)
            is_duplicate = any(_cosine_similarity(emb, existing) >= threshold for existing in seen_embeddings)
            if not is_duplicate:
                seen_embeddings.append(emb)
                out.append(item)
        else:
            shingles = _shingle_set(text, k=min_shingle_size)
            sig = _minhash_signature(shingles, num_hashes=num_hashes)
            is_duplicate = any(_minhash_similarity(sig, existing) >= threshold for existing in seen_signatures)
            if not is_duplicate:
                seen_signatures.append(sig)
                out.append(item)
    return out


def dedupe_chunks(
    chunks: Iterable[Dict[str, Any]],
    key_fields: Tuple[str, ...] = ("chunk_text",),
) -> List[Dict[str, Any]]:
    """Dedupe chunk dicts by key fields."""
    seen = set()
    out = []
    for item in chunks:
        key_vals = tuple(item.get(k, "") for k in key_fields)
        key_text = "||".join(map(str, key_vals))
        h = _chunk_hash(key_text)
        if h in seen:
            continue
        seen.add(h)
        out.append(item)
    return out
