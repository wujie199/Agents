from typing import Dict, List


def rrf_merge_messages(
    ranked_lists: List[List[dict]],
    *,
    id_key: str = "message_id",
    k: int = 60,
) -> List[dict]:
    """Reciprocal Rank Fusion for L2 message dicts."""
    if not ranked_lists:
        return []
    if len(ranked_lists) == 1:
        return ranked_lists[0]

    scores: Dict[str, float] = {}
    message_map: Dict[str, dict] = {}

    for result_list in ranked_lists:
        for rank, message in enumerate(result_list, start=1):
            message_id = message.get(id_key)
            if not message_id:
                continue
            scores[message_id] = scores.get(message_id, 0.0) + 1.0 / (k + rank)
            if message_id not in message_map:
                message_map[message_id] = message

    ordered_ids = sorted(scores.keys(), key=lambda mid: scores[mid], reverse=True)
    return [message_map[mid] for mid in ordered_ids]
