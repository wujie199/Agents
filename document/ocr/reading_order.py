# -*- coding: utf-8 -*-
"""阅读顺序：XY-Cut 多栏排序 + layout order 优先。"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


def _bbox_from_box(box: Dict[str, Any]) -> Tuple[float, float, float, float]:
    coord = box.get("coordinate") or [0, 0, 0, 0]
    if len(coord) < 4:
        return (0.0, 0.0, 0.0, 0.0)
    return (float(coord[0]), float(coord[1]), float(coord[2]), float(coord[3]))


def _center_x(bbox: Tuple[float, float, float, float]) -> float:
    return (bbox[0] + bbox[2]) / 2


def _center_y(bbox: Tuple[float, float, float, float]) -> float:
    return (bbox[1] + bbox[3]) / 2


def _xy_cut_columns(
    boxes: List[Dict[str, Any]],
    *,
    page_width: float | None = None,
) -> List[List[Dict[str, Any]]]:
    """按 x 中心聚类为栏（简易 XY-Cut）。"""
    if len(boxes) <= 1:
        return [boxes]

    bboxes = [_bbox_from_box(b) for b in boxes]
    if page_width is None:
        page_width = max(b[2] for b in bboxes) if bboxes else 1000.0

    # 两栏判定：x 中心存在明显间隙
    centers = sorted(_center_x(b) for b in bboxes)
    if len(centers) < 4:
        return [boxes]

    mid = page_width / 2
    left = [b for b, bb in zip(boxes, bboxes) if _center_x(bb) < mid]
    right = [b for b, bb in zip(boxes, bboxes) if _center_x(bb) >= mid]

    if not left or not right:
        return [boxes]
    # 若左右数量过于失衡，可能不是双栏
    ratio = min(len(left), len(right)) / max(len(left), len(right))
    if ratio < 0.15:
        return [boxes]
    return [left, right]


def sort_boxes_reading_order(
    boxes: List[Dict[str, Any]],
    *,
    page_width: float | None = None,
    page_height: float | None = None,
) -> List[Dict[str, Any]]:
    """
    阅读顺序：
    1. 若 layout 提供 order，优先 order
    2. 否则 XY-Cut 分栏 → 栏内 (y, x) 排序
    """
    if not boxes:
        return []

    has_order = any(b.get("order") is not None for b in boxes)
    if has_order:
        def sort_key(box: Dict[str, Any]) -> tuple:
            order = box.get("order")
            coord = box.get("coordinate") or [0, 0, 0, 0]
            if order is None:
                return (1, coord[1] if len(coord) > 1 else 0, coord[0] if coord else 0)
            return (0, order)

        return sorted(boxes, key=sort_key)

    if page_width is None:
        bboxes = [_bbox_from_box(b) for b in boxes]
        page_width = max((bb[2] for bb in bboxes), default=1000.0)

    columns = _xy_cut_columns(boxes, page_width=page_width)
    ordered: List[Dict[str, Any]] = []
    for col in columns:
        col_sorted = sorted(
            col,
            key=lambda b: (_center_y(_bbox_from_box(b)), _center_x(_bbox_from_box(b))),
        )
        ordered.extend(col_sorted)

    # 重写 order 便于下游一致
    result: List[Dict[str, Any]] = []
    for i, box in enumerate(ordered):
        nb = dict(box)
        nb["order"] = i
        result.append(nb)
    return result
