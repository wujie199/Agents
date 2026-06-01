from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import logging

from core.domain.evidence import Evidence


class FusionStrategy(ABC):
    """Module docstring."""
    
    @abstractmethod
    def fuse(
        self,
        results: List[List[Evidence]],
        weights: Optional[List[float]] = None
    ) -> List[Evidence]:
        ...


class RRFFusion(FusionStrategy):
    """Reciprocal Rank Fusion: score(d) = sum 1/(k + rank_i(d))."""
    
    def __init__(self, k: int = 60):
        self._k = k
        self._logger = logging.getLogger("rag.fusion.rrf")
    
    def fuse(
        self,
        results: List[List[Evidence]],
        weights: Optional[List[float]] = None
    ) -> List[Evidence]:
        if not results:
            return []
        
        if len(results) == 1:
            return results[0]
        
        scores: Dict[str, float] = {}
        evidence_map: Dict[str, Evidence] = {}
        
        for list_idx, evidence_list in enumerate(results):
            weight = weights[list_idx] if weights and list_idx < len(weights) else 1.0
            
            for rank, evidence in enumerate(evidence_list, start=1):
                doc_id = evidence.id or f"evidence_{id(evidence)}"
                
                rrf_score = weight / (self._k + rank)
                
                if doc_id in scores:
                    scores[doc_id] += rrf_score
                else:
                    scores[doc_id] = rrf_score
                    evidence_map[doc_id] = evidence
        
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        fused = []
        for doc_id, score in sorted_docs:
            evidence = evidence_map[doc_id]
            
            merged_metadata = {**evidence.metadata, "rrf_score": score}
            
            fused.append(Evidence(
                id=evidence.id,
                content=evidence.content,
                score=score,
                source_type=evidence.source_type,
                metadata=merged_metadata,
            ))
        
        return fused


class WeightedFusion(FusionStrategy):
    """
    
    ç´æ¥å¯¹åæ°è¿è¡å ææ±åï¼éè¦åæ°å¯æ¯è¾
    """
    
    def __init__(self, normalize: bool = True):
        self._normalize = normalize
        self._logger = logging.getLogger("rag.fusion.weighted")
    
    def fuse(
        self,
        results: List[List[Evidence]],
        weights: Optional[List[float]] = None
    ) -> List[Evidence]:
        if not results:
            return []
        
        if len(results) == 1:
            return results[0]
        
        if weights is None:
            weights = [1.0 / len(results)] * len(results)
        
        if self._normalize:
            results = [self._normalize_scores(r) for r in results]
        
        scores: Dict[str, float] = {}
        evidence_map: Dict[str, Evidence] = {}
        
        for list_idx, evidence_list in enumerate(results):
            weight = weights[list_idx] if list_idx < len(weights) else 1.0
            
            for evidence in evidence_list:
                doc_id = evidence.id or f"evidence_{id(evidence)}"
                
                weighted_score = (evidence.score or 0.0) * weight
                
                if doc_id in scores:
                    scores[doc_id] += weighted_score
                else:
                    scores[doc_id] = weighted_score
                    evidence_map[doc_id] = evidence
        
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        fused = []
        for doc_id, score in sorted_docs:
            evidence = evidence_map[doc_id]
            
            merged_metadata = {**evidence.metadata, "fusion_score": score}
            
            fused.append(Evidence(
                id=evidence.id,
                content=evidence.content,
                score=score,
                source_type=evidence.source_type,
                metadata=merged_metadata,
            ))
        
        return fused
    
    def _normalize_scores(self, evidences: List[Evidence]) -> List[Evidence]:
        if not evidences:
            return []
        
        scores = [e.score or 0.0 for e in evidences]
        max_score = max(scores) if scores else 1.0
        min_score = min(scores) if scores else 0.0
        
        if max_score == min_score:
            return [
                Evidence(
                    id=e.id,
                    content=e.content,
                    score=1.0,
                    source_type=e.source_type,
                    metadata=e.metadata,
                )
                for e in evidences
            ]
        
        normalized = []
        for e in evidences:
            norm_score = ((e.score or 0.0) - min_score) / (max_score - min_score)
            normalized.append(Evidence(
                id=e.id,
                content=e.content,
                score=norm_score,
                source_type=e.source_type,
                metadata=e.metadata,
            ))
        
        return normalized


class CascadeFusion(FusionStrategy):
    """
    
    æé¡ºåºä½¿ç¨ååç«¯ç»æï¼åèä¼å?    """
    
    def __init__(self, deduplicate: bool = True):
        self._deduplicate = deduplicate
        self._logger = logging.getLogger("rag.fusion.cascade")
    
    def fuse(
        self,
        results: List[List[Evidence]],
        weights: Optional[List[float]] = None
    ) -> List[Evidence]:
        if not results:
            return []
        
        if len(results) == 1:
            return results[0]
        
        fused = []
        seen_ids = set()
        
        for evidence_list in results:
            for evidence in evidence_list:
                doc_id = evidence.id or f"evidence_{id(evidence)}"
                
                if self._deduplicate:
                    if doc_id in seen_ids:
                        continue
                    seen_ids.add(doc_id)
                
                fused.append(evidence)
        
        return fused


class FirstMatchFusion(FusionStrategy):
    """
    
    ä½¿ç¨ç¬¬ä¸ä¸ªéç©ºç»æé
    """
    
    def fuse(
        self,
        results: List[List[Evidence]],
        weights: Optional[List[float]] = None
    ) -> List[Evidence]:
        for evidence_list in results:
            if evidence_list:
                return evidence_list
        
        return []


class FusionFactory:
    """Module docstring."""
    
    _strategies = {
        "rrf": RRFFusion,
        "weighted": WeightedFusion,
        "cascade": CascadeFusion,
        "first_match": FirstMatchFusion,
    }
    
    @classmethod
    def create(
        cls,
        strategy: str,
        **kwargs
    ) -> FusionStrategy:
        if strategy not in cls._strategies:
            raise ValueError(f"Unknown fusion strategy: {strategy}")
        
        return cls._strategies[strategy](**kwargs)
    
    @classmethod
    def register(
        cls,
        name: str,
        strategy_cls: type
    ) -> None:
        cls._strategies[name] = strategy_cls
