from enum import Enum
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import re
import logging


class QueryType(str, Enum):
    FACTUAL_EXACT = "factual_exact"
    SEMANTIC_DOC = "semantic_doc"
    RELATIONAL = "relational"
    HYBRID = "hybrid"
    OPERATIONAL = "operational"
    GRAPH = "graph"


@dataclass
class ClassificationResult:
    query_type: QueryType
    confidence: float
    features: Dict[str, Any]
    entities: List[str]
    keywords: List[str]


class QueryClassifier:
    """Classify queries by rules, keywords, and optional LLM fallback."""
    
    def __init__(
        self,
        enable_llm_fallback: bool = False,
        llm_model: Optional[Any] = None,
        confidence_threshold: float = 0.7
    ):
        self._enable_llm = enable_llm_fallback
        self._llm = llm_model
        self._threshold = confidence_threshold
        self._logger = logging.getLogger("rag.classifier")
        
        self._exact_patterns = [
            r"^\d+$",
            r"^[A-Z]{2,4}-\d+$",
            r"^[A-Z]{2,4}\d+$",
            r"等于\s*[^\s]+",
            r"is\s+equal\s+to",
            r"编号[为是]\s*[^\s]+",
            r"ID[为是]\s*[^\s]+",
        ]
        
        self._relational_patterns = [
            r"统计",
            r"数量",
            r"总计",
            r"平均",
            r"最大",
            r"最小",
            r"count",
            r"sum",
            r"average",
            r"max",
            r"min",
            r"有多少",
            r"列出所有",
            r"全部列表",
        ]
        
        self._graph_patterns = [
            r"依赖",
            r"负责",
            r"依赖",
            r"影响",
            r"关联",
            r"上下游",
            r"关系",
            r"连接",
            r"路径",
            r"who\s+is\s+responsible",
            r"depends\s+on",
            r"related\s+to",
        ]
        
        self._semantic_patterns = [
            r'ç±»ä¼¼',
            r'ç¸ä¼¼',
            r'ç¸å³',
            r'æè¿°',
            r'è§è',
            r'æ¡ä¾',
            r'similar',
            r'related',
            r'description',
            r'å¦ä½',
            r'ææ ·',
            r'æ¹æ³',
        ]
    
    def classify(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ClassificationResult:
        features = self._extract_features(query)
        
        result = self._rule_based_classify(query, features)
        
        if result.confidence < self._threshold and self._enable_llm:
            llm_result = self._llm_classify(query, features)
            if llm_result.confidence > result.confidence:
                result = llm_result
        
        return result
    
    def _extract_features(self, query: str) -> Dict[str, Any]:
        features = {}
        
        features['length'] = len(query)
        features['word_count'] = len(query.split())
        
        features['has_digits'] = bool(re.search(r'\d+', query))
        features['has_id_pattern'] = bool(re.search(r'[A-Z]{2,4}-?\d+', query))
        features['has_chinese'] = bool(re.search(r'[\u4e00-\u9fff]', query))
        
        features['has_question_word'] = bool(
            re.search(r'(è°|ä»ä¹|åªé|å¦ä½|å¤å°|ä¸ºä»ä¹|which|what|who|where|how|why)', query, re.IGNORECASE)
        )
        
        entities = self._extract_entities(query)
        features['entity_count'] = len(entities)
        features['entities'] = entities
        
        keywords = self._extract_keywords(query)
        features['keywords'] = keywords
        
        return features
    
    def _extract_entities(self, query: str) -> List[str]:
        entities = []
        
        id_matches = re.findall(r'[A-Z]{2,4}-?\d+', query)
        entities.extend(id_matches)
        
        number_matches = re.findall(r'\b\d{4,}\b', query)
        entities.extend(number_matches)
        
        return entities
    
    def _extract_keywords(self, query: str) -> List[str]:
        stopwords = {
            "的", "了", "在", "是", "我", "有", "和", "就", "不", "人",
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "must", "of", "to", "in",
            "for", "on", "with", "at", "by", "from", "as", "into", "through",
        }
        
        words = re.findall(r'[\w\u4e00-\u9fff]+', query.lower())
        keywords = [w for w in words if w not in stopwords and len(w) > 1]
        
        return keywords
    
    def _rule_based_classify(
        self,
        query: str,
        features: Dict[str, Any]
    ) -> ClassificationResult:
        query_lower = query.lower()
        
        for pattern in self._exact_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return ClassificationResult(
                    query_type=QueryType.FACTUAL_EXACT,
                    confidence=0.95,
                    features=features,
                    entities=features.get('entities', []),
                    keywords=features.get('keywords', [])
                )
        
        for pattern in self._graph_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return ClassificationResult(
                    query_type=QueryType.GRAPH,
                    confidence=0.90,
                    features=features,
                    entities=features.get('entities', []),
                    keywords=features.get('keywords', [])
                )
        
        for pattern in self._relational_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return ClassificationResult(
                    query_type=QueryType.RELATIONAL,
                    confidence=0.85,
                    features=features,
                    entities=features.get('entities', []),
                    keywords=features.get('keywords', [])
                )
        
        for pattern in self._semantic_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return ClassificationResult(
                    query_type=QueryType.SEMANTIC_DOC,
                    confidence=0.80,
                    features=features,
                    entities=features.get('entities', []),
                    keywords=features.get('keywords', [])
                )
        
        if features.get('has_question_word') and features.get('entity_count', 0) > 0:
            return ClassificationResult(
                query_type=QueryType.HYBRID,
                confidence=0.75,
                features=features,
                entities=features.get('entities', []),
                keywords=features.get('keywords', [])
            )
        
        return ClassificationResult(
            query_type=QueryType.SEMANTIC_DOC,
            confidence=0.5,
            features=features,
            entities=features.get('entities', []),
            keywords=features.get('keywords', [])
        )
    
    def _llm_classify(
        self,
        query: str,
        features: Dict[str, Any]
    ) -> ClassificationResult:
        if self._llm is None:
            return ClassificationResult(
                query_type=QueryType.SEMANTIC_DOC,
                confidence=0.3,
                features=features,
                entities=[],
                keywords=[]
            )
        
        try:
            prompt = f"""Classify the following query into one of these types:
- factual_exact: Exact ID/number lookup
- semantic_doc: Document similarity search
- relational: Database aggregation/statistics
- graph: Relationship/path queries
- hybrid: Mixed query requiring multiple sources
- operational: Real-time status lookup

Query: {query}

Return only the classification type."""
            
            if hasattr(self._llm, 'ainvoke'):
                import asyncio
                response = asyncio.run(self._llm.ainvoke(prompt))
            else:
                response = self._llm.invoke(prompt)
            
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            for qt in QueryType:
                if qt.value in response_text.lower():
                    return ClassificationResult(
                        query_type=qt,
                        confidence=0.85,
                        features=features,
                        entities=features.get('entities', []),
                        keywords=features.get('keywords', [])
                    )
            
        except Exception as e:
            self._logger.warning(f"LLM classification failed: {e}")
        
        return ClassificationResult(
            query_type=QueryType.SEMANTIC_DOC,
            confidence=0.4,
            features=features,
            entities=features.get('entities', []),
            keywords=features.get('keywords', [])
        )
