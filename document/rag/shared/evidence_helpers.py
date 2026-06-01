from typing import Any, Dict, List, Optional

from core.domain.evidence import DegradedReason, Evidence, EvidenceBundle, SourceType


def source_type_to_str(value: Any) -> str:
    if isinstance(value, SourceType):
        return value.value
    return str(value)


def source_type_from_value(value: Any) -> SourceType:
    if isinstance(value, SourceType):
        return value
    if value is None:
        return SourceType.VECTOR
    return SourceType(str(value))


def degraded_from_value(value: Any) -> Optional[DegradedReason]:
    if value is None:
        return None
    if isinstance(value, DegradedReason):
        return value
    return DegradedReason(str(value))


def evidence_to_dict(evidence: Evidence) -> Dict[str, Any]:
    return {
        "id": evidence.id,
        "content": evidence.content,
        "source_type": source_type_to_str(evidence.source_type),
        "score": evidence.score,
        "citation": evidence.citation,
        "metadata": evidence.metadata,
    }


def evidence_from_dict(data: Dict[str, Any]) -> Evidence:
    return Evidence(
        id=data["id"],
        content=data.get("content", ""),
        source_type=source_type_from_value(data.get("source_type")),
        score=float(data.get("score", 0.0)),
        citation=data.get("citation"),
        metadata=data.get("metadata") or {},
    )


def bundle_to_cache_dict(bundle: EvidenceBundle) -> Dict[str, Any]:
    return {
        "evidences": [evidence_to_dict(e) for e in bundle.evidences],
        "plan": bundle.plan,
        "empty": bundle.empty,
        "degraded_reason": (
            bundle.degraded_reason.value
            if bundle.degraded_reason is not None
            else None
        ),
        "error_code": bundle.error_code,
    }


def bundle_from_cache_dict(data: Dict[str, Any]) -> EvidenceBundle:
    evidences = [evidence_from_dict(e) for e in data.get("evidences", [])]
    degraded = degraded_from_value(data.get("degraded_reason"))
    return EvidenceBundle(
        evidences=evidences,
        plan=data.get("plan"),
        empty=data.get("empty", len(evidences) == 0),
        degraded_reason=degraded,
        error_code=data.get("error_code"),
    )
