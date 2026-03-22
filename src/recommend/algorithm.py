from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List


@dataclass
class RecommendationItem:
    category: str
    name: str
    reason: str
    estimated_price: str


@dataclass
class RecommendationResult:
    # 외부 알고리즘은 dict 패키지 리스트를 줄 수 있음 (graph.node_chat_result가 dict 처리)
    recommendation_list: List[Any]
    total_estimated_budget: str


def _normalize_external_result(raw: Any) -> RecommendationResult:
    """외부 템플릿 반환값 → RecommendationResult (템플릿 폴더의 파일은 수정하지 않음)."""
    if isinstance(raw, RecommendationResult):
        return raw
    if isinstance(raw, dict):
        lst = (
            raw.get("recommendation_list")
            or raw.get("packages")
            or raw.get("data")
        )
        if not isinstance(lst, list):
            lst = []
        budget = (
            raw.get("total_estimated_budget")
            or raw.get("total_budget")
            or raw.get("totalEstimatedBudget")
            or ""
        )
        return RecommendationResult(
            recommendation_list=list(lst),
            total_estimated_budget=str(budget),
        )
    if isinstance(raw, list):
        return RecommendationResult(
            recommendation_list=raw,
            total_estimated_budget="",
        )
    raise TypeError(
        "외부 추천 알고리즘 반환 형식을 해석할 수 없습니다. "
        "dict(recommendation_list, total_estimated_budget) 또는 list(패키지들)를 기대합니다. "
        f"got: {type(raw)!r}"
    )


def rerank_and_filter(
    *,
    user_info: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> RecommendationResult:
    """
    1) 환경 변수로 지정한 **외부 폴더**(예: recommendation_algorithm)의 코드를 호출 — 그 안의 파일은 편집하지 않음.
    2) 미설정 시 NotImplementedError → graph.node_chat_result에서 LLM 폴백.

    환경 변수:
    - RECOMMENDATION_ALGORITHM_PATH: 예 /data/recommendation_algorithm (sys.path에 추가)
    - RECOMMENDATION_ALGORITHM_ENTRYPOINT (선택):
        - 비어 있거나 pipeline:run_full_pipeline → pipeline.run_full_pipeline(input_data, engine) (pipeline_adapter)
        - 그 외 모듈:함수 → (user_info, candidates) 직접 호출
    """
    from .external_loader import try_run_external

    raw = try_run_external(user_info=user_info, candidates=candidates)
    if raw is not None:
        return _normalize_external_result(raw)
    raise NotImplementedError(
        "외부 추천이 설정되지 않았습니다. "
        ".env에 RECOMMENDATION_ALGORITHM_PATH (필수) 및 필요 시 RECOMMENDATION_ALGORITHM_ENTRYPOINT 를 설정하세요."
    )
