from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class RecommendationItem:
    category: str
    name: str
    reason: str
    estimated_price: str


@dataclass
class RecommendationResult:
    recommendation_list: list[RecommendationItem]
    total_estimated_budget: str


def rerank_and_filter(
    *,
    user_info: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> RecommendationResult:
    """
    TODO(ipynb 이식 지점):
    - candidates(벡터 검색/필터링으로 가져온 상품들)를 입력으로 받아
      ipynb의 점수 계산/제약조건/패키지 구성 로직을 적용해 최종 추천을 만든다.

    주의: 서버에서 import 가능한 .py 코드여야 하므로, ipynb 로직은 여기로 옮겨야 함.
    """
    raise NotImplementedError("ipynb 추천 알고리즘 로직을 여기에 이식하세요.")

