"""
추천 리스트를 이미 받은 뒤 RECOMMEND_RAG에서 '다른 가전/가구', '유사 상품', '같은 가격대' 등
추가 추천 의도를 키워드로 분류하고 DB 조회를 수행한다.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.ext.asyncio import AsyncSession

from .product_details import (
    fetch_products_bundle_details,
    merge_product_bundles,
    search_products_by_keywords,
    search_products_by_name_query,
    search_products_by_price_range,
    semantic_search_products,
)


class FollowupRecommendKind(str, Enum):
    NONE = "none"
    MORE_APPLIANCE = "more_appliance"
    MORE_FURNITURE = "more_furniture"
    MORE_MIXED = "more_mixed"
    SIMILAR_PRODUCT = "similar_product"
    SAME_PRICE_BAND = "same_price_band"


# DB 키워드 검색에 쓸 canonical 샘플(가전/가구 카테고리 대표)
_APPLIANCE_KWS = [
    "TV", "에어컨", "세탁기", "건조기", "냉장고", "공기청정기", "청소기",
    "제습기", "가습기", "식기세척기", "전기레인지", "전자레인지", "밥솥",
]
_FURNITURE_KWS = [
    "소파", "침대", "매트리스", "식탁", "책상", "수납장", "옷장", "의자",
    "거실장", "선반", "화장대", "협탁", "책장",
]


@dataclass
class FollowupRecommendIntent:
    kind: FollowupRecommendKind
    reference_phrase: Optional[str] = None


def _qnorm(s: str) -> str:
    return (s or "").lower().replace(" ", "").replace("\n", "").replace("\t", "")


# 제품·상품·가전·가구·가격대 + 구어체(같은 걸/것/거)
# '거'는 '거리' 등 오인 방지: 거 뒤가 공백·조사·구두점·끝일 때만 (걸·것은 그대로)
_REF_TAIL = r"(?:제품|상품|가전|가구|가격대|걸|것|거(?=\s|[를을이가은는도,.!?]|$))"


def extract_reference_product_phrase(question: str) -> Optional[str]:
    """
    사용자 문장에서 기준 상품명 후보 추출.
    - 'X와/과 같은 (제품|상품|가전|가구|가격대|걸|것|거)'
    - 'X 같은 …' — '침대같은 가구를', '오븐같은 가전을', '침대같은 걸' 등
    - 앞부분이 길면 첫 구절만 (쉼표·+ 전까지)
    """
    s = (question or "").strip()
    if not s:
        return None

    m = re.search(
        rf"(.+?)(?:\s*과|\s*와)\s*같은\s*{_REF_TAIL}?",
        s,
    )
    if m:
        cand = m.group(1).strip()
        cand = re.sub(r"^(?:그리고|또|그럼|좀|제발)\s*", "", cand).strip()
        cand = _trim_trailing_intent_noise(cand)
        return cand if len(cand) >= 2 else None

    m = re.search(rf"(.+?)\s*같은\s*{_REF_TAIL}", s)
    if m:
        cand = m.group(1).strip()
        cand = _trim_trailing_intent_noise(cand)
        return cand if len(cand) >= 2 else None

    return None


def _trim_trailing_intent_noise(s: str) -> str:
    """기준 상품명 뒤에 붙은 요청어 제거."""
    return re.sub(
        r"(?:의)?\s*(?:같은\s*)?(?:가전|가구)?\s*(?:의)?\s*(?:같은\s*)?가격대.*$",
        "",
        s,
        flags=re.I,
    ).strip()


def classify_followup_recommend_intent(question: str) -> FollowupRecommendIntent:
    """
    의도 파악 규칙(키워드·공백 제거 문자열 기준, 우선순위 위→아래).

    - SAME_PRICE_BAND: '같은 가격대' / '비슷한 가격' + (다른|추천|제품) 류 + 기준 상품명 가능
    - SIMILAR_PRODUCT: '같은 제품/가전/가구', '비슷한', '유사', '대안', '대체', 'X와 같은'
    - MORE_APPLIANCE: '다른 가전', 또는 가전+더/추가/추천 조합
    - MORE_FURNITURE: '다른 가구', 또는 가구+더/추가/추천
    - MORE_MIXED: 가전+가구 동시 언급+리스트/더, 또는 '리스트 더'·'더 추천' 등
    """
    q = _qnorm(question)
    if not q:
        return FollowupRecommendIntent(FollowupRecommendKind.NONE)

    # 다음 페이지 의도와 충돌 방지 (순수 페이징만)
    if ("다음추천" in q or "다음패키지" in q or "더보여줘" == q or "더보여" == q) and "다른" not in q:
        return FollowupRecommendIntent(FollowupRecommendKind.NONE)

    ref = extract_reference_product_phrase(question)

    # 1) 같은 가격대
    if (
        ("같은가격대" in q or "비슷한가격" in q or "가격대가" in q)
        and any(x in q for x in ("다른", "추천", "제품", "가전", "가구", "비슷"))
    ):
        if ref:
            return FollowupRecommendIntent(
                FollowupRecommendKind.SAME_PRICE_BAND,
                reference_phrase=ref,
            )

    # 2) 유사·동일 류 (공백 제거 q에 '같은상품' 등이 붙어 있어도 잡힘)
    if (
        any(
            k in q
            for k in (
                "같은제품",
                "같은상품",
                "같은가전",
                "같은가구",
                "비슷한제품",
                "비슷한상품",
                "비슷한가전",
                "비슷한가구",
                "같은걸",
                "같은것",
                "같은거",
                "비슷한걸",
                "비슷한것",
                "비슷한거",
                "유사한",
                "유사제품",
                "대안",
                "대체",
            )
        )
        or re.search(r"(?:과|와)\s*같은", question or "")
    ):
        if ref:
            return FollowupRecommendIntent(
                FollowupRecommendKind.SIMILAR_PRODUCT,
                reference_phrase=ref,
            )

    # 3) 카테고리별 더보기 (가전+가구 동시 언급은 단일 카테고리보다 먼저 — '가전/가구 리스트' 등)
    if ("가전" in q and "가구" in q) and any(
        x in q for x in ("리스트", "더", "추가", "추천")
    ):
        return FollowupRecommendIntent(FollowupRecommendKind.MORE_MIXED)
    if ("리스트" in q or "목록" in q) and ("더" in q or "추가" in q) and "추천" in q:
        return FollowupRecommendIntent(FollowupRecommendKind.MORE_MIXED)

    if "다른가전" in q or ("가전" in q and "추천" in q and ("더" in q or "추가" in q or "다른" in q)):
        return FollowupRecommendIntent(FollowupRecommendKind.MORE_APPLIANCE)
    if "다른가구" in q or ("가구" in q and "추천" in q and ("더" in q or "추가" in q or "다른" in q)):
        return FollowupRecommendIntent(FollowupRecommendKind.MORE_FURNITURE)
    if "더추천" in q or "추가로추천" in q or "추가추천" in q:
        return FollowupRecommendIntent(FollowupRecommendKind.MORE_MIXED)

    return FollowupRecommendIntent(FollowupRecommendKind.NONE)


def _to_positive_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    try:
        n = int(v)
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def _ref_price_parts(
    prod: Dict[str, Any],
) -> tuple[Optional[int], Optional[int], Optional[int]]:
    """
    (discount_price, original_price, effective).
    effective는 DB의 가격대 필터와 동일하게 할인가가 있으면 할인가, 없으면 정가.
    """
    p = prod.get("product") or {}
    dp = _to_positive_int(p.get("discount_price"))
    op = _to_positive_int(p.get("original_price"))
    eff = dp if dp is not None else op
    return dp, op, eff


def _format_ref_price_caption(dp: Optional[int], op: Optional[int], eff: Optional[int]) -> str:
    """LLM 안내용: 할인가·정가가 모두 있으면 둘 다 표시, 밴드 산정 규칙은 DB와 동일(coalesce)."""
    if eff is None or eff <= 0:
        return ""
    parts: List[str] = []
    if dp is not None:
        parts.append(f"할인가 {dp:,}원")
    if op is not None:
        parts.append(f"정가 {op:,}원")
    if not parts:
        return f"가격 약 {eff:,}원"
    rule = "할인가 우선, 없으면 정가"
    return f"{' · '.join(parts)} (±15% 가격대는 {rule}과 동일하게 {eff:,}원 기준)"


async def run_followup_recommendation_queries(
    session: AsyncSession,
    intent: FollowupRecommendIntent,
    question_str: str,
) -> Tuple[Dict[str, Any], Optional[List[Dict[str, Any]]], str]:
    """
    Returns:
      (searched_products_bundle, semantic_results_or_none, instruction_note)
    """
    note = ""
    empty: Dict[str, Any] = {"products": []}
    sem: Optional[List[Dict[str, Any]]] = None

    if intent.kind == FollowupRecommendKind.MORE_APPLIANCE:
        bundle = await search_products_by_keywords(
            session, keywords=_APPLIANCE_KWS, limit=15
        )
        note = (
            "사용자는 이미 받은 추천 패키지 외에 **다른 가전** 후보를 원합니다. "
            "아래 DB 검색 결과(가전 카테고리 키워드 기반)만 근거로, 질문에 맞게 소개하세요."
        )
        return bundle, None, note

    if intent.kind == FollowupRecommendKind.MORE_FURNITURE:
        bundle = await search_products_by_keywords(
            session, keywords=_FURNITURE_KWS, limit=15
        )
        note = (
            "사용자는 **다른 가구** 후보를 원합니다. 아래 DB 검색 결과만 근거로 소개하세요."
        )
        return bundle, None, note

    if intent.kind == FollowupRecommendKind.MORE_MIXED:
        b1 = await search_products_by_keywords(
            session, keywords=_APPLIANCE_KWS[:8], limit=8
        )
        b2 = await search_products_by_keywords(
            session, keywords=_FURNITURE_KWS[:8], limit=8
        )
        bundle = merge_product_bundles(b1, b2)
        note = (
            "사용자는 **가전·가구를 더 넓게** 보고 싶어합니다. 아래 후보만 근거로 답하세요."
        )
        return bundle, None, note

    ref = intent.reference_phrase or ""
    if intent.kind in (
        FollowupRecommendKind.SIMILAR_PRODUCT,
        FollowupRecommendKind.SAME_PRICE_BAND,
    ):
        if not ref.strip():
            return empty, None, ""

    if intent.kind == FollowupRecommendKind.SIMILAR_PRODUCT:
        name_bundle = await search_products_by_name_query(
            session, query=ref, include_spec_and_prices=True
        )
        try:
            from openai import AsyncOpenAI

            oai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            emb = await oai.embeddings.create(
                model="text-embedding-3-small",
                input=f"{ref} 유사 대안 추천",
            )
            vec = emb.data[0].embedding
            sem = await semantic_search_products(
                session, query_embedding=vec, top_k=8
            )
        except Exception:
            sem = None
        note = (
            f"기준 표현: 「{ref}」. [키워드 검색 결과]=상품명 직접 검색, "
            "[유사 상품 검색 결과]=벡터 검색(있을 때). 제공 데이터만으로 비슷한 대안을 설명하세요."
        )
        return name_bundle, sem, note

    if intent.kind == FollowupRecommendKind.SAME_PRICE_BAND:
        ref_bundle = await search_products_by_name_query(
            session, query=ref, include_spec_and_prices=True
        )
        prods = ref_bundle.get("products") or []
        if not prods:
            return empty, None, ""
        first = prods[0]
        pid = (first.get("product") or {}).get("product_id")
        dp, op, price = _ref_price_parts(first)
        if price is None or price <= 0:
            note = "기준 상품의 가격 정보가 DB에 없어 같은 가격대 검색을 생략했습니다."
            return ref_bundle, None, note
        lo = max(0, int(price * 0.85))
        hi = int(price * 1.15)
        alt = await search_products_by_price_range(
            session,
            min_price=lo,
            max_price=hi,
            exclude_product_ids=[int(pid)] if pid is not None else None,
            limit=12,
        )
        bundle = merge_product_bundles(ref_bundle, alt)
        price_cap = _format_ref_price_caption(dp, op, price)
        note = (
            f"기준 상품: 「{ref}」({price_cap}). "
            f"같은 가격대(약 {lo:,}~{hi:,}원) **다른 제품** 후보만 근거로 비교·추천하세요."
        )
        return bundle, None, note

    return empty, None, ""
