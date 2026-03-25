from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional

import re

from sqlalchemy import and_, case, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from .products_repo import (
    ProductEntity,
    ProductHit,
    ProductSpecEntity,
    ProductTagsEntity,
    SubscribePriceEntity,
    vector_search_products,
)


async def fetch_products_bundle_details(
    session: AsyncSession,
    *,
    model_ids: Iterable[str],
) -> Dict[str, Any]:
    """
    model_id 목록으로 product/product_spec/subscribe_price를 조회해 묶어서 반환.

    반환 형태(요약):
    {
      "products": [
        {
          "model_id": "...",
          "product": {...},
          "spec": {...} | None,
          "subscribe_prices": [{...}, ...]
        }
      ]
    }
    """
    model_id_list = [m for m in (str(x).strip() for x in model_ids) if m]
    if not model_id_list:
        return {"products": []}

    # 1) product 조회
    prod_rows = (
        await session.execute(
            select(ProductEntity).where(ProductEntity.model_id.in_(model_id_list))
        )
    ).scalars().all()

    by_product_id: dict[int, ProductEntity] = {p.product_id: p for p in prod_rows}
    by_model_id: dict[str, ProductEntity] = {p.model_id: p for p in prod_rows}

    product_ids = list(by_product_id.keys())
    if not product_ids:
        return {"products": []}

    # 2) spec 조회
    spec_rows = (
        await session.execute(
            select(ProductSpecEntity).where(ProductSpecEntity.product_id.in_(product_ids))
        )
    ).scalars().all()
    spec_by_product_id: dict[int, ProductSpecEntity] = {s.product_id: s for s in spec_rows}

    # 3) subscribe_price 조회(1:N)
    sub_rows = (
        await session.execute(
            select(SubscribePriceEntity).where(
                SubscribePriceEntity.product_id.in_(product_ids)
            )
        )
    ).scalars().all()
    subs_by_product_id: dict[int, list[SubscribePriceEntity]] = defaultdict(list)
    for s in sub_rows:
        subs_by_product_id[s.product_id].append(s)

    # 4) 응답 조립(model_id 순서를 최대한 유지)
    out: List[Dict[str, Any]] = []
    for mid in model_id_list:
        p = by_model_id.get(mid)
        if not p:
            continue
        spec = spec_by_product_id.get(p.product_id)
        subs = subs_by_product_id.get(p.product_id, [])

        out.append(
            {
                "model_id": p.model_id,
                "product": {
                    "product_id": p.product_id,
                    "model_id": p.model_id,
                    "product_name": p.product_name,
                    "category": p.category,
                    "product_category": p.product_category,
                    "brand": p.brand,
                    "original_price": p.original_price,
                    "discount_rate": p.discount_rate,
                    "discount_price": p.discount_price,
                    "is_subscribe": p.is_subscribe,
                    "review_score": p.review_score,
                    "review_cnt": p.review_cnt,
                    "product_url": p.product_url,
                    "product_image_url": p.product_image_url,
                },
                "spec": (
                    None
                    if spec is None
                    else {
                        "width": spec.width,
                        "height": spec.height,
                        "depth": spec.depth,
                    }
                ),
                "subscribe_prices": [
                    {
                        "month": s.month,
                        "price": s.price,
                        "contract_period_year": s.contract_period_year,
                        "mandatory_period_year": s.mandatory_period_year,
                        "visit_service_type": s.visit_service_type,
                        "visit_cycle_month": s.visit_cycle_month,
                    }
                    for s in subs
                ],
            }
        )

    return {"products": out}


def _safe_like_fragment(s: str) -> str:
    """ILIKE 패턴에 쓸 사용자 입력에서 % _ \\ 를 제거해 과도한 매칭·이스케이프 문제를 줄임."""
    return re.sub(r"[%_\\]+", " ", (s or "").strip()).strip()


# 최소 불용어(과도 제거 시 fallback용)
_REQUEST_WORDS_MINIMAL: frozenset[str] = frozenset({
    "알려줘", "알려주세요", "보여줘", "보여주세요", "찾아줘", "찾아주세요",
    "상세정보", "알려줄래", "알려줄래요",
})

# 검색 전처리용 불용어(요청·질문·완충어 등). 상품명 토큰과 겹치면 _is_protected_product_token 으로 보존.
_REQUEST_WORDS: frozenset[str] = frozenset({
    *_REQUEST_WORDS_MINIMAL,
    "알려주시", "보여주시", "찾아주시",
    "추천해줘", "추천해주세요", "추천", "추천좀",
    "설명해줘", "설명해주세요", "설명", "설명좀",
    "정리해줘", "정리해주세요",
    "알고싶어", "알고싶어요", "궁금", "궁금해", "궁금합니다",
    "부탁", "부탁해", "부탁해요", "부탁드려요", "부탁드립니다",
    "해줘", "해주세요", "해주시", "해줄래", "해줄래요",
    "주세요", "주시", "드려요", "드립니다", "합니다", "해요", "세요",
    "상세", "정보", "스펙", "사양", "규격", "치수", "크기", "크기가", "사이즈",
    "가격", "비교", "문의", "질문",
    "후기", "리뷰", "평", "평점",
    "뭐야", "뭐예요", "뭔가요", "무엇", "어떤", "어때", "어떤가", "어떻게", "왜",
    "있나", "있나요", "있어", "있어요", "맞나", "맞나요", "맞아", "맞아요",
    "좀", "약간", "대충", "그냥", "일단", "한번", "혹시", "정도", "정도로",
    "제가", "저는", "나는", "우리",
    "그리고", "또한", "또", "근데", "그런데", "그래서",
    "please", "tell", "me", "about", "the", "a", "an", "what", "how", "which", "can", "you",
    "vs",
})

_WEAK_SEARCH_TOKENS: frozenset[str] = frozenset({
    "일반", "기본", "보급형", "표준형", "스탠드", "스탠드형", "벽걸이", "형",
})


def _is_protected_product_token(t: str) -> bool:
    """
    모델명·규격·브랜드 약어 등으로 보이는 토큰은 불용어에서 제외하지 않음.
    (예: SS, QNED, UHD, 2in1, LG)
    """
    if len(t) < 2 or len(t) > 16:
        return False
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.\-+]{0,15}", t):
        return False
    if any(c.isdigit() for c in t):
        return True
    if t.isupper():
        return True
    return False


def _strip_korean_josa(token: str) -> str:
    """토큰 끝 조사 제거 (예: SS에 -> SS)."""
    t = (token or "").strip()
    if len(t) <= 1:
        return t
    return re.sub(
        r"(에서|에게서|에게|으로|로|에|의|을|를|은|는|이|가|도|와|과|랑)$",
        "",
        t,
    ).strip()


def _strip_sentence_noise(s: str) -> str:
    """구두점·문장 꼬리·영문 요청구 제거(토큰 단위 불용어 전)."""
    x = (s or "").strip()
    if not x:
        return ""
    x = re.sub(r"[\(\)\[\]\{\}<>]", " ", x)
    x = re.sub(r"[?!.,，、]+", " ", x)
    x = re.sub(r"\b(?:please|tell\s+me\s+about|tell\s+me)\b", " ", x, flags=re.I)
    x = re.sub(r"(에\s*대한|에\s*대해|대해서|관련(?:된)?)\s*", " ", x)
    x = re.sub(
        r"(?:알려줘|알려주세요|알려주시|알려줄래|알려줄래요|"
        r"보여줘|보여주세요|보여주시|"
        r"찾아줘|찾아주세요|찾아주시|"
        r"설명해줘|설명해주세요|"
        r"해줘|해주세요|해주시|해줄래|해줄래요|"
        r"주세요|주시|드려요|드립니다|합니다|해요)\s*$",
        "",
        x,
    ).strip()
    return re.sub(r"\s+", " ", x).strip()


def _filter_tokens_with_stopwords(
    toks: list[str],
    stopwords: frozenset[str],
) -> list[str]:
    out: list[str] = []
    for tok in toks:
        base = _strip_korean_josa(tok)
        if not base:
            continue
        if _is_protected_product_token(base):
            out.append(base)
            continue
        if base in stopwords:
            continue
        out.append(base)
    return out


def _normalize_query_segment(seg: str) -> str:
    """
    검색용 세그먼트 정리:
    - 문장 꼬리·메타 표현 제거
    - 불용어 토큰 제거(모델/규격 토큰은 보존)
    - 과도하게 비면 최소 불용어만 적용해 fallback
    """
    s = _strip_sentence_noise(seg)
    if not s:
        return ""
    raw_toks = [t for t in re.split(r"\s+", s) if t]
    cleaned = _filter_tokens_with_stopwords(raw_toks, _REQUEST_WORDS)
    joined = " ".join(cleaned).strip()
    if len(joined) >= 2 and (len(cleaned) >= 2 or len(joined) >= 4):
        return joined
    # 너무 많이 지워짐 → 최소 불용어만
    cleaned_min = _filter_tokens_with_stopwords(raw_toks, _REQUEST_WORDS_MINIMAL)
    jm = " ".join(cleaned_min).strip()
    if len(jm) >= 2:
        return jm
    return " ".join(raw_toks).strip()


def _split_product_name_segments(query: str) -> list[str]:
    """
    사용자 입력을 여러 상품 후보로 나눔 (침대 + 매트리스 등).
    """
    q = (query or "").strip()
    if not q:
        return []
    parts = re.split(r"[\n,，、+＋]+", q)
    out: list[str] = []
    for p in parts:
        t = p.strip()
        if len(t) >= 2:
            out.append(t)
    return out if out else ([q] if len(q) >= 2 else [])


def _extract_search_tokens(seg: str) -> list[str]:
    base = _normalize_query_segment(seg)
    if not base:
        return []
    parts = [
        _strip_korean_josa(t)
        for t in re.split(r"[\s/|]+", base)
        if t and len(t.strip()) >= 2
    ]
    cleaned: list[str] = []
    seen: set[str] = set()
    for tok in parts:
        t = re.sub(r"[^0-9A-Za-z가-힣]+", "", tok).strip()
        if len(t) < 2:
            continue
        if t in _REQUEST_WORDS or t in _REQUEST_WORDS_MINIMAL:
            continue
        if t in seen:
            continue
        seen.add(t)
        cleaned.append(t)
    strong = [t for t in cleaned if t not in _WEAK_SEARCH_TOKENS]
    return strong if strong else cleaned


def _product_entity_to_bundle_item_minimal(p: ProductEntity) -> Dict[str, Any]:
    """리뷰 전용: product 행만 — spec / subscribe_price 조회 없음."""
    return {
        "model_id": p.model_id,
        "product": {
            "product_id": p.product_id,
            "model_id": p.model_id,
            "product_name": p.product_name,
            "category": p.category,
            "product_category": p.product_category,
            "brand": p.brand,
            "original_price": p.original_price,
            "discount_rate": p.discount_rate,
            "discount_price": p.discount_price,
            "is_subscribe": p.is_subscribe,
            "review_score": p.review_score,
            "review_cnt": p.review_cnt,
            "product_url": p.product_url,
            "product_image_url": p.product_image_url,
        },
        "spec": None,
        "subscribe_prices": [],
    }


async def search_products_by_name_query(
    session: AsyncSession,
    *,
    query: str,
    limit_per_segment: int = 8,
    max_segments: int = 8,
    include_spec_and_prices: bool = True,
) -> Dict[str, Any]:
    """
    사용자가 입력한 상품명(자유 텍스트)으로 product 테이블 검색 후 bundle 상세 반환.
    product_name / brand / category / product_category 에 ILIKE.
    세그먼트(+, 쉼표 등)별로 검색해 여러 상품을 한 번에 묶을 수 있음.
    include_spec_and_prices=False 이면 product 행만(spec/subscribe_price 쿼리 없음) — 리뷰 RAG용.
    """
    segments = _split_product_name_segments(query)
    if not segments:
        return {"products": []}
    segments = segments[:max_segments]

    collected: list[str] = []
    seen: set[str] = set()

    async def _rows_for_segment(seg: str) -> list:
        norm_seg = _normalize_query_segment(seg)
        safe = _safe_like_fragment(norm_seg)
        tokens = _extract_search_tokens(seg)[:6]

        phrase_conditions = []
        if len(safe) >= 2:
            like = f"%{safe}%"
            phrase_conditions.extend([
                ProductEntity.product_name.ilike(like),
                ProductEntity.brand.ilike(like),
                ProductEntity.category.ilike(like),
                ProductEntity.product_category.ilike(like),
            ])

        compact = _safe_like_fragment(re.sub(r"\s+", "", norm_seg))
        if len(compact) >= 2 and compact != safe.replace(" ", ""):
            compact_like = f"%{compact}%"
            phrase_conditions.append(
                func.replace(ProductEntity.product_name, " ", "").ilike(compact_like)
            )

        if phrase_conditions:
            res = await session.execute(
                select(ProductEntity).where(or_(*phrase_conditions)).limit(limit_per_segment)
            )
            rows = list(res.scalars().all())
            if rows:
                return rows

        if not tokens:
            return []

        token_match_exprs = []
        score_parts = []
        for tok in tokens:
            frag = _safe_like_fragment(tok)
            if len(frag) < 2:
                continue
            pn = ProductEntity.product_name.ilike(f"%{frag}%")
            brand = ProductEntity.brand.ilike(f"%{frag}%")
            category = ProductEntity.category.ilike(f"%{frag}%")
            product_category = ProductEntity.product_category.ilike(f"%{frag}%")
            token_match_exprs.append(or_(pn, brand, category, product_category))
            score_parts.extend([
                case((pn, 6), else_=0),
                case((brand, 4), else_=0),
                case((category, 3), else_=0),
                case((product_category, 3), else_=0),
            ])

        if not token_match_exprs:
            return []

        score_expr = score_parts[0]
        for part in score_parts[1:]:
            score_expr = score_expr + part
        match_count_expr = case((token_match_exprs[0], 1), else_=0)
        for expr in token_match_exprs[1:]:
            match_count_expr = match_count_expr + case((expr, 1), else_=0)
        min_match = 1 if len(token_match_exprs) == 1 else min(2, len(token_match_exprs))
        stmt = (
            select(ProductEntity)
            .where(match_count_expr >= min_match)
            .order_by(score_expr.desc(), ProductEntity.product_name.asc())
            .limit(limit_per_segment)
        )
        res2 = await session.execute(stmt)
        return list(res2.scalars().all())

    for seg in segments:
        for p in await _rows_for_segment(seg):
            mid = (p.model_id or "").strip()
            if mid and mid not in seen:
                seen.add(mid)
                collected.append(mid)

    if not collected:
        return {"products": []}

    if not include_spec_and_prices:
        res = await session.execute(
            select(ProductEntity).where(ProductEntity.model_id.in_(collected))
        )
        by_mid = {r.model_id: r for r in res.scalars().all()}
        ordered = [by_mid[mid] for mid in collected if mid in by_mid]
        return {
            "products": [_product_entity_to_bundle_item_minimal(p) for p in ordered],
        }

    return await fetch_products_bundle_details(session, model_ids=collected)


def merge_product_bundles(*bundles: Dict[str, Any]) -> Dict[str, Any]:
    """여러 {products: [...]} 번들을 model_id 기준으로 병합."""
    seen: set[str] = set()
    out: List[Dict[str, Any]] = []
    for b in bundles:
        if not isinstance(b, dict):
            continue
        for item in (b.get("products") or []):
            if not isinstance(item, dict):
                continue
            mid = str(item.get("model_id") or "").strip()
            if not mid or mid in seen:
                continue
            seen.add(mid)
            out.append(item)
    return {"products": out}


async def search_products_by_price_range(
    session: AsyncSession,
    *,
    min_price: int,
    max_price: int,
    exclude_product_ids: Optional[List[int]] = None,
    limit: int = 12,
) -> Dict[str, Any]:
    """
    discount_price 우선, 없으면 original_price 로 가격 잡아 min~max 범위 필터.
    """
    if min_price > max_price:
        min_price, max_price = max_price, min_price
    price_expr = func.coalesce(
        ProductEntity.discount_price, ProductEntity.original_price
    )
    stmt = (
        select(ProductEntity)
        .where(price_expr.isnot(None))
        .where(price_expr >= min_price)
        .where(price_expr <= max_price)
    )
    if exclude_product_ids:
        stmt = stmt.where(ProductEntity.product_id.notin_(exclude_product_ids))
    stmt = stmt.limit(limit)
    rows = list((await session.execute(stmt)).scalars().all())
    model_ids = [p.model_id for p in rows if p.model_id]
    if not model_ids:
        return {"products": []}
    return await fetch_products_bundle_details(session, model_ids=model_ids)


# ── 리뷰 태그 조회 ──────────────────────────────────────────────────


async def fetch_product_review_tags(
    session: AsyncSession,
    *,
    product_ids: Iterable[int],
) -> Dict[int, List[str]]:
    """
    product_tags 테이블에서 상품별 리뷰 태그를 조회.
    반환: {product_id: ["태그1", "태그2", ...]}
    """
    pid_list = list(product_ids)
    if not pid_list:
        return {}

    rows = (
        await session.execute(
            select(ProductTagsEntity).where(
                ProductTagsEntity.product_id.in_(pid_list)
            )
        )
    ).scalars().all()

    return {r.product_id: (r.tags or []) for r in rows}


# ── 상품 비교 (여러 상품의 스펙/가격을 한 번에) ─────────────────────


async def fetch_products_comparison(
    session: AsyncSession,
    *,
    model_ids: Iterable[str],
) -> List[Dict[str, Any]]:
    """
    model_id 목록으로 product + spec + subscribe_price + review_tags 를
    비교 가능한 형태로 묶어 반환.
    """
    details = await fetch_products_bundle_details(session, model_ids=model_ids)
    products = details.get("products") or []
    if not products:
        return []

    pid_list = [
        p["product"]["product_id"]
        for p in products
        if isinstance(p, dict) and isinstance(p.get("product"), dict)
    ]
    tags_map = await fetch_product_review_tags(session, product_ids=pid_list)

    for item in products:
        prod = item.get("product") or {}
        pid = prod.get("product_id")
        item["review_tags"] = tags_map.get(pid, [])

    return products


# ── pgvector 시맨틱 검색 래퍼 ────────────────────────────────────────


async def semantic_search_products(
    session: AsyncSession,
    *,
    query_embedding: List[float],
    top_k: int = 10,
    category: str | None = None,
) -> List[Dict[str, Any]]:
    """
    pgvector cosine 검색 → 매칭된 product_id로 상세 정보(spec/subscribe)까지 묶어 반환.
    """
    hits: list[ProductHit] = await vector_search_products(
        session, query_embedding, top_k=top_k, category=category,
    )
    if not hits:
        return []

    hit_names = [h.name for h in hits]
    hit_scores = {h.name: h.score for h in hits}

    prod_rows = (
        await session.execute(
            select(ProductEntity).where(
                or_(*[ProductEntity.product_name.ilike(f"%{n}%") for n in hit_names])
            ).limit(top_k)
        )
    ).scalars().all()

    if not prod_rows:
        return [
            {
                "product_name": h.name,
                "category": h.category,
                "brand": h.brand,
                "price": h.price,
                "similarity_score": round(h.score, 4),
            }
            for h in hits
        ]

    model_ids = [p.model_id for p in prod_rows if p.model_id]
    if model_ids:
        details = await fetch_products_bundle_details(session, model_ids=model_ids)
        products = details.get("products") or []
        for item in products:
            pname = (item.get("product") or {}).get("product_name", "")
            item["similarity_score"] = hit_scores.get(pname, 0)
        return products

    return []


async def search_products_by_keywords(
    session: AsyncSession,
    *,
    keywords: List[str],
    limit: int = 10,
    include_spec_and_prices: bool = True,
) -> Dict[str, Any]:
    """
    product 테이블에서 키워드(product_name, category, brand)로 ILIKE 검색 후
    fetch_products_bundle_details 와 동일한 형태로 반환.
    include_spec_and_prices=False 이면 product 행만 — 리뷰 RAG 보조 검색용.
    """
    if not keywords:
        return {"products": []}

    conditions = []
    for kw in keywords:
        kw = kw.strip()
        if not kw:
            continue
        like = f"%{kw}%"
        conditions.append(ProductEntity.product_name.ilike(like))
        conditions.append(ProductEntity.category.ilike(like))
        conditions.append(ProductEntity.product_category.ilike(like))
        conditions.append(ProductEntity.brand.ilike(like))

    if not conditions:
        return {"products": []}

    prod_rows = (
        await session.execute(
            select(ProductEntity).where(or_(*conditions)).limit(limit)
        )
    ).scalars().all()

    if not prod_rows:
        return {"products": []}

    if not include_spec_and_prices:
        return {
            "products": [_product_entity_to_bundle_item_minimal(p) for p in prod_rows],
        }

    # 기존 bundle_details 와 동일한 조립 로직 재사용
    by_product_id: dict[int, ProductEntity] = {p.product_id: p for p in prod_rows}
    product_ids = list(by_product_id.keys())

    spec_rows = (
        await session.execute(
            select(ProductSpecEntity).where(ProductSpecEntity.product_id.in_(product_ids))
        )
    ).scalars().all()
    spec_by_pid: dict[int, ProductSpecEntity] = {s.product_id: s for s in spec_rows}

    sub_rows = (
        await session.execute(
            select(SubscribePriceEntity).where(
                SubscribePriceEntity.product_id.in_(product_ids)
            )
        )
    ).scalars().all()
    subs_by_pid: dict[int, list[SubscribePriceEntity]] = defaultdict(list)
    for s in sub_rows:
        subs_by_pid[s.product_id].append(s)

    out: List[Dict[str, Any]] = []
    for p in prod_rows:
        spec = spec_by_pid.get(p.product_id)
        subs = subs_by_pid.get(p.product_id, [])
        out.append(
            {
                "model_id": p.model_id,
                "product": {
                    "product_id": p.product_id,
                    "model_id": p.model_id,
                    "product_name": p.product_name,
                    "category": p.category,
                    "product_category": p.product_category,
                    "brand": p.brand,
                    "original_price": p.original_price,
                    "discount_rate": p.discount_rate,
                    "discount_price": p.discount_price,
                    "is_subscribe": p.is_subscribe,
                    "review_score": p.review_score,
                    "review_cnt": p.review_cnt,
                    "product_url": p.product_url,
                    "product_image_url": p.product_image_url,
                },
                "spec": (
                    None
                    if spec is None
                    else {
                        "width": spec.width,
                        "height": spec.height,
                        "depth": spec.depth,
                    }
                ),
                "subscribe_prices": [
                    {
                        "month": s.month,
                        "price": s.price,
                        "contract_period_year": s.contract_period_year,
                        "mandatory_period_year": s.mandatory_period_year,
                        "visit_service_type": s.visit_service_type,
                        "visit_cycle_month": s.visit_cycle_month,
                    }
                    for s in subs
                ],
            }
        )

    return {"products": out}
