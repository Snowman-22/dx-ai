from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from .products_repo import (
    Product,
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
) -> Dict[str, Any]:
    """
    product 테이블에서 키워드(product_name, category, brand)로 ILIKE 검색 후
    fetch_products_bundle_details 와 동일한 형태로 반환.
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

