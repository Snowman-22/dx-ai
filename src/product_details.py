from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .products_repo import ProductEntity, ProductSpecEntity, SubscribePriceEntity


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

