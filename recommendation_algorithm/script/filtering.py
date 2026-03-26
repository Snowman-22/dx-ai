"""
filtering.py
- DB에서 품목 필터링으로 제품 조회
- 예산 배분 및 예산 필터링

DB 테이블:
  - product              : 제품 기본 정보
  - electronics_derived  : 가전 파생컬럼
  - furniture_derived    : 가구 파생컬럼
  - subscribe_price      : 구독 가격 정보
  - category_stats       : 카테고리별 컬럼 중앙값 (null 처리용)
  - category_price_stats : 카테고리별 대표 가격 (예산 필터링용)
"""

import pandas as pd
from sqlalchemy import text


# ================================================================== #
#  DB 조회
# ================================================================== #

def fetch_electronics(engine, needed_categories: list) -> pd.DataFrame:
    """
    품목 필터로 가전 제품 + 파생컬럼 조회
    product 테이블과 electronics_derived, subscribe_price 테이블 JOIN
    """
    placeholders = ", ".join([f":cat{i}" for i in range(len(needed_categories))])
    params = {f"cat{i}": cat for i, cat in enumerate(needed_categories)}

    query = text(f"""
        SELECT
            p.product_id,
            p.model_id,
            p.product_name          AS name,
            p.product_category      AS category,
            p.brand,
            p.original_price,
            p.discount_price        AS price,
            p.discount_rate         AS raw_discount_rate,
            p.is_subscribe,
            p.review_score,
            p.review_cnt,
            p.product_url,
            p.product_image_url,
            s.price                 AS subscription_price,
            s.contract_period_year,
            s.mandatory_period_year,
            s.visit_service_type,
            s.visit_cycle_month,
            e.discount_rate,
            e.value_score,
            e.popularity_score,
            e.review_reliability,
            e.has_ai,
            e.premium_line,
            e.color_series,
            e.design_style,
            e.size_grade,
            e.recommended_area,
            e.single_score,
            e.large_family_score,
            e.busy_worker_score,
            e.pet_score,
            e.energy_grade
        FROM product p
        JOIN electronics_derived e ON p.product_id = e.product_id
        LEFT JOIN (
            SELECT DISTINCT ON (product_id)
                product_id, price, contract_period_year,
                mandatory_period_year, visit_service_type, visit_cycle_month
            FROM subscribe_price
            ORDER BY product_id, price ASC
        ) s ON p.product_id = s.product_id
        WHERE p.product_category IN ({placeholders})
          AND p.category = 'APPLIANCE'
          AND p.discount_price IS NOT NULL
          AND (p.is_used IS NULL OR p.is_used = true)
    """)

    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params=params)

    return df


def fetch_furniture(engine, needed_categories: list) -> pd.DataFrame:
    """
    품목 필터로 가구 제품 + 파생컬럼 조회
    product 테이블과 furniture_derived 테이블 JOIN
    """
    placeholders = ", ".join([f":cat{i}" for i in range(len(needed_categories))])
    params = {f"cat{i}": cat for i, cat in enumerate(needed_categories)}

    query = text(f"""
        SELECT
            p.product_id,
            p.model_id,
            p.product_name          AS name,
            p.product_category      AS category,
            p.brand,
            p.original_price,
            p.discount_price        AS price,
            p.discount_rate         AS raw_discount_rate,
            p.review_score,
            p.review_cnt,
            p.product_url,
            p.product_image_url,
            f.discount_rate,
            f.material_grade,
            f.is_eco_friendly,
            f.maintenance_score,
            f.color_series,
            f.design_style,
            f.size_grade,
            f.bed_size,
            f.sofa_capacity,
            f.dining_capacity,
            f.single_score,
            f.newlywed_score,
            f.large_family_score,
            f.space_saving_score,
            f.pet_score,
            f.is_installation_included,
            f.delivery_score
        FROM product p
        JOIN furniture_derived f ON p.product_id = f.product_id
        WHERE p.product_category IN ({placeholders})
          AND p.category = 'furniture'
          AND p.discount_price IS NOT NULL
          AND (p.is_used IS NULL OR p.is_used = true)
    """)

    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params=params)

    return df


def fetch_category_medians(engine) -> dict:
    """
    category_stats 테이블에서 카테고리별 컬럼 중앙값 조회
    반환: {(category, col_name): median_value}
    """
    query = text("SELECT category, column_name, median_value FROM category_stats")
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)

    return {
        (row["category"], row["column_name"]): row["median_value"]
        for _, row in df.iterrows()
    }


def fetch_category_price_stats(engine) -> dict:
    """
    category_price_stats 테이블에서 품목별 대표 가격 조회
    반환: {category: median_price}
    """
    query = text("SELECT category, median_price FROM category_price_stats")
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)

    return dict(zip(df["category"], df["median_price"]))


# ================================================================== #
#  예산 배분 및 필터링
# ================================================================== #

def allocate_budget(needed_items: list, budget: int, category_price_stats: dict) -> dict:
    """
    품목별 대표 가격 비율로 총예산을 배분.
    각 카테고리에 최소 대표 가격만큼은 배정하여 저가 카테고리가 탈락하지 않도록 보장.
    """
    rep_prices = {item: category_price_stats.get(item, 0) for item in needed_items}
    total_rep = sum(rep_prices.values())

    if total_rep == 0:
        per_item = budget / len(needed_items) if needed_items else 0
        return {item: per_item for item in needed_items}

    allocated = {
        item: budget * (price / total_rep)
        for item, price in rep_prices.items()
    }

    # 최소 보장: 배정 예산이 대표 가격(중앙값) 미만이면 중앙값으로 올림
    # (예산 총합이 초과될 수 있으나, filter_by_budget의 110% 여유 + 구독 폴백으로 처리)
    for item in allocated:
        median = rep_prices.get(item, 0)
        if median > 0 and allocated[item] < median:
            allocated[item] = median

    return allocated


def filter_by_budget(df: pd.DataFrame, allocated_budget: dict, price_col: str = "price") -> pd.DataFrame:
    """
    각 제품의 카테고리별 배정 예산의 110% 이하인 제품만 남김.
    예산 초과지만 구독 가능한 제품은 유지하여 구독 추천 후보로 활용.
    """
    if df.empty:
        return df.copy()
    budget_series = df["category"].map(allocated_budget)
    within_budget = budget_series.isna() | (df[price_col] <= budget_series * 1.1)

    # 예산 초과지만 구독 가능한 제품은 유지 (단, 배정 예산의 3배 이하)
    if "is_subscribe" in df.columns:
        is_sub = df["is_subscribe"].fillna(False).astype(bool)
        sub_cap = budget_series.notna() & (df[price_col] <= budget_series * 3)
        mask = within_budget | (is_sub & sub_cap)
    else:
        mask = within_budget

    result = df[mask].copy()

    # 구독으로 살아남은 제품에 추천 플래그
    if "is_subscribe" in result.columns:
        budget_filtered = result["category"].map(allocated_budget)
        over_budget = budget_filtered.notna() & (result[price_col] > budget_filtered * 1.1)
        result["subscribe_recommended"] = over_budget & result["is_subscribe"].fillna(False).astype(bool)
    return result