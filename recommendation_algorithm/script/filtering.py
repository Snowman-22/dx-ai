"""
filtering.py
- DB에서 품목 필터링으로 제품 조회
- 예산 배분 및 예산 필터링

DB 테이블:
  - product              : 제품 기본 정보
  - electronics_derived  : 가전 파생컬럼
  - furniture_derived    : 가구 파생컬럼
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
    product 테이블과 electronics_derived 테이블 JOIN
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
        WHERE p.product_category IN ({placeholders})
          AND p.category = 'APPLIANCE'
          AND p.discount_price IS NOT NULL
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
    품목별 대표 가격 비율로 총예산을 배분
    category_price_stats: {category: median_price}
    반환: {category: allocated_budget}
    """
    rep_prices = {item: category_price_stats.get(item, 0) for item in needed_items}
    total_rep = sum(rep_prices.values())

    if total_rep == 0:
        per_item = budget / len(needed_items) if needed_items else 0
        return {item: per_item for item in needed_items}

    return {
        item: budget * (price / total_rep)
        for item, price in rep_prices.items()
    }


def filter_by_budget(df: pd.DataFrame, allocated_budget: dict, price_col: str = "price") -> pd.DataFrame:
    """
    각 제품의 카테고리별 배정 예산의 110% 이하인 제품만 남김
    """
    def is_within_budget(row):
        cat = row.get("category")
        budget_for_cat = allocated_budget.get(cat)
        if budget_for_cat is None:
            return True
        return row[price_col] <= budget_for_cat * 1.1

    mask = df.apply(is_within_budget, axis=1)
    return df[mask].copy()