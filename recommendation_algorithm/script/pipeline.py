"""
pipeline.py
- 전체 추천 흐름 조율

전체 흐름:
    1. 품목/예산 필터링  → filtering.py
    2. DerivedScore 계산 → derived_score.py
    3. ReviewScore 계산  → review_score.py
    4. ImageScore 계산   → image_score.py
    5. 총점 합산 + 재정렬 + 조합 → scoring.py
    6. 추천 이유 생성 → recommendation_reason.py
"""

from typing import Optional

from db import get_engine
from filtering import (
    fetch_electronics,
    fetch_furniture,
    fetch_category_medians,
    fetch_category_price_stats,
    allocate_budget,
    filter_by_budget,
)
from derived_score import (
    calc_electronics_derived_score,
    calc_furniture_derived_score,
)
from image_score import (
    calc_image_scores,
    calc_final_score_furniture,
)
from review_score import (
    calc_review_scores,
    calc_final_score_electronics,
)
from scoring import run_scoring


def parse_input(input_data: dict) -> dict:
    """
    input JSON 키 정규화 (camelCase / snake_case 둘 다 허용)
    반환: 정규화된 파라미터 딕셔너리
    """
    products = input_data.get("products") or input_data.get("product", {})
    electronics = products.get("electronics", {})
    furniture   = products.get("furniture", {})

    return {
        "starter":          input_data.get("starter_package") or input_data.get("starterPackage", ""),
        "budget":           input_data.get("budget", 0),
        "square_footage":   input_data.get("square_footage") or input_data.get("squareFootage"),
        "preferences":      input_data.get("preferences", []),
        "style":            input_data.get("style"),
        "needed_electronics": electronics.get("needed", []),
        "needed_furniture":   furniture.get("needed", []),
    }


def run_pipeline(input_data: dict, engine) -> dict:
    """
    추천 파이프라인 실행
    input_data: API 입력 JSON
    반환: { 카테고리명: df }  예) { "냉장고": df, "세탁기": df, "침대": df, ... }
    각 df는 해당 카테고리 후보 제품들을 derived_score 내림차순으로 정렬
    """
    p = parse_input(input_data)

    # ── 공통 데이터 로드 ──────────────────────────────────────────
    category_medians     = fetch_category_medians(engine)
    category_price_stats = fetch_category_price_stats(engine)

    all_needed = p["needed_electronics"] + p["needed_furniture"]
    allocated  = allocate_budget(all_needed, p["budget"], category_price_stats)

    results = {}

    # ── 가전 ─────────────────────────────────────────────────────
    if p["needed_electronics"]:
        df_e = fetch_electronics(engine, p["needed_electronics"])
        df_e = filter_by_budget(df_e, allocated)

        df_e["derived_score"] = df_e.apply(
            lambda row: calc_electronics_derived_score(
                row,
                starter_package=p["starter"],
                preferences=p["preferences"],
                style=p["style"],
                category_medians=category_medians,
                square_footage=p["square_footage"],
            ),
            axis=1,
        )

        # ReviewScore 계산
        df_e = calc_review_scores(df_e, p["starter"], p["preferences"], engine)

        # FinalScore = 0.7 * derived_score + 0.3 * review_score
        df_e = calc_final_score_electronics(df_e)

        # 카테고리별로 분리해서 반환
        for cat, df_cat in df_e.groupby("category"):
            results[cat] = df_cat.sort_values("final_score", ascending=False).reset_index(drop=True)

    # ── 가구 ─────────────────────────────────────────────────────
    if p["needed_furniture"]:
        df_f = fetch_furniture(engine, p["needed_furniture"])
        df_f = filter_by_budget(df_f, allocated)

        # 후보 없음 / 스키마 이상 → 가구만 건너뛰고 가전·스코어링은 계속
        if df_f.empty or "category" not in df_f.columns:
            pass
        else:
            df_f["derived_score"] = df_f.apply(
                lambda row: calc_furniture_derived_score(
                    row,
                    starter_package=p["starter"],
                    preferences=p["preferences"],
                    style=p["style"],
                    category_medians=category_medians,
                ),
                axis=1,
            )

            if df_f.empty or "category" not in df_f.columns:
                pass
            else:
                # ImageScore 계산
                df_f = calc_image_scores(df_f, p["style"], engine)

                # FinalScore = 0.6 * derived_score + 0.4 * image_score
                df_f = calc_final_score_furniture(df_f)

                # 카테고리별로 분리해서 반환
                for cat, df_cat in df_f.groupby("category"):
                    results[cat] = df_cat.sort_values("final_score", ascending=False).reset_index(drop=True)

    return results


def run_full_pipeline(input_data: dict, engine, use_llm: bool = True) -> dict:
    p       = parse_input(input_data)
    results = run_pipeline(input_data, engine)
    return run_scoring(
        results,
        budget         = p["budget"],
        starter        = p["starter"],
        preferences    = p["preferences"],
        square_footage = p["square_footage"] or 0,
        use_llm        = use_llm,
    )


# ================================================================== #
#  실행 예시
# ================================================================== #

if __name__ == "__main__":
    input_data = {
        "starter_package": "혼자 사는 라이프",
        "budget": 3_000_000,
        "square_footage": 15,
        "products": {
            "electronics": {
                "owned":  ["냉장고", "세탁기", "청소기"],
                "needed": ["광파오븐/전자레인지", "공기청정기"],
            },
            "furniture": {
                "needed": ["책상", "의자"],
            },
        },
        "preferences": [
            "공간 활용이 중요해요",
            "가격보다 만족도가 중요해요",
            "청소와 관리가 쉬운 게 좋아요",
            "반려동물과 함께 살아요",
        ],
        "style": "모던/미니멀",
    }

    import json
    with get_engine() as engine:
        output = run_full_pipeline(input_data, engine, use_llm=True)

    print(json.dumps(output, ensure_ascii=False, indent=2))