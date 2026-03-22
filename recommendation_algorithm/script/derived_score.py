"""
derived_score.py
- 가전/가구 파생변수 기반 DerivedScore 계산

계산식:
    DerivedScore = Σ [ GroupScore(i) × BaseWeight(i) × StarterMult(i) ] (재정규화)
    GroupScore(i) = Σ [ ColScore(j) × ColWeight(j) ] / Σ ColWeight(j)
"""

import pandas as pd
import numpy as np
from typing import Optional


# ------------------------------------------------------------------ #
#  각 컬럼의 실제 min/max (0~1 정규화용)
#  카테고리형 컬럼은 별도 인코딩 함수로 처리
# ------------------------------------------------------------------ #

ELECTRONICS_COL_RANGE = {
    "value_score":        (0.0, 10.0),
    "popularity_score":   (0.0, 100.0),
    "review_reliability": (0.0, 1.0),
    "single_score":       (0.0, 4.0),
    "large_family_score": (0.0, 5.0),
    "busy_worker_score":  (0.0, 3.0),
    "pet_score":          (0.0, 5.0),
    "discount_rate":      (0.0, 100.0),
}

FURNITURE_COL_RANGE = {
    "value_score":          (0.0, 10.0),
    "maintenance_score":    (2.0, 5.0),
    "delivery_score":       (0.0, 2.0),
    "single_score":         (0.0, 4.0),
    "newlywed_score":       (0.0, 4.0),
    "large_family_score":   (0.0, 4.0),
    "space_saving_score":   (0.0, 5.0),
    "pet_score":            (0.0, 5.0),
    "sofa_capacity":        (1.0, 4.0),
    "dining_capacity":      (1.0, 6.0),
    "discount_rate":        (0.0, 100.0),
}

# ------------------------------------------------------------------ #
#  카테고리형 컬럼 인코딩 맵
# ------------------------------------------------------------------ #

SIZE_GRADE_MAP_SMALL = {"소": 1.0, "중": 0.5, "대": 0.0, "미정": 0.5}  # 작을수록 선호
SIZE_GRADE_MAP_LARGE = {"소": 0.0, "중": 0.5, "대": 1.0, "미정": 0.5}  # 클수록 선호

ENERGY_GRADE_MAP = {
    "1등급": 1.0, "2등급": 0.8, "3등급": 0.6,
    "4등급": 0.4, "5등급": 0.2, "비대상": 0.5
}

MATERIAL_GRADE_MAP = {"프리미엄": 1.0, "일반": 0.5, "보급형": 0.2, "미분류": 0.5}

BED_SIZE_MAP_SINGLE = {"싱글": 1.0, "슈퍼싱글": 0.8, "더블": 0.4, "퀸": 0.2, "킹": 0.0}
BED_SIZE_MAP_COUPLE = {"싱글": 0.0, "슈퍼싱글": 0.2, "더블": 0.6, "퀸": 1.0, "킹": 1.0}

PREMIUM_LINE_MAP = {"일반": 0.2,  "오브제": 0.7, "시그니처": 1.0}

# ------------------------------------------------------------------ #
#  스타터 패키지 → 가구 유형 분류
#  sofa_capacity / dining_capacity / bed_size 인코딩 방향 결정에 사용
# ------------------------------------------------------------------ #

STARTER_HOUSEHOLD_TYPE = {
    "혼자 사는 라이프":       "single",
    "여유로운 시니어 라이프":  "couple",
    "둘이 함께 시작하는 집":  "couple",
    "아기와 함께하는 집":     "family",
    "자녀와 함께하는 집":     "family",
    "부모님과 함께하는 집":   "family",
}

# ------------------------------------------------------------------ #
#  스타터 패키지별 개별 컬럼 가중치
#  선택지 ColWeight와 동일하게 max로 병합
# ------------------------------------------------------------------ #

STARTER_COL_WEIGHTS = {
    "electronics": {
        "혼자 사는 라이프":       {"single_score": 1.5},
        "둘이 함께 시작하는 집":  {},
        "아기와 함께하는 집":     {"large_family_score": 1.3},
        "자녀와 함께하는 집":     {"large_family_score": 1.5},
        "부모님과 함께하는 집":   {"large_family_score": 1.3},
        "여유로운 시니어 라이프":  {},
    },
    "furniture": {
        "혼자 사는 라이프":       {"single_score": 1.5, "space_saving_score": 1.3},
        "둘이 함께 시작하는 집":  {"newlywed_score": 1.5},
        "아기와 함께하는 집":     {"large_family_score": 1.3},
        "자녀와 함께하는 집":     {"large_family_score": 1.5},
        "부모님과 함께하는 집":   {"large_family_score": 1.3},
        "여유로운 시니어 라이프":  {},
    },
}

# ------------------------------------------------------------------ #
#  스타터 패키지별 그룹 가중치 보정계수
#  가전: G1 가격, G2 인기도, G3 생활패턴, G4 기능/AI, G5 공간/크기, G6 스타일
#  가구: G1 가격, G2 편의성, G3 생활패턴, G4 소재/품질, G5 공간/크기, G6 스타일
# ------------------------------------------------------------------ #

STARTER_MULT = {
    "electronics": {
        "혼자 사는 라이프":       {"G1": 1.35, "G2": 0.90, "G3": 1.20, "G4": 0.80, "G5": 1.20, "G6": 0.80},
        "둘이 함께 시작하는 집":   {"G1": 1.00, "G2": 1.10, "G3": 1.10, "G4": 1.10, "G5": 1.00, "G6": 1.30},
        "아기와 함께하는 집":      {"G1": 0.80, "G2": 1.00, "G3": 1.40, "G4": 1.10, "G5": 0.90, "G6": 0.80},
        "자녀와 함께하는 집":      {"G1": 0.90, "G2": 1.00, "G3": 1.30, "G4": 1.00, "G5": 1.00, "G6": 0.90},
        "부모님과 함께하는 집":    {"G1": 0.90, "G2": 1.20, "G3": 1.20, "G4": 0.90, "G5": 0.90, "G6": 1.00},
        "여유로운 시니어 라이프":   {"G1": 1.10, "G2": 1.20, "G3": 0.90, "G4": 0.80, "G5": 1.00, "G6": 1.10},
    },
    "furniture": {
        "혼자 사는 라이프":       {"G1": 1.30, "G2": 1.20, "G3": 1.20, "G4": 0.80, "G5": 1.20, "G6": 0.90},
        "둘이 함께 시작하는 집":   {"G1": 1.00, "G2": 1.10, "G3": 1.20, "G4": 1.20, "G5": 1.20, "G6": 1.40},
        "아기와 함께하는 집":      {"G1": 0.80, "G2": 0.90, "G3": 1.30, "G4": 1.50, "G5": 0.90, "G6": 0.80},
        "자녀와 함께하는 집":      {"G1": 0.90, "G2": 0.90, "G3": 1.20, "G4": 1.30, "G5": 1.10, "G6": 0.90},
        "부모님과 함께하는 집":    {"G1": 0.90, "G2": 1.10, "G3": 1.20, "G4": 1.10, "G5": 1.00, "G6": 1.00},
        "여유로운 시니어 라이프":   {"G1": 1.10, "G2": 1.20, "G3": 0.80, "G4": 1.00, "G5": 1.00, "G6": 1.10},
    },
}

# ------------------------------------------------------------------ #
#  그룹 기본 가중치
# ------------------------------------------------------------------ #

BASE_WEIGHT = {
    "electronics": {"G1": 0.22, "G2": 0.22, "G3": 0.26, "G4": 0.10, "G5": 0.10, "G6": 0.10},
    "furniture":   {"G1": 0.18, "G2": 0.18, "G3": 0.22, "G4": 0.18, "G5": 0.14, "G6": 0.10},
}

# ------------------------------------------------------------------ #
#  선택지 → ColWeight 조정 테이블
#  같은 컬럼에 여러 선택지가 걸리면 max 값 적용
# ------------------------------------------------------------------ #

PREFERENCE_COL_WEIGHTS = {
    "공간 활용이 중요해요": {
        "electronics": {"size_grade": 1.5},        # 방향은 size_dir("small")로 결정
        "furniture":   {"size_grade": 1.5, "space_saving_score": 1.5},
    },
    "큰 제품도 괜찮아요": {
        "electronics": {"size_grade": 1.5, "large_family_score": 1.3},
        "furniture":   {"size_grade": 1.5, "large_family_score": 1.3},
    },
    "수납이 넉넉했으면 좋겠어요": {
        "electronics": {},
        "furniture":   {"space_saving_score": 1.5},
    },
    "가성비가 중요해요": {
        "electronics": {"value_score": 1.5},
        "furniture":   {},
    },
    "할인 혜택이 중요해요": {
        "electronics": {"discount_rate": 1.5},
        "furniture":   {"discount_rate": 1.5},
    },
    "가격보다 만족도가 중요해요": {
        "electronics": {"popularity_score": 1.5, "review_reliability": 1.5},
        "furniture":   {},
    },
    "프리미엄 제품도 고려해요": {
        "electronics": {"premium_line": 1.5},
        "furniture":   {"material_grade": 1.5, "brand_grade": 1.3},
    },
    "간단 요리를 자주 해요": {
        "electronics": {"single_score": 1.3, "busy_worker_score": 1.3},
        "furniture":   {"single_score": 1.3},
    },
    "집에서 보내는 시간이 많아요": {
        "electronics": {"popularity_score": 1.2},
        "furniture":   {},
    },
    "집에서 일하는 시간이 많아요": {
        "electronics": {"busy_worker_score": 1.5},
        "furniture":   {},
    },
    "청소와 관리가 쉬운 게 좋아요": {
        "electronics": {},
        "furniture":   {"maintenance_score": 1.5, "delivery_score": 1.3},
    },
    "자동화 기능(AI)이 필요해요": {
        "electronics": {"has_ai": 2.0},
        "furniture":   {},
    },
    "사용이 쉬운 제품이 좋아요": {
        "electronics": {},
        "furniture":   {"maintenance_score": 1.5},
    },
    "소음이 적은 제품이 좋아요": {
        "electronics": {},
        "furniture":   {},
    },
    "에너지 효율이 중요해요": {
        "electronics": {"energy_grade": 2.0},
        "furniture":   {},
    },
    "친환경 소재를 선호해요": {
        "electronics": {},
        "furniture":   {"is_eco_friendly": 2.0, "is_natural_material": 1.5},
    },
    "내추럴/우드 스타일이 좋아요": {
        "electronics": {"design_style_natural": 1.5},
        "furniture":   {"color_series_wood": 1.5, "design_style_natural": 1.5},
    },
    "화이트/밝은 톤이 좋아요": {
        "electronics": {"color_series_white": 1.5},
        "furniture":   {"color_series_white": 1.5},
    },
    "반려동물과 함께 살아요": {
        "electronics": {"pet_score": 2.0},
        "furniture":   {"pet_score": 2.0},
    },
}

STYLE_COL_WEIGHTS = {
    "모던/미니멀": {
        "electronics": {"design_style_modern_minimal": 1.5},
        "furniture":   {"design_style_modern_minimal": 1.5},
    },
    "내추럴/우드": {
        "electronics": {"design_style_natural": 1.5},
        "furniture":   {"color_series_wood": 1.5, "design_style_natural": 1.5},
    },
    "화이트/클린": {
        "electronics": {"color_series_white": 1.5},
        "furniture":   {"color_series_white": 1.5},
    },
}

# ------------------------------------------------------------------ #
#  필수 컬럼 정의
#  null 값 → 카테고리 중앙값 대체 / 비필수는 0.5 고정
# ------------------------------------------------------------------ #

REQUIRED_COLS_BY_CATEGORY = {
    "에어컨":       ["recommended_area", "size_grade"],
    "공기청정기":   ["recommended_area", "size_grade"],
    "제습기":       ["recommended_area", "size_grade"],
    "가습기":       ["recommended_area"],
    "냉장고":       ["size_grade"],
    "세탁기":       ["size_grade"],
    "워시타워":     ["size_grade"],
    "워시콤보":     ["size_grade"],
    "의류건조기":   ["size_grade"],
    "침대":         ["bed_size", "size_grade"],
    "소파":         ["sofa_capacity", "size_grade"],
    "식탁·테이블":  ["dining_capacity", "size_grade"],
    "매트리스·토퍼": ["bed_size"],
}


# ================================================================== #
#  Helper
# ================================================================== #

def minmax_normalize(value, min_val, max_val):
    if pd.isna(value):
        return None
    if max_val == min_val:
        return 0.5
    return float(np.clip((value - min_val) / (max_val - min_val), 0.0, 1.0))


def safe_map(value, mapping, default=0.5):
    if pd.isna(value):
        return None
    return mapping.get(str(value).strip(), default)


# ================================================================== #
#  ColScore 인코딩 — 가전
# ================================================================== #

def encode_electronics_col(col_name, value, preferences, size_direction="small", square_footage=None):
    """가전 파생컬럼 하나를 ColScore(0~1)로 변환. null이면 None 반환."""

    if col_name == "recommended_area":
        if pd.isna(value) or square_footage is None:
            return None
        diff_ratio = abs(float(value) - float(square_footage)) / float(square_footage)
        if diff_ratio <= 0.10:
            return 1.0
        elif diff_ratio <= 0.25:
            return 0.8
        elif diff_ratio <= 0.50:
            return 0.5
        else:
            return 0.2

    if col_name in ELECTRONICS_COL_RANGE:
        mn, mx = ELECTRONICS_COL_RANGE[col_name]
        return minmax_normalize(value, mn, mx)

    if col_name == "size_grade":
        m = SIZE_GRADE_MAP_SMALL if size_direction == "small" else SIZE_GRADE_MAP_LARGE
        return safe_map(value, m)

    if col_name == "energy_grade":
        return safe_map(value, ENERGY_GRADE_MAP)

    if col_name == "premium_line":
        return safe_map(value, PREMIUM_LINE_MAP)

    if col_name == "has_ai":
        if pd.isna(value):
            return None
        return 1.0 if str(value).upper() in ("TRUE", "1", "T") else 0.0

    if col_name == "design_style_natural":
        if pd.isna(value):
            return None
        return 1.0 if str(value) == "내추럴 홈" else 0.0

    if col_name == "design_style_modern_minimal":
        if pd.isna(value):
            return None
        return 1.0 if str(value) == "모던 미니멀" else 0.0

    if col_name == "color_series_white":
        if pd.isna(value):
            return None
        return 1.0 if str(value) == "화이트계" else 0.0

    return None


# ================================================================== #
#  ColScore 인코딩 — 가구
# ================================================================== #

def encode_furniture_col(col_name, value, size_direction="small", household_type="single"):
    """
    가구 파생컬럼 하나를 ColScore(0~1)로 변환. null이면 None 반환.
    household_type: "single" | "couple" | "family"
    """

    if col_name == "sofa_capacity":
        if pd.isna(value):
            return None
        mn, mx = FURNITURE_COL_RANGE["sofa_capacity"]
        norm = float(np.clip((value - mn) / (mx - mn), 0.0, 1.0))
        if household_type == "single":
            return 1.0 - norm
        elif household_type == "couple":
            peak = (2.0 - mn) / (mx - mn)
            return float(np.clip(1.0 - abs(norm - peak) / max(peak, 1.0 - peak), 0.0, 1.0))
        else:
            return norm

    if col_name == "dining_capacity":
        if pd.isna(value):
            return None
        mn, mx = FURNITURE_COL_RANGE["dining_capacity"]
        norm = float(np.clip((value - mn) / (mx - mn), 0.0, 1.0))
        if household_type == "single":
            return 1.0 - norm
        elif household_type == "couple":
            peak = (2.0 - mn) / (mx - mn)
            return float(np.clip(1.0 - abs(norm - peak) / max(peak, 1.0 - peak), 0.0, 1.0))
        else:
            return norm

    if col_name in FURNITURE_COL_RANGE:
        mn, mx = FURNITURE_COL_RANGE[col_name]
        return minmax_normalize(value, mn, mx)

    if col_name == "size_grade":
        m = SIZE_GRADE_MAP_SMALL if size_direction == "small" else SIZE_GRADE_MAP_LARGE
        return safe_map(value, m)

    if col_name == "material_grade":
        return safe_map(value, MATERIAL_GRADE_MAP)

    if col_name == "bed_size":
        bed_map = BED_SIZE_MAP_SINGLE if household_type == "single" else BED_SIZE_MAP_COUPLE
        return safe_map(value, bed_map)

    if col_name in ("is_eco_friendly", "is_natural_material", "is_free_delivery", "is_installation_included"):
        if pd.isna(value):
            return None
        return 1.0 if str(value).upper() in ("TRUE", "1", "T") else 0.0

    if col_name == "brand_grade":
        return safe_map(value, {"프리미엄": 1.0, "중급": 0.6, "일반": 0.3}, default=0.5)

    if col_name == "design_style_natural":
        if pd.isna(value):
            return None
        return 1.0 if str(value) == "내추럴 우드" else 0.0

    if col_name == "design_style_modern_minimal":
        if pd.isna(value):
            return None
        return 1.0 if str(value) == "모던 미니멀" else 0.0

    if col_name == "color_series_white":
        if pd.isna(value):
            return None
        return 1.0 if str(value) == "화이트계" else 0.0

    if col_name == "color_series_wood":
        if pd.isna(value):
            return None
        return 1.0 if str(value) == "우드·브라운계" else 0.0

    return None


# ================================================================== #
#  ColWeight 계산
# ================================================================== #

def build_col_weights(
    preferences: list,
    style: Optional[str],
    product_type: str,
    starter_package: str = "",
) -> dict:
    """
    선택지 + 스타일 + 스타터 패키지 → {col_name: ColWeight}
    같은 컬럼에 여러 소스가 걸리면 max 적용
    적용 순서: preferences → style → starter_package
    """
    weights = {}

    for pref in preferences:
        if pref not in PREFERENCE_COL_WEIGHTS:
            continue
        for col, w in PREFERENCE_COL_WEIGHTS[pref].get(product_type, {}).items():
            weights[col] = max(weights.get(col, 1.0), w)

    if style and style in STYLE_COL_WEIGHTS:
        for col, w in STYLE_COL_WEIGHTS[style].get(product_type, {}).items():
            weights[col] = max(weights.get(col, 1.0), w)

    if starter_package and starter_package in STARTER_COL_WEIGHTS.get(product_type, {}):
        for col, w in STARTER_COL_WEIGHTS[product_type][starter_package].items():
            weights[col] = max(weights.get(col, 1.0), w)

    return weights


# ================================================================== #
#  GroupScore 계산
# ================================================================== #

def calc_group_score(
    group_cols: list,
    col_weights: dict,
    encode_fn,
    category_medians: dict,
    category: str,
    required_cols: list,
) -> float:
    """
    GroupScore = Σ(ColScore × ColWeight) / Σ(ColWeight)
    - ColWeight=0 이면 해당 컬럼 제외
    - null 처리: 필수 컬럼 → 카테고리 중앙값 / 비필수 → 0.5
    """
    numerator = 0.0
    denominator = 0.0

    for col_name, raw_value in group_cols:
        col_weight = col_weights.get(col_name, 1.0)
        if col_weight == 0.0:
            continue

        col_score = encode_fn(col_name, raw_value)

        if col_score is None:
            if col_name in required_cols:
                median_val = category_medians.get((category, col_name))
                col_score = encode_fn(col_name, median_val) if median_val is not None else 0.5
            else:
                col_score = 0.5

        numerator   += col_score * col_weight
        denominator += col_weight

    return numerator / denominator if denominator > 0 else 0.5


# ================================================================== #
#  DerivedScore 계산 — 가전
# ================================================================== #

def calc_electronics_derived_score(
    row: pd.Series,
    starter_package: str,
    preferences: list,
    style: Optional[str],
    category_medians: dict,
    square_footage: Optional[float] = None,
) -> float:
    """
    가전 제품 1개의 DerivedScore 반환 (0~1)
    square_footage: 사용자 집 평수 (recommended_area 적합도 계산에 사용)
    """
    cat = row.get("category", "")
    required_cols = REQUIRED_COLS_BY_CATEGORY.get(cat, [])
    size_dir = "large" if "큰 제품도 괜찮아요" in preferences else "small"
    col_weights = build_col_weights(preferences, style, "electronics", starter_package)

    def enc(col, val):
        return encode_electronics_col(col, val, preferences, size_dir, square_footage)

    G1_cols = [
        ("value_score",   row.get("value_score")),
        ("discount_rate", row.get("discount_rate")),
    ]
    G2_cols = [
        ("popularity_score",   row.get("popularity_score")),
        ("review_reliability", row.get("review_reliability")),
    ]
    G3_cols = [
        ("single_score",       row.get("single_score")),
        ("large_family_score", row.get("large_family_score")),
        ("busy_worker_score",  row.get("busy_worker_score")),
        ("pet_score",          row.get("pet_score")),
    ]
    G4_cols = [
        ("has_ai",       row.get("has_ai")),
        ("premium_line", row.get("premium_line")),
        ("energy_grade", row.get("energy_grade")),
    ]
    G5_cols = [
        ("size_grade",       row.get("size_grade")),
        ("recommended_area", row.get("recommended_area")),
    ]
    G6_cols = [
        ("design_style_natural",        row.get("design_style")),
        ("design_style_modern_minimal",  row.get("design_style")),
        ("color_series_white",          row.get("color_series")),
    ]

    # energy_grade, G6 컬럼: 선택 시에만 활성화 (기본 ColWeight=0)
    if "energy_grade" not in col_weights:
        col_weights["energy_grade"] = 0.0
    for col, _ in G6_cols:
        if col not in col_weights:
            col_weights[col] = 0.0

    group_scores = {
        "G1": calc_group_score(G1_cols, col_weights, enc, category_medians, cat, required_cols),
        "G2": calc_group_score(G2_cols, col_weights, enc, category_medians, cat, required_cols),
        "G3": calc_group_score(G3_cols, col_weights, enc, category_medians, cat, required_cols),
        "G4": calc_group_score(G4_cols, col_weights, enc, category_medians, cat, required_cols),
        "G5": calc_group_score(G5_cols, col_weights, enc, category_medians, cat, required_cols),
        "G6": calc_group_score(G6_cols, col_weights, enc, category_medians, cat, required_cols),
    }

    starter_mult = STARTER_MULT["electronics"].get(starter_package, {g: 1.0 for g in BASE_WEIGHT["electronics"]})
    base_w = BASE_WEIGHT["electronics"]
    raw_weights = {g: base_w[g] * starter_mult.get(g, 1.0) for g in base_w}
    total = sum(raw_weights.values())
    norm_weights = {g: w / total for g, w in raw_weights.items()}

    return float(np.clip(sum(group_scores[g] * norm_weights[g] for g in group_scores), 0.0, 1.0))


# ================================================================== #
#  DerivedScore 계산 — 가구
# ================================================================== #

def calc_furniture_derived_score(
    row: pd.Series,
    starter_package: str,
    preferences: list,
    style: Optional[str],
    category_medians: dict,
) -> float:
    """가구 제품 1개의 DerivedScore 반환 (0~1)"""
    cat = row.get("category", "")
    required_cols = REQUIRED_COLS_BY_CATEGORY.get(cat, [])
    size_dir = "large" if "큰 제품도 괜찮아요" in preferences else "small"
    household_type = STARTER_HOUSEHOLD_TYPE.get(starter_package, "single")
    if "큰 제품도 괜찮아요" in preferences:
        household_type = "family"
    col_weights = build_col_weights(preferences, style, "furniture", starter_package)

    def enc(col, val):
        return encode_furniture_col(col, val, size_dir, household_type)

    G1_cols = [
        ("discount_rate", row.get("discount_rate")),
    ]
    G2_cols = [
        ("is_installation_included", row.get("is_installation_included")),
        ("delivery_score",           row.get("delivery_score")),
    ]
    G3_cols = [
        ("single_score",       row.get("single_score")),
        ("newlywed_score",     row.get("newlywed_score")),
        ("large_family_score", row.get("large_family_score")),
        ("space_saving_score", row.get("space_saving_score")),
        ("pet_score",          row.get("pet_score")),
    ]
    G4_cols = [
        ("material_grade",    row.get("material_grade")),
        ("is_eco_friendly",   row.get("is_eco_friendly")),
        ("maintenance_score", row.get("maintenance_score")),
    ]
    G5_cols = [
        ("size_grade",      row.get("size_grade")),
        ("sofa_capacity",   row.get("sofa_capacity")),
        ("dining_capacity", row.get("dining_capacity")),
        ("bed_size",        row.get("bed_size")),
    ]
    G6_cols = [
        ("design_style_natural",        row.get("design_style")),
        ("design_style_modern_minimal",  row.get("design_style")),
        ("color_series_white",          row.get("color_series")),
        ("color_series_wood",           row.get("color_series")),
    ]
    for col, _ in G6_cols:
        if col not in col_weights:
            col_weights[col] = 0.0

    group_scores = {
        "G1": calc_group_score(G1_cols, col_weights, enc, category_medians, cat, required_cols),
        "G2": calc_group_score(G2_cols, col_weights, enc, category_medians, cat, required_cols),
        "G3": calc_group_score(G3_cols, col_weights, enc, category_medians, cat, required_cols),
        "G4": calc_group_score(G4_cols, col_weights, enc, category_medians, cat, required_cols),
        "G5": calc_group_score(G5_cols, col_weights, enc, category_medians, cat, required_cols),
        "G6": calc_group_score(G6_cols, col_weights, enc, category_medians, cat, required_cols),
    }

    starter_mult = STARTER_MULT["furniture"].get(starter_package, {g: 1.0 for g in BASE_WEIGHT["furniture"]})
    base_w = BASE_WEIGHT["furniture"]
    raw_weights = {g: base_w[g] * starter_mult.get(g, 1.0) for g in base_w}
    total = sum(raw_weights.values())
    norm_weights = {g: w / total for g, w in raw_weights.items()}

    return float(np.clip(sum(group_scores[g] * norm_weights[g] for g in group_scores), 0.0, 1.0))
