"""
scoring.py
- 재정렬(Re-ranking): 구독 가능 제품 상단 배치
- 조합 생성: 카테고리별 상위 5개 조합 탐색 → 다양성 패널티 적용 후 선별
- 출력 포맷: 프론트 요청 형태

PackageScore = 0.8 * 상품점수평균 + 0.2 * budget_fit
budget_fit   = 1 - abs(총조합가격 - 총예산) / 총예산
"""

import itertools
import math
from recommendation_reason import generate_reasons
import numpy as np
import pandas as pd
from typing import Optional


# ── 설정 ────────────────────────────────────────────────────────────

TOP_N_PER_CATEGORY = 5    # 카테고리별 상위 N개 후보
N_PACKAGES          = 60   # 전체 조합 풀 크기
N_THEMES            = 3    # 테마 수
N_PER_THEME         = 4    # 테마별 패키지 수
N_DISPLAY           = N_THEMES  # _determine_themes용
MIN_PACKAGES        = 3    # 최종 최소 패키지 수
MIN_PRODUCTS_PER_PACKAGE = 3  # 패키지당 최소 상품 수
DIVERSITY_PENALTY   = 0.15  # 이미 등장한 제품 1개당 패널티


# ================================================================== #
#  테마 정의
# ================================================================== #

PREFERENCE_THEME_MAP = {
    "가성비가 중요해요":         "가성비",
    "할인 혜택이 중요해요":       "가성비",
    "프리미엄 제품도 고려해요":    "프리미엄",
    "에너지 효율이 중요해요":      "효율",
    "자동화 기능(AI)이 필요해요":  "효율",
    "반려동물과 함께 살아요":      "펫 프렌들리",
    "공간 활용이 중요해요":        "공간 최적화",
    "큰 제품도 괜찮아요":          "공간 최적화",
    "친환경 소재를 선호해요":      "친환경",
    "청소와 관리가 쉬운 게 좋아요": "효율",
}

DEFAULT_THEMES = ["밸런스", "가성비", "프리미엄"]


# ================================================================== #
#  재정렬 — 구독 가능 제품 상단 배치
# ================================================================== #

def rerank(results: dict) -> dict:
    """
    카테고리별 df에서 is_subscribe=True 제품을 상단으로 재정렬
    정렬 기준:
        1순위: is_subscribe (True 먼저)
        2순위: final_score 내림차순
    """
    reranked = {}
    for cat, df in results.items():
        df = df.copy()
        score_col = "final_score" if "final_score" in df.columns else "derived_score"

        if "is_subscribe" in df.columns:
            df["is_subscribe"] = df["is_subscribe"].fillna(False).astype(bool)
            df = df.sort_values(
                ["is_subscribe", score_col],
                ascending=[False, False]
            ).reset_index(drop=True)
        else:
            df = df.sort_values(score_col, ascending=False).reset_index(drop=True)

        reranked[cat] = df

    return reranked


# ================================================================== #
#  조합 생성
# ================================================================== #

def _get_candidates(results: dict) -> dict:
    """카테고리별 상위 TOP_N_PER_CATEGORY개 후보 추출"""
    return {
        cat: df.head(TOP_N_PER_CATEGORY).to_dict("records")
        for cat, df in results.items()
        if not df.empty
    }


def _calc_package_score(products: list, budget: int) -> float:
    """
    PackageScore = 0.8 * 상품점수평균 + 0.2 * budget_fit
    budget_fit   = 1 - abs(총조합가격 - 총예산) / 총예산
    """
    score_col  = "final_score" if "final_score" in products[0] else "derived_score"
    avg_score  = np.mean([p.get(score_col, 0.0) for p in products])

    total_price = sum(p.get("price", 0) for p in products)
    if budget > 0:
        budget_fit = max(0.0, 1.0 - abs(total_price - budget) / budget)
    else:
        budget_fit = 1.0

    return 0.8 * avg_score + 0.2 * budget_fit


def generate_packages(results: dict, budget: int) -> list:
    """
    카테고리별 상위 5개 후보로 모든 조합 탐색
    다양성 보장: 이미 선택된 제품이 포함된 조합에 패널티 부여
    → N_PACKAGES개 반환
    """
    candidates = _get_candidates(results)
    if not candidates:
        return []

    categories      = list(candidates.keys())
    candidate_lists = [candidates[cat] for cat in categories]

    # 모든 조합 탐색
    all_combos = []
    for combo in itertools.product(*candidate_lists):
        products      = list(combo)
        package_score = _calc_package_score(products, budget)
        total_price   = sum(p.get("price", 0) for p in products)
        all_combos.append({
            "products":      products,
            "package_score": package_score,
            "total_price":   total_price,
        })

    # PackageScore 내림차순 정렬
    all_combos.sort(key=lambda x: x["package_score"], reverse=True)

    # 다양성 보장: 이미 선택된 product_id에 패널티 적용하며 순차 선택
    selected  = []
    used_pids = {}  # {product_id: 등장 횟수}

    for combo in all_combos:
        if len(selected) >= N_PACKAGES:
            break

        pids    = [p.get("product_id") for p in combo["products"]]
        penalty = sum(used_pids.get(pid, 0) * DIVERSITY_PENALTY for pid in pids)
        adjusted_score = combo["package_score"] - penalty

        selected.append({**combo, "adjusted_score": adjusted_score})

        for pid in pids:
            used_pids[pid] = used_pids.get(pid, 0) + 1

    return selected


# ================================================================== #
#  테마 기반 패키지 선별
# ================================================================== #

def _determine_themes(preferences: list) -> list:
    """사용자 선택지 → 테마 3개 결정. 부족하면 기본 테마로 채움"""
    themes = []
    for pref in preferences:
        theme = PREFERENCE_THEME_MAP.get(pref)
        if theme and theme not in themes:
            themes.append(theme)
    for t in DEFAULT_THEMES:
        if len(themes) >= N_DISPLAY:
            break
        if t not in themes:
            themes.append(t)
    return themes[:N_DISPLAY]


def _score_by_theme(pkg: dict, theme: str, budget: int) -> float:
    """테마별 패키지 점수 계산"""
    products    = pkg["products"]
    total_price = pkg["total_price"]
    score_col   = "final_score" if "final_score" in products[0] else "derived_score"

    if theme == "가성비":
        return 1.0 - (total_price / budget) if budget > 0 else 0.0
    elif theme == "프리미엄":
        return float(np.mean([p.get(score_col, 0.0) for p in products]))
    elif theme == "효율":
        has_subscribe   = any(p.get("is_subscribe") for p in products)
        subscribe_bonus = 0.2 if has_subscribe else 0.0
        return pkg["package_score"] + subscribe_bonus
    elif theme == "펫 프렌들리":
        return float(sum(p.get("pet_score", 0) or 0 for p in products))
    elif theme == "공간 최적화":
        small_cnt = sum(1 for p in products if p.get("size_grade") == "소")
        return small_cnt / len(products) if products else 0.0
    elif theme == "친환경":
        eco_cnt = sum(1 for p in products if p.get("is_eco_friendly"))
        return eco_cnt / len(products) if products else 0.0
    else:  # 밸런스
        return pkg.get("adjusted_score", pkg["package_score"])


def _product_identity_key(p: dict) -> str:
    pid = p.get("product_id")
    if pid is not None and str(pid).strip():
        return f"pid:{str(pid).strip()}"
    mid = p.get("model_id")
    if mid is not None and str(mid).strip():
        return f"mid:{str(mid).strip()}"
    name = str(p.get("name") or "").strip().lower()
    cat  = str(p.get("category") or "").strip().lower()
    if name or cat:
        return f"namecat:{name}|{cat}"
    return ""


def _package_product_keys(pkg: dict) -> set:
    products = pkg.get("products") or []
    keys = set()
    for p in products:
        if not isinstance(p, dict):
            continue
        k = _product_identity_key(p)
        if k:
            keys.add(k)
    return keys


def _package_signature(pkg: dict) -> tuple:
    return tuple(sorted(_package_product_keys(pkg)))


def _build_global_product_pool(reranked: dict) -> list:
    pool = []
    seen = set()
    for _, df in reranked.items():
        rows = df.head(TOP_N_PER_CATEGORY).to_dict("records")
        for p in rows:
            if not isinstance(p, dict):
                continue
            k = _product_identity_key(p)
            if not k or k in seen:
                continue
            seen.add(k)
            pool.append(p)
    return pool


def _ensure_min_products(pkg: dict, global_pool: list, min_count: int) -> dict:
    out      = dict(pkg)
    products = list(out.get("products") or [])
    used     = {_product_identity_key(p) for p in products if isinstance(p, dict)}

    if len(products) < min_count:
        for gp in global_pool:
            if not isinstance(gp, dict):
                continue
            k = _product_identity_key(gp)
            if not k or k in used:
                continue
            products.append(gp)
            used.add(k)
            if len(products) >= min_count:
                break

    out["products"]    = products
    out["total_price"] = int(sum(p.get("price", 0) for p in products if isinstance(p, dict)))
    out["package_score"] = _calc_package_score(products, 0) if products else 0.0
    return out


def _enforce_minimum_output(themed_packages: list, all_packages: list, reranked: dict) -> list:
    global_pool = _build_global_product_pool(reranked)
    adjusted    = []
    seen_sig    = set()

    for item in themed_packages:
        if not isinstance(item, dict):
            continue
        theme     = item.get("theme") or "밸런스"
        pkg       = item.get("package") or {}
        fixed_pkg = _ensure_min_products(pkg, global_pool, MIN_PRODUCTS_PER_PACKAGE)
        sig       = _package_signature(fixed_pkg)
        if sig and sig in seen_sig:
            continue
        if sig:
            seen_sig.add(sig)
        adjusted.append({"theme": theme, "package": fixed_pkg})

    if len(adjusted) < MIN_PACKAGES:
        for pkg in all_packages:
            if len(adjusted) >= MIN_PACKAGES:
                break
            if not isinstance(pkg, dict):
                continue
            fixed_pkg = _ensure_min_products(pkg, global_pool, MIN_PRODUCTS_PER_PACKAGE)
            sig       = _package_signature(fixed_pkg)
            if sig and sig in seen_sig:
                continue
            if sig:
                seen_sig.add(sig)
            adjusted.append({"theme": "밸런스", "package": fixed_pkg})

    return adjusted


def select_themed_packages(all_packages: list, preferences: list, budget: int) -> list:
    """
    전체 조합 풀에서 테마별로 상위 N_PER_THEME개씩 선별
    중복 패키지 방지 + 상품당 최대 2회 등장 제한
    반환: [{"theme": str, "package": dict}, ...]  총 N_THEMES × N_PER_THEME개
    """
    themes             = _determine_themes(preferences)
    selected           = []
    used_idx           = set()
    product_key_counts = {}

    for theme in themes:
        scored = sorted(
            [
                (i, _score_by_theme(pkg, theme, budget))
                for i, pkg in enumerate(all_packages)
                if i not in used_idx
            ],
            key=lambda x: x[1],
            reverse=True,
        )
        picked_in_theme = 0
        for i, _ in scored:
            if picked_in_theme >= N_PER_THEME:
                break
            candidate_pkg  = all_packages[i]
            candidate_keys = _package_product_keys(candidate_pkg)

            # 상품당 최대 2회까지만 허용
            if any(product_key_counts.get(k, 0) >= 2 for k in candidate_keys):
                continue

            used_idx.add(i)
            for k in candidate_keys:
                product_key_counts[k] = product_key_counts.get(k, 0) + 1
            selected.append({"theme": theme, "package": candidate_pkg})
            picked_in_theme += 1

    return selected


# ================================================================== #
#  출력 포맷 변환
# ================================================================== #

def _safe_int(val) -> int:
    """None / NaN → 0, 나머지 → int"""
    try:
        v = float(val)
        return 0 if math.isnan(v) else int(v)
    except (TypeError, ValueError):
        return 0


ELECTRONICS_CATEGORIES = {
    "TV", "스탠바이미", "냉장고", "전기레인지", "광파오븐/전자레인지",
    "식기세척기", "정수기", "세탁기", "워시타워", "워시콤보",
    "의류관리기", "의류건조기", "청소기", "에어컨", "공기청정기",
    "제습기", "가습기",
}


def _format_appliance(p: dict) -> dict:
    return {
        "name":                 p.get("name", ""),
        "category":             p.get("category", ""),
        "totalPrice":           _safe_int(p.get("original_price") or p.get("price", 0)),
        "subscriptionPrice":    _safe_int(p.get("subscription_price")),
        "contractPeriodYear":   _safe_int(p.get("contract_period_year")),
        "mandatoryPeriodYear":  _safe_int(p.get("mandatory_period_year")),
        "visitServiceType":     p.get("visit_service_type"),
        "visitCycleMonth":      _safe_int(p.get("visit_cycle_month")),
        "image":                p.get("product_image_url", ""),
        "productUrl":           p.get("product_url", ""),
        "popularityScore":      round(float(p.get("popularity_score", 0) or 0), 1),
    }


def _format_furniture(p: dict) -> dict:
    return {
        "name":       p.get("name", ""),
        "category":   p.get("category", ""),
        "price":      int(p.get("price", 0)),
        "image":      p.get("product_image_url", ""),
        "productUrl": p.get("product_url", ""),
    }


def format_output(themed_packages: list, reasons: list) -> dict:
    """
    테마별 패키지 + 추천 이유 → 프론트 요청 형태로 변환
    """
    output_packages = []

    for item, reason in zip(themed_packages, reasons):
        theme      = item["theme"]
        pkg        = item["package"]
        appliances = []
        furniture  = []

        for p in pkg["products"]:
            cat = p.get("category", "")
            if cat in ELECTRONICS_CATEGORIES:
                appliances.append(_format_appliance(p))
            else:
                furniture.append(_format_furniture(p))

        output_packages.append({
            "theme":                theme,
            "appliances":           appliances,
            "furniture":            furniture,
            "recommendationReason": reason,
        })

    return {"packages": output_packages}


# ================================================================== #
#  메인 함수 — pipeline.py에서 호출
# ================================================================== #

def run_scoring(
    results: dict,
    budget: int,
    starter: str = "",
    preferences: list = None,
    square_footage: int = 0,
    use_llm: bool = True,
) -> dict:
    """
    재정렬 → 조합 생성 → 추천 이유 생성 → 출력 포맷 변환
    """
    preferences = preferences or []

    # 1. 재정렬
    reranked = rerank(results)

    # 2. 전체 조합 생성 (다양성 패널티 적용)
    all_packages = generate_packages(reranked, budget)

    # 3. 테마별 패키지 선별
    themed_packages = select_themed_packages(all_packages, preferences, budget)
    themed_packages = _enforce_minimum_output(themed_packages, all_packages, reranked)

    # 4. 추천 이유 생성
    if use_llm and themed_packages:
        pkg_list = [item["package"] for item in themed_packages]
        themes   = [item["theme"] for item in themed_packages]
        reasons  = generate_reasons(pkg_list, starter, preferences, budget, square_footage, themes)
    else:
        reasons = ["test"] * len(themed_packages)

    # 5. 출력 포맷
    return format_output(themed_packages, reasons)