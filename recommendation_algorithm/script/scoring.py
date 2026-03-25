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

TOP_N_PER_CATEGORY  = 7    # 카테고리별 상위 N개 후보
N_PACKAGES          = 60   # 전체 조합 풀 크기
N_THEMES            = 3    # 테마 수
N_PER_THEME         = 4    # 테마별 패키지 수
N_DISPLAY           = N_THEMES  # _determine_themes용
MIN_PACKAGES        = 12   # 최종 최소 패키지 수 (3테마 × 4개)
MIN_PRODUCTS_PER_PACKAGE = 3  # 패키지당 최소 상품 수
DIVERSITY_PENALTY   = 1.0   # 이미 등장한 제품 1개당 패널티
MIN_DIFF_RATIO      = 0.3    # 패키지 간 최소 30% 제품이 달라야 함
MAX_PER_BRAND       = 2    # 카테고리당 같은 브랜드 최대 후보 수
ELEC_WEIGHT         = 1.3  # 패키지 점수 계산 시 가전 가중치
FURN_WEIGHT         = 0.8  # 패키지 점수 계산 시 가구 가중치


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
    카테고리별 df에서 구독 가능 제품을 상단으로 재정렬하되,
    상위 50%만 구독 우선 배치하여 비구독 고품질 제품도 후보에 남도록 함.
    """
    reranked = {}
    for cat, df in results.items():
        df = df.copy()
        score_col = "final_score" if "final_score" in df.columns else "derived_score"

        if "is_subscribe" in df.columns:
            df["is_subscribe"] = df["is_subscribe"].fillna(False).astype(bool)
            # 구독 우선 슬롯: 후보 수의 50% (최소 2, 최대 TOP_N의 절반)
            max_sub_slots = max(2, TOP_N_PER_CATEGORY // 2)
            sub_df = df[df["is_subscribe"]].sort_values(score_col, ascending=False).head(max_sub_slots)
            nonsub_df = df[~df.index.isin(sub_df.index)].sort_values(score_col, ascending=False)
            df = pd.concat([sub_df, nonsub_df]).reset_index(drop=True)
        else:
            df = df.sort_values(score_col, ascending=False).reset_index(drop=True)

        reranked[cat] = df

    return reranked


# ================================================================== #
#  조합 생성
# ================================================================== #

def _get_candidates(results: dict) -> dict:
    """카테고리별 상위 TOP_N_PER_CATEGORY개 후보 추출 (브랜드 다양성 적용)"""
    candidates = {}
    for cat, df in results.items():
        if df.empty:
            continue
        # 브랜드가 2개 이상일 때만 다양성 제한 적용
        unique_brands = df["brand"].dropna().nunique() if "brand" in df.columns else 0
        apply_brand_limit = unique_brands >= 2

        selected = []
        brand_count = {}
        for _, row in df.iterrows():
            if len(selected) >= TOP_N_PER_CATEGORY:
                break
            brand = str(row.get("brand", "") or "").strip()
            if apply_brand_limit and brand and brand_count.get(brand, 0) >= MAX_PER_BRAND:
                continue
            selected.append(row.to_dict())
            if brand:
                brand_count[brand] = brand_count.get(brand, 0) + 1
        candidates[cat] = selected
    return candidates


def _get_theme_candidates(reranked: dict, theme: str) -> dict:
    """테마별로 다른 정렬 기준으로 후보 선정 (멀티 전략 앙상블)"""
    candidates = {}
    for cat, df in reranked.items():
        if df.empty:
            continue
        df = df.copy()
        score_col = "final_score" if "final_score" in df.columns else "derived_score"

        # 테마별 보조 점수 계산
        if theme == "가성비":
            vs = df["value_score"].fillna(5.0) / 10.0 if "value_score" in df.columns else 0.5
            dr = df["discount_rate"].fillna(0) / 100.0 if "discount_rate" in df.columns else 0.0
            df["_sort"] = 0.4 * df[score_col] + 0.3 * vs + 0.3 * dr
        elif theme == "프리미엄":
            prem = pd.Series(0.0, index=df.index)
            if "premium_line" in df.columns:
                prem = df["premium_line"].isin(["오브제", "시그니처"]).astype(float)
            if "material_grade" in df.columns:
                prem = np.maximum(prem, (df["material_grade"] == "프리미엄").astype(float))
            df["_sort"] = 0.5 * df[score_col] + 0.5 * prem
        elif theme == "효율":
            energy = df["energy_grade"].isin(["1등급", "2등급"]).astype(float) if "energy_grade" in df.columns else 0.0
            sub = df["is_subscribe"].fillna(False).astype(float) if "is_subscribe" in df.columns else 0.0
            df["_sort"] = 0.5 * df[score_col] + 0.3 * energy + 0.2 * sub
        elif theme == "펫 프렌들리":
            pet = df["pet_score"].fillna(0) / 5.0 if "pet_score" in df.columns else 0.0
            df["_sort"] = 0.5 * df[score_col] + 0.5 * pet
        elif theme == "공간 최적화":
            space = df["space_saving_score"].fillna(0) / 5.0 if "space_saving_score" in df.columns else 0.0
            small = (df["size_grade"] == "소").astype(float) if "size_grade" in df.columns else 0.0
            df["_sort"] = 0.4 * df[score_col] + 0.3 * space + 0.3 * small
        elif theme == "친환경":
            eco = df["is_eco_friendly"].fillna(False).astype(float) if "is_eco_friendly" in df.columns else 0.0
            df["_sort"] = 0.5 * df[score_col] + 0.5 * eco
        else:  # 밸런스
            df["_sort"] = df[score_col].copy()

        df = df.sort_values("_sort", ascending=False)

        # 브랜드 다양성 적용
        unique_brands = df["brand"].dropna().nunique() if "brand" in df.columns else 0
        apply_brand_limit = unique_brands >= 2
        selected = []
        brand_count = {}
        for _, row in df.iterrows():
            if len(selected) >= TOP_N_PER_CATEGORY:
                break
            brand = str(row.get("brand", "") or "").strip()
            if apply_brand_limit and brand and brand_count.get(brand, 0) >= MAX_PER_BRAND:
                continue
            selected.append(row.to_dict())
            if brand:
                brand_count[brand] = brand_count.get(brand, 0) + 1
        candidates[cat] = selected

    return candidates


def _package_feature_vector(pkg: dict) -> np.ndarray:
    """
    패키지의 특성 벡터 생성 (MMR 유사도 계산용).
    제품별 특성을 평균하여 패키지 수준 벡터로 집약.
    """
    products = pkg.get("products", [])
    if not products:
        return np.zeros(8)

    score_col = "final_score" if "final_score" in products[0] else "derived_score"
    n = len(products)

    # 8차원 특성 벡터
    avg_score = np.mean([p.get(score_col, 0.0) for p in products])
    avg_price = np.mean([p.get("price", 0) for p in products]) / 5_000_000  # 정규화
    sub_ratio = sum(1 for p in products if p.get("is_subscribe")) / n
    ai_ratio = sum(1 for p in products if p.get("has_ai")) / n
    energy_ratio = sum(1 for p in products if p.get("energy_grade") in ("1등급", "2등급")) / n
    premium_ratio = sum(1 for p in products
                        if p.get("premium_line") in ("오브제", "시그니처")
                        or p.get("material_grade") == "프리미엄") / n
    pet_avg = np.mean([float(p.get("pet_score", 0) or 0) for p in products]) / 5.0
    small_ratio = sum(1 for p in products if p.get("size_grade") == "소") / n

    return np.array([avg_score, avg_price, sub_ratio, ai_ratio,
                     energy_ratio, premium_ratio, pet_avg, small_ratio])


MMR_LAMBDA = 0.5  # 점수 50% + 다양성 50%


def _jaccard_similarity(keys_a: set, keys_b: set) -> float:
    """두 패키지의 제품 ID 기반 Jaccard 유사도"""
    if not keys_a or not keys_b:
        return 0.0
    return len(keys_a & keys_b) / len(keys_a | keys_b)


def _cosine_similarity_vec(a: np.ndarray, b: np.ndarray) -> float:
    """두 특성 벡터의 코사인 유사도"""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _mmr_pick_packages(
    packages: list,
    n: int,
    existing_themed: list,
) -> list:
    """
    MMR(Maximum Marginal Relevance) 기반 패키지 선택.
    유사도 = 0.5 × 특성벡터 코사인 + 0.5 × 제품ID Jaccard
    score = λ × relevance - (1-λ) × max_similarity
    """
    if not packages:
        return []

    # 시그니처 중복 제거 (동일 조합 제외)
    existing_sigs = {_package_signature(tp["package"]) for tp in existing_themed}
    candidates = []
    for pkg in packages:
        sig = _package_signature(pkg)
        if sig not in existing_sigs:
            candidates.append(pkg)

    if not candidates:
        return []

    # 사전 계산: 특성 벡터 + 제품 키 + 점수
    vectors = [_package_feature_vector(pkg) for pkg in candidates]
    key_sets = [_package_product_keys(pkg) for pkg in candidates]
    scores = np.array([pkg.get("theme_score", pkg.get("package_score", 0.0))
                       for pkg in candidates])

    # 점수 정규화 (0~1)
    s_min, s_max = scores.min(), scores.max()
    if s_max > s_min:
        norm_scores = (scores - s_min) / (s_max - s_min)
    else:
        norm_scores = np.ones(len(scores))

    # 기존 선택 정보 (크로스 테마 다양성)
    selected_vectors = [_package_feature_vector(tp["package"]) for tp in existing_themed]
    selected_keys = [_package_product_keys(tp["package"]) for tp in existing_themed]

    selected = []
    used_indices = set()

    for _ in range(n):
        best_idx = -1
        best_mmr = -float("inf")

        for i in range(len(candidates)):
            if i in used_indices:
                continue

            relevance = norm_scores[i]

            # 하이브리드 유사도: 특성벡터 코사인 + 제품ID Jaccard
            if selected_vectors:
                max_sim = 0.0
                for sv, sk in zip(selected_vectors, selected_keys):
                    cos_sim = _cosine_similarity_vec(vectors[i], sv)
                    jac_sim = _jaccard_similarity(key_sets[i], sk)
                    hybrid = 0.5 * cos_sim + 0.5 * jac_sim
                    if hybrid > max_sim:
                        max_sim = hybrid
            else:
                max_sim = 0.0

            mmr = MMR_LAMBDA * relevance - (1 - MMR_LAMBDA) * max_sim

            if mmr > best_mmr:
                best_mmr = mmr
                best_idx = i

        if best_idx < 0:
            break

        selected.append(candidates[best_idx])
        selected_vectors.append(vectors[best_idx])
        selected_keys.append(key_sets[best_idx])
        used_indices.add(best_idx)

    return selected


def _effective_price(p: dict, prefer_subscription: bool = False) -> int:
    """
    제품의 실효 가격 반환.
    prefer_subscription=True: 구독 가능 제품은 구독 총비용(월×개월) 사용.
    (예산 부족 시 자동 활성화)
    """
    sub_price = p.get("subscription_price")
    # NaN / None 체크
    try:
        sub_price_valid = sub_price is not None and int(float(sub_price)) > 0
    except (TypeError, ValueError):
        sub_price_valid = False

    use_sub = sub_price_valid and (
        p.get("subscribe_recommended") or prefer_subscription
    )
    if use_sub:
        years = p.get("contract_period_year") or 3
        try:
            return int(float(sub_price)) * int(float(years)) * 12
        except (TypeError, ValueError):
            pass
    try:
        return int(float(p.get("price", 0) or 0))
    except (TypeError, ValueError):
        return 0


def _calc_package_score(products: list, budget: int) -> float:
    """
    PackageScore = 0.8 * 가중평균점수 + 0.2 * budget_fit
    가전 제품에 ELEC_WEIGHT, 가구에 FURN_WEIGHT 가중
    구독 추천 제품은 구독 총비용으로 budget_fit 계산
    """
    score_col  = "final_score" if "final_score" in products[0] else "derived_score"
    weighted_sum = 0.0
    weight_sum = 0.0
    for p in products:
        w = ELEC_WEIGHT if p.get("category", "") in ELECTRONICS_CATEGORIES else FURN_WEIGHT
        weighted_sum += p.get(score_col, 0.0) * w
        weight_sum += w
    avg_score = weighted_sum / weight_sum if weight_sum > 0 else 0.0

    total_price = sum(_effective_price(p) for p in products)
    if budget > 0:
        budget_fit = max(0.0, 1.0 - abs(total_price - budget) / budget)
    else:
        budget_fit = 1.0

    return 0.8 * avg_score + 0.2 * budget_fit


def generate_packages(results: dict, budget: int, candidates: dict = None, max_packages: int = None) -> list:
    """
    numpy 벡터화 전수탐색으로 조합 생성
    - candidates: 미리 선정된 후보 (None이면 _get_candidates 사용)
    - max_packages: 생성할 최대 패키지 수 (None이면 N_PACKAGES 사용)
    """
    MAX_COMBOS = 2_000_000
    if max_packages is None:
        max_packages = N_PACKAGES

    if candidates is None:
        candidates = _get_candidates(results)
    if not candidates:
        return []

    categories      = list(candidates.keys())
    candidate_lists = [candidates[cat] for cat in categories]
    n_categories    = len(candidate_lists)

    # 카테고리 수에 따라 적응형 다양성 제약
    # 12개 패키지 기준 최대 등장 횟수를 강하게 제한
    if n_categories <= 6:
        MAX_PRODUCT_APPEARANCES = 2
        adaptive_diff_ratio = MIN_DIFF_RATIO          # 0.3
    elif n_categories <= 9:
        MAX_PRODUCT_APPEARANCES = 2
        adaptive_diff_ratio = 0.25
    else:
        MAX_PRODUCT_APPEARANCES = 2
        adaptive_diff_ratio = 0.2

    # 총 조합 수 계산, 너무 많으면 카테고리당 후보 수를 줄임
    def _calc_total(clists):
        t = 1
        for cl in clists:
            t *= max(len(cl), 1)
        return t

    effective_lists = [list(cl) for cl in candidate_lists]
    while _calc_total(effective_lists) > MAX_COMBOS:
        # 가구 카테고리부터 먼저 축소하여 가전 다양성 유지
        max_furn_len = 0
        max_furn_idx = -1
        max_elec_len = 0
        max_elec_idx = -1
        for ci, cl in enumerate(effective_lists):
            if len(cl) <= 1:
                continue
            if categories[ci] in ELECTRONICS_CATEGORIES:
                if len(cl) > max_elec_len:
                    max_elec_len = len(cl)
                    max_elec_idx = ci
            else:
                if len(cl) > max_furn_len:
                    max_furn_len = len(cl)
                    max_furn_idx = ci
        # 가구 먼저 축소, 가구가 더 줄일 수 없으면 가전 축소
        if max_furn_idx >= 0 and max_furn_len > 2:
            effective_lists[max_furn_idx].pop()
        elif max_elec_idx >= 0:
            effective_lists[max_elec_idx].pop()
        elif max_furn_idx >= 0:
            effective_lists[max_furn_idx].pop()
        else:
            break
        if all(len(cl) <= 1 for cl in effective_lists):
            break

    total_combos = _calc_total(effective_lists)

    # score_col 결정
    first_product = effective_lists[0][0] if effective_lists[0] else {}
    score_col = "final_score" if "final_score" in first_product else "derived_score"

    # ── 예산 초과 판정: 최저가 조합이 예산을 넘으면 구독 가격 모드 ──
    min_prices = [min((p.get("price", 0) for p in cl), default=0) for cl in effective_lists]
    prefer_subscription = budget > 0 and sum(min_prices) > budget
    # 구독 모드 시 예산을 65%로 축소 (구독 총비용에는 서비스비/이자 포함)
    effective_budget = int(budget * 0.65) if prefer_subscription else budget

    # ── 카테고리별 점수/가격 배열 준비 ───────────────────────────
    cat_scores = []  # [np.array for each category]
    cat_prices = []
    for cl in effective_lists:
        cat_scores.append(np.array([p.get(score_col, 0.0) for p in cl]))
        cat_prices.append(np.array([_effective_price(p, prefer_subscription) for p in cl]))

    # ── numpy meshgrid로 모든 조합의 인덱스 생성 ────────────────
    ranges = [np.arange(len(cl)) for cl in effective_lists]
    grids  = np.meshgrid(*ranges, indexing="ij")
    # 각 grid를 flatten → (total_combos,) 배열
    idx_arrays = [g.ravel() for g in grids]

    # ── 가전/가구 가중치 배열 ────────────────────────────────────
    cat_weights = np.array([
        ELEC_WEIGHT if cat in ELECTRONICS_CATEGORIES else FURN_WEIGHT
        for cat in categories
    ])

    # ── 벡터화 점수/가격 계산 ────────────────────────────────────
    score_sum = np.zeros(total_combos)
    price_sum = np.zeros(total_combos, dtype=np.int64)

    for cat_i in range(n_categories):
        score_sum += cat_scores[cat_i][idx_arrays[cat_i]] * cat_weights[cat_i]
        price_sum += cat_prices[cat_i][idx_arrays[cat_i]]

    avg_scores = score_sum / cat_weights.sum()

    if effective_budget > 0:
        budget_fits = np.maximum(0.0, 1.0 - np.abs(price_sum - effective_budget) / effective_budget)
    else:
        budget_fits = np.ones(total_combos)

    package_scores = 0.8 * avg_scores + 0.2 * budget_fits

    # ── PID 인덱스 매핑 (numpy 레벨에서 빠른 다양성 체크) ────────
    # cat_pid_arrays[cat_i] = idx_arrays[cat_i]에 대응하는 product_id 배열
    cat_pid_arrays = []
    for cat_i in range(n_categories):
        pids_for_cat = np.array([p.get("product_id", 0) for p in effective_lists[cat_i]])
        cat_pid_arrays.append(pids_for_cat[idx_arrays[cat_i]])  # (total_combos,)

    # 후보가 1개뿐인 카테고리 인덱스
    single_cat_indices = set()
    for cat_i, cat in enumerate(categories):
        if len(candidates.get(cat, [])) <= 1:
            single_cat_indices.add(cat_i)

    # ── 가전/가구 그룹 인덱스 분리 (그룹별 다양성 강제용) ────────
    elec_cat_indices = [ci for ci, cat in enumerate(categories)
                        if cat in ELECTRONICS_CATEGORIES and ci not in single_cat_indices]
    furn_cat_indices = [ci for ci, cat in enumerate(categories)
                        if cat not in ELECTRONICS_CATEGORIES and ci not in single_cat_indices]

    # 그룹별 최소 차이 수: 그룹 내 카테고리가 2개 이상이면 최소 1개 차이 요구
    min_elec_diff = max(1, int(len(elec_cat_indices) * adaptive_diff_ratio)) if len(elec_cat_indices) >= 2 else 0
    min_furn_diff = max(1, int(len(furn_cat_indices) * adaptive_diff_ratio)) if len(furn_cat_indices) >= 2 else 0

    # ── 전체 정렬 ────────────────────────────────────────────────
    sorted_indices = np.argsort(package_scores)[::-1]

    # ── 스트리밍 다양성 필터 (가전/가구 그룹별 체크) ──────────────
    selected       = []
    used_pids      = {}
    cat_pid_counts = {}  # (cat_i, pid) -> count
    # 선택된 패키지별 가전/가구 pid set (그룹별 차이 비율 계산용)
    selected_elec_sets = []
    selected_furn_sets = []

    # 카테고리 수에 따라 최소 차이 제품 수 계산
    min_diff_count = max(1, int(n_categories * adaptive_diff_ratio))

    def _extract_group_pids(flat_idx, group_indices):
        return set(int(cat_pid_arrays[ci][flat_idx]) for ci in group_indices)

    for scan_i in range(total_combos):
        if len(selected) >= max_packages:
            break

        flat_idx = int(sorted_indices[scan_i])

        # 다양성 체크 1: 카테고리별 동일 제품 등장 횟수 제한
        skip = False
        for cat_i in range(n_categories):
            if cat_i in single_cat_indices:
                continue
            pid = int(cat_pid_arrays[cat_i][flat_idx])
            if cat_pid_counts.get((cat_i, pid), 0) >= MAX_PRODUCT_APPEARANCES:
                skip = True
                break
        if skip:
            continue

        # 현재 조합의 가전/가구 pid set 구성
        cur_elec = _extract_group_pids(flat_idx, elec_cat_indices)
        cur_furn = _extract_group_pids(flat_idx, furn_cat_indices)

        # 다양성 체크 2: 가전/가구 각 그룹별 최소 차이 검증
        too_similar = False
        for prev_i in range(len(selected)):
            elec_diff = len(cur_elec - selected_elec_sets[prev_i])
            furn_diff = len(cur_furn - selected_furn_sets[prev_i])

            if elec_diff < min_elec_diff or furn_diff < min_furn_diff:
                too_similar = True
                break
        if too_similar:
            continue

        # 통과한 조합만 dict 생성
        products = [effective_lists[cat_i][idx_arrays[cat_i][flat_idx]]
                    for cat_i in range(n_categories)]

        pkg_score   = float(package_scores[flat_idx])
        total_price = int(price_sum[flat_idx])

        pids    = [p.get("product_id") for p in products]
        penalty = sum(used_pids.get(pid, 0) * DIVERSITY_PENALTY for pid in pids)

        selected.append({
            "products":      products,
            "package_score": pkg_score,
            "total_price":   total_price,
            "adjusted_score": pkg_score - penalty,
        })

        # 등장 횟수 업데이트
        selected_elec_sets.append(cur_elec)
        selected_furn_sets.append(cur_furn)
        for cat_i in range(n_categories):
            pid = int(cat_pid_arrays[cat_i][flat_idx])
            cat_pid_counts[(cat_i, pid)] = cat_pid_counts.get((cat_i, pid), 0) + 1
        for pid in pids:
            used_pids[pid] = used_pids.get(pid, 0) + 1

    # 패키지 수가 부족하면 다양성 제약을 단계적으로 완화해서 재시도
    if len(selected) < max_packages:
        relaxed_elec = max(1, min_elec_diff - 1) if min_elec_diff > 1 else 0
        relaxed_furn = max(1, min_furn_diff - 1) if min_furn_diff > 1 else 0
        for scan_i in range(total_combos):
            if len(selected) >= max_packages:
                break
            flat_idx = int(sorted_indices[scan_i])

            cur_elec = _extract_group_pids(flat_idx, elec_cat_indices)
            cur_furn = _extract_group_pids(flat_idx, furn_cat_indices)

            too_similar = False
            for prev_i in range(len(selected)):
                elec_diff = len(cur_elec - selected_elec_sets[prev_i])
                furn_diff = len(cur_furn - selected_furn_sets[prev_i])
                if elec_diff < relaxed_elec or furn_diff < relaxed_furn:
                    too_similar = True
                    break
            if too_similar:
                continue

            products = [effective_lists[cat_i][idx_arrays[cat_i][flat_idx]]
                        for cat_i in range(n_categories)]
            pkg_score   = float(package_scores[flat_idx])
            total_price = int(price_sum[flat_idx])
            pids    = [p.get("product_id") for p in products]
            penalty = sum(used_pids.get(pid, 0) * DIVERSITY_PENALTY for pid in pids)

            selected.append({
                "products":      products,
                "package_score": pkg_score,
                "total_price":   total_price,
                "adjusted_score": pkg_score - penalty,
            })
            selected_elec_sets.append(cur_elec)
            selected_furn_sets.append(cur_furn)
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
    """테마별 패키지 점수 계산 (복합 지표)"""
    products    = pkg["products"]
    total_price = pkg["total_price"]
    n           = len(products) if products else 1
    score_col   = "final_score" if "final_score" in products[0] else "derived_score"
    avg_score   = float(np.mean([p.get(score_col, 0.0) for p in products]))

    if theme == "가성비":
        # 가격 저렴함 50% + 가성비 점수(value_score) 50%
        price_ratio = 1.0 - (total_price / budget) if budget > 0 else 0.0
        value_avg = float(np.mean([p.get("value_score", 5.0) or 5.0 for p in products])) / 10.0
        return 0.5 * max(0.0, price_ratio) + 0.5 * value_avg
    elif theme == "프리미엄":
        # 상품 점수 50% + 프리미엄 속성 50%
        premium_cnt = sum(1 for p in products
                         if p.get("premium_line") in ("오브제", "시그니처")
                         or p.get("material_grade") == "프리미엄")
        premium_ratio = premium_cnt / n
        return 0.5 * avg_score + 0.5 * premium_ratio
    elif theme == "효율":
        has_subscribe   = any(p.get("is_subscribe") for p in products)
        subscribe_bonus = 0.15 if has_subscribe else 0.0
        energy_cnt = sum(1 for p in products if p.get("energy_grade") in ("1등급", "2등급"))
        energy_ratio = energy_cnt / n
        return 0.5 * pkg["package_score"] + 0.3 * energy_ratio + 0.2 * subscribe_bonus
    elif theme == "펫 프렌들리":
        # 가전 pet_score + 가구 pet_score 통합 (0~1 정규화)
        pet_scores = [float(p.get("pet_score", 0) or 0) for p in products]
        max_pet = 5.0
        pet_avg = float(np.mean(pet_scores)) / max_pet if pet_scores else 0.0
        return 0.6 * pet_avg + 0.4 * avg_score
    elif theme == "공간 최적화":
        small_cnt = sum(1 for p in products if p.get("size_grade") == "소")
        space_scores = [float(p.get("space_saving_score", 0) or 0) for p in products]
        space_avg = float(np.mean(space_scores)) / 5.0 if space_scores else 0.0
        small_ratio = small_cnt / n
        return 0.4 * small_ratio + 0.3 * space_avg + 0.3 * avg_score
    elif theme == "친환경":
        eco_cnt = sum(1 for p in products if p.get("is_eco_friendly"))
        eco_ratio = eco_cnt / n
        return 0.5 * eco_ratio + 0.5 * avg_score
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
    used_cats = {p.get("category", "") for p in products if isinstance(p, dict)}

    if len(products) < min_count:
        for gp in global_pool:
            if not isinstance(gp, dict):
                continue
            k = _product_identity_key(gp)
            cat = gp.get("category", "")
            if not k or k in used:
                continue
            if cat and cat in used_cats:
                continue
            products.append(gp)
            used.add(k)
            used_cats.add(cat)
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
        # 1차: 중복 시그니처 제외하고 채움
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

    if len(adjusted) < MIN_PACKAGES:
        # 2차: 시그니처 중복도 허용하고 채움
        for pkg in all_packages:
            if len(adjusted) >= MIN_PACKAGES:
                break
            if not isinstance(pkg, dict):
                continue
            fixed_pkg = _ensure_min_products(pkg, global_pool, MIN_PRODUCTS_PER_PACKAGE)
            adjusted.append({"theme": "밸런스", "package": fixed_pkg})

    return adjusted


def select_themed_packages(all_packages: list, preferences: list, budget: int) -> list:
    """
    전체 조합 풀에서 테마별로 상위 N_PER_THEME개씩 순차 선별
    - 이전 테마에서 선택된 조합은 다음 테마 풀에서 제거
    - 공간 최적화: 60개 → 4개 선택 → 56개 남음
    - 효율:        56개 → 4개 선택 → 52개 남음
    - 펫 프렌들리: 52개 → 4개 선택
    반환: [{"theme": str, "package": dict}, ...]
    """
    themes    = _determine_themes(preferences)
    selected  = []
    remaining = list(range(len(all_packages)))  # 아직 선택 안 된 조합 인덱스

    # 전역 제품 등장 횟수 추적 → 반복 등장 시 점수 페널티
    global_pid_counts = {}
    REPEAT_PENALTY = 0.15  # 이미 등장한 제품 1회당 페널티

    # 카테고리 수 파악 (적응형 다양성 비율 결정)
    sample_n = len(all_packages[0].get("products", [])) if all_packages else 0
    if sample_n <= 6:
        theme_diff_ratio = MIN_DIFF_RATIO
    elif sample_n <= 9:
        theme_diff_ratio = 0.25
    else:
        theme_diff_ratio = 0.2

    def _split_group_keys(pkg):
        """패키지의 제품을 가전/가구 그룹별 key set으로 분리"""
        products = pkg.get("products") or []
        elec_keys = set()
        furn_keys = set()
        for p in products:
            if not isinstance(p, dict):
                continue
            k = _product_identity_key(p)
            if not k:
                continue
            cat = p.get("category", "")
            if cat in ELECTRONICS_CATEGORIES:
                elec_keys.add(k)
            else:
                furn_keys.add(k)
        return elec_keys, furn_keys

    def _global_repeat_penalty(pkg):
        """패키지 내 제품의 전역 반복 등장에 대한 페널티 합산"""
        penalty = 0.0
        for p in pkg.get("products", []):
            if not isinstance(p, dict):
                continue
            k = _product_identity_key(p)
            if k:
                penalty += global_pid_counts.get(k, 0) * REPEAT_PENALTY
        return penalty

    def _update_global_counts(pkg):
        """패키지 선택 후 전역 등장 횟수 갱신"""
        for p in pkg.get("products", []):
            if not isinstance(p, dict):
                continue
            k = _product_identity_key(p)
            if k:
                global_pid_counts[k] = global_pid_counts.get(k, 0) + 1

    for theme in themes:
        scored = sorted(
            [
                (i, _score_by_theme(all_packages[i], theme, budget) - _global_repeat_penalty(all_packages[i]))
                for i in remaining
            ],
            key=lambda x: x[1],
            reverse=True,
        )

        picked_in_theme = 0
        picked_indices  = []
        theme_elec_sets = []  # 테마 내 가전 다양성 검증용
        theme_furn_sets = []  # 테마 내 가구 다양성 검증용

        # 1차: 가전/가구 각 그룹별 다양성 기준 (전역 페널티는 scored에 반영됨)
        for i, _ in scored:
            if picked_in_theme >= N_PER_THEME:
                break

            cur_elec, cur_furn = _split_group_keys(all_packages[i])
            min_elec = max(1, int(len(cur_elec) * theme_diff_ratio)) if len(cur_elec) >= 2 else 0
            min_furn = max(1, int(len(cur_furn) * theme_diff_ratio)) if len(cur_furn) >= 2 else 0

            too_similar = False
            for prev_i in range(len(theme_elec_sets)):
                elec_diff = len(cur_elec - theme_elec_sets[prev_i])
                furn_diff = len(cur_furn - theme_furn_sets[prev_i])
                if elec_diff < min_elec or furn_diff < min_furn:
                    too_similar = True
                    break
            if too_similar:
                continue

            selected.append({"theme": theme, "package": all_packages[i]})
            picked_indices.append(i)
            theme_elec_sets.append(cur_elec)
            theme_furn_sets.append(cur_furn)
            _update_global_counts(all_packages[i])
            picked_in_theme += 1

        # 2차: 부족하면 가전/가구 각각 최소 1개 차이로 완화
        if picked_in_theme < N_PER_THEME:
            for i, _ in scored:
                if picked_in_theme >= N_PER_THEME:
                    break
                if i in picked_indices:
                    continue

                cur_elec, cur_furn = _split_group_keys(all_packages[i])
                relaxed_elec = 1 if len(cur_elec) >= 2 else 0
                relaxed_furn = 1 if len(cur_furn) >= 2 else 0

                too_similar = False
                for prev_i in range(len(theme_elec_sets)):
                    elec_diff = len(cur_elec - theme_elec_sets[prev_i])
                    furn_diff = len(cur_furn - theme_furn_sets[prev_i])
                    if elec_diff < relaxed_elec or furn_diff < relaxed_furn:
                        too_similar = True
                        break
                if too_similar:
                    continue

                selected.append({"theme": theme, "package": all_packages[i]})
                picked_indices.append(i)
                theme_elec_sets.append(cur_elec)
                theme_furn_sets.append(cur_furn)
                _update_global_counts(all_packages[i])
                picked_in_theme += 1

        # 3차: 다양성 해제, 점수순으로 채움 (페널티 반영된 순서)
        if picked_in_theme < N_PER_THEME:
            for i, _ in scored:
                if picked_in_theme >= N_PER_THEME:
                    break
                if i in picked_indices:
                    continue
                selected.append({"theme": theme, "package": all_packages[i]})
                picked_indices.append(i)
                picked_in_theme += 1

        # 선택된 조합을 풀에서 제거
        for i in picked_indices:
            remaining.remove(i)

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
    "TV", "스탠바이미", "냉장고", "전기레인지", "오븐", "전자레인지",
    "식기세척기", "정수기", "세탁기", "워시타워", "워시콤보",
    "의류관리기", "의류건조기", "청소기", "에어컨", "공기청정기",
    "제습기", "가습기",
}


def _format_appliance(p: dict) -> dict:
    return {
        "productId":            p.get("product_id"),
        "modelId":              p.get("model_id"),
        "brand":                p.get("brand", ""),
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
        "productId":  p.get("product_id"),
        "modelId":    p.get("model_id"),
        "brand":      p.get("brand", ""),
        "name":       p.get("name", ""),
        "category":   p.get("category", ""),
        "price":      int(p.get("price", 0)),
        "image":      p.get("product_image_url", ""),
        "productUrl": p.get("product_url", ""),
    }


def format_output(
    themed_packages: list,
    reasons: list,
    needed_categories: set = None,
    budget: int = 0,
) -> dict:
    """
    테마별 패키지 + 추천 이유 → 프론트 요청 형태로 변환
    """
    needed_categories = needed_categories or set()
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

        # ── recommendationPlus 생성 ──
        plus_parts = []

        # 1. 구독 추천 안내: 패키지 내 구독 제품이 절반 이상이면
        if budget > 0 and appliances:
            sub_count = sum(1 for a in appliances if a.get("subscriptionPrice", 0) > 0)
            if sub_count >= len(appliances) * 0.5:
                plus_parts.append(
                    "예산 대비 필요한 가전이 많아 구독 상품을 중심으로 추천했어요. "
                    "월 구독으로 부담 없이 시작할 수 있어요."
                )

        # 2. 누락 카테고리 안내
        pkg_cats = {a.get("category", "") for a in appliances} | {f.get("category", "") for f in furniture}
        missing = needed_categories - pkg_cats - {""}
        if missing:
            missing_str = ", ".join(sorted(missing))
            plus_parts.append(
                f"{missing_str}은(는) 예산 내 적합한 제품을 찾지 못해 이 패키지에 포함되지 않았어요."
            )

        output_packages.append({
            "theme":                theme,
            "appliances":           appliances,
            "furniture":            furniture,
            "recommendationReason": reason,
            "recommendationPlus":   " ".join(plus_parts) if plus_parts else None,
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
    needed_categories: list = None,
) -> dict:
    """
    멀티 전략 앙상블 + 탐욕 다양성 구성.
    테마별로 다른 후보를 선정하고, 탐욕적으로 다양한 패키지를 구성.
    """
    preferences = preferences or []

    # 1. 재정렬
    reranked = rerank(results)

    # 2. 테마 결정
    themes = _determine_themes(preferences)

    # 3. 멀티 전략 앙상블 + MMR 다양성 선택
    themed_packages = []

    for theme in themes:
        # 테마별 후보 선정 (다른 정렬 기준)
        theme_candidates = _get_theme_candidates(reranked, theme)

        # 테마별 조합 생성 (풀 20개)
        packages = generate_packages(
            reranked, budget,
            candidates=theme_candidates,
            max_packages=30,
        )

        # 테마 점수로 재정렬
        for pkg in packages:
            pkg["theme_score"] = _score_by_theme(pkg, theme, budget)
        packages.sort(key=lambda p: p["theme_score"], reverse=True)

        # 탐욕적으로 4개 선택
        picked = _mmr_pick_packages(packages, N_PER_THEME, themed_packages)
        for pkg in picked:
            themed_packages.append({"theme": theme, "package": pkg})

    # 4. 최소 12개 보장 — 부족하면 밸런스 풀에서 추가
    if len(themed_packages) < MIN_PACKAGES:
        fallback = generate_packages(reranked, budget)
        themed_packages = _enforce_minimum_output(themed_packages, fallback, reranked)

    # 5. 추천 이유 생성
    if use_llm and themed_packages:
        pkg_list = [item["package"] for item in themed_packages]
        themes   = [item["theme"] for item in themed_packages]
        reasons  = generate_reasons(pkg_list, starter, preferences, budget, square_footage, themes)
    else:
        reasons = ["test"] * len(themed_packages)

    # 6. 출력 포맷
    all_needed = set(needed_categories or list(results.keys()))
    return format_output(themed_packages, reasons, all_needed, budget)