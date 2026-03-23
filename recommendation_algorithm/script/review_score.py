"""
review_score.py
- 저장된 클러스터 모델, 상품 태그, 리뷰 임베딩을 활용해 ReviewScore 계산

계산식:
    ReviewScore = 0.4 * cluster_preference + 0.6 * review_profile_match
    review_profile_match = 0.7 * embedding_similarity + 0.3 * tag_overlap_score
"""

import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sqlalchemy import text


# ── 설정 ────────────────────────────────────────────────────────────

def _algorithm_root() -> Path:
    env_root = os.getenv("RECOMMENDATION_ALGORITHM_PATH", "").strip()
    if env_root:
        return Path(env_root)
    return Path(__file__).resolve().parents[1]


# CLUSTER_MODEL_PATH = r"C:\Users\4243\LGDX\0.Project\3.DX_project\recommendation_algorithm\data\review\review_cluster_model.pkl"
CLUSTER_MODEL_PATH = _algorithm_root() / "file" / "review_cluster_model.pkl"
EMBED_MODEL_NAME   = "jhgan/ko-sroberta-multitask"

# 선택지 → 사용자 프로필 키워드
PREFERENCE_KEYWORDS = {
    "공간 활용이 중요해요":      ["공간", "작다", "컴팩트", "슬림", "절약"],
    "큰 제품도 괜찮아요":        ["대용량", "크다", "넉넉하다", "여유"],
    "수납이 넉넉했으면 좋겠어요":  ["수납", "정리", "서랍", "공간"],
    "가성비가 중요해요":         ["가성비", "저렴하다", "합리적", "가격"],
    "할인 혜택이 중요해요":       ["할인", "세일", "저렴하다"],
    "가격보다 만족도가 중요해요":  ["만족하다", "추천", "좋다", "재구매"],
    "프리미엄 제품도 고려해요":    ["고급", "프리미엄", "디자인", "완성도"],
    "간단 요리를 자주 해요":      ["간편", "혼밥", "빠르다", "간단"],
    "집에서 보내는 시간이 많아요": ["집콕", "오래", "만족하다"],
    "집에서 일하는 시간이 많아요": ["재택", "오래", "편리하다"],
    "청소와 관리가 쉬운 게 좋아요": ["청소", "관리", "편하다", "유지"],
    "자동화 기능(AI)이 필요해요":  ["자동", "AI", "스마트", "편리하다"],
    "사용이 쉬운 제품이 좋아요":   ["간단", "조작", "직관적", "편하다"],
    "소음이 적은 제품이 좋아요":   ["조용하다", "소음", "무소음"],
    "에너지 효율이 중요해요":      ["절전", "에너지", "전기세", "효율"],
    "친환경 소재를 선호해요":      ["친환경", "안전", "무해"],
    "내추럴/우드 스타일이 좋아요":  ["우드", "내추럴", "자연"],
    "화이트/밝은 톤이 좋아요":     ["화이트", "밝다", "깔끔하다"],
    "반려동물과 함께 살아요":      ["펫", "반려동물", "털", "청소"],
}

STARTER_KEYWORDS = {
    "혼자 사는 라이프":       ["혼자", "1인", "원룸", "소형", "간편"],
    "둘이 함께 시작하는 집":   ["둘", "커플", "신혼", "2인"],
    "아기와 함께하는 집":      ["아기", "유아", "안전", "친환경"],
    "자녀와 함께하는 집":      ["가족", "아이", "대용량", "넓다"],
    "부모님과 함께하는 집":    ["부모님", "어르신", "편리하다", "간단"],
    "여유로운 시니어 라이프":   ["시니어", "편리하다", "건강", "조용하다"],
}


# ================================================================== #
#  모델/리소스 로드 (싱글톤)
# ================================================================== #

_cluster_bundle = None
_embed_model    = None


def _load_cluster_bundle() -> dict:
    global _cluster_bundle
    if _cluster_bundle is None:
        try:
            with open(CLUSTER_MODEL_PATH, "rb") as f:
                _cluster_bundle = pickle.load(f)
        except FileNotFoundError:
            print(f"[review_score] 클러스터 모델 없음: {CLUSTER_MODEL_PATH}")
            _cluster_bundle = {}
    return _cluster_bundle


def _load_embed_model() -> Optional[SentenceTransformer]:
    global _embed_model
    if _embed_model is None:
        try:
            _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
        except Exception as e:
            print(f"[review_score] 임베딩 모델 로드 실패: {e}")
    return _embed_model


# ================================================================== #
#  사용자 프로필 텍스트/문장 생성
# ================================================================== #

def build_user_profile_text(starter: str, preferences: list) -> str:
    """TF-IDF 비교용 공백 구분 키워드 문자열"""
    words = STARTER_KEYWORDS.get(starter, [])
    for pref in preferences:
        words += PREFERENCE_KEYWORDS.get(pref, [])
    return " ".join(words)


def build_user_profile_sentence(starter: str, preferences: list) -> str:
    """임베딩용 자연어 문장"""
    parts = []
    if starter in STARTER_KEYWORDS:
        parts.append(" ".join(STARTER_KEYWORDS[starter]))
    for pref in preferences:
        kws = PREFERENCE_KEYWORDS.get(pref, [])
        if kws:
            parts.append(" ".join(kws))
    return " ".join(parts)


# ================================================================== #
#  1. cluster_preference
# ================================================================== #

def _classify_cluster(user_text: str, bundle: dict) -> int:
    """사용자 프로필 → 가장 가까운 클러스터 ID"""
    vec = bundle["vectorizer"].transform([user_text])
    return int(bundle["kmeans"].predict(vec)[0])


def calc_cluster_preference(df: pd.DataFrame, cluster_products: list) -> pd.Series:
    """
    클러스터 추천 상품 순위에 따라 선형 감소 점수 부여
    1위 = 1.0, 꼴찌 = 0.5, 미포함 = 0.0
    """
    n = len(cluster_products)
    rank_score = {
        pid: 1.0 - 0.5 * (i / max(n - 1, 1))
        for i, pid in enumerate(cluster_products)
    }
    return df["product_id"].map(lambda pid: rank_score.get(pid, 0.0))


# ================================================================== #
#  2. tag_overlap_score
# ================================================================== #

def fetch_product_tags(engine, product_ids: list) -> dict:
    """DB의 product_tags 테이블에서 상품별 태그 조회"""
    placeholders = ", ".join([f":p{i}" for i in range(len(product_ids))])
    params = {f"p{i}": pid for i, pid in enumerate(product_ids)}
    with engine.connect() as conn:
        rows = conn.execute(
            text(f"SELECT product_id, tags FROM product_tags WHERE product_id IN ({placeholders})"),
            params
        ).fetchall()
    return {row[0]: row[1] for row in rows}


def calc_tag_overlap_score(df: pd.DataFrame, user_text: str, product_tags: dict) -> pd.Series:
    """
    상품별 태그와 사용자 키워드 간 겹침 비율
    tag_overlap_score = 겹친 태그 수 / 상품 태그 전체 수
    """
    user_words = set(user_text.split())

    def score(product_id):
        tags = product_tags.get(product_id)
        if not tags:
            return 0.5  # 태그 없으면 중립
        overlap = len(user_words & set(tags))
        return min(1.0, overlap / len(tags))

    return df["product_id"].map(score)


# ================================================================== #
#  3. embedding_similarity
# ================================================================== #

def fetch_review_vectors(engine, product_ids: list) -> dict:
    """DB의 product_review_embeddings 테이블에서 리뷰 벡터 조회"""
    placeholders = ", ".join([f":p{i}" for i in range(len(product_ids))])
    params = {f"p{i}": pid for i, pid in enumerate(product_ids)}
    with engine.connect() as conn:
        rows = conn.execute(
            text(f"""
                SELECT product_id, review_vector
                FROM product_review_embeddings
                WHERE product_id IN ({placeholders})
            """),
            params
        ).fetchall()

    result = {}
    for row in rows:
        val = row[1]
        if isinstance(val, str):
            val = val.strip("[]").split(",")
        result[row[0]] = np.array(val, dtype=np.float32)
    return result


def calc_embedding_similarity(
    df: pd.DataFrame,
    user_sentence: str,
    review_vectors: dict,
    embed_model,
) -> pd.Series:
    """
    사용자 프로필 문장 임베딩 vs 상품별 리뷰 임베딩 코사인 유사도
    (-1~1) → (0~1) 변환
    """
    user_vec = embed_model.encode(
        [user_sentence], normalize_embeddings=True
    )[0]

    def score(product_id):
        vec = review_vectors.get(product_id)
        if vec is None:
            return 0.5  # 벡터 없으면 중립
        sim = float(np.dot(user_vec, vec))   # L2 정규화된 벡터 → 내적 = 코사인 유사도
        return (sim + 1.0) / 2.0             # -1~1 → 0~1

    return df["product_id"].map(score)


# ================================================================== #
#  ReviewScore 계산 메인 함수
# ================================================================== #

def calc_review_scores(
    df: pd.DataFrame,
    starter: str,
    preferences: list,
    engine,
) -> pd.DataFrame:
    """
    가전 후보 df에 review_score 컬럼 추가

    ReviewScore = 0.4 * cluster_preference
                + 0.6 * (0.7 * embedding_similarity + 0.3 * tag_overlap_score)
    """
    df = df.copy()

    # 리소스 로드
    bundle      = _load_cluster_bundle()
    embed_model = _load_embed_model()

    # 리소스 없으면 중립값
    if not bundle or embed_model is None:
        df["review_score"] = 0.5
        return df

    user_text     = build_user_profile_text(starter, preferences)
    user_sentence = build_user_profile_sentence(starter, preferences)

    if not user_text.strip():
        df["review_score"] = 0.5
        return df

    product_ids = df["product_id"].tolist()

    # 1. cluster_preference
    cluster_id    = _classify_cluster(user_text, bundle)
    cluster_prods = bundle["cluster_products"].get(cluster_id, [])
    cp_score      = calc_cluster_preference(df, cluster_prods)

    # 2. tag_overlap_score
    product_tags  = fetch_product_tags(engine, product_ids)
    tag_score     = calc_tag_overlap_score(df, user_text, product_tags)

    # 3. embedding_similarity
    review_vectors = fetch_review_vectors(engine, product_ids)
    emb_score      = calc_embedding_similarity(df, user_sentence, review_vectors, embed_model)

    # review_profile_match
    profile_match = 0.7 * emb_score + 0.3 * tag_score

    # ReviewScore
    df["review_score"] = 0.4 * cp_score + 0.6 * profile_match

    return df


# ================================================================== #
#  FinalScore
# ================================================================== #

def calc_final_score_electronics(df: pd.DataFrame) -> pd.DataFrame:
    """FinalScore = 0.7 * derived_score + 0.3 * review_score"""
    df = df.copy()
    df["final_score"] = 0.7 * df["derived_score"] + 0.3 * df["review_score"]
    return df
