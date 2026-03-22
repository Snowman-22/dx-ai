"""
image_score.py
- 사용자 스타일 선택 → 인테리어 대표 벡터 로드
- DB에서 가구 후보 image_vector 조회
- 코사인 유사도 → ImageScore 계산

사용:
    from image_score import calc_image_scores
    df_furniture = calc_image_scores(df_furniture, style, engine)
"""

import numpy as np
import pandas as pd
from sqlalchemy import text
from typing import Optional


# ------------------------------------------------------------------ #
#  인테리어 스타일 벡터 파일 경로
# ------------------------------------------------------------------ #

# INTERIOR_VECTORS_PATH = r"C:\Users\4243\LGDX\0.Project\3.DX_project\recommendation_algorithm\data\image\interior_style_vectors.npy"
INTERIOR_VECTORS_PATH = "/data/recommendation_algorithm/file/interior_style_vectors.npy"

# input style 값 → npy 키 매핑
STYLE_KEY_MAP = {
    "모던/미니멀": "모던미니멀",
    "내추럴/우드": "내추럴우드",
    "컬러풀":      "컬러풀",
}


# ================================================================== #
#  인테리어 스타일 벡터 로드
# ================================================================== #

def load_style_vector(style: str) -> Optional[np.ndarray]:
    """
    사용자 선택 스타일에 해당하는 인테리어 대표 벡터 반환
    style: input_data의 style 값 (예: "모던/미니멀")
    반환: np.ndarray (512,) 또는 None (스타일 없으면)
    """
    key = STYLE_KEY_MAP.get(style)
    if key is None:
        return None

    try:
        style_vectors = np.load(INTERIOR_VECTORS_PATH, allow_pickle=True).item()
        return style_vectors.get(key)
    except FileNotFoundError:
        print(f"[image_score] 벡터 파일 없음: {INTERIOR_VECTORS_PATH}")
        return None


# ================================================================== #
#  DB에서 image_vector 조회
# ================================================================== #

def fetch_image_vectors(engine, product_ids: list) -> dict:
    """
    furniture_derived 테이블에서 image_vector 조회
    반환: {product_id: np.ndarray(512,)}
    """
    placeholders = ", ".join([f":pid{i}" for i in range(len(product_ids))])
    params = {f"pid{i}": pid for i, pid in enumerate(product_ids)}
 
    query = text(f"""
        SELECT product_id, image_vector
        FROM furniture_derived
        WHERE product_id IN ({placeholders})
          AND image_vector IS NOT NULL
    """)
 
    with engine.connect() as conn:
        rows = conn.execute(query, params).fetchall()
 
    result = {}
    for row in rows:
        val = row[1]
        if isinstance(val, str):
            # pgvector가 '[0.1, 0.2, ...]' 문자열로 반환하는 경우 파싱
            val = val.strip("[]").split(",")
        result[row[0]] = np.array(val, dtype=np.float32)
    return result


# ================================================================== #
#  코사인 유사도 계산
# ================================================================== #

def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    두 벡터의 코사인 유사도 반환 (-1 ~ 1)
    두 벡터 모두 L2 정규화된 상태이면 내적만으로 계산 가능
    """
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def similarity_to_score(similarity: float) -> float:
    """
    코사인 유사도(-1~1) → ImageScore(0~1) 변환
    (-1~1) → (0~1) 선형 변환
    """
    return float((similarity + 1.0) / 2.0)


# ================================================================== #
#  ImageScore 계산 메인 함수
# ================================================================== #

def calc_image_scores(
    df: pd.DataFrame,
    style: Optional[str],
    engine,
) -> pd.DataFrame:
    """
    가구 후보 df에 image_score 컬럼 추가
    - style이 없거나 벡터 파일이 없으면 image_score = 0.5 (중립값)
    - image_vector가 없는 제품도 image_score = 0.5

    df: furniture 후보 DataFrame (product_id 컬럼 필요)
    style: 사용자 스타일 선택값 (예: "모던/미니멀")
    engine: sqlalchemy engine
    반환: image_score 컬럼이 추가된 df
    """
    df = df.copy()

    # 스타일 미선택 or 벡터 파일 없으면 중립값
    style_vec = load_style_vector(style) if style else None
    if style_vec is None:
        df["image_score"] = 0.5
        return df

    # DB에서 image_vector 조회
    product_ids  = df["product_id"].tolist()
    image_vectors = fetch_image_vectors(engine, product_ids)

    # 코사인 유사도 → ImageScore
    def get_score(product_id):
        vec = image_vectors.get(product_id)
        if vec is None:
            return 0.5  # 벡터 없는 제품은 중립값
        sim = cosine_similarity(style_vec, vec)
        return similarity_to_score(sim)

    df["image_score"] = df["product_id"].apply(get_score)
    return df


# ================================================================== #
#  FinalScore 계산
# ================================================================== #

def calc_final_score_furniture(df: pd.DataFrame) -> pd.DataFrame:
    """
    FinalScore = 0.6 * derived_score + 0.4 * image_score
    df: derived_score, image_score 컬럼 필요
    """
    df = df.copy()
    df["final_score"] = 0.6 * df["derived_score"] + 0.4 * df["image_score"]
    return df
