"""
recommendation_algorithm 폴더의 pipeline.run_full_pipeline 을 호출하기 위한 어댑터.
템플릿(pipeline.py, db.py 등)은 수정하지 않습니다.

환경 변수 RECOMMENDATION_ALGORITHM_PATH 가 설정되어 있고
RECOMMENDATION_ALGORITHM_ENTRYPOINT 가 비어 있거나 pipeline:run_full_pipeline 이면 이 경로를 탑니다.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def _parse_square_meters(size_val: Any) -> int:
    """user_info['size'] 예: '10~20평', '15' → 대략 평수 숫자."""
    s = str(size_val or "").strip()
    if not s:
        return 15
    m = re.search(r"(\d+)\s*~\s*(\d+)", s)
    if m:
        return (int(m.group(1)) + int(m.group(2))) // 2
    m2 = re.search(r"(\d+)", s)
    if m2:
        return int(m2.group(1))
    return 15


def _budget_won(user_info: Dict[str, Any]) -> int:
    """진단 user_info → 원화 예산."""
    if user_info.get("budget_manwon") is not None:
        try:
            return int(float(str(user_info["budget_manwon"]).replace(",", ""))) * 10_000
        except (TypeError, ValueError):
            pass
    rng = user_info.get("budget_range_manwon")
    if isinstance(rng, dict) and rng.get("max") is not None:
        try:
            return int(float(str(rng["max"]).replace(",", ""))) * 10_000
        except (TypeError, ValueError):
            pass
    return 3_000_000


def _str_list(val: Any) -> List[str]:
    if val is None:
        return []
    if isinstance(val, list):
        return [str(x).strip() for x in val if str(x).strip()]
    if isinstance(val, str):
        if "," in val:
            return [x.strip() for x in val.split(",") if x.strip()]
        return [val.strip()] if val.strip() else []
    return []


def build_input_data_from_user_info(user_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph user_info → 파이프라인 2.9.1 input JSON 형태.
    """
    owned = _str_list(user_info.get("owned_appliances"))
    needed_el = _str_list(user_info.get("needed_appliances"))
    furniture_needed = _str_list(user_info.get("furniture_needed"))
    if not furniture_needed:
        furniture_needed = _str_list(user_info.get("purchase_plans"))

    lifestyle = user_info.get("lifestyle") or ""
    preferences: List[str] = []
    if isinstance(lifestyle, str) and lifestyle.strip():
        preferences = [x.strip() for x in lifestyle.split(",") if x.strip()]
    extra = user_info.get("preferences")
    if isinstance(extra, list):
        preferences = [str(x) for x in extra if str(x).strip()] or preferences

    starter = (
        user_info.get("starter_package")
        or user_info.get("starterPackage")
        or (preferences[0] if preferences else None)
        or "혼자 사는 라이프"
    )

    style = (
        user_info.get("style")
        or user_info.get("interior_style")
        or "모던/미니멀"
    )

    return {
        "starterPackage": str(starter),
        "budget": _budget_won(user_info),
        "square_footage": _parse_square_meters(user_info.get("size")),
        "product": {
            "electronics": {
                "owned": owned,
                "needed": needed_el,
            },
            "furniture": {
                "needed": furniture_needed,
            },
        },
        "preferences": preferences or ["공간 활용이 중요해요"],
        "style": str(style),
    }


def _map_appliance_to_product(a: Dict[str, Any]) -> Dict[str, Any]:
    img = a.get("image") or ""
    url = a.get("productUrl") or a.get("product_url") or ""
    return {
        "product_id": a.get("product_id"),
        "category": "appliance",
        "product_name": a.get("name") or "",
        "name": a.get("name") or "",
        "model_id": str(a.get("model_id") or a.get("modelId") or ""),
        "brand": a.get("brand") or "",
        "price_normal": int(a.get("totalPrice") or a.get("total_price") or 0),
        "price_subscription": int(a.get("subscriptionPrice") or a.get("subscription_price") or 0),
        "product_url": url,
        "product_image_url": img if str(img).startswith("http") else img,
        "popularityScore": a.get("popularityScore"),
    }


def _map_furniture_to_product(f: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "product_id": f.get("product_id"),
        "category": "furniture",
        "product_name": f.get("name") or "",
        "name": f.get("name") or "",
        "model_id": str(f.get("model_id") or ""),
        "brand": f.get("brand") or "",
        "price": int(f.get("price") or 0),
        "product_url": f.get("productUrl") or f.get("product_url") or "",
        "product_image_url": f.get("image") or f.get("product_image_url") or "",
    }


def pipeline_output_to_recommendation_list(output: Any) -> Dict[str, Any]:
    """
    파이프라인 output( packages: 12세트 등 ) → graph.node_chat_result 가 쓰는 형태.

    프론트 요구: packages[].appliances, furniture, recommendationReason
    graph 후처리: products 병합 + appliances/furniture 자동 분리
    """
    if not isinstance(output, dict):
        raise TypeError(f"pipeline output must be dict, got {type(output)!r}")

    packages = output.get("packages")
    if not isinstance(packages, list):
        packages = []

    recommendation_list: List[Dict[str, Any]] = []
    for i, pkg in enumerate(packages):
        if not isinstance(pkg, dict):
            continue
        theme = pkg.get("theme") or f"패키지 {i + 1}"
        reason = pkg.get("recommendationReason") or pkg.get("reason") or ""
        appliances_raw = pkg.get("appliances") if isinstance(pkg.get("appliances"), list) else []
        furniture_raw = pkg.get("furniture") if isinstance(pkg.get("furniture"), list) else []

        products: List[Dict[str, Any]] = []
        for a in appliances_raw:
            if isinstance(a, dict):
                products.append(_map_appliance_to_product(a))
        for f in furniture_raw:
            if isinstance(f, dict):
                products.append(_map_furniture_to_product(f))

        recommendation_list.append(
            {
                "name": theme,
                "package_name": theme,
                "theme": theme,
                "reason": reason,
                "recommendationReason": reason,
                "appliances": appliances_raw,
                "furniture": furniture_raw,
                "products": products,
            }
        )

    total_budget = ""
    if recommendation_list:
        # 대략 표시용
        try:
            tot = 0
            for pkg in recommendation_list:
                for p in pkg.get("products") or []:
                    if not isinstance(p, dict):
                        continue
                    if p.get("category") == "appliance":
                        tot += int(p.get("price_normal") or 0)
                        tot += int(p.get("price_subscription") or 0)
                    else:
                        tot += int(p.get("price") or 0)
            total_budget = f"약 {tot:,}원"
        except (TypeError, ValueError):
            total_budget = output.get("total_estimated_budget") or ""

    return {
        "recommendation_list": recommendation_list,
        "total_estimated_budget": str(output.get("total_estimated_budget") or total_budget),
    }


def run_full_pipeline_wrapped(
    *,
    user_info: Dict[str, Any],
    candidates: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    recommendation_algorithm 내 pipeline.run_full_pipeline(input_data, engine, use_llm=...) 호출.
    use_llm 은 기본 False (템플릿 내 추천 이유 LLM 비활성). 켜려면 RECOMMENDATION_PIPELINE_USE_LLM=1.
    candidates는 파이프라인이 DB에서 직접 조회한다면 사용하지 않음(시그니처 호환용).
    """
    _ = candidates
    root = os.getenv("RECOMMENDATION_ALGORITHM_PATH", "").strip()
    if not root or not os.path.isdir(root):
        raise FileNotFoundError(f"RECOMMENDATION_ALGORITHM_PATH 유효하지 않음: {root!r}")

    # 템플릿은 보통 recommendation_algorithm/script/ 아래에 pipeline.py, db.py 가 있음
    # (from db import … 형태이므로 script 디렉터리를 sys.path 에 넣어야 함)
    script_dir = os.path.join(root, "script")
    path_to_prepend = script_dir if os.path.isdir(script_dir) else root
    if path_to_prepend not in sys.path:
        sys.path.insert(0, path_to_prepend)

    input_data = build_input_data_from_user_info(user_info)

    logger.info("pipeline input_data keys: %s", list(input_data.keys()))

    # 템플릿 모듈 (수정 금지 — import만)
    from pipeline import run_full_pipeline
    from db import get_engine

    # 템플릿 scoring.run_scoring(use_llm=True) 시 recommendation_reason.generate_reasons(LLM) 호출됨.
    # 기본은 LLM 끔 — 이유만 필요하면 RECOMMENDATION_PIPELINE_USE_LLM=1
    use_llm_reasons = os.getenv("RECOMMENDATION_PIPELINE_USE_LLM", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )

    with get_engine() as engine:
        output = run_full_pipeline(input_data, engine, use_llm=use_llm_reasons)

    if not use_llm_reasons and isinstance(output, dict):
        for pkg in output.get("packages") or []:
            if not isinstance(pkg, dict):
                continue
            r = pkg.get("recommendationReason") or pkg.get("reason") or ""
            if r in ("", "test"):
                theme = pkg.get("theme") or "추천"
                pkg["recommendationReason"] = f"「{theme}」 패키지 구성입니다."

    return pipeline_output_to_recommendation_list(output)
