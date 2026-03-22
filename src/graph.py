from __future__ import annotations

import json
import re
from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START

from .db import get_session_maker
from .product_details import fetch_products_bundle_details
from .state_store import get_checkpointer
from .prompt import (
    build_package_reason_prompt,
    build_blueprint_rag_prompt,
    build_recommendation_prompt,
    build_rag_prompt,
    build_rag_prompt_with_package_context,
)
from .recommend import generate_recommendation_result
from .recommend.service import fetch_candidate_products


class ChatStep(str, Enum):
    # UI 순서: 예산 → 평수 → 라이프스타일 → 보유+필요 가전(한 단계) → 구매계획 → 추천
    CHAT_0 = "CHAT_0"  # 진단 시작(인트로)
    CHAT_1 = "CHAT_1"  # 총 예산(만원, 숫자)
    CHAT_2 = "CHAT_2"  # 평수(칩/텍스트)
    CHAT_3 = "CHAT_3"  # 라이프스타일(칩/텍스트)
    CHAT_4 = "CHAT_4"  # 보유 + 필요(필수) 가전 — 한 요청(owned/needed)
    CHAT_5 = "CHAT_5"  # 구매 계획(가구/가전 텍스트)
    CHAT_6 = "CHAT_6"  # 추천 리스트 생성
    CHAT_RESULT = "CHAT_RESULT"  # 추천 결과(내부 상태)
    RECOMMEND_RAG = "RECOMMEND_RAG"  # 추천 이후 일반 질의(도면·배치 없음)
    BLUEPRINT_RAG = "BLUEPRINT_RAG"  # 도면 단계: 선택 도면 + 패키지 + 스티커 좌표 RAG
    CHAT_11 = "CHAT_11"  # 진단 종료


class ChatState(TypedDict, total=False):
    step: ChatStep
    user_info: Dict[str, Any]
    messages: List[Dict[str, Any]]
    incoming_assistant_message: Any
    last_user_input: Any
    data: Dict[str, Any]
    is_completed: bool
    ai_response: str
    requested_step_code: Optional[str]


llm_json = ChatOpenAI(model="gpt-4o-mini", temperature=0, response_format={"type": "json_object"})
llm_text = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

# --- 헬퍼 함수들 ---
def _to_str(x: Any) -> str:
    return str(x).strip() if x is not None else ""

def _extract_json_from_llm_response(resp: Any) -> dict[str, Any]:
    import json as _json
    if hasattr(resp, "content"):
        content = resp.content
    else:
        try:
            content = resp.dict().get("content")
        except Exception:
            content = None

    if isinstance(content, str):
        return _json.loads(content)
    if isinstance(content, dict):
        return content
    return _json.loads(str(content))


def _to_int_price(x: Any) -> int:
    if x is None: return 0
    if isinstance(x, bool): return int(x)
    if isinstance(x, (int, float)): return int(x)
    if isinstance(x, str):
        s = x.strip().replace(",", "")
        try:
            return int(float(s)) if "." in s else int(s)
        except ValueError:
            return 0
    return 0


def _compute_budget_breakdown(item: Dict[str, Any]) -> Dict[str, Any]:
    products = item.get("products")
    if isinstance(products, list):
        appliance_normal_sum = 0
        appliance_subscription_sum = 0
        furniture_price_sum = 0
        for p in products:
            if not isinstance(p, dict): continue
            category = p.get("category")
            is_appliance = category == "appliance" or ("price_normal" in p or "price_subscription" in p)
            is_furniture = category == "furniture" or ("price" in p and not is_appliance)
            if is_appliance:
                appliance_normal_sum += _to_int_price(p.get("price_normal"))
                appliance_subscription_sum += _to_int_price(p.get("price_subscription"))
            if is_furniture:
                furniture_price_sum += _to_int_price(p.get("price"))
        return {
            "appliance_price_normal_sum": appliance_normal_sum,
            "appliance_price_subscription_sum": appliance_subscription_sum,
            "furniture_price_sum": furniture_price_sum,
        }
    return {
        "appliance_price_normal_sum": _to_int_price(item.get("price_normal")),
        "appliance_price_subscription_sum": _to_int_price(item.get("price_subscription")),
        "furniture_price_sum": _to_int_price(item.get("price")),
    }

def _append_message(state: ChatState, *, role: str, content: Any) -> List[Dict[str, Any]]:
    messages = list(state.get("messages") or [])
    messages.append({"role": role, "content": content})
    return messages


def _normalize_name(s: Any) -> str:
    return _to_str(s).lower().replace(" ", "")


def _enforce_products_from_catalog(
    products: Any,
    *,
    catalog_by_model_id: Dict[str, Dict[str, Any]],
    catalog_by_name: Dict[str, Dict[str, Any]],
) -> list[Dict[str, Any]]:
    """
    LLM fallback 결과를 DB 카탈로그 기준으로 보정합니다.
    - model_id 우선 매칭, 없으면 product_name 매칭
    - 카탈로그에 없는 항목은 제거(임의 상품 방지)
    """
    if not isinstance(products, list):
        return []

    out: list[Dict[str, Any]] = []
    for p in products:
        if not isinstance(p, dict):
            continue
        raw_mid = _to_str(p.get("model_id") or p.get("model"))
        raw_name = _to_str(p.get("product_name") or p.get("name"))

        matched: Optional[Dict[str, Any]] = None
        if raw_mid and raw_mid in catalog_by_model_id:
            matched = catalog_by_model_id[raw_mid]
        elif raw_name:
            key = _normalize_name(raw_name)
            matched = catalog_by_name.get(key)

        if matched is None:
            continue

        cat = p.get("category") or matched.get("category")
        price = matched.get("price")
        out.append({
            "category": cat,
            "product_name": matched.get("name"),
            "model_id": matched.get("model_id") or "",
            "brand": matched.get("brand"),
            "price_normal": price if cat == "appliance" else 0,
            "price_subscription": 0,
            "price": price,
            "product_image_url": matched.get("image_url") or "",
            "product_url": matched.get("url") or "",
        })
    return out


def _get_budget_max_won(user_info: Dict[str, Any]) -> Optional[int]:
    """
    user_info의 예산 정보를 원화 상한으로 변환합니다.
    - budget_manwon: 숫자 입력(만원 단위)
    - budget_range_manwon.max: 선택 범위 상한(만원 단위)
    """
    if user_info.get("budget_manwon") is not None:
        return _to_int_price(user_info.get("budget_manwon")) * 10_000
    rng = user_info.get("budget_range_manwon")
    if isinstance(rng, dict):
        mx = rng.get("max")
        if mx is not None:
            return _to_int_price(mx) * 10_000
    return None


def _product_price_won(p: Dict[str, Any]) -> int:
    category = p.get("category")
    if category == "appliance":
        # 일시불 기준 예산 컷오프를 기본으로 적용
        return _to_int_price(p.get("price_normal") or p.get("price"))
    return _to_int_price(p.get("price"))


def _apply_budget_cap_to_products(
    products: list[Dict[str, Any]],
    *,
    budget_max_won: Optional[int],
) -> list[Dict[str, Any]]:
    if budget_max_won is None or budget_max_won <= 0:
        return products

    picked: list[Dict[str, Any]] = []
    total = 0
    # 카테고리 순서를 유지한 채 누적 컷오프로 예산 상한 적용
    for p in products:
        if not isinstance(p, dict):
            continue
        price = _product_price_won(p)
        if price <= 0:
            continue
        if total + price <= budget_max_won:
            picked.append(p)
            total += price
    # 상한이 너무 타이트해서 아무것도 못 담는 경우, 원본을 유지해 빈 패키지 방지
    return picked or products

def _parse_budget(user_text: Any) -> Dict[str, Any]:
    """
    사용자 예산 입력을 LLM 프롬프트가 쓰는 형태로 정규화합니다.

    반환 스키마(기존 호환):
    - budget_choice: 문자열(프롬프트 '예산 입력'에 사용)
    - budget_range_manwon: {"min": int|None, "max": int|None} (프롬프트 '예산 범위(만원)'에 사용)
    - budget_type: "number" | "choice" | "subscription"
    """
    def _range_dict(min_v: Any, max_v: Any) -> Dict[str, Any]:
        return {"min": None if min_v is None else int(min_v), "max": None if max_v is None else int(max_v)}

    # 1) 이미 프론트가 dict로 준 경우(버튼 선택 결과 구조화)
    if isinstance(user_text, dict):
        budget_choice = user_text.get("budget_choice")
        budget_range = user_text.get("budget_range_manwon")
        if isinstance(budget_choice, str) and isinstance(budget_range, dict):
            return {
                "budget_type": user_text.get("budget_type") or "choice",
                "budget_choice": budget_choice,
                "budget_range_manwon": _range_dict(budget_range.get("min"), budget_range.get("max")),
            }

        mode = user_text.get("budget_mode") or user_text.get("budget_type") or user_text.get("mode")
        mode_s = _to_str(mode).lower()

        # 구독 월 납입 범위 -> 일시불 추정(가정값)
        # (프론트/정책에서 확정 계약 기간을 주면 그 값을 쓰도록 확장 가능)
        if mode_s in {"subscription", "subscribe", "monthly_subscription", "월구독", "구독"}:
            monthly = user_text.get("monthly_range_manwon") or user_text.get("monthly_range") or user_text.get("monthly")
            if isinstance(monthly, dict) and ("min" in monthly or "max" in monthly):
                factor = int(_to_str(user_text.get("subscription_months_factor") or 48))  # 기본 48개월 가정
                mn = monthly.get("min")
                mx = monthly.get("max")
                one_min = None if mn is None else int(mn) * factor
                one_max = None if mx is None else int(mx) * factor
                return {
                    "budget_type": "subscription",
                    "budget_choice": user_text.get("budget_choice") or f"월 구독({mn}~{mx}만 원대)",
                    "budget_range_manwon": _range_dict(one_min, one_max),
                }

        # 일시불 범위
        if mode_s in {"one_time", "one-time", "fixed", "cash", "일시불"}:
            rng = (
                user_text.get("one_time_range_manwon")
                or user_text.get("one_time_range")
                or user_text.get("fixed_range_manwon")
                or user_text.get("fixed_range")
            )
            if isinstance(rng, dict) and ("min" in rng or "max" in rng):
                return {
                    "budget_type": "choice",
                    "budget_choice": user_text.get("budget_choice") or "일시불",
                    "budget_range_manwon": _range_dict(rng.get("min"), rng.get("max")),
                }

        # dict인데 위 케이스로 파싱이 안 되면 문자열로도 한 번 더 시도
        # (예: {"btn":"btn2"} 같은 형태)
        user_text = user_text.get("text") or user_text.get("value") or user_text.get("btn") or user_text

    # 2) 문자열/숫자 파싱
    s = _to_str(user_text)
    if not s:
        raise ValueError("empty")

    if s.isdigit():
        return {"budget_type": "number", "budget_manwon": int(s)}

    normalized = s.replace(" ", "")

    # 기존(구버전) 예산 범위 매핑
    legacy_mapping = {
        "50만원이하": {"min": 0, "max": 50},
        "50~150만원": {"min": 50, "max": 150},
        "150~300만원": {"min": 150, "max": 300},
        "300만원이상": {"min": 300, "max": None},
        "아직정하지않았어요": {"min": None, "max": None},
    }

    if normalized in legacy_mapping:
        return {
            "budget_type": "choice",
            "budget_choice": user_text,
            "budget_range_manwon": legacy_mapping[normalized],
        }

    lowered = normalized.lower()

    # 새 UI: 잘 모르겠어요
    if "모르" in lowered or "잘모" in lowered or lowered in {"아직정하지않았어요", "아직모르겠어요"}:
        return {
            "budget_type": "choice",
            "budget_choice": "아직정하지않았어요",
            "budget_range_manwon": {"min": None, "max": None},
        }

    # 새 UI: 구독 월 범위 -> 일시불 추정(48개월 가정)
    # 예: "월3~5만 원대", "월 5~10만 원대"
    if "월" in lowered and "만" in lowered and ("~" in normalized or "-" in normalized):
        m = re.search(r"월.*?([0-9]+)\s*[~\-]\s*([0-9]+)\s*만", normalized)
        if m:
            monthly_min = int(m.group(1))
            monthly_max = int(m.group(2))
            factor = 48
            return {
                "budget_type": "subscription",
                "budget_choice": f"월구독({monthly_min}~{monthly_max}만 원대)",
                "budget_range_manwon": _range_dict(monthly_min * factor, monthly_max * factor),
            }

        # "월3만" 처럼 단일 값으로 들어오는 경우(드물지만 방어)
        m2 = re.search(r"월.*?([0-9]+)\s*만", normalized)
        if m2:
            monthly = int(m2.group(1))
            factor = 48
            return {
                "budget_type": "subscription",
                "budget_choice": f"월구독({monthly}만)",
                "budget_range_manwon": _range_dict(monthly * factor, None),
            }

    # 새 UI: 일시불 범위
    # 예: "100만 원 이하", "100~200만 원", "200만 원 이상"
    m_low = re.search(r"([0-9]+)\s*만[^0-9]*이하", normalized)
    if m_low:
        mx = int(m_low.group(1))
        return {"budget_type": "choice", "budget_choice": f"일시불(100만원 이하)", "budget_range_manwon": _range_dict(0, mx)}

    m_high = re.search(r"([0-9]+)\s*만[^0-9]*이상", normalized)
    if m_high:
        mn = int(m_high.group(1))
        return {"budget_type": "choice", "budget_choice": f"일시불({mn}만원 이상)", "budget_range_manwon": _range_dict(mn, None)}

    m_range = re.search(r"([0-9]+)\s*[~\-]\s*([0-9]+)\s*만", normalized)
    if m_range:
        mn = int(m_range.group(1))
        mx = int(m_range.group(2))
        return {"budget_type": "choice", "budget_choice": f"일시불({mn}~{mx}만원)", "budget_range_manwon": _range_dict(mn, mx)}

    # 새 UI 버튼값만 온 경우(btn1/btn2/btn3)
    if lowered in {"btn1", "btn2", "btn3"}:
        if lowered == "btn1":
            return {"budget_type": "subscription", "budget_choice": "월 구독(범위 미입력)", "budget_range_manwon": {"min": None, "max": None}}
        if lowered == "btn2":
            return {"budget_type": "choice", "budget_choice": "일시불(범위 미입력)", "budget_range_manwon": {"min": None, "max": None}}
        return {"budget_type": "choice", "budget_choice": "아직정하지않았어요", "budget_range_manwon": {"min": None, "max": None}}

    raise ValueError("invalid budget")


def _extract_owned_list(user_text: Any) -> List[str]:
    if isinstance(user_text, dict):
        raw = user_text.get("owned")
        if isinstance(raw, list):
            return [_to_str(x) for x in raw if _to_str(x)]
        return []
    if isinstance(user_text, list):
        return [_to_str(x) for x in user_text if _to_str(x)]
    s = _to_str(user_text)
    if not s:
        return []
    if "," in s:
        return [x.strip() for x in s.split(",") if x.strip()]
    return [s]


def _extract_needed_list(user_text: Any) -> List[str]:
    if isinstance(user_text, dict):
        raw = user_text.get("needed")
        if raw is None:
            raw = user_text.get("required") or user_text.get("essential")
        if isinstance(raw, list):
            return [_to_str(x) for x in raw if _to_str(x)]
        return []
    if isinstance(user_text, list):
        return [_to_str(x) for x in user_text if _to_str(x)]
    s = _to_str(user_text)
    if not s:
        return []
    if "," in s:
        return [x.strip() for x in s.split(",") if x.strip()]
    return [s]


def _dict_has_needed_key(d: dict) -> bool:
    return any(
        k in d
        for k in ("needed", "required", "essential", "requiredAppliances", "essentialAppliances")
    )


def _get_needed_raw_from_dict(d: dict) -> Any:
    if "needed" in d:
        return d.get("needed")
    if "required" in d:
        return d.get("required")
    if "requiredAppliances" in d:
        return d.get("requiredAppliances")
    if "essentialAppliances" in d:
        return d.get("essentialAppliances")
    if "essential" in d:
        return d.get("essential")
    return None


# --- 노드 정의 ---

def node_chat_0(state: ChatState) -> ChatState:
    return {**state, "step": ChatStep.CHAT_1, "data": {}, "is_completed": False}


def node_chat_1(state: ChatState) -> ChatState:
    """CHAT_1: 총 예산(만원, 숫자 등) — UI 첫 입력."""
    user_text = state.get("last_user_input")
    user_info = dict(state.get("user_info") or {})
    try:
        parsed = _parse_budget(user_text)
        user_info.update(parsed)
        return {**state, "step": ChatStep.CHAT_2, "user_info": user_info, "messages": _append_message(state, role="user", content=user_text)}
    except ValueError:
        return {**state, "step": ChatStep.CHAT_1, "data": {"error": "예산을 다시 입력해주세요."}, "messages": _append_message(state, role="user", content=user_text)}


def node_chat_2(state: ChatState) -> ChatState:
    """CHAT_2: 평수 (10평 이하 / 10~20평 … 또는 텍스트)."""
    user_text = state.get("last_user_input")
    user_info = dict(state.get("user_info") or {})
    user_info["size"] = user_text
    return {**state, "step": ChatStep.CHAT_3, "user_info": user_info, "messages": _append_message(state, role="user", content=user_text)}


def node_chat_3(state: ChatState) -> ChatState:
    """CHAT_3: 라이프스타일 (칩 다중 선택 시 list 가능)."""
    user_text = state.get("last_user_input")
    user_info = dict(state.get("user_info") or {})
    if isinstance(user_text, list):
        user_info["lifestyle"] = ", ".join(_to_str(x) for x in user_text)
    else:
        user_info["lifestyle"] = user_text
    return {**state, "step": ChatStep.CHAT_4, "user_info": user_info, "messages": _append_message(state, role="user", content=user_text)}


def node_chat_4(state: ChatState) -> ChatState:
    """CHAT_4: 보유 + 필요(필수) 가전 — owned / needed 를 한 요청에."""
    user_text = state.get("last_user_input")
    user_info = dict(state.get("user_info") or {})
    if isinstance(user_text, dict):
        if "owned" in user_text:
            user_info["owned_appliances"] = _extract_owned_list({"owned": user_text.get("owned")})
        if _dict_has_needed_key(user_text):
            user_info["needed_appliances"] = _extract_needed_list(
                {"needed": _get_needed_raw_from_dict(user_text)}
            )
    else:
        user_info["needed_appliances"] = _extract_needed_list(user_text)
    return {**state, "step": ChatStep.CHAT_5, "user_info": user_info, "messages": _append_message(state, role="user", content=user_text)}


def node_chat_5(state: ChatState) -> ChatState:
    """CHAT_5: 새 공간용 구매 계획(가구/가전) 자유 텍스트."""
    user_text = state.get("last_user_input")
    user_info = dict(state.get("user_info") or {})
    if isinstance(user_text, list):
        user_info["purchase_plans"] = ", ".join(_to_str(x) for x in user_text)
    else:
        user_info["purchase_plans"] = user_text
    if _to_str(user_info.get("purchase_plans")):
        user_info["need_furniture"] = True
    return {**state, "step": ChatStep.CHAT_6, "user_info": user_info, "messages": _append_message(state, role="user", content=user_text)}


async def node_chat_6(state: ChatState) -> ChatState:
    """CHAT_6: 추천 리스트 생성."""
    user_text = state.get("last_user_input")
    if user_text is not None:
        state = {**state, "messages": _append_message(state, role="user", content=user_text)}
    return await node_chat_result(state)


def node_chat_11(state: ChatState) -> ChatState:
    """
    진단 종료 단계.
    프론트에서 종료/저장 UI를 처리하므로 여기서는 종료 상태만 플래그로 남깁니다.
    """
    return {**state, "step": ChatStep.CHAT_11, "is_completed": True}


async def node_chat_result(state: ChatState) -> ChatState:
    user_info = state.get("user_info") or {}
    messages = list(state.get("messages") or [])
    candidate_products: list[dict[str, Any]] = []
    budget_max_won = _get_budget_max_won(user_info)

    try:
        result = await generate_recommendation_result(user_info=user_info)
        full_recommendation_list = []
        for item in result.recommendation_list:
            if isinstance(item, dict):
                full_recommendation_list.append(item)
                continue
            full_recommendation_list.append({
                "category": getattr(item, "category", None),
                "package_name": getattr(item, "package_name", None),
                "name": getattr(item, "package_name", getattr(item, "name", "")) or getattr(item, "name", ""),
                "products": getattr(item, "products", None),
                "reason": getattr(item, "reason", None),
                "estimated_price": getattr(item, "estimated_price", None),
            })
        total_estimated_budget = result.total_estimated_budget
    except NotImplementedError:
        candidate_products = await fetch_candidate_products(limit=300)
        prompt = build_recommendation_prompt(user_info, candidate_products=candidate_products)
        resp = await llm_json.ainvoke(prompt) # 비동기 환경이므로 ainvoke 추천
        content_json = _extract_json_from_llm_response(resp)
        full_recommendation_list = content_json.get("recommendation_list") or content_json.get("data") or []
        total_estimated_budget = content_json.get("total_estimated_budget", "")

    if not isinstance(full_recommendation_list, list):
        full_recommendation_list = []

    catalog_by_model_id: Dict[str, Dict[str, Any]] = {}
    catalog_by_name: Dict[str, Dict[str, Any]] = {}
    if candidate_products:
        for c in candidate_products:
            if not isinstance(c, dict):
                continue
            mid = _to_str(c.get("model_id"))
            nm = _normalize_name(c.get("name"))
            if mid:
                catalog_by_model_id[mid] = c
            if nm:
                catalog_by_name[nm] = c

    for item in full_recommendation_list:
        if isinstance(item, dict):
            item.setdefault("name", item.get("package_name") or item.get("name") or item.get("title") or "")
            raw_products = item.get("products") or []
            if catalog_by_model_id or catalog_by_name:
                item["products"] = _enforce_products_from_catalog(
                    raw_products,
                    catalog_by_model_id=catalog_by_model_id,
                    catalog_by_name=catalog_by_name,
                )
            else:
                # 추천 알고리즘 경로(향후 구현) 또는 카탈로그 미사용 시 기존 정규화
                normalized_products: list[dict[str, Any]] = []
                if isinstance(raw_products, list):
                    for p in raw_products:
                        if not isinstance(p, dict):
                            continue
                        normalized_products.append({
                            **p,
                            "product_name": p.get("product_name") or p.get("name") or "",
                            "product_url": p.get("product_url") or p.get("url") or "",
                            "product_image_url": p.get("product_image_url") or p.get("image_url") or "",
                        })
                item["products"] = normalized_products

            # 예산 상한(일시불 기준) 적용: 패키지 총액이 상한을 넘지 않도록 제품 컷오프
            if isinstance(item.get("products"), list):
                item["products"] = _apply_budget_cap_to_products(
                    item["products"],
                    budget_max_won=budget_max_won,
                )
            item.update(_compute_budget_breakdown(item))
            # GUI-2 추천 카드 렌더 편의를 위해 제품을 가전/가구/소품으로 분리합니다.
            products = item.get("products") or []
            appliances: list[dict[str, Any]] = []
            furniture: list[dict[str, Any]] = []
            if isinstance(products, list):
                for p in products:
                    if not isinstance(p, dict):
                        continue
                    p_cat = p.get("category")
                    if p_cat == "appliance":
                        appliances.append(p)
                    elif p_cat == "furniture":
                        furniture.append(p)
                    else:
                        # category가 불명확한 경우, price_normal/price_subscription 존재를 기준으로 분류
                        if "price_normal" in p or "price_subscription" in p:
                            appliances.append(p)
                        else:
                            furniture.append(p)
            item["appliances"] = appliances
            item["furniture"] = furniture

    if full_recommendation_list:
        packages_payload: list[Dict[str, Any]] = []
        for i, item in enumerate(full_recommendation_list):
            if not isinstance(item, dict): continue
            package_name = item.get("name") or item.get("package_name") or f"패키지_{i+1}"
            products = item.get("products") or []
            simplified_products = []
            if isinstance(products, list):
                for p in products:
                    if not isinstance(p, dict): continue
                    simplified_products.append({
                        "category": p.get("category"),
                        "product_name": p.get("product_name") or p.get("name"),
                        "model_id": p.get("model_id") or p.get("model"),
                        "brand": p.get("brand"),
                        "price_normal": p.get("price_normal"),
                        "price_subscription": p.get("price_subscription"),
                        "price": p.get("price"),
                        "product_image_url": p.get("product_image_url") or p.get("image_url"),
                        "product_url": p.get("product_url") or p.get("url"),
                    })

            packages_payload.append({
                "index": i,
                "package_name": package_name,
                "products": simplified_products,
                "appliance_price_normal_sum": item.get("appliance_price_normal_sum"),
                "appliance_price_subscription_sum": item.get("appliance_price_subscription_sum"),
                "furniture_price_sum": item.get("furniture_price_sum"),
            })

        reason_prompt = build_package_reason_prompt(user_info, packages_payload)
        resp = await llm_json.ainvoke(reason_prompt)
        reasons_json = _extract_json_from_llm_response(resp)
        reasons = reasons_json.get("reasons") or []

        for i, item in enumerate(full_recommendation_list):
            if isinstance(item, dict):
                item["reason"] = item.get("reason") or ""
        for r in reasons:
            if not isinstance(r, dict): continue
            idx = r.get("index")
            if isinstance(idx, int) and 0 <= idx < len(full_recommendation_list):
                if isinstance(full_recommendation_list[idx], dict):
                    full_recommendation_list[idx]["reason"] = r.get("reason") or full_recommendation_list[idx].get("reason")

    display_recommendations = full_recommendation_list[:3]
    next_index = len(display_recommendations)

    messages.append({
        "role": "assistant",
        "content": {
            "type": "recommendation_result",
            "display_recommendations": display_recommendations,
            "total_estimated_budget": total_estimated_budget,
        },
    })

    return {
        **state,
        "step": ChatStep.CHAT_RESULT,
        "data": {
            "all_recommendations": full_recommendation_list,
            "display_recommendations": display_recommendations,
            "next_index": next_index,
            "total_estimated_budget": total_estimated_budget,
        },
        "messages": messages,
        "is_completed": True,
    }


def _parse_recommend_rag_user_input(user_text: Any) -> tuple[str, Optional[int]]:
    """
    RECOMMEND_RAG용 userText — 질문 + (선택) 패키지 인덱스만.
    도면·배치(floorPlanId, placements 등)는 무시합니다. → BLUEPRINT_RAG 단계에서 처리.
    """
    if isinstance(user_text, dict):
        q = (
            _to_str(user_text.get("message"))
            or _to_str(user_text.get("question"))
            or _to_str(user_text.get("text"))
        )
        pkg_idx: Optional[int] = None
        raw_pi = user_text.get("packageIndex")
        if raw_pi is None:
            raw_pi = user_text.get("package_index")
        if raw_pi is not None and raw_pi != "":
            try:
                pkg_idx = int(raw_pi)
            except (TypeError, ValueError):
                pkg_idx = None
        return q, pkg_idx
    return _to_str(user_text), None


def _parse_placement_payload(user_text: Any) -> Dict[str, Any]:
    """도면 배치 화면에서 보내는 userText(dict) 정규화. camelCase/snake_case 모두."""
    if not isinstance(user_text, dict):
        return {}
    out = dict(user_text)
    if "floor_plan_id" not in out and user_text.get("floorPlanId"):
        out["floor_plan_id"] = user_text.get("floorPlanId")
    if "package_index" not in out and user_text.get("packageIndex") is not None:
        out["package_index"] = user_text.get("packageIndex")
    if "floor_plan_image_url" not in out and user_text.get("floorPlanImageUrl"):
        out["floor_plan_image_url"] = user_text.get("floorPlanImageUrl")
    if "canvas_size" not in out and user_text.get("canvasSize"):
        out["canvas_size"] = user_text.get("canvasSize")
    return out


async def _execute_floor_plan_package_rag(
    state: ChatState,
    *,
    result_step: ChatStep,
    response_key: str,
    assistant_type: str,
) -> ChatState:
    """
    도면 ID + 선택 패키지 인덱스 + 스티커(placements/utilities) + DB 제품 상세로 배치 적합성 RAG.
    """
    user_info = dict(state.get("user_info") or {})
    new_data = dict(state.get("data") or {})
    raw = state.get("last_user_input")
    payload = _parse_placement_payload(raw)

    floor_plan_id = _to_str(payload.get("floor_plan_id"))
    pkg_idx_raw = payload.get("package_index")
    try:
        package_index = int(pkg_idx_raw) if pkg_idx_raw is not None else -1
    except (TypeError, ValueError):
        package_index = -1

    placements = payload.get("placements")
    if not isinstance(placements, list):
        placements = []
    utilities = payload.get("utilities")
    if not isinstance(utilities, list):
        utilities = []

    all_recs = new_data.get("all_recommendations") or []
    if not isinstance(all_recs, list) or not all_recs:
        err = "추천 패키지가 없습니다. 같은 convId로 CHAT_6(추천 리스트)를 먼저 완료해 주세요."
        return {
            **state,
            "step": result_step,
            "data": {**new_data, "error": err},
            "ai_response": None,
            "is_completed": True,
        }

    if not floor_plan_id:
        return {
            **state,
            "step": result_step,
            "data": {**new_data, "error": "floor_plan_id(또는 floorPlanId)가 필요합니다."},
            "ai_response": None,
            "is_completed": True,
        }

    if package_index < 0 or package_index >= len(all_recs):
        return {
            **state,
            "step": result_step,
            "data": {
                **new_data,
                "error": f"package_index는 0~{len(all_recs) - 1} 범위여야 합니다.",
            },
            "ai_response": None,
            "is_completed": True,
        }

    pkg = all_recs[package_index]
    if not isinstance(pkg, dict):
        return {
            **state,
            "step": result_step,
            "data": {**new_data, "error": "선택한 패키지 데이터 형식이 올바르지 않습니다."},
            "ai_response": None,
            "is_completed": True,
        }

    user_info["selected_floor_plan_id"] = floor_plan_id
    user_info["selected_package_index"] = package_index
    user_info["last_placement_payload"] = payload

    model_ids: List[str] = []
    products = pkg.get("products") or []
    if isinstance(products, list):
        for p in products:
            if isinstance(p, dict):
                mid = p.get("model_id")
                if mid:
                    model_ids.append(str(mid))

    products_details: Dict[str, Any] = {"products": []}
    if model_ids:
        try:
            session_maker = get_session_maker()
            async with session_maker() as session:
                products_details = await fetch_products_bundle_details(session, model_ids=model_ids)
        except Exception:
            products_details = {"products": []}

    floor_plan_ctx: Dict[str, Any] = {
        "floor_plan_id": floor_plan_id,
        "floor_plan_image_url": payload.get("floor_plan_image_url"),
        "canvas_size": payload.get("canvas_size"),
        "placements": placements,
        "utilities": utilities,
        "extra_note": payload.get("note") or payload.get("userNote"),
    }

    messages = _append_message(state, role="user", content=raw)
    prompt = build_blueprint_rag_prompt(
        user_info,
        floor_plan=floor_plan_ctx,
        selected_package=pkg,
        products_details=products_details,
    )
    resp = await llm_json.ainvoke(prompt)
    data_json = _extract_json_from_llm_response(resp)
    answer_text = _to_str(data_json.get("answer"))

    block = {
        "floor_plan_id": floor_plan_id,
        "package_index": package_index,
        "package_name": pkg.get("name") or pkg.get("package_name"),
        "llm": data_json,
    }
    new_data[response_key] = block

    messages.append({"role": "assistant", "content": {"type": assistant_type, **block}})
    return {
        **state,
        "step": result_step,
        "user_info": user_info,
        "ai_response": answer_text or None,
        "messages": messages,
        "data": new_data,
        "is_completed": True,
    }


async def node_blueprint_rag(state: ChatState) -> ChatState:
    """도면 단계: 선택 도면 + 패키지 + 스티커 좌표 RAG."""
    return await _execute_floor_plan_package_rag(
        state,
        result_step=ChatStep.BLUEPRINT_RAG,
        response_key="blueprint_rag",
        assistant_type="blueprint_rag",
    )


def _detect_next_recommendation_page_intent(text: str) -> bool:
    """
    Spring이 보관한 전체 추천 목록의 '다음 3개'로 넘길 의도인지 키워드로 보조 판별.
    LLM 플래그와 OR 로 사용.
    """
    t = _to_str(text).replace(" ", "").lower()
    if not t:
        return False
    keys = (
        "더보여", "더보여줘", "다시보여", "다시보여줘", "다시추천", "추천다시",
        "다른걸", "다른거", "다른패키지", "다른조합", "다음추천", "다음패키지",
        "추천더", "다시추천해", "다른걸추천", "패키지더",
    )
    return any(k in t for k in keys)


def _norm_match_text(s: str) -> str:
    return _to_str(s).lower().replace(" ", "").replace("\n", "").replace("\t", "")


def _score_product_name_vs_question(product_name: str, question: str) -> int:
    """질문과 추천 상품명 유사도(간단 휴리스틱)."""
    pn = _norm_match_text(product_name)
    qn = _norm_match_text(question)
    if not pn or not qn:
        return 0
    if pn in qn:
        return 1000 + len(pn)
    if qn in pn:
        return 500 + len(qn)
    score = 0
    for L in range(min(len(pn), 40), 2, -1):
        for i in range(len(pn) - L + 1):
            sub = pn[i : i + L]
            if len(sub) >= 3 and sub in qn:
                score = max(score, L * 15)
    for tok in _to_str(product_name).split():
        t = _norm_match_text(tok)
        if len(t) >= 2 and t in qn:
            score = max(score, len(t) * 20)
    return score


def _find_model_ids_by_product_name(
    question: str,
    all_recs: Any,
    *,
    max_models: int = 5,
) -> tuple[List[str], Optional[int]]:
    """
    all_recommendations 안의 product_name/name과 질문을 매칭해 model_id 목록.
    Returns (model_ids, 대표 패키지 인덱스 또는 None).
    """
    if not isinstance(all_recs, list) or not _to_str(question).strip():
        return [], None

    qn_flat = _norm_match_text(question)
    candidates: list[tuple[int, int, str, str]] = []  # score, pkg_idx, model_id, name

    for pi, pkg in enumerate(all_recs):
        if not isinstance(pkg, dict):
            continue
        products = pkg.get("products") or []
        if not isinstance(products, list):
            continue
        for p in products:
            if not isinstance(p, dict):
                continue
            mid = _to_str(p.get("model_id"))
            if not mid:
                continue
            pname = _to_str(p.get("product_name") or p.get("name"))
            sc = _score_product_name_vs_question(pname, question) if pname else 0
            if mid.lower() in qn_flat or mid in question:
                sc = max(sc, 900 + len(mid))
            if sc > 0:
                candidates.append((sc, pi, mid, pname))

    if not candidates:
        return [], None

    candidates.sort(key=lambda x: -x[0])
    best = candidates[0][0]
    threshold = max(30, int(best * 0.35))

    out_ids: list[str] = []
    seen: set[str] = set()
    first_pkg: Optional[int] = None
    for sc, pi, mid, _ in candidates:
        if sc >= threshold and mid not in seen:
            seen.add(mid)
            out_ids.append(mid)
            if first_pkg is None:
                first_pkg = pi
            if len(out_ids) >= max_models:
                break

    return out_ids, first_pkg


# ⭐ 수정 포인트: async def 로 변경
async def node_recommend_rag(state: ChatState) -> ChatState:
    user_info = state.get("user_info") or {}
    raw_user = state.get("last_user_input") or ""
    question_str, explicit_pkg_idx = _parse_recommend_rag_user_input(raw_user)
    messages = _append_message(state, role="user", content=raw_user)
    new_data = dict(state.get("data") or {})

    def _extract_package_index(text: str) -> Optional[int]:
        m = re.search(r"(\d+)\s*번", text)
        if m:
            try: return int(m.group(1)) - 1
            except ValueError: return None
        if "첫" in text: return 0
        if "두" in text: return 1
        if "세" in text: return 2
        return None

    package_idx: Optional[int] = explicit_pkg_idx
    if package_idx is None:
        package_idx = _extract_package_index(_to_str(question_str))
    if package_idx is None and user_info.get("selected_package_index") is not None:
        try:
            package_idx = int(user_info["selected_package_index"])
        except (TypeError, ValueError):
            package_idx = None

    prompt = None
    if package_idx is not None:
        all_recs = new_data.get("all_recommendations") or []
        if isinstance(all_recs, list) and 0 <= package_idx < len(all_recs):
            pkg = all_recs[package_idx]
            if isinstance(pkg, dict):
                model_ids = []
                products = pkg.get("products") or []
                if isinstance(products, list):
                    for p in products:
                        if isinstance(p, dict):
                            mid = p.get("model_id")
                            if mid: model_ids.append(str(mid))
                if model_ids:
                    try:
                        session_maker = get_session_maker()
                        async with session_maker() as session:
                            details = await fetch_products_bundle_details(
                                session, model_ids=model_ids
                            )
                        package_context = {
                            "package_index": package_idx,
                            "package_budgets": {
                                "appliance_price_normal_sum": pkg.get("appliance_price_normal_sum"),
                                "appliance_price_subscription_sum": pkg.get("appliance_price_subscription_sum"),
                                "furniture_price_sum": pkg.get("furniture_price_sum"),
                            },
                            "products_details": details,
                        }
                        prompt = build_rag_prompt_with_package_context(
                            user_info,
                            question_str,
                            package_context=package_context,
                        )
                    except Exception:
                        prompt = None

    # 패키지 인덱스 없이 질문에 상품명이 있으면 all_recommendations에서 매칭 → DB 스펙
    if prompt is None:
        all_recs = new_data.get("all_recommendations") or []
        mids_name, pkg_by_name = _find_model_ids_by_product_name(question_str, all_recs)
        if mids_name:
            try:
                session_maker = get_session_maker()
                async with session_maker() as session:
                    details = await fetch_products_bundle_details(
                        session, model_ids=mids_name
                    )
                pkg_slice: Dict[str, Any] = {}
                if (
                    pkg_by_name is not None
                    and isinstance(all_recs, list)
                    and 0 <= pkg_by_name < len(all_recs)
                ):
                    p0 = all_recs[pkg_by_name]
                    if isinstance(p0, dict):
                        pkg_slice = p0
                package_context = {
                    "package_index": pkg_by_name,
                    "matched_by": "product_name",
                    "package_budgets": {
                        "appliance_price_normal_sum": pkg_slice.get("appliance_price_normal_sum"),
                        "appliance_price_subscription_sum": pkg_slice.get(
                            "appliance_price_subscription_sum"
                        ),
                        "furniture_price_sum": pkg_slice.get("furniture_price_sum"),
                    },
                    "products_details": details,
                }
                prompt = build_rag_prompt_with_package_context(
                    user_info,
                    question_str,
                    package_context=package_context,
                )
            except Exception:
                prompt = None

    if prompt is None:
        prompt = build_rag_prompt(user_info, question_str)

    resp = await llm_json.ainvoke(prompt)
    data_json = _extract_json_from_llm_response(resp)
    answer_text = data_json.get("answer", "")
    show_next = bool(data_json.get("show_next_recommendation_page"))
    if not show_next:
        show_next = _detect_next_recommendation_page_intent(_to_str(question_str))

    new_data["show_next_recommendation_page"] = show_next
    # Spring/프론트 편의 (동일 의미)
    new_data["showNextRecommendationPage"] = show_next

    messages.append({"role": "assistant", "content": answer_text})
    return {
        **state,
        "step": ChatStep.RECOMMEND_RAG,
        "ai_response": answer_text,
        "messages": messages,
        "data": new_data,
        "is_completed": True,
    }

def route_from_step(state: ChatState) -> str:
    requested = state.get("requested_step_code")
    if isinstance(requested, str):
        # 예시 문서 표기는 CHAT-6 처럼 하이픈이 올 수 있어, 프론트/문서 불일치를 방어합니다.
        requested = requested.replace("-", "_").upper()
    step = ChatStep(requested) if requested and requested in ChatStep.__members__ else state.get("step", ChatStep.CHAT_0)
    mapping = {
        ChatStep.CHAT_0: "chat_0",
        ChatStep.CHAT_1: "chat_1",
        ChatStep.CHAT_2: "chat_2",
        ChatStep.CHAT_3: "chat_3",
        ChatStep.CHAT_4: "chat_4",
        ChatStep.CHAT_5: "chat_5",
        ChatStep.CHAT_6: "chat_6",
        ChatStep.CHAT_11: "chat_11",
        ChatStep.CHAT_RESULT: "chat_result",
        ChatStep.RECOMMEND_RAG: "recommend_rag",
        ChatStep.BLUEPRINT_RAG: "blueprint_rag",
    }
    return mapping.get(step, "chat_0")

def build_graph():
    workflow = StateGraph(ChatState)
    nodes = {
        "chat_0": node_chat_0, "chat_1": node_chat_1, "chat_2": node_chat_2,
        "chat_3": node_chat_3, "chat_4": node_chat_4,
        "chat_5": node_chat_5, "chat_6": node_chat_6,
        "chat_11": node_chat_11,
        "chat_result": node_chat_result,
        "recommend_rag": node_recommend_rag,
        "blueprint_rag": node_blueprint_rag,
    }
    for name, func in nodes.items():
        workflow.add_node(name, func)
    workflow.add_conditional_edges(START, route_from_step)
    for name in nodes.keys():
        workflow.add_edge(name, END)
    return workflow.compile(checkpointer=get_checkpointer())

chat_app = build_graph()