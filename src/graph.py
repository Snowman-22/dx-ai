from __future__ import annotations

import inspect
import json
import logging
import re
from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START

from .db import get_session_maker
from .product_details import fetch_products_bundle_details
from .state_store import get_checkpointer
from .prompt import (
    build_blueprint_rag_prompt,
    build_rag_prompt,
    build_rag_prompt_with_package_context,
)
from .recommend import generate_recommendation_result
from .recommend.service import fetch_recommendation_catalog_maps

logger = logging.getLogger(__name__)


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

APPLIANCE_KEYWORDS: dict[str, list[str]] = {
    "TV": [
        "tv", "티비", "텔레비전", "oled", "qned", "나노셀", "올레드",
        "led tv", "qned tv", "울트라hd", "uhd tv", "스탠바이미",
    ],
    "에어컨": [
        "에어컨", "벽걸이에어컨", "스탠드에어컨", "사계절에어컨",
        "2in1에어컨", "벽걸이형 에어컨", "스탠드형 에어컨", "휘센",
    ],
    "세탁기": [
        "세탁기", "드럼세탁기", "드럼 세탁기", "통돌이세탁기", "통돌이 세탁기",
        "통돌이", "미니워시", "워시타워", "워시콤보", "트롬 세탁기",
    ],
    "건조기": [
        "건조기", "의류건조기", "의류 건조기", "트롬 건조기", "건조기세트",
    ],
    "냉장고": [
        "냉장고", "양문형냉장고", "양문형 냉장고", "일반냉장고", "일반 냉장고",
        "김치냉장고", "김치 냉장고", "냉동고", "프렌치도어", "매직스페이스",
        "더블매직스페이스", "노크온", "디오스", "무드업",
    ],
    "공기청정기": [
        "공기청정기", "공청기", "에어로타워", "퓨리케어", "퓨리케어 공기청정기",
    ],
    "정수기": [
        "정수기", "얼음정수기", "냉온정수기", "냉온정", "냉정수기", "온정수기",
        "정수전용", "정수기기", "퓨리케어 정수기",
    ],
    "청소기": [
        "청소기", "로봇청소기", "무선청소기", "유선청소기", "스틱청소기",
        "코드제로", "로보킹", "사이킹",
    ],
    "제습기": [
        "제습기", "대용량제습기", "소형제습기",
    ],
    "가습기": [
        "가습기", "자연기화 가습기", "하이드로타워", "하이드로에센셜",
    ],
    "식기세척기": [
        "식기세척기", "식세기", "빌트인 식기세척기", "프리스탠딩 식기세척기",
    ],
    "전기레인지": [
        "전기레인지", "인덕션", "하이브리드레인지", "레인지", "빌트인 레인지",
    ],
    "전자레인지": [
        "전자레인지", "전자 레인지", "광파오븐", "광파 오븐", "오븐", "전자오븐",
    ],
    "의류관리기": [
        "의류관리기", "스타일러", "스티머", "의류 케어기",
    ],
    "밥솥": [
        "밥솥", "전기밥솥", "압력밥솥", "쿠쿠 밥솥",
    ],
}

FURNITURE_KEYWORDS: dict[str, list[str]] = {
    "소파": [
        "소파", "1인소파", "2인소파", "3인소파", "4인소파", "모듈소파",
        "카우치소파", "패브릭소파", "가죽소파", "리클라이너", "전동리클라이너",
        "리클라이너소파", "코너소파", "벤치소파", "키즈소파",
    ],
    "의자": [
        "의자", "체어", "다이닝체어", "식탁의자", "데스크체어", "컴퓨터의자",
        "사무의자", "학생의자", "바스툴", "홈바의자", "화장대의자",
        "암체어", "라운지체어", "패브릭체어", "가죽체어", "스툴", "유아의자",
    ],
    "침대": [
        "침대", "패브릭침대", "가죽침대", "저상형침대", "패밀리침대", "데이베드",
        "가드침대", "유아침대", "싱글침대", "슈퍼싱글침대", "퀸침대",
        "킹침대", "라지킹침대",
    ],
    "매트리스": [
        "매트리스", "토퍼", "스프링매트리스", "라텍스매트리스",
        "싱글 매트리스", "퀸 매트리스", "킹 매트리스",
    ],
    "식탁": [
        "식탁", "세라믹식탁", "원형식탁", "타원형식탁", "4인식탁", "6인식탁",
        "2인식탁", "다이닝테이블", "원목식탁", "마블식탁",
    ],
    "테이블": [
        "테이블", "사이드테이블", "티테이블", "소파테이블", "소파탁자",
        "거실테이블", "커피테이블", "야외테이블",
    ],
    "수납장": [
        "수납장", "서랍장", "장식장", "진열장", "선반장", "수납선반", "와이드서랍장",
        "사이드보드", "이동식수납", "캐비닛", "콘솔장",
    ],
    "책장": [
        "책장", "북케이스", "오픈책장", "책꽂이",
    ],
    "옷장": [
        "옷장", "행거형옷장", "서랍형옷장", "거울옷장", "붙박이장",
    ],
    "책상": [
        "책상", "데스크", "서재책상", "컴퓨터책상", "학생책상", "1인책상",
        "파티션책상", "작업책상",
    ],
    "거실장": [
        "거실장", "tv거실장", "tv장", "tv콘솔", "tv캐비넷", "tv선반", "미디어콘솔",
    ],
    "행거": [
        "행거", "옷행거", "코트행거", "드레스룸", "스탠드행거",
    ],
    "선반": [
        "선반", "벽선반", "모니터선반", "책상선반", "수납선반",
    ],
    "화장대": [
        "화장대", "거울화장대", "콘솔", "콘솔선반",
    ],
    "협탁": [
        "협탁", "침대협탁", "사이드협탁",
    ],
    "트롤리": [
        "트롤리", "이동식트롤리", "카트형 수납장",
    ],
    "벤치": [
        "벤치", "벤치의자", "벤치소파",
    ],
    "아웃도어가구": [
        "아웃도어가구", "야외의자", "야외테이블", "테라스가구",
    ],
}

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
    catalog_by_product_id: Optional[Dict[int, Dict[str, Any]]] = None,
    strict: bool = False,
) -> list[Dict[str, Any]]:
    """
    DB 카탈로그에 있는 상품만 남기고, 필드는 RDS 스냅샷으로 통일합니다.
    - strict=True: product_id 또는 model_id 로만 매칭 (이름만 있는 가짜 상품 제거)
    - strict=False: model_id → 이름 정규화 매칭 (레거시)
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
        if catalog_by_product_id:
            try:
                raw_pid = p.get("product_id")
                pid = int(raw_pid) if raw_pid is not None and str(raw_pid).strip() != "" else None
            except (TypeError, ValueError):
                pid = None
            if pid is not None and pid in catalog_by_product_id:
                matched = catalog_by_product_id[pid]
        if matched is None and raw_mid and raw_mid in catalog_by_model_id:
            matched = catalog_by_model_id[raw_mid]
        if matched is None and not strict and raw_name:
            key = _normalize_name(raw_name)
            matched = catalog_by_name.get(key)

        if matched is None:
            continue

        cat_hint = p.get("category")
        if cat_hint in ("appliance", "furniture"):
            cat = cat_hint
        else:
            db_cat = _to_str(matched.get("category")).upper()
            if db_cat == "APPLIANCE":
                cat = "appliance"
            elif db_cat == "FURNITURE":
                cat = "furniture"
            else:
                cat = "appliance"
        price = matched.get("price")
        out.append({
            "product_id": matched.get("id"),
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


def _unique_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = _to_str(item)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _match_keywords(text: str, mapping: dict[str, list[str]]) -> list[str]:
    lowered = _to_str(text).lower()
    matched: list[str] = []
    for canonical, aliases in mapping.items():
        for alias in aliases:
            if alias.lower() in lowered:
                matched.append(canonical)
                break
    return _unique_keep_order(matched)


async def _classify_chat5_items(user_text: Any) -> dict[str, list[str]]:
    text = _to_str(user_text)
    if not text:
        return {"appliances": [], "furniture": []}

    appliances = _match_keywords(text, APPLIANCE_KEYWORDS)
    furniture = _match_keywords(text, FURNITURE_KEYWORDS)

    # 룰베이스로 못 잡은 경우에만 OpenAI에 "추출/정규화"만 맡깁니다.
    if appliances or furniture:
        return {"appliances": appliances, "furniture": furniture}

    prompt = f"""
사용자 문장에서 가전/가구 품목명만 추출해서 JSON으로 반환하세요.
- 새 품목을 상상하지 마세요.
- 문장에 직접 언급된 품목만 추출하세요.
- 없으면 빈 배열을 반환하세요.
- 가능한 경우 아래 정규 품목명 중 하나로 맞추세요.

가전 후보: {list(APPLIANCE_KEYWORDS.keys())}
가구 후보: {list(FURNITURE_KEYWORDS.keys())}

반환 형식:
{{
  "appliances": ["품목명"],
  "furniture": ["품목명"]
}}

사용자 문장:
\"\"\"{text}\"\"\"
""".strip()

    try:
        resp = await llm_json.ainvoke(prompt)
        parsed = _extract_json_from_llm_response(resp)
    except Exception:
        return {"appliances": appliances, "furniture": furniture}

    appliances = _unique_keep_order(
        [x for x in parsed.get("appliances", []) if _to_str(x) in APPLIANCE_KEYWORDS]
    )
    furniture = _unique_keep_order(
        [x for x in parsed.get("furniture", []) if _to_str(x) in FURNITURE_KEYWORDS]
    )
    return {"appliances": appliances, "furniture": furniture}


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
        return {
            **state,
            "step": ChatStep.CHAT_2,
            "user_info": user_info,
            "data": {},
            "ai_response": None,
            "messages": _append_message(state, role="user", content=user_text),
        }
    except ValueError:
        return {**state, "step": ChatStep.CHAT_1, "data": {"error": "예산을 다시 입력해주세요."}, "messages": _append_message(state, role="user", content=user_text)}


def node_chat_2(state: ChatState) -> ChatState:
    """CHAT_2: 평수 (10평 이하 / 10~20평 … 또는 텍스트)."""
    user_text = state.get("last_user_input")
    user_info = dict(state.get("user_info") or {})
    user_info["size"] = user_text
    return {
        **state,
        "step": ChatStep.CHAT_3,
        "user_info": user_info,
        "data": {},
        "ai_response": None,
        "messages": _append_message(state, role="user", content=user_text),
    }


def node_chat_3(state: ChatState) -> ChatState:
    """CHAT_3: 라이프스타일 (칩 다중 선택 시 list 가능)."""
    user_text = state.get("last_user_input")
    user_info = dict(state.get("user_info") or {})
    if isinstance(user_text, list):
        user_info["lifestyle"] = ", ".join(_to_str(x) for x in user_text)
    else:
        user_info["lifestyle"] = user_text
    return {
        **state,
        "step": ChatStep.CHAT_4,
        "user_info": user_info,
        "data": {},
        "ai_response": None,
        "messages": _append_message(state, role="user", content=user_text),
    }


def _coerce_chat4_user_text(user_text: Any) -> Any:
    """
    프론트가 JSON 객체 대신 문자열로 일부만 보낸 경우 보정.
    예: '  "owned": [...], "needed": [...]  ' → dict
    """
    if isinstance(user_text, dict):
        return user_text
    if not isinstance(user_text, str):
        return user_text
    s = user_text.strip()
    if not s:
        return user_text
    if s.startswith("{"):
        try:
            return json.loads(s)
        except Exception:
            return user_text
    if "owned" in s or "needed" in s:
        try:
            return json.loads("{" + s + "}")
        except Exception:
            return user_text
    return user_text


def node_chat_4(state: ChatState) -> ChatState:
    """CHAT_4: 보유 + 필요(필수) 가전 — owned / needed 를 한 요청에."""
    user_text = _coerce_chat4_user_text(state.get("last_user_input"))
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
    return {
        **state,
        "step": ChatStep.CHAT_5,
        "user_info": user_info,
        "data": {},
        "ai_response": None,
        "messages": _append_message(state, role="user", content=user_text),
    }


async def node_chat_5(state: ChatState) -> ChatState:
    """CHAT_5: 새 공간용 구매 계획(가구/가전) 자유 텍스트."""
    user_text = state.get("last_user_input")
    user_info = dict(state.get("user_info") or {})
    if isinstance(user_text, list):
        user_info["purchase_plans"] = ", ".join(_to_str(x) for x in user_text)
    else:
        user_info["purchase_plans"] = user_text

    classified = await _classify_chat5_items(user_text)
    classified_appliances = classified.get("appliances") or []
    classified_furniture = classified.get("furniture") or []

    if classified_appliances:
        existing = _unique_keep_order(_extract_needed_list(user_info.get("needed_appliances")))
        user_info["needed_appliances"] = _unique_keep_order(existing + classified_appliances)
    if classified_furniture:
        user_info["furniture_needed"] = classified_furniture

    if _to_str(user_info.get("purchase_plans")):
        user_info["need_furniture"] = True
    return {
        **state,
        "step": ChatStep.CHAT_6,
        "user_info": user_info,
        "data": {},
        "ai_response": None,
        "messages": _append_message(state, role="user", content=user_text),
    }


async def node_chat_6(state: ChatState) -> ChatState:
    """CHAT_6: 추천 리스트 생성."""
    user_text = state.get("last_user_input")
    if user_text is not None:
        state = {
            **state,
            "data": {},
            "ai_response": None,
            "messages": _append_message(state, role="user", content=user_text),
        }
    return await node_chat_result(state)


def node_chat_11(state: ChatState) -> ChatState:
    """
    진단 종료 단계.
    프론트에서 종료/저장 UI를 처리하므로 여기서는 종료 상태만 플래그로 남깁니다.
    """
    return {**state, "step": ChatStep.CHAT_11, "data": {}, "ai_response": None, "is_completed": True}


async def node_chat_result(state: ChatState) -> ChatState:
    user_info = state.get("user_info") or {}
    messages = list(state.get("messages") or [])
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
        return {
            **state,
            "step": ChatStep.CHAT_6,
            "data": {
                "error": (
                    "추천 엔진이 설정되지 않았습니다. 서버에 RECOMMENDATION_ALGORITHM_PATH(및 필요 시 "
                    "RECOMMENDATION_ALGORITHM_ENTRYPOINT)를 설정한 뒤 다시 시도해 주세요. "
                    "LLM으로 가짜 상품을 만들지 않습니다."
                ),
            },
            "messages": messages,
            "is_completed": False,
        }
    except Exception as e:
        return {
            **state,
            "step": ChatStep.CHAT_6,
            "data": {"error": f"추천 생성에 실패했습니다: {e!s}"},
            "messages": messages,
            "is_completed": False,
        }

    if not isinstance(full_recommendation_list, list):
        full_recommendation_list = []

    # RDS product 테이블에 없는 항목은 절대 내려보내지 않음 (LLM/오염된 필드 제거)
    catalog_by_pid, catalog_by_mid = await fetch_recommendation_catalog_maps(full_recommendation_list)
    catalog_by_name: Dict[str, Dict[str, Any]] = {}

    # 추천 이유 문자열은 알고리즘/DB 결과만 사용(LLM으로 패키지 문구 생성 안 함).
    for item in full_recommendation_list:
        if isinstance(item, dict):
            item.setdefault("name", item.get("package_name") or item.get("name") or item.get("title") or "")
            raw_products = item.get("products") or []
            item["products"] = _enforce_products_from_catalog(
                raw_products,
                catalog_by_model_id=catalog_by_mid,
                catalog_by_name=catalog_by_name,
                catalog_by_product_id=catalog_by_pid if catalog_by_pid else None,
                strict=True,
            )

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
            item["reason"] = item.get("reason") or ""

    # 검증 후 상품이 하나도 없는 패키지는 제거
    full_recommendation_list = [
        item
        for item in full_recommendation_list
        if isinstance(item, dict)
        and isinstance(item.get("products"), list)
        and len(item["products"]) > 0
    ]

    if not full_recommendation_list:
        return {
            **state,
            "step": ChatStep.CHAT_6,
            "data": {
                "error": (
                    "추천 결과에 포함된 상품이 RDS `product` 테이블에서 확인되지 않았습니다. "
                    "product_id·model_id가 없거나 DB에 없는 행이면 응답에서 제외됩니다."
                ),
            },
            "messages": messages,
            "is_completed": False,
        }

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

def _step_to_route_key(raw: Any) -> str:
    """
    DynamoDB/JSON 복원 시 step이 문자열로만 들어오면 Enum 키와 매칭되지 않아
    잘못된 노드(chat_0)로 가거나, 폴백 step과 꼬일 수 있음 → 문자열 키로 통일.
    """
    if raw is None:
        return "CHAT_0"
    if isinstance(raw, ChatStep):
        return raw.name
    if isinstance(raw, str):
        s = raw.strip().replace("-", "_").upper()
        if s in ChatStep.__members__:
            return s
    return "CHAT_0"


async def dispatch_node(state: ChatState) -> ChatState:
    """
    START → conditional_edges 대신 단일 진입점.
    LangGraph가 체크포인트와 병합할 때 `requested_step_code`가 라우터에 안 넘어가
    CHAT_1(예산)만 실행되던 문제를 막기 위해, 여기서 직접 stepCode에 맞는 핸들러만 호출합니다.
    """
    raw = state.get("requested_step_code")
    if isinstance(raw, str):
        raw = raw.strip().replace("-", "_").upper()
    if raw and raw in ChatStep.__members__:
        key = raw
    else:
        key = _step_to_route_key(state.get("step"))

    logger.info("dispatch: requested_step_code=%r → resolved=%r", state.get("requested_step_code"), key)

    handlers: Dict[str, Any] = {
        "CHAT_0": node_chat_0,
        "CHAT_1": node_chat_1,
        "CHAT_2": node_chat_2,
        "CHAT_3": node_chat_3,
        "CHAT_4": node_chat_4,
        "CHAT_5": node_chat_5,
        "CHAT_6": node_chat_6,
        "CHAT_11": node_chat_11,
        "CHAT_RESULT": node_chat_result,
        "RECOMMEND_RAG": node_recommend_rag,
        "BLUEPRINT_RAG": node_blueprint_rag,
    }
    fn = handlers.get(key)
    if fn is None:
        logger.warning("dispatch: unknown key=%r, fallback CHAT_0", key)
        return node_chat_0(state)
    if inspect.iscoroutinefunction(fn):
        return await fn(state)
    return fn(state)


def build_graph():
    workflow = StateGraph(ChatState)
    workflow.add_node("dispatch", dispatch_node)
    workflow.add_edge(START, "dispatch")
    workflow.add_edge("dispatch", END)
    return workflow.compile(checkpointer=get_checkpointer())

chat_app = build_graph()
