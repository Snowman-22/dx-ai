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
    build_recommendation_prompt,
    build_rag_prompt,
    build_rag_prompt_with_package_context,
)
from .recommend import generate_recommendation_result


class ChatStep(str, Enum):
    CHAT_0 = "CHAT_0"  # 진단 시작
    CHAT_1 = "CHAT_1"  # 주거공간 크기(평수)
    CHAT_2 = "CHAT_2"  # 보유/필요 가전
    CHAT_3 = "CHAT_3"  # 가구/소품 추천 필요 여부
    CHAT_3_1 = "CHAT_3_1"  # 인테리어 스타일
    CHAT_4 = "CHAT_4"  # 라이프스타일
    CHAT_5 = "CHAT_5"  # 예산
    CHAT_6 = "CHAT_6"  # 진단 완료
    CHAT_RESULT = "CHAT_RESULT"  # 추천 결과
    RAG_CHAT = "RAG_CHAT"  # 추천 이후 자유대화


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

def _parse_budget(user_text: Any) -> Dict[str, Any]:
    s = _to_str(user_text)
    if not s: raise ValueError("empty")
    if s.isdigit(): return {"budget_type": "number", "budget_manwon": int(s)}
    normalized = s.replace(" ", "")
    mapping = {
        "50만원이하": {"min": 0, "max": 50},
        "50~150만원": {"min": 50, "max": 150},
        "150~300만원": {"min": 150, "max": 300},
        "300만원이상": {"min": 300, "max": None},
        "아직정하지않았어요": {"min": None, "max": None},
    }
    if normalized in mapping:
        return {"budget_type": "choice", "budget_choice": user_text, "budget_range_manwon": mapping[normalized]}
    raise ValueError("invalid budget")

# --- 노드 정의 ---

def node_chat_0(state: ChatState) -> ChatState:
    return {**state, "step": ChatStep.CHAT_1, "data": {}, "is_completed": False}

def node_chat_1(state: ChatState) -> ChatState:
    user_text = state.get("last_user_input")
    user_info = dict(state.get("user_info") or {})
    user_info["size"] = user_text
    return {**state, "step": ChatStep.CHAT_2, "user_info": user_info, "messages": _append_message(state, role="user", content=user_text)}

def node_chat_2(state: ChatState) -> ChatState:
    user_text = state.get("last_user_input")
    user_info = dict(state.get("user_info") or {})
    owned = user_text.get("owned", []) if isinstance(user_text, dict) else (user_text if isinstance(user_text, list) else [])
    needed = user_text.get("needed", []) if isinstance(user_text, dict) else []
    user_info["owned_appliances"], user_info["needed_appliances"] = owned, needed
    return {**state, "step": ChatStep.CHAT_3, "user_info": user_info, "messages": _append_message(state, role="user", content=user_text)}

def node_chat_3(state: ChatState) -> ChatState:
    user_text = state.get("last_user_input")
    user_info = dict(state.get("user_info") or {})
    s = _to_str(user_text)
    need_furniture = False if "없" in s else True
    user_info["need_furniture"] = need_furniture
    next_step = ChatStep.CHAT_3_1 if need_furniture else ChatStep.CHAT_4
    return {**state, "step": next_step, "user_info": user_info, "messages": _append_message(state, role="user", content=user_text)}

def node_chat_3_1(state: ChatState) -> ChatState:
    user_text = state.get("last_user_input")
    user_info = dict(state.get("user_info") or {})
    user_info["interior_style"] = user_text
    return {**state, "step": ChatStep.CHAT_4, "user_info": user_info, "messages": _append_message(state, role="user", content=user_text)}

def node_chat_4(state: ChatState) -> ChatState:
    user_text = state.get("last_user_input")
    user_info = dict(state.get("user_info") or {})
    user_info["lifestyle"] = user_text
    return {**state, "step": ChatStep.CHAT_5, "user_info": user_info, "messages": _append_message(state, role="user", content=user_text)}

def node_chat_5(state: ChatState) -> ChatState:
    user_text = state.get("last_user_input")
    user_info = dict(state.get("user_info") or {})
    try:
        parsed = _parse_budget(user_text)
        user_info.update(parsed)
        return {**state, "step": ChatStep.CHAT_6, "user_info": user_info, "messages": _append_message(state, role="user", content=user_text)}
    except ValueError:
        return {**state, "step": ChatStep.CHAT_5, "data": {"error": "예산을 다시 입력해주세요."}, "messages": _append_message(state, role="user", content=user_text)}

def node_chat_6(state: ChatState) -> ChatState:
    user_text = state.get("last_user_input")
    s = _to_str(user_text)
    if "추천" in s and ("보기" in s or "리스트" in s):
        return {**state, "step": ChatStep.CHAT_RESULT, "messages": _append_message(state, role="user", content=user_text)}
    return {**state, "step": ChatStep.CHAT_6, "messages": _append_message(state, role="user", content=user_text)}

async def node_chat_result(state: ChatState) -> ChatState:
    user_info = state.get("user_info") or {}
    messages = list(state.get("messages") or [])

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
        prompt = build_recommendation_prompt(user_info)
        resp = await llm_json.ainvoke(prompt) # 비동기 환경이므로 ainvoke 추천
        content_json = _extract_json_from_llm_response(resp)
        full_recommendation_list = content_json.get("recommendation_list") or content_json.get("data") or []
        total_estimated_budget = content_json.get("total_estimated_budget", "")

    if not isinstance(full_recommendation_list, list):
        full_recommendation_list = []

    for item in full_recommendation_list:
        if isinstance(item, dict):
            item.setdefault("name", item.get("package_name") or item.get("name") or item.get("title") or "")
            item.update(_compute_budget_breakdown(item))

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
                        "name": p.get("name"),
                        "model_id": p.get("model_id") or p.get("model"),
                        "brand": p.get("brand"),
                        "price_normal": p.get("price_normal"),
                        "price_subscription": p.get("price_subscription"),
                        "price": p.get("price"),
                        "image_url": p.get("image_url"),
                        "url": p.get("url"),
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

# ⭐ 수정 포인트: async def 로 변경
async def node_rag_chat(state: ChatState) -> ChatState:
    user_info = state.get("user_info") or {}
    user_text = state.get("last_user_input") or ""
    messages = _append_message(state, role="user", content=user_text)
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

    prompt = None
    package_idx = _extract_package_index(_to_str(user_text))
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
                        # 이제 async def 안이므로 정상 작동합니다!
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
                            user_info, _to_str(user_text), package_context=package_context
                        )
                    except Exception:
                        prompt = None

    if prompt is None:
        prompt = build_rag_prompt(user_info, user_text)

    resp = await llm_json.ainvoke(prompt)
    data_json = _extract_json_from_llm_response(resp)
    request_more = bool(data_json.get("request_more_packages"))
    answer_text = data_json.get("answer", "")

    if request_more:
        all_recs = new_data.get("all_recommendations") or []
        next_index = int(new_data.get("next_index", 0))
        if next_index >= len(all_recs):
            new_data["display_recommendations"] = []
            new_data["error"] = "모든 패키지 조합을 보여드렸습니다."
        else:
            batch = all_recs[next_index : next_index + 3]
            new_data["display_recommendations"] = batch
            new_data["next_index"] = next_index + len(batch)

    messages.append({"role": "assistant", "content": answer_text})
    return {
        **state,
        "step": ChatStep.RAG_CHAT,
        "ai_response": answer_text,
        "messages": messages,
        "data": new_data,
        "is_completed": True,
    }

def route_from_step(state: ChatState) -> str:
    requested = state.get("requested_step_code")
    step = ChatStep(requested) if requested and requested in ChatStep.__members__ else state.get("step", ChatStep.CHAT_0)
    mapping = {
        ChatStep.CHAT_0: "chat_0", ChatStep.CHAT_1: "chat_1", ChatStep.CHAT_2: "chat_2",
        ChatStep.CHAT_3: "chat_3", ChatStep.CHAT_3_1: "chat_3_1", ChatStep.CHAT_4: "chat_4",
        ChatStep.CHAT_5: "chat_5", ChatStep.CHAT_6: "chat_6",
        ChatStep.CHAT_RESULT: "chat_result", ChatStep.RAG_CHAT: "rag_chat"
    }
    return mapping.get(step, "chat_0")

def build_graph():
    workflow = StateGraph(ChatState)
    nodes = {
        "chat_0": node_chat_0, "chat_1": node_chat_1, "chat_2": node_chat_2,
        "chat_3": node_chat_3, "chat_3_1": node_chat_3_1, "chat_4": node_chat_4,
        "chat_5": node_chat_5, "chat_6": node_chat_6,
        "chat_result": node_chat_result, "rag_chat": node_rag_chat
    }
    for name, func in nodes.items():
        workflow.add_node(name, func)
    workflow.add_conditional_edges(START, route_from_step)
    for name in nodes.keys():
        workflow.add_edge(name, END)
    return workflow.compile(checkpointer=get_checkpointer())

chat_app = build_graph()