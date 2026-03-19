from __future__ import annotations

import json
from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START

from .state_store import get_checkpointer
from .prompt import build_recommendation_prompt, build_rag_prompt
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
    # checkpointer가 복구한 기존 state의 `step`이 요청 값을 덮어쓰는 케이스가 있어,
    # 요청에서 온 step_code를 별도 필드로 보관해 라우팅 우선순위를 강제합니다.
    requested_step_code: Optional[str]


llm_json = ChatOpenAI(model="gpt-4o-mini", temperature=0, response_format={"type": "json_object"})
llm_text = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

# --- 헬퍼 함수들 ---
def _to_str(x: Any) -> str:
    return str(x).strip() if x is not None else ""

def _extract_json_from_llm_response(resp: Any) -> dict[str, Any]:
    """
    ChatOpenAI(response_format={"type":"json_object"}) 응답을
    JSON dict로 안전하게 추출합니다.
    - resp.content: 보통 JSON 문자열
    - resp.dict()["content"]: 환경/버전에 따라 문자열로 내려올 수 있음
    """
    import json as _json

    # 1) 권장 경로: AIMessage.content
    if hasattr(resp, "content"):
        content = resp.content
    else:
        # 2) 차선: dict 구조에서 content 키 추출
        try:
            content = resp.dict().get("content")
        except Exception:
            content = None

    if isinstance(content, str):
        return _json.loads(content)
    if isinstance(content, dict):
        return content

    # 마지막 방어: str로 강제 변환 후 시도(실패하면 예외 발생)
    return _json.loads(str(content))

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

# --- 노드 정의 (각 노드는 작업 후 'END'로 갈 수 있게 설계) ---

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
    # 가전 정보 처리 로직 (생략 가능하지만 유지)
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
    """
    추천 결과 생성.
    - DB 기반 추천(재랭킹/패키징) 시도
    - NotImplementedError(알고리즘 미이식)면 LLM JSON 모드로 폴백
    """
    user_info = state.get("user_info") or {}
    messages = list(state.get("messages") or [])

    try:
        result = await generate_recommendation_result(user_info=user_info)
        # 알고리즘 결과를 통일된 패키지 리스트 형태로 변환
        full_recommendation_list = [
            {
                "package_name": getattr(item, "package_name", getattr(item, "name", "")),
                "products": getattr(item, "products", [getattr(item, "name", "")]),
                "reason": item.reason,
                "estimated_price": item.estimated_price,
            }
            for item in result.recommendation_list
        ]
        total_estimated_budget = result.total_estimated_budget
    except NotImplementedError:
        prompt = build_recommendation_prompt(user_info)
        resp = llm_json.invoke(prompt)

        content_json = _extract_json_from_llm_response(resp)

        # LLM 폴백 결과는 프롬프트 스키마에 맞춰 recommendation_list/total_estimated_budget만 추출
        full_recommendation_list = content_json.get("recommendation_list", [])
        total_estimated_budget = content_json.get("total_estimated_budget", "")

    # 처음에는 상위 3개만 노출하고, 나머지는 RAG_CHAT에서 점진적으로 사용
    display_recommendations = full_recommendation_list[:3]
    next_index = len(display_recommendations)

    # 추천 결과를 messages 로그에 남김
    messages.append(
        {
            "role": "assistant",
            "content": {
                "type": "recommendation_result",
                "display_recommendations": display_recommendations,
                "total_estimated_budget": total_estimated_budget,
            },
        }
    )

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

def node_rag_chat(state: ChatState) -> ChatState:
    """
    추천 이후 자유대화.
    - 사용자의 의도를 JSON으로 판별(request_more_packages / answer)
    - request_more_packages=true면 다음 3개 배치를 display_recommendations로 업데이트
    """
    user_info = state.get("user_info") or {}
    user_text = state.get("last_user_input") or ""

    # 현재 턴의 사용자 발화를 기록
    messages = _append_message(state, role="user", content=user_text)

    prompt = build_rag_prompt(user_info, user_text)
    resp = llm_json.invoke(prompt)

    data_json = _extract_json_from_llm_response(resp)

    request_more = bool(data_json.get("request_more_packages"))
    answer_text = data_json.get("answer", "")

    new_data = dict(state.get("data") or {})

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

    # assistant 응답을 메시지 로그에 추가
    messages.append({"role": "assistant", "content": answer_text})

    return {
        **state,
        "step": ChatStep.RAG_CHAT,
        "ai_response": answer_text,
        "messages": messages,
        "data": new_data,
        "is_completed": True,
    }

# --- 라우터 함수 ---
def route_from_step(state: ChatState) -> str:
    requested = state.get("requested_step_code")
    if requested:
        try:
            step = ChatStep(requested)
        except ValueError:
            step = state.get("step", ChatStep.CHAT_0)
    else:
        step = state.get("step", ChatStep.CHAT_0)
    mapping = {
        ChatStep.CHAT_0: "chat_0", ChatStep.CHAT_1: "chat_1",
        ChatStep.CHAT_2: "chat_2", ChatStep.CHAT_3: "chat_3",
        ChatStep.CHAT_3_1: "chat_3_1", ChatStep.CHAT_4: "chat_4",
        ChatStep.CHAT_5: "chat_5", ChatStep.CHAT_6: "chat_6",
        ChatStep.CHAT_RESULT: "chat_result", ChatStep.RAG_CHAT: "rag_chat"
    }
    return mapping.get(step, "chat_0")

# --- 그래프 빌드 ---
def build_graph():
    workflow = StateGraph(ChatState)

    # 1. 노드 추가
    nodes = {
        "chat_0": node_chat_0, "chat_1": node_chat_1, "chat_2": node_chat_2,
        "chat_3": node_chat_3, "chat_3_1": node_chat_3_1, "chat_4": node_chat_4,
        "chat_5": node_chat_5, "chat_6": node_chat_6,
        "chat_result": node_chat_result, "rag_chat": node_rag_chat
    }
    for name, func in nodes.items():
        workflow.add_node(name, func)

    # 2. 시작점 설정 (현재 step을 보고 바로 해당 노드로 점프!)
    workflow.add_conditional_edges(START, route_from_step)

    # 3. 모든 노드의 끝을 END로 연결 (이게 무한루프 해결 포인트!)
    for name in nodes.keys():
        workflow.add_edge(name, END)

    app = workflow.compile(checkpointer=get_checkpointer())
    return app

chat_app = build_graph()