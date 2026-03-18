from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from .state_store import get_checkpointer
from .prompt import build_recommendation_prompt, build_rag_prompt
from .recommend import generate_recommendation_result


class ChatStep(str, Enum):
    CHAT_0 = "CHAT_0"  # 진단 시작
    CHAT_1 = "CHAT_1"  # 주거공간 크기(평수)
    CHAT_2 = "CHAT_2"  # 보유/필요 가전
    CHAT_3 = "CHAT_3"  # 가구/소품 추천 필요 여부(+ 자유 입력)
    CHAT_3_1 = "CHAT_3_1"  # 인테리어 스타일
    CHAT_4 = "CHAT_4"  # 라이프스타일
    CHAT_5 = "CHAT_5"  # 예산
    CHAT_6 = "CHAT_6"  # 진단 완료(추천 리스트 보기 버튼)
    CHAT_RESULT = "CHAT_RESULT"  # 추천 결과
    RAG_CHAT = "RAG_CHAT"  # 추천 이후 자유대화


class ChatState(TypedDict, total=False):
    step: ChatStep
    user_info: Dict[str, Any]
    messages: List[Dict[str, Any]]
    # SpringBoot/프론트가 "이번 턴에 화면에 띄운 assistant 문구"를 같이 보내면,
    # 체크포인터로 복구된 messages에 append한 뒤 pop 처리한다.
    incoming_assistant_message: Any
    last_user_input: Any
    data: Dict[str, Any]
    is_completed: bool
    ai_response: str


llm_json = ChatOpenAI(model="gpt-4o-mini", temperature=0, response_format={"type": "json_object"})
llm_text = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)


def _to_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def _append_message(state: ChatState, *, role: str, content: Any) -> List[Dict[str, Any]]:
    """
    DynamoDB에 저장될 메시지 로그용 헬퍼.
    - role: "user" | "assistant" 등
    - content: 원본 입력(문자열/객체)을 그대로 저장
    """
    messages = list(state.get("messages") or [])
    messages.append({"role": role, "content": content})
    return messages


def _parse_budget(user_text: Any) -> Dict[str, Any]:
    """
    예산 입력:
    - 숫자 문자열: "100" -> 100 (만원)
    - 선택지 문자열: "50만 원 이하" 등 -> 범주/범위 저장
    """
    raw = user_text
    s = _to_str(user_text)
    if not s:
        raise ValueError("empty")

    if s.isdigit():
        return {"budget_type": "number", "budget_manwon": int(s)}

    normalized = s.replace(" ", "")
    mapping = {
        "50만원이하": {"min": 0, "max": 50},
        "50~150만원": {"min": 50, "max": 150},
        "150~300만원": {"min": 150, "max": 300},
        "300만원이상": {"min": 300, "max": None},
        "아직정하지않았어요": {"min": None, "max": None},
    }
    if normalized in mapping:
        return {
            "budget_type": "choice",
            "budget_choice": raw,
            "budget_range_manwon": mapping[normalized],
        }

    raise ValueError("invalid budget")


def node_chat_0(state: ChatState) -> ChatState:
    return {
        **state,
        "step": ChatStep.CHAT_1,
        "data": {},
        "is_completed": False,
    }


def node_chat_1(state: ChatState) -> ChatState:
    # 평수(선택/자유 입력) 저장
    user_text = state.get("last_user_input")
    user_info = dict(state.get("user_info") or {})
    user_info["size"] = user_text
    messages = _append_message(state, role="user", content=user_text)

    return {
        **state,
        "step": ChatStep.CHAT_2,
        "user_info": user_info,
        "messages": messages,
        "data": {},
        "is_completed": False,
    }


def node_chat_2(state: ChatState) -> ChatState:
    # 보유/필요 가전: 프론트 탭 UI에 따라 다양한 형태가 올 수 있어 방어적으로 처리
    # 권장 형태: {"owned": [...], "needed": [...]} 또는 {"owned_appliances": [...], "needed_appliances": [...]}
    # 임시 허용: 리스트만 오면 owned로 간주
    user_text = state.get("last_user_input")
    user_info = dict(state.get("user_info") or {})

    if isinstance(user_text, dict):
        owned = user_text.get("owned") or user_text.get("owned_appliances") or []
        needed = user_text.get("needed") or user_text.get("needed_appliances") or []
    elif isinstance(user_text, list):
        owned = user_text
        needed = []
    else:
        owned = []
        needed = []

    user_info["owned_appliances"] = owned
    user_info["needed_appliances"] = needed
    messages = _append_message(state, role="user", content=user_text)

    return {
        **state,
        "step": ChatStep.CHAT_3,
        "user_info": user_info,
        "messages": messages,
        "data": {},
        "is_completed": False,
    }


def node_chat_3(state: ChatState) -> ChatState:
    # 가구/소품 추천 필요 여부
    # 허용 형태:
    # - {"need_furniture": true, "furniture_note": "..."}  (네 + 입력)
    # - {"need_furniture": false}                         (아직 필요없어요)
    # - 문자열 "네, 추천 해주세요" / "아직 필요 없어요" 도 방어적으로 허용
    user_text = state.get("last_user_input")
    user_info = dict(state.get("user_info") or {})
    need_furniture = None
    furniture_note = None

    if isinstance(user_text, dict):
        if "need_furniture" in user_text:
            need_furniture = bool(user_text.get("need_furniture"))
        if "furniture_note" in user_text:
            furniture_note = user_text.get("furniture_note")
        if "note" in user_text and furniture_note is None:
            furniture_note = user_text.get("note")
    else:
        s = _to_str(user_text)
        if "없" in s:
            need_furniture = False
        elif "네" in s or "추천" in s:
            need_furniture = True

    user_info["need_furniture"] = need_furniture
    if furniture_note:
        user_info["furniture_note"] = furniture_note

    next_step = ChatStep.CHAT_3_1 if need_furniture else ChatStep.CHAT_4
    messages = _append_message(state, role="user", content=user_text)

    return {
        **state,
        "step": next_step,
        "user_info": user_info,
        "messages": messages,
        "data": {},
        "is_completed": False,
    }


def node_chat_3_1(state: ChatState) -> ChatState:
    # 인테리어 스타일 선택
    user_text = state.get("last_user_input")
    user_info = dict(state.get("user_info") or {})
    user_info["interior_style"] = user_text
    messages = _append_message(state, role="user", content=user_text)

    return {
        **state,
        "step": ChatStep.CHAT_4,
        "user_info": user_info,
        "messages": messages,
        "data": {},
        "is_completed": False,
    }


def node_chat_4(state: ChatState) -> ChatState:
    # 라이프스타일 선택
    user_text = state.get("last_user_input")
    user_info = dict(state.get("user_info") or {})
    user_info["lifestyle"] = user_text
    messages = _append_message(state, role="user", content=user_text)

    return {
        **state,
        "step": ChatStep.CHAT_5,
        "user_info": user_info,
        "messages": messages,
        "data": {},
        "is_completed": False,
    }


def node_chat_5(state: ChatState) -> ChatState:
    # 예산 입력(선택지 또는 숫자)
    user_text = state.get("last_user_input")
    user_info = dict(state.get("user_info") or {})
    try:
        parsed = _parse_budget(user_text)
        user_info.update(parsed)
        messages = _append_message(state, role="user", content=user_text)
        return {
            **state,
            "step": ChatStep.CHAT_6,
            "user_info": user_info,
            "messages": messages,
            "data": {},
            "is_completed": False,
        }
    except ValueError:
        return {
            **state,
            "step": ChatStep.CHAT_5,
            "messages": _append_message(state, role="user", content=user_text),
            "data": {"error": "예산은 숫자만(단위 제외) 또는 제공된 선택지 중 하나로 입력해주세요."},
            "is_completed": False,
        }


def node_chat_6(state: ChatState) -> ChatState:
    # "추천 리스트 보기" 버튼을 누르면 결과 생성으로 이동
    user_text = state.get("last_user_input")
    s = _to_str(user_text)
    messages = _append_message(state, role="user", content=user_text)
    if "추천" in s and ("보기" in s or "리스트" in s):
        return {
            **state,
            "step": ChatStep.CHAT_RESULT,
            "messages": messages,
            "data": {},
            "is_completed": False,
        }
    return {
        **state,
        "step": ChatStep.CHAT_6,
        "messages": messages,
        "data": {},
        "is_completed": False,
    }


async def node_chat_result(state: ChatState) -> ChatState:
    """
    추천 결과 생성.
    - 1순위: DB 기반 추천(상품 후보 조회 + ipynb 알고리즘 재랭킹)
    - 2순위(폴백): OpenAI JSON 모드로 추천 리스트 생성
    """
    user_info = state.get("user_info") or {}
    messages = list(state.get("messages") or [])

    try:
        result = await generate_recommendation_result(user_info=user_info)
        # 알고리즘 결과를 통일된 패키지 리스트 형태로 변환
        full_recommendation_list = [
            {
                "package_name": getattr(item, "package_name", item.name),
                "products": getattr(item, "products", [item.name]),
                "reason": item.reason,
                "estimated_price": item.estimated_price,
            }
            for item in result.recommendation_list
        ]
        total_estimated_budget = result.total_estimated_budget
    except NotImplementedError:
        prompt = build_recommendation_prompt(user_info)
        resp = llm_json.invoke(prompt)

        content = resp.dict()["content"][0]["text"]
        if isinstance(content, str):
            import json

            content_json = json.loads(content)
        else:
            content_json = content

        # LLM에서도 패키지 리스트 스키마를 맞춘다고 가정
        full_recommendation_list = content_json.get("recommendation_list", [])
        total_estimated_budget = content_json.get("total_estimated_budget", "")

    # 처음에는 상위 3개만 노출하고, 나머지는 RAG_CHAT에서 점진적으로 사용
    display_recommendations = full_recommendation_list[:3]
    next_index = len(display_recommendations)

    # 추천 결과가 생성되었음을 assistant 메시지로 로그에 남김
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
    추천 이후 자유 대화.
    - LLM이 사용자의 의도를 보고 "다른 패키지/조합" 요청 여부를 JSON으로 알려줌.
    - request_more_packages=true 이면, 저장된 추천 리스트에서 다음 3개를 잘라서 전달.
    """
    import json

    user_info = state.get("user_info") or {}
    user_text = state.get("last_user_input") or ""
    # 현재 턴의 사용자 발화를 user role로 기록
    messages = _append_message(state, role="user", content=user_text)

    prompt = build_rag_prompt(user_info, user_text)
    # 의도 판별 및 답변 생성을 JSON 모드로 요청
    resp = llm_json.invoke(prompt)
    content = resp.dict()["content"][0]["text"]
    if isinstance(content, str):
        data_json = json.loads(content)
    else:
        data_json = content

    request_more = bool(data_json.get("request_more_packages"))
    answer_text = data_json.get("answer", "")

    new_data = dict(state.get("data") or {})

    if request_more:
        all_recs = new_data.get("all_recommendations") or []
        next_index = int(new_data.get("next_index", 0))

        # 이미 모든 패키지를 다 보여준 상태라면, 추가 패키지가 없다는 메시지를 보낸다.
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


def route_from_step(state: ChatState) -> str:
    step = state.get("step", ChatStep.CHAT_0)

    if step == ChatStep.CHAT_0:
        return "chat_0"
    if step == ChatStep.CHAT_1:
        return "chat_1"
    if step == ChatStep.CHAT_2:
        return "chat_2"
    if step == ChatStep.CHAT_3:
        return "chat_3"
    if step == ChatStep.CHAT_3_1:
        return "chat_3_1"
    if step == ChatStep.CHAT_4:
        return "chat_4"
    if step == ChatStep.CHAT_5:
        return "chat_5"
    if step == ChatStep.CHAT_6:
        return "chat_6"
    if step == ChatStep.CHAT_RESULT:
        return "chat_result"
    if step == ChatStep.RAG_CHAT:
        return "rag_chat"
    return "chat_0"


def build_graph():
    workflow = StateGraph(ChatState)

    workflow.add_node("chat_0", node_chat_0)
    workflow.add_node("chat_1", node_chat_1)
    workflow.add_node("chat_2", node_chat_2)
    workflow.add_node("chat_3", node_chat_3)
    workflow.add_node("chat_3_1", node_chat_3_1)
    workflow.add_node("chat_4", node_chat_4)
    workflow.add_node("chat_5", node_chat_5)
    workflow.add_node("chat_6", node_chat_6)
    workflow.add_node("chat_result", node_chat_result)
    workflow.add_node("rag_chat", node_rag_chat)

    # step 기반 라우팅
    workflow.set_entry_point("router")

    def router(state: ChatState) -> ChatState:
        # 단순 패스용; 실제 라우팅은 conditional_edges에서 처리
        # 프론트가 "assistant_text"를 보내면, 여기서 messages에 누적 저장한다.
        incoming = state.get("incoming_assistant_message")
        if incoming is not None and _to_str(incoming) != "":
            state = {
                **state,
                "messages": _append_message(state, role="assistant", content=incoming),
            }
        # 동일 메시지가 다음 턴에 중복 저장되지 않도록 제거
        if "incoming_assistant_message" in state:
            state.pop("incoming_assistant_message", None)
        return state

    workflow.add_node("router", router)

    workflow.add_conditional_edges(
        "router",
        route_from_step,
        {
            "chat_0": "chat_0",
            "chat_1": "chat_1",
            "chat_2": "chat_2",
            "chat_3": "chat_3",
            "chat_3_1": "chat_3_1",
            "chat_4": "chat_4",
            "chat_5": "chat_5",
            "chat_6": "chat_6",
            "chat_result": "chat_result",
            "rag_chat": "rag_chat",
        },
    )

    # 각 노드는 다시 router로 돌아가거나 종료
    for node in ["chat_0", "chat_1", "chat_2", "chat_3", "chat_3_1", "chat_4", "chat_5", "chat_6"]:
        workflow.add_edge(node, "router")

    workflow.add_edge("chat_result", END)
    workflow.add_edge("rag_chat", END)

    app = workflow.compile(checkpointer=get_checkpointer())
    return app


chat_app = build_graph()

