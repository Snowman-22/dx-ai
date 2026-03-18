import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.graph import chat_app, ChatState, ChatStep
from src.state_store import get_checkpointer
from src.db import get_engine
from src.recommend.service import ensure_chat_metadata, save_recommendations_to_db


class ChatRequest(BaseModel):
    # 하나의 conv_id(문자열)로 채팅 세션을 식별
    conv_id: str
    # SpringBoot/프론트가 화면에 표시한 assistant 문구를 그대로 저장하고 싶을 때 전달
    assistant_text: Optional[object] = None
    user_text: Optional[object] = None


class ChatResponse(BaseModel):
    chat_type: str
    step_code: str
    data: dict
    is_completed: bool
    ai_response: Optional[str] = None


def _chat_type_from_step(step_code: str) -> str:
    # CHAT_0~CHAT_6: 사용자정보 파악 채팅
    if step_code in {
        "CHAT_0",
        "CHAT_1",
        "CHAT_2",
        "CHAT_3",
        "CHAT_3_1",
        "CHAT_4",
        "CHAT_5",
        "CHAT_6",
    }:
        return "USER_INFO_CHAT"
    # CHAT_RESULT~RAG_CHAT: 패키지 추천/추천 이후 대화
    if step_code in {"CHAT_RESULT", "RAG_CHAT"}:
        return "PACKAGE_RECO_CHAT"
    return "UNKNOWN"


def create_app() -> FastAPI:
    app = FastAPI(title="FastAPI LangGraph Chatbot")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    checkpointer = get_checkpointer()

    @app.on_event("startup")
    async def _startup():
        # DB 커넥션이 유효한지 빠르게 확인(실패 시 로그로 확인 가능)
        engine = get_engine()
        async with engine.begin() as conn:
            await conn.run_sync(lambda _: None)

    @app.post("/ai/chat", response_model=ChatResponse)
    async def chat_endpoint(payload: ChatRequest):
        if not payload.conv_id:
            raise HTTPException(status_code=400, detail="conv_id is required")

        # LangGraph state 초기화 / 복구
        # conv_id를 그대로 LangGraph thread_id로 사용하고,
        # DynamoDB 체크포인터 및 Postgres Chat.conv_id와 동일한 값으로 연결합니다.
        config = {"configurable": {"thread_id": payload.conv_id}}

        # 입력을 LangGraph용 상태로 변환
        # 중요: checkpointer(DynamoDB/메모리)가 이전 state(user_info 등)를 복구하므로,
        # 여기서 빈 dict로 덮어쓰지 않도록 "이번 턴에 새로 들어온 정보"만 전달합니다.
        state: ChatState = {
            "incoming_assistant_message": payload.assistant_text,
            "last_user_input": payload.user_text,
        }
        result = await chat_app.ainvoke(state, config=config)

        # 어떤 단계에서 끊기더라도 Postgres Chat 메타데이터는 남도록 conv_id로 upsert
        try:
            await ensure_chat_metadata(chat_session_id=payload.conv_id)
        except Exception:
            # 메타데이터 저장 실패는 챗 응답에는 영향 없도록 무시
            import logging

            logging.getLogger(__name__).exception("Failed to ensure chat metadata")

        # 추천 결과 단계가 완료되었다면, Postgres에 Recommendation/Chat 저장
        step = result.get("step")
        if step == ChatStep.CHAT_RESULT and result.get("is_completed"):
            rec_data = result.get("data") or {}
            recommendation_list = rec_data.get("recommendation_list") or []
             # user_info 기반으로 Chat 제목 생성 시도
            user_info = result.get("user_info") or {}
            size = user_info.get("size") or ""
            lifestyle = user_info.get("lifestyle") or ""
            budget_range = user_info.get("budget_range_manwon") or {}
            min_budget = budget_range.get("min")
            max_budget = budget_range.get("max")

            if min_budget is not None and max_budget is not None:
                budget_text = f"{min_budget}~{max_budget}만원"
            elif min_budget is not None and max_budget is None:
                budget_text = f"{min_budget}만원 이상"
            elif min_budget is None and max_budget is not None:
                budget_text = f"{max_budget}만원 이하"
            else:
                budget_text = "예산 미정"

            # 예: "10~20평 1인 자취, 예산 150~300만원 진단"
            # lifestyle가 없으면 생략
            title_parts = []
            if size:
                title_parts.append(str(size))
            if lifestyle:
                title_parts.append(str(lifestyle))
            title_prefix = " ".join(title_parts).strip()
            if title_prefix:
                chat_title = f"{title_prefix}, 예산 {budget_text} 진단"
            else:
                chat_title = f"가전/가구 추천, 예산 {budget_text} 진단"

            try:
                await save_recommendations_to_db(
                    chat_session_id=payload.conv_id,
                    recommendation_list=recommendation_list,
                    chat_title=chat_title,
                )
            except Exception as e:
                # 저장 실패는 추천 응답 자체에는 영향을 주지 않도록 로그만 남기는 것이 안전
                # (여기서는 단순 예시라 re-raise 하지 않음)
                import logging

                logging.getLogger(__name__).exception(
                    "Failed to save recommendations to DB: %s", e
                )

        return ChatResponse(
            chat_type=_chat_type_from_step(str(result["step"])),
            step_code=result["step"],
            data=result.get("data", {}),
            is_completed=result.get("is_completed", False),
            ai_response=result.get("ai_response"),
        )

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=True,
    )

