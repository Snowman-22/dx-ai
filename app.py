import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.graph import chat_app, ChatState, ChatStep
from src.state_store import get_checkpointer

from sqlalchemy import text
from src.db import get_engine  # 이미 있다면 패스!
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import Depends

class ChatRequest(BaseModel):
    # 하나의 conv_id(문자열)로 채팅 세션을 식별
    conv_id: str
    # 프론트/스프링이 "현재 단계"를 들고 있고, FastAPI는 그 단계에 맞는 노드를 실행
    # (기존에는 FastAPI가 step_code를 응답으로 내려주며 다음 단계를 주도)
    step_code: str
    # SpringBoot/프론트가 화면에 표시한 assistant 문구를 그대로 저장하고 싶을 때 전달
    assistant_text: Optional[object] = None
    user_text: Optional[object] = None


class ChatResponse(BaseModel):
    data: dict
    ai_response: Optional[str] = None


def create_app() -> FastAPI:
    app = FastAPI(title="FastAPI LangGraph Chatbot")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # checkpointer는 graph.compile 시 사용되므로 여기서는 생성만 보장(사이드이펙트 없음)
    _ = get_checkpointer()

    @app.post("/ai/chat", response_model=ChatResponse)
    async def chat_endpoint(payload: ChatRequest):
        if not payload.conv_id:
            raise HTTPException(status_code=400, detail="conv_id is required")
        if not payload.step_code:
            raise HTTPException(status_code=400, detail="step_code is required")

        # LangGraph state 초기화 / 복구
        # conv_id를 그대로 LangGraph thread_id로 사용하고,
        # DynamoDB 체크포인터 및 Postgres Chat.conv_id와 동일한 값으로 연결합니다.
        config = {"configurable": {"thread_id": payload.conv_id}}

        # 입력을 LangGraph용 상태로 변환
        # 중요: checkpointer(DynamoDB/메모리)가 이전 state(user_info 등)를 복구하므로,
        # 여기서 빈 dict로 덮어쓰지 않도록 "이번 턴에 새로 들어온 정보"만 전달합니다.
        #
        # step_code는 "프론트/스프링이 가진 현재 단계"이며, FastAPI는 이를 기준으로 라우팅합니다.
        # (checkpoint에 저장된 step이 있더라도, 요청으로 들어온 step_code를 우선합니다.)
        state: ChatState = {
            # checkpointer가 복구한 기존 step이 라우팅에 우선 적용될 수 있어,
            # 요청에서 넘어온 step_code를 별도 필드로 보관하고 그래프 라우팅에서 우선순위를 강제합니다.
            "requested_step_code": payload.step_code,
            "incoming_assistant_message": payload.assistant_text,
            "last_user_input": payload.user_text,
        }
        result = await chat_app.ainvoke(state, config=config)

        data = result.get("data", {}) or {}
        ai_response = result.get("ai_response")

        # CHAT_0~CHAT_6 구간은 일반적으로 data가 비어있으므로,
        # 프론트/스프링에서 성공 여부를 단순하게 처리할 수 있도록 message를 내려준다.
        if payload.step_code in {
            "CHAT_0",
            "CHAT_1",
            "CHAT_2",
            "CHAT_3",
            "CHAT_3_1",
            "CHAT_4",
            "CHAT_5",
            "CHAT_6",
        }:
            if isinstance(data, dict) and "error" not in data:
                data = {"message": "성공적으로 저장되었습니다."}

        return ChatResponse(data=data, ai_response=ai_response)




    # DB 세션을 가져오는 의존성 주입 함수 (만약 이미 src.db 등에 있다면 그걸 써도 돼!)
    async def get_db():
        engine = get_engine()
        async with AsyncSession(engine) as session:
            try:
                yield session
            finally:
                await session.close()

    @app.get("/users/all")
    async def get_all_test_users(db: AsyncSession = Depends(get_db)):
        try:
            # 안전하게 public.test_user라고 명시
            query = text("SELECT * FROM public.test_user")
            result = await db.execute(query)
            
            # 조회 결과를 딕셔너리 리스트로 변환
            users = [dict(row) for row in result.mappings().all()]
            
            return {
                "status": "success",
                "count": len(users),
                "data": users
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"RDS 조회 실패: {str(e)}")


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

