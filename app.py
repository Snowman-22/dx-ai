import os
import sys
import logging
import json
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from src.graph import ChatStep, chat_app, ChatState
from src.state_store import get_checkpointer


def _normalize_step_code(raw: str) -> str:
    """하이픈/공백/대소문자 차이로 ChatStep 매칭이 실패해 잘못된 노드로 가는 것을 방지."""
    return (raw or "").strip().replace("-", "_").upper()

from sqlalchemy import text
from src.db import get_engine  # 이미 있다면 패스!
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import Depends


class ChatRequest(BaseModel):
    """snake_case(user_text) 또는 프론트 camelCase(userText) 모두 수용."""

    model_config = ConfigDict(populate_by_name=True)

    # 하나의 conv_id(문자열)로 채팅 세션을 식별
    conv_id: str = Field(alias="convId")
    # 프론트/스프링이 "현재 단계"를 들고 있고, FastAPI는 그 단계에 맞는 노드를 실행
    # (기존에는 FastAPI가 step_code를 응답으로 내려주며 다음 단계를 주도)
    step_code: str = Field(alias="stepCode")
    # SpringBoot/프론트가 화면에 표시한 assistant 문구를 그대로 저장하고 싶을 때 전달
    assistant_text: Optional[object] = Field(default=None, alias="assistantText")
    user_text: Optional[object] = Field(default=None, alias="userText")


class ChatResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    data: dict
    ai_response: Optional[str] = Field(default=None, alias="aiResponse")


def create_app() -> FastAPI:
    app = FastAPI(title="FastAPI LangGraph Chatbot")

    # Docker 로그(stdout)로 남기기 위한 로깅 설정
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
        )
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # checkpointer는 graph.compile 시 사용되므로 여기서는 생성만 보장(사이드이펙트 없음)
    _ = get_checkpointer()

    @app.post("/ai/chat", response_model=ChatResponse, response_model_by_alias=True)
    async def chat_endpoint(payload: ChatRequest):
        if not payload.conv_id:
            raise HTTPException(status_code=400, detail="conv_id is required")
        if not payload.step_code:
            raise HTTPException(status_code=400, detail="step_code is required")

        norm_step = _normalize_step_code(payload.step_code)
        if norm_step not in ChatStep.__members__:
            return ChatResponse(
                data={
                    "error": (
                        f"지원하지 않는 stepCode입니다: {payload.step_code!r}. "
                        f"허용: {', '.join(sorted(ChatStep.__members__))}"
                    )
                },
                ai_response=None,
            )

        # LangGraph state 초기화 / 복구
        # conv_id를 그대로 LangGraph thread_id로 사용하고,
        # DynamoDB 체크포인터 및 Postgres Chat.conv_id와 동일한 값으로 연결합니다.
        config = {"configurable": {"thread_id": payload.conv_id}}

        def _to_jsonable(x):
            # pydantic/ORM/기타 객체 직렬화에 대비
            return x if isinstance(x, (dict, list, str, int, float, bool)) or x is None else str(x)

        # 요청 입력 로깅(요청/입력/merge까지 전부 stdout로 출력)
        logger = logging.getLogger("fastapi_chat")
        logger.info(
            "INCOMING /ai/chat payload(by_alias)=%s",
            json.dumps(
                payload.model_dump(by_alias=True),
                ensure_ascii=False,
                default=_to_jsonable,
            ),
        )

        # 입력을 LangGraph용 상태로 변환
        # 중요: checkpointer(DynamoDB/메모리)가 이전 state(user_info 등)를 복구하므로,
        # 여기서 빈 dict로 덮어쓰지 않도록 "이번 턴에 새로 들어온 정보"만 전달합니다.
        #
        # step_code는 "프론트/스프링이 가진 현재 단계"이며, FastAPI는 이를 기준으로 라우팅합니다.
        # (checkpoint에 저장된 step이 있더라도, 요청으로 들어온 step_code를 우선합니다.)
        # 이번 요청 필드만 넘기면 체크포인트 병합 순서에 따라 requested_step_code가
        # 사라지거나 step만 남아 CHAT_5 입력이 CHAT_1(예산 파싱)으로 갈 수 있음 → 스냅샷과 병합.
        incoming: ChatState = {
            "requested_step_code": norm_step,
            "incoming_assistant_message": payload.assistant_text,
            "last_user_input": payload.user_text,
        }
        merged: ChatState = dict(incoming)
        snap = None
        try:
            if hasattr(chat_app, "aget_state"):
                snap = await chat_app.aget_state(config)
            elif hasattr(chat_app, "get_state"):
                snap = chat_app.get_state(config)
        except Exception:
            snap = None
        if snap is not None and getattr(snap, "values", None):
            merged = {**dict(snap.values), **incoming}

        logger.info(
            "FASTAPI merged_state=%s",
            json.dumps(
                merged,
                ensure_ascii=False,
                default=_to_jsonable,
            ),
        )
        result = await chat_app.ainvoke(merged, config=config)

        data = result.get("data", {}) or {}
        ai_response = result.get("ai_response")

        logger.info(
            "RESULT /ai/chat result=%s",
            json.dumps(
                {
                    "step": result.get("step"),
                    "is_completed": result.get("is_completed"),
                    "data": result.get("data"),
                    "ai_response": result.get("ai_response"),
                },
                ensure_ascii=False,
                default=_to_jsonable,
            ),
        )

        # CHAT_0~CHAT_5: 추천(CHAT_6) 전까지는 보통 data가 비어 있음.
        # 프론트가 "저장 완료" 여부를 쉽게 판단할 수 있도록 메시지를 채워줍니다.
        if norm_step in {
            "CHAT_0",
            "CHAT_1",
            "CHAT_2",
            "CHAT_3",
            "CHAT_4",
            "CHAT_5",
        }:
            if isinstance(data, dict) and not data and "error" not in data:
                data = {"message": "저장 완료, 다음 단계로 진행해주세요."}

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
                "data": users,
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"RDS 조회 실패: {str(e)}")

    @app.get("/health")
    async def health():
        """
        추천 엔진 설정 여부 — 배포 후 여기서 path_exists / will_use_pipeline 를 꼭 확인하세요.
        (로컬 .env 에만 있고 Docker ENV/secret 에 없으면 컨테이너에서는 비어 있음 → 파이프라인 미실행)
        """
        ra_path = os.getenv("RECOMMENDATION_ALGORITHM_PATH", "").strip()
        ra_entry = os.getenv("RECOMMENDATION_ALGORITHM_ENTRYPOINT", "").strip().lower()
        path_exists = bool(ra_path and os.path.isdir(ra_path))
        use_pipeline = bool(ra_path) and (not ra_entry or ra_entry == "pipeline:run_full_pipeline")
        return {
            "status": "ok",
            "recommendation_engine": {
                "path_configured": bool(ra_path),
                "path": ra_path or None,
                "path_exists": path_exists,
                "entrypoint": ra_entry or None,
                "will_use_pipeline": use_pipeline and path_exists,
            },
        }

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
