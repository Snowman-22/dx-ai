from __future__ import annotations

from typing import Any, Optional, Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import get_session_maker
from ..products_repo import Chat, Product, Recommendation
from .algorithm import RecommendationResult, rerank_and_filter


async def _fetch_candidate_products(*, limit: int = 200) -> list[dict[str, Any]]:
    """
    TODO: 실제로는 여기서
    - pgvector Top-K 검색(쿼리 임베딩 기반)
    - 또는 조건 필터링 + 벡터 검색
    을 수행해 candidates를 좁혀야 합니다.
    """
    session_maker = get_session_maker()
    async with session_maker() as session:
        rows = (
            await session.execute(
                select(
                    Product.id,
                    Product.name,
                    Product.category,
                    Product.brand,
                    Product.description,
                    Product.price,
                ).limit(limit)
            )
        ).all()

    return [
        {
            "id": r.id,
            "name": r.name,
            "category": r.category,
            "brand": r.brand,
            "description": r.description,
            "price": r.price,
        }
        for r in rows
    ]


async def generate_recommendation_result(*, user_info: dict[str, Any]) -> RecommendationResult:
    """
    DB 기반 추천 진입점.

    최종 목표:
    1) user_info로 검색 쿼리/필터 생성
    2) Postgres에서 후보 상품을 벡터검색으로 Top-K 조회
    3) ipynb 알고리즘(rerank_and_filter)로 재랭킹/패키징
    """
    candidates = await _fetch_candidate_products()
    return rerank_and_filter(user_info=user_info, candidates=candidates)


async def _get_or_create_chat(
    session: AsyncSession,
    *,
    conv_id: str,
    chat_title: Optional[str] = None,
) -> Chat:
    """
    conv_id 기준으로 Chat row를 조회하거나 생성.
    """
    result = await session.execute(select(Chat).where(Chat.conv_id == conv_id))
    chat: Optional[Chat] = result.scalar_one_or_none()

    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)

    if chat is not None:
        # 마지막 조회 시점을 end_date로 갱신
        chat.end_date = now
        # CHAT_RESULT 시점에 더 구체적인 제목이 들어왔다면 덮어씀
        if chat_title:
            chat.chat_title = chat_title
        await session.flush()
        return chat

    # 새 Chat 생성: 제목이 없으면 기본 패턴으로 생성
    if not chat_title:
        chat_title = f"가전/가구 추천 {now.strftime('%Y-%m-%d %H:%M')}"

    chat = Chat(
        conv_id=conv_id,
        chat_title=chat_title,
        start_date=now,
        end_date=now,
    )
    session.add(chat)
    await session.flush()
    return chat


async def save_recommendations_to_db(
    *,
    conv_id: str,
    recommendation_list: Sequence[dict[str, Any]],
    chat_title: Optional[str] = None,
) -> None:
    """
    LangGraph에서 생성한 recommendation_list를 Postgres Chat/Recommendation 테이블에 저장.

    - conv_id: 프론트/체크포인터에서 사용하는 conv_id (LangGraph thread_id와 동일)
    - recommendation_list: graph.node_chat_result에서 내려오는 리스트
      각 원소 예시:
      {"category": "...", "name": "...", "reason": "...", "estimated_price": ...}
    """
    if not recommendation_list:
        return

    session_maker = get_session_maker()
    async with session_maker() as session:
        async with session.begin():
            chat = await _get_or_create_chat(
                session,
                conv_id=conv_id,
                chat_title=chat_title,
            )

            for item in recommendation_list:
                reason = str(item.get("reason") or "") or None
                name = str(item.get("name") or "") or None

                # 현재 스키마상 products가 배열이므로, 단일 추천도 1개짜리 배열로 저장
                products: Optional[list[str]] = None
                if name:
                    products = [name]

                rec = Recommendation(
                    chat_id=chat.chat_id,
                    reason=reason,
                    products=products,
                )
                session.add(rec)


async def ensure_chat_metadata(
    *,
    conv_id: str,
    chat_title: Optional[str] = None,
) -> None:
    """
    추천이 아직 생성되지 않은 단계에서도 Chat row를 만들어 두기 위한 헬퍼.
    - 어떤 단계에서든 conv_id 기준으로 Chat이 존재하도록 보장.
    - chat_title 이 주어지면 제목을 업데이트(또는 생성 시 사용)한다.
    """
    session_maker = get_session_maker()
    async with session_maker() as session:
        async with session.begin():
            await _get_or_create_chat(
                session,
                conv_id=conv_id,
                chat_title=chat_title,
            )

