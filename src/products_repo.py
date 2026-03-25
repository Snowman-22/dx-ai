from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from datetime import datetime


class Base(DeclarativeBase):
    pass


# --- RDS 스키마 기반 상세 조회용 모델들 ---
# NOTE: 실제 스키마가 아래 테이블/컬럼명과 다르면 조정이 필요합니다.


class ProductEntity(Base):
    """
    RDS의 product 테이블(이미지 스키마 기준) 매핑.
    RAG 단계에서 패키지 상세설명용으로 사용.
    """

    __tablename__ = "product"

    product_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    product_name: Mapped[str] = mapped_column(String(255), nullable=False)

    category: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    product_category: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    brand: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    original_price: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    discount_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    discount_price: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    is_subscribe: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)

    review_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    review_cnt: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    product_url: Mapped[Optional[str]] = mapped_column(String(2048), nullable=True)
    product_image_url: Mapped[Optional[str]] = mapped_column(String(2048), nullable=True)


class ProductSpecEntity(Base):
    __tablename__ = "product_spec"

    product_spec_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    product_id: Mapped[int] = mapped_column(Integer, index=True)

    width: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    height: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    depth: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    weight: Mapped[Optional[float]] = mapped_column(Float, nullable=True)


class SubscribePriceEntity(Base):
    __tablename__ = "subscribe_price"

    subscribe_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    product_id: Mapped[int] = mapped_column(Integer, index=True)

    month: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    price: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    contract_period_year: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    mandatory_period_year: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    visit_service_type: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    visit_cycle_month: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)


class ProductTagsEntity(Base):
    """product_tags 테이블: 상품별 리뷰 기반 태그."""
    __tablename__ = "product_tags"

    product_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tags: Mapped[Optional[list[str]]] = mapped_column(ARRAY(String), nullable=True)


class Chat(Base):
    """
    진단/추천 세션 메타데이터.
    DynamoDB의 conv_id(LangGraph thread_id)와 Postgres를 연결하는 용도.
    """

    __tablename__ = "chat"

    chat_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conv_id: Mapped[str] = mapped_column(String(255), unique=True, index=True)

    guest_session_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    chat_title: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    start_date: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    end_date: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    starterpackage_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    is_select_blueprint: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    blueprint_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)


class Recommendation(Base):
    """
    추천 결과 테이블.
    - chat_id: Chat 테이블 FK
    - reason: 추천 이유(텍스트)
    - products: 추천 상품 목록 (간단히 문자열 배열로 저장)
    """

    __tablename__ = "recommendation"

    recommendation_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    chat_id: Mapped[int] = mapped_column(Integer, ForeignKey("chat.chat_id"), nullable=False, index=True)
    reason: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    products: Mapped[Optional[list[str]]] = mapped_column(ARRAY(String), nullable=True)


@dataclass
class ProductHit:
    id: int
    name: str
    category: Optional[str]
    brand: Optional[str]
    price: Optional[float]
    score: float


async def vector_search_products(
    session: AsyncSession,
    query_embedding: list[float],
    *,
    top_k: int = 10,
    category: Optional[str] = None,
) -> list[ProductHit]:
    """
    시맨틱(벡터) 상품 검색. 현재 RDS `product` 테이블에는 embedding 컬럼이 없어
    DB 벡터 검색은 수행하지 않고 빈 목록을 반환합니다.
    (나중에 `product`에 pgvector 컬럼을 두면 여기서 조회하도록 연결하면 됩니다.)
    """
    return []

