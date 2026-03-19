from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    select,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from datetime import datetime


class Base(DeclarativeBase):
    pass


class Product(Base):
    """
    실제 상품 테이블 스키마는 프로젝트에 맞게 수정 필요.
    지금은 "벡터 검색 가능한 형태"의 예시 모델만 제공합니다.
    """

    __tablename__ = "products"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    category: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    brand: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    embedding: Mapped[Optional[list[float]]] = mapped_column(Vector(1536), nullable=True)


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


async def ensure_pgvector(session: AsyncSession) -> None:
    # pgvector 확장 (권한이 없으면 실패할 수 있음)
    await session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))


async def vector_search_products(
    session: AsyncSession,
    query_embedding: list[float],
    *,
    top_k: int = 10,
    category: Optional[str] = None,
) -> list[ProductHit]:
    """
    pgvector cosine distance 기반 Top-K 검색 예시.
    - embedding 컬럼 및 인덱스(권장) 필요
    """
    stmt = select(
        Product.id,
        Product.name,
        Product.category,
        Product.brand,
        Product.price,
        (1.0 - Product.embedding.cosine_distance(query_embedding)).label("score"),
    ).where(Product.embedding.isnot(None))

    if category:
        stmt = stmt.where(Product.category == category)

    stmt = stmt.order_by(text("score DESC")).limit(top_k)

    rows = (await session.execute(stmt)).all()
    return [
        ProductHit(
            id=r.id,
            name=r.name,
            category=r.category,
            brand=r.brand,
            price=r.price,
            score=float(r.score),
        )
        for r in rows
    ]

