import os


def get_database_url() -> str:
    """
    SQLAlchemy async URL을 반환합니다.

    우선순위:
    1) DATABASE_URL (예: postgresql+asyncpg://user:pass@host:5432/db)
    2) POSTGRES_* 조합
    """
    url = os.getenv("DATABASE_URL", "").strip()
    if url:
        return url

    host = os.getenv("POSTGRES_HOST", "localhost").strip()
    port = os.getenv("POSTGRES_PORT", "5432").strip()
    db = os.getenv("POSTGRES_DB", "").strip()
    user = os.getenv("POSTGRES_USER", "").strip()
    password = os.getenv("POSTGRES_PASSWORD", "").strip()

    if not db or not user:
        raise RuntimeError("DATABASE_URL 또는 POSTGRES_DB/POSTGRES_USER 환경 변수가 필요합니다.")

    return f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{db}"

