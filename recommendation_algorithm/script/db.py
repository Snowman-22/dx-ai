"""
db.py
- RDS PostgreSQL 직접 연결 (EC2 내부에서 실행 시)
- SSH 터널 불필요 (EC2 -> RDS 같은 VPC)
"""

import os
from contextlib import contextmanager

from sqlalchemy import create_engine


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required env var: {name}")
    return value


RDS_CONFIG = {
    "host": require_env("POSTGRES_HOST"),
    "port": int(os.getenv("POSTGRES_PORT", "5432")),
    "dbname": require_env("POSTGRES_DB"),
    "user": require_env("POSTGRES_USER"),
    "password": require_env("POSTGRES_PASSWORD"),
}


def create_db_engine():
    url = (
        f"postgresql+psycopg2://{RDS_CONFIG['user']}:{RDS_CONFIG['password']}"
        f"@{RDS_CONFIG['host']}:{RDS_CONFIG['port']}/{RDS_CONFIG['dbname']}"
    )
    return create_engine(url, pool_pre_ping=True)


@contextmanager
def get_engine():
    engine = create_db_engine()
    try:
        yield engine
    finally:
        engine.dispose()
