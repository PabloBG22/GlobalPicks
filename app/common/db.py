from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Optional

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.common.config import settings

_ENGINE: Optional[Engine] = None
_SESSION_MAKER: Optional[sessionmaker] = None


def get_engine() -> Engine:
    global _ENGINE
    if _ENGINE is None:
        if not settings.database_url:
            raise RuntimeError("DATABASE_URL is not set")
        _ENGINE = create_engine(settings.database_url, pool_pre_ping=True)
    return _ENGINE


def get_sessionmaker() -> sessionmaker:
    global _SESSION_MAKER
    if _SESSION_MAKER is None:
        _SESSION_MAKER = sessionmaker(
            bind=get_engine(),
            autoflush=False,
            autocommit=False,
            future=True,
        )
    return _SESSION_MAKER


@contextmanager
def db_session() -> Iterator[Session]:
    session = get_sessionmaker()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
