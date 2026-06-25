from __future__ import annotations

from contextvars import ContextVar, Token
from typing import Optional


_REQUEST_ID: ContextVar[str] = ContextVar("request_id", default="-")


def get_request_id() -> str:
    return _REQUEST_ID.get()


def set_request_id(request_id: str) -> Token[str]:
    return _REQUEST_ID.set(request_id or "-")


def reset_request_id(token: Optional[Token[str]]) -> None:
    if token is not None:
        _REQUEST_ID.reset(token)
