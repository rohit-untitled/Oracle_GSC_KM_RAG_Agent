import os
from typing import Optional

def require_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Required env var {name} not set")
    return v

def get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    return os.getenv(name, default)