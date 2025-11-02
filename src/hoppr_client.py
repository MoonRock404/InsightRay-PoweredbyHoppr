from functools import lru_cache
from .config import get_api_key

# Prefer PyPI name 'hopprai'
try:
    from hopprai import HOPPR, HOPPRError
except Exception:
    from hopprai import HOPPR  # type: ignore
    class HOPPRError(Exception):
        pass

__all__ = ["get_hoppr", "HOPPR", "HOPPRError"]

@lru_cache()
def get_hoppr() -> HOPPR:
    api_key = get_api_key()
    return HOPPR(api_key)
