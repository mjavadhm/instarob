import asyncio
import httpx
from functools import wraps
from typing import Callable, Any, Coroutine

from app.core.logging import get_logger

logger = get_logger()

T = Callable[..., Coroutine[Any, Any, Any]]

def async_retry(max_retries: int = 3, delay: int = 2, backoff: int = 2):
    """
    A decorator for retrying an async function if it raises an exception.
    """
    def decorator(func: T) -> T:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            retries = 0
            current_delay = delay
            while retries < max_retries:
                try:
                    return await func(*args, **kwargs)
                except httpx.HTTPStatusError as e:
                    if e.response.status_code in [429, 500, 502, 503, 504]:
                        logger.warning(f"Retryable HTTP error: {e.response.status_code}. Retrying in {current_delay}s...")
                    else:
                        logger.error(f"Non-retryable HTTP error: {e.response.status_code}. Aborting.")
                        raise
                except (httpx.RequestError, asyncio.TimeoutError) as e:
                    logger.warning(f"Network error ('{type(e).__name__}'). Retrying in {current_delay}s...")
                except Exception as e:
                    logger.error(f"An unexpected error occurred in '{func.__name__}': {e}", exc_info=True)
                    raise # Re-raise unexpected exceptions immediately

                retries += 1
                if retries >= max_retries:
                    logger.error(f"Function '{func.__name__}' failed after {max_retries} retries.")
                    raise

                await asyncio.sleep(current_delay)
                current_delay *= backoff
        return wrapper
    return decorator
