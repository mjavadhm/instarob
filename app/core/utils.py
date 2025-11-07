import asyncio
from functools import wraps
from typing import Callable, Awaitable, Any
from openai import APIConnectionError, APITimeoutError, APIStatusError
from app.core.logging import get_logger

logger = get_logger()

def async_retry(max_retries: int = 2, delay: int = 1):
    def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except (APIConnectionError, APITimeoutError) as e:
                    if attempt == max_retries:
                        logger.error(f"Attempt {attempt + 1}/{max_retries + 1} failed with network error. Max retries reached.")
                        raise
                    logger.warning(f"Attempt {attempt + 1}/{max_retries + 1} failed with network error: {e}. Retrying in {delay}s...")
                    await asyncio.sleep(delay)
                except APIStatusError as e:
                    if e.status_code >= 500 and attempt < max_retries:
                        logger.warning(f"Attempt {attempt + 1}/{max_retries + 1} failed with server error {e.status_code}. Retrying in {delay}s...")
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"Attempt {attempt + 1}/{max_retries + 1} failed with status {e.status_code}. Not retrying.")
                        raise
                except Exception as e:
                    logger.error(f"An unexpected error occurred in '{func.__name__}' on attempt {attempt + 1}.")
                    raise
        return wrapper
    return decorator
