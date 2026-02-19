import sys
from loguru import logger
from contextvars import ContextVar

request_id_var = ContextVar("request_id", default="[no-request-id]")

def setup_logging():
    logger.remove()
    # Console sink
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "<yellow>rid:{extra[request_id]}</yellow> - <level>{message}</level>",
        enqueue=True,
        backtrace=True,
        level="INFO",
    )
    # File sink
    logger.add(
        "uvicorn.log",
        rotation="10 MB",
        retention="10 days",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | rid:{extra[request_id]} - {message}",
        enqueue=True,
        backtrace=True,
        level="INFO",
    )
    # Use a lambda to ensure the value is retrieved at log time, making it pickleable.
    logger.configure(patcher=lambda record: record["extra"].update(request_id=request_id_var.get()))

def get_logger():
    return logger

class RequestContextLogMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        from uuid import uuid4

        headers = {k.decode('latin-1'): v.decode('latin-1') for k, v in scope.get('headers', [])}
        request_id = headers.get("x-request-id") or str(uuid4())

        token = request_id_var.set(request_id)

        try:
            await self.app(scope, receive, send)
        finally:
            request_id_var.reset(token)
