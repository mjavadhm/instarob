from fastapi import FastAPI
from app.api.router import api_router
from app.core.logging import setup_logging, RequestContextLogMiddleware

setup_logging()

app = FastAPI(title="Reels Input Service")
app.add_middleware(RequestContextLogMiddleware)
app.include_router(api_router, prefix="/api")

@app.get("/")
async def root():
    return {"message": "Reels Input Service is running"}
