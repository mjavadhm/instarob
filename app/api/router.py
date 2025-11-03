from fastapi import APIRouter
from app.api.v1 import reels, health, frames

api_router = APIRouter()
api_router.include_router(reels.router, prefix="/v1/reels", tags=["reels"])
api_router.include_router(health.router, prefix="/v1/health", tags=["health"])
api_router.include_router(frames.router, prefix="/v1/frames", tags=["frames"])
