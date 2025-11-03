from fastapi import APIRouter
from app.api.v1 import reels, health

api_router = APIRouter()
api_router.include_router(reels.router, prefix="/v1", tags=["reels"])
api_router.include_router(health.router, prefix="/v1", tags=["health"])
