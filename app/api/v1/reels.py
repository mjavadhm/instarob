from fastapi import APIRouter, Depends
from app.models.reel_in import ReelIn
from app.models.reel_out import ReelOut
from app.services.reels_service import reels_service, ReelsService

router = APIRouter()

@router.post("/reels", response_model=ReelOut, status_code=201)
async def create_reel(
    reel_in: ReelIn,
    service: ReelsService = Depends(lambda: reels_service)
):
    return await service.process_reel(reel_in)
