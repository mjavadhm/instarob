from fastapi import APIRouter, Depends, HTTPException
from app.models.reel_in import ReelIn
from app.models.frame_analysis import FrameAnalysis
from app.services.reels_service import reels_service, ReelsService

router = APIRouter()

@router.post("/reels", response_model=FrameAnalysis, status_code=200)
async def analyze_reel(
    reel_in: ReelIn,
    service: ReelsService = Depends(lambda: reels_service)
):
    try:
        analysis_result = await service.analyze_reel_video(reel_in)
        return analysis_result
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")
