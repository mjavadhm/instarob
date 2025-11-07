from fastapi import APIRouter, Depends, HTTPException, status
from app.models.reel_in import ReelIn
from app.models.suggestions_out import SuggestionsOut
from app.services.reels_service import reels_service, ReelsService

router = APIRouter()

@router.post("/", response_model=SuggestionsOut, status_code=status.HTTP_200_OK)
async def analyze_reel(
    reel_in: ReelIn,
    service: ReelsService = Depends(lambda: reels_service)
):
    try:
        analysis_result = await service.analyze_reel_video(reel_in)
        return analysis_result
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except Exception:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred.")
